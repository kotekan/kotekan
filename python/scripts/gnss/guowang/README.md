# GuoWang code search (#218)

China's GuoWang LEO satellites (CelesTrak `GROUP=hulianwang`, "HULIANWANG DIGUI-nn") broadcast a
continuous BPSK(10.23)-like signal on 1268.52 MHz (nulls at +-10.23 MHz) and a ~2.05 Mcps lobe on
1202.025 MHz. They burst through the dish beam (az 180, el 62.70) at ~0.3 deg/s. These tools plan a
baseband capture through such a pass and search the capture for the spreading code.

Python: `/home/kvand/gnss/venv/bin/python` (numpy, scipy, skyfield). Run anything heavy on cf06 under
`nice -n 19` with `OMP_NUM_THREADS=4 GUOWANG_FFT_WORKERS=4`; the home directory is shared.

| file | what |
|---|---|
| `guowang_passes.py` | pass planner: passes near boresight, capture windows, validation against burst lists |
| `guowang_search.py` | the code search on complex baseband (or stitched channels) |
| `guowang_codes.py` | candidate code families (B3I, all B3I-structured codes, B1I, all B1I-structured, L3OC) and Berlekamp-Massey |
| `guowang_synth.py` | synthetic captures and the validation table |
| `test_guowang.py` | checks (time conversion, codes against the repo's code checks, a short synthetic search) |

## Plan a capture

1. The day before, refresh the TLEs (CelesTrak serves a group once per 2 h per client; `--fetch` does
   nothing when the cache in `~/.cache/guowang` is younger than that):

       guowang_passes.py passes --fetch --start 2026-11-02 --days 1

2. List the windows: one per in-plane train (consecutive satellites ~12.7 min apart), centred on the
   train's closest pass, which is the strongest burst:

       guowang_passes.py capture --start 2026-11-02 --days 2 --length 80 --per-day 3

   Columns: the closest pass (`sep` from boresight, `el`, angular `rate`), the geometric Doppler and
   its rate at 1268.52 and 1202.025 MHz at closest approach (`-f range_rate / c`; the transmitter's own
   offset is unknown), the train, the launch (COSPAR) and other passes inside the window.
   Satellites seen silent in the beam (`SILENT` in the script: DIGUI-01 and -03, both of the first
   polar launch 2024-240) are skipped; a `?` marks a suspect batch. Prefer passes < 1 deg.

3. Timing. The burst is > 2x for 25-50 s around the closest approach (FWHM 7-12 s). An 80-100 s
   capture centred on it covers the rise and fall; the train's neighbours are 12.7 min away. Pass
   times are UTC; a capture stamped in GPS time is 18 s ahead.

## What the capture must contain

- Both lobes: 1258.3-1278.7 MHz (better 1248-1292, the first sidelobes) and 1200.1-1204.1 MHz (better
  1194-1210). Complex samples at >= 2x the bandwidth kept: the search resamples the 1268.52 lobe to
  20.46 MSPS (2 samples per chip) and needs content to +-10.23 MHz; the chip-rate line is measured at
  4 samples per chip.
- At least 0.5 s at the burst peak; 2-5 s helps the weak-signal paths; the whole 80-100 s lets the
  code Doppler be tracked through the pass.
- One element is enough at the peak of a burst (x3-50 band power per dish: +1.6 to +8 dB per sample
  over 25 MHz, C/N0 ~75-82 dB-Hz). A beam formed toward the TLE track (`dop*` and the pass geometry
  give it to ~0.05 deg) adds up to ~+18-21 dB and reaches the satellites in the dish sidelobes. Keep a
  few single elements as well: a beam steered off the satellite loses it.
- The sample clock and the absolute time (GPS or UTC, say which), and for channelised data the channel
  centre frequencies and whether the samples are conjugated (the CHORD F-engine's are: `--conj`).

## Run the search

    guowang_search.py run capture.npz --lobe b3 --out res_b3          # x, fs, fc in the npz
    guowang_search.py run capture.npy --fs 25e6 --fc 1268.52e6 --t0 30 --t-len 2 --lobe b3
    guowang_search.py run chans.npz --conj --lobe l3                   # chan, freq_hz, fs_chan

Pick `--t0` at the burst peak. The steps, in the order printed:

1. **chip rate**: the line at the chip rate in `x(t) x*(t - d)`, needs no code; its offset from 10.23
   Mcps gives the code Doppler.
2. **period**: the lag of the strongest peak of the segmented delay-and-multiply (0.05-2.2 ms). The
   `coherent` column is the same lag from the whole record: near the segmented value only if the
   code carries no data (then the spectrum has lines at 1/period).
3. **carrier**: the line at 2x Doppler in `x^2` (4x in `x^4` if `x^2` shows nothing: QPSK), searched
   over +-30 kHz and +-200 Hz/s, refined coherently. A DIGUI pass near boresight shows -115..-140
   Hz/s; a MEO BeiDou in the same band shows < 1 Hz/s.
4. **fold** (needs the carrier): the record folded at the period, one template plus a sign per
   sub-block, the code epoch found where the signs change, the chips read from it. Then
   `identify` correlates the chips with every code of each family (`rho` ~0.9 for the right code,
   ~0.05 for the rest), `sign changes ... NH20 agreement` tests for the B3I/B1I secondary code, and
   Berlekamp-Massey gives the linear complexity per 600-chip window (26 and divisible by `B3I G1` /
   `B3I G2` for any B3I-structured code; ~300 for a memory code; for an LFSR code of other
   polynomials, a small L and the connection polynomial itself). The chips go to `<out>_chips.npy`.
   The fold needs ~-15 dB per sample (25 MSPS) for 0.5 s of data, -20 dB for 2 s, -25 dB for 4 s
   (`--t-blind`).
5. **pcps**: parallel-code-phase acquisition per family. With a carrier the stream is wiped and each
   code is searched at the known Doppler (8192 codes of `b3i_all` over 20 periods in ~25 s); without one
   the 63 PRNs are searched over the full Doppler / rate grid (~4 min for 100 periods); the large families need
   `--full-family-blind` (hours on CPU). `margin` is dB above the 1e-3 false-alarm level. A strong
   signal passes it on every code of a family through cross-correlation (its code need not be in the
   family at all), so a code is `IDENTIFIED` only when it stands >= 6 dB over the family's runner-up.
6. **verify**: the best code's delay tracked over the record gives the received chip rate, the period
   and the code Doppler, next to the carrier Doppler; equal = code and carrier from one clock.

Output: the log, and `<out>.json` with every number.

## Validate

    guowang_synth.py validate --out /path/val     # the table (~35 min on cf06): SNR sweep, Gold, random code,
                                                  # 4-bit, off-clock chips, noise, full grid, 1202, channelised
    guowang_synth.py validate --only 1202         # the cases whose name contains "1202"
    test_guowang.py                               # ~10 s
