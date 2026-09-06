# CHORD GNSS beam-cube archive — raw, L0, and the trim ladder

The beam cube is the (PRN slot × subband bin × element) pair of sums — incoherent |A|² (the
beam) and coherent Σ A·rot (the arc) — over one ~1.007 s window, produced by every
`GnssGpuRecordAssemble` instance in the fleet (15 per node, 14 on cx44 = 89 senders) and
PUSHED to cf06 (`bufferSend` → `bufferRecv:11070` → `rawFileWrite`). This document is the
archive that receives it: where the bytes are, what each level means, how to read it, and how to
tell a hole from a loss. The plan it implements is `fixtures/recording_plan_20260905.md`.

```
/mnt/cs00/data/kvand/gnss_cube/
  raw/gnss_cube_NNNNNNN.raw          rawFileWrite bundles, 900 frames each (~10 s of fleet)
  l0/manifest.json                   which raw files have been folded (size, mtime, epoch)
  l0/<pointing>/<sender>/<YYYYMMDD>.h5      L0: every live (window, slot) cell, float32 as shipped
  rung12/<pointing>/<sender>/<YYYYMMDD>.h5  exact 12-window (~12 s) sums
  rung60/<pointing>/<sender>/<YYYYMMDD>.h5  exact 60-window (~60 s) sums
```

`<sender>` is `cx19_gnss0_b2b_n2assemble` — the node and the stage, i.e. the chain's full
address as it travels in the frame. `<pointing>` comes from `config/pointings.yaml`, a DECLARED
epoch table (a map builder must never sum two pointings by accident; undeclared time lands under
`unknown/`, visibly).

## 1. Sizes, measured 2026-09-05

| level  | logical              | on cs00 (compresses at rest) | ratio to raw |
|--------|----------------------|------------------------------|--------------|
| raw    | 9 MB/s = 780 GB/day  | ~260 GB/day (3.0× at rest)   | 1            |
| L0     | ~190 GB/day          | ~185 GB/day (≈1.03×)         | 0.7 on disk  |
| rung12 | ~16 GB/day           | ~16 GB/day                   | 1/16         |
| rung60 | ~3.3 GB/day          | ~3.3 GB/day                  | 1/80         |

⚠️ THE RAW ARCHIVE IS ALREADY 3× COMPRESSED BY THE FILESYSTEM (the zero pad and the dead
slots), so L0 buys STRUCTURE — per-sender day files, a UTC axis, indexable rows — much more than
it buys space. The space is in the rungs. Retention is a P6 decision (KV); nothing here deletes.
cs00 had 207 TB free on 2026-09-05: raw alone is ~2 years.

L0 keeps 12.3 live slots per window on average (11.8–12.7 per chain), out of 24–32 slots.

## 2. What is in a frame, and what is NOT

`lib/stages/gnss/gnssRecord.hpp` "BEAM-CUBE FRAME" is the layout. Two things worth restating:

- **A slot with `n_rec == 0` is SILENCE, not a zero.** The producer leaves prn = 0 and every
  array zero for a PRN that did not run this window; L0 drops those rows entirely. Reading a
  zero row as "no power" is the one measurement a transport must never manufacture.
- **The frame carries linear SUMS with their term counts.** `incoh/w` is a mean power;
  windows ADD; decibels do not. Nothing in this archive is a dB or a mean.
- **v2 frames (archived 2026-09-05 17:03 → the v3 cycle) carry no UTC.** The time axis is
  F-engine samples (`wstart`, `sample_rate`) and the epoch of sample 0 lived in the node config
  (`frame0_utc`, GPS-week-rollover corrected). **v3** (CUBE_VERSION 3, header +8 bytes: `double
  utc0` at 96, chain string at 104, frame 101 208 B) puts it in the frame. The reader
  (`gnss_cube_read.py`) accepts both; the compactor needs `--utc0` for v2 and GATES it (§4).

## 3. The tools

All Python here runs with `/home/kvand/gnss/venv/bin/python` (h5py, numpy; venv-ft has no
h5py) **on cf06** with absolute paths — the nodes are production trackers.

- `python/scripts/gnss/gnss_cube_read.py ls|head|npz` — the raw files, frame by frame.
- `python/scripts/gnss/gnss_cube_compact.py`
  - `compact` — raw → L0, idempotent through `l0/manifest.json`; only CLOSED raw files (every
    file but the newest) unless `--include-open`. ~3400 frames/s on cf06 = ~38× real time.
  - `ls [--gaps]` — completeness per L0 file with every hole ATTRIBUTED (§5).
  - `rung --n 12|60` — L0 → exact sums per (absolute block = idx // n, slot, prn). A PRN swap
    inside a block yields two rows (`n_win` < n), never a mixed sum.
  - `verify --raw F.raw` — N random live cells re-read from L0 and compared BIT-FOR-BIT.
- `scripts/gnss/cubecompact_up.sh` (cf06 stack style, restarts the loop) →
  `cubecompact_loop.sh`: `compact` every 300 s; after 00:20 UTC, rungs for YESTERDAY once.
  Log `/tmp/gnss_cubecompact.log` on cf06. Refusals ("REFUSED") in that log are the epoch gate.

## 4. The epoch gate (why a v2 file can be REFUSED)

rawFileWrite closes a file within seconds of its last frame — measured `mtime − utc(last
window)` = +1.2 … +2.6 s on 09-05. So `|utc(last window) − mtime| > 60 s` means the epoch does
not describe this file: wrong `--utc0`, or the F-engine restarted since (new frame0). The
compactor refuses the file rather than dating it wrong; an L0 day file also refuses a second
`utc0` (two epochs in one day = an F-engine restart; keep them apart deliberately).

time0 for the 2026-09-05 archive: **1788541059.000002870** s (frame0_nano 1169225859000002870
+ GPS-week rollover; identical on all six nodes, exposed at `/telescope/time0_ns`).

## 5. Holes are three different faults

`ls` never adds them together:

- **sender dropped N** — the frame's cumulative `dropped` rose across the gap: the assembler's
  output buffer was full, the tracker outran the archive. The node's fault; sized exactly.
- **host-wide (node down/restart)** — every OTHER sender file on that host that covers the gap
  has a gap overlapping it by ≥80% of the longer one: a node_up (≈75 windows), a fleet cycle, a
  wedge. Not a recording-chain loss. (The first rule, "same gap ±2 windows", called every one
  of 2354 real host-wide holes DOWNSTREAM on 09-06: GPU 1's instances come up ~5 windows after
  GPU 0's, and a day file that ends before the gap has no vote — only files COVERING [a, b] do.)
- **DOWNSTREAM (transport/archiver)** — one sender, counter flat: `bufferSend drop_frames`, the
  network, or the archiver. The recording chain's fault and the only one worth chasing; `ls`
  prints these even without `--gaps`.

09-05/06 state (13:20 UTC 09-06): 89/89 senders, 2354 holes, all host-wide (two cx19 restarts:
17:44 09-05 fleet cycle, 06:12 09-06), 0 sender drops, 0 downstream. First pass: 1156 holes, all host-wide (cx19's 15 senders × 76–77 windows
at 17:44:14 — the fleet cycle), 0 sender drops, 0 downstream.

## 6. L0 / rung contents

L0 (`format = gnss_cube_l0`), extensible datasets, one row per LIVE (window, slot):

```
win/   idx i8, utc f8, wstart0 i8, wstart1 i8, dropped i8 (cumulative), n_live i4,
       freq_id_lo/hi i4[n_bin]           -- one row per window received (completeness)
rows/  idx i8, slot i2, prn i2, n_rec i4, n_reanchor i4, phi0 f8,
       w f4[n_bin], energy f4[n_bin], coh c8[n_bin,n_elem], incoh f4[n_bin,n_elem]
attrs  chain, sender, gpu, n_bin, n_elem, bin_width, win_samples, sample_rate, utc0,
       utc0_source (frame|flag), pointing, day, cube_versions, source_files, units
```

Rung (`format = gnss_cube_rung`, `rung_n`, `ref_elem`): per (block, slot, prn) row —
`idx0/idx1`, `utc0/utc1`, `n_win`, `n_rec`, `n_reanchor`, `w`, `energy`, `incoh` (sums),
`cohpow = Σ_windows |coh|²`, and `cohref = Σ_windows coh[e]·conj(coh[ref_elem])`. The last
exists because a coherent sum ACROSS windows is meaningless — `phi0` is a per-window reference
— while the cross-element product is blind to that common phase and is exactly the per-element
relative response (amplitude AND phase) the phased-array calibration wants. `ref_elem` defaults
to the strongest mean-incoh element of the file and is recorded. `blk/n_win_present` says how
many windows each block actually had.

## 7. Deploying v3 (archiver FIRST, then nodes — the two sizes cannot coexist)

`bufferRecv` has one `frame_size`; a v2 sender into a v3 archiver (or vice versa) is CLOSED
with a frame_size message and delivers nothing (the offline gate shows a v2 producer in a
v3-sized buffer losing every other window). The v3 binary refuses a v2 config loudly
(`cube_buf needs a 101208 B frame`). Order, at the next fleet cycle:

1. me: `mv build/kotekan/kotekan.v3cube_20260905 build/kotekan/kotekan` (same for
   `build_nodpdk`), regenerate `config/generated/` (the generator reads CUBE_HEADER_BYTES from
   the hpp → frame_size 101208 everywhere), `ssh cf06 cubearch_up.sh` (archiver on v3);
2. KV: `node_up.sh` the six nodes;
3. gate in minutes: `gnss_cube_read.py head <newest raw>` shows `version 3` and a `utc(wstart0)`
   that matches the wall clock; `gnss_cube_compact.py` then needs no `--utc0` for those files.

Offline gate for the code itself: `scripts/gnss/cube_e2e.py --binary <nodpdk binary>` on cf06
(PASS on the v3 binary; on the v2 binary it fails on version, utc0 and the lost windows).

## 8. The map builder (P5): `gnss_beam_cube.py build --source l0`

The healpix builder reads the archive directly — L0 or, ~10× faster and identical in result,
`rung12/` (each 12-window block is one sample). `--archive` is a POINTING directory
(`.../rung12/p0_dec40p73`); it refuses to mix two. Per (sender file, 5-min, bin, element) it
takes the **median over all slots** as the pedestal F (11 of 12 slots are noise at any moment,
and the cube frame carries no probe flag), then accumulates `(P − F)/F` — **pedestal units** —
into the (subband, pixel, element) cells. Samples below F are dropped by default (the elem
builder's convention; leaves a ~0.4σ floor), `--unbiased` keeps them signed.

⚠️⚠️ **THE BINS ARE NOT IN A COMMON UNIT.** `incoh` is Σ|G/E_c|², so each bin's scale is
∝ 1/E_c² of its own channel: for a BPSK(10) signal the centre channels sit ~20 dB below the
edge channels *in these units*, at equal sky power. A raw sum over bins is just the edge bins
— it put a 21 dB "peak" at the horizon and made five instances look "per-instance bright"
(they were the 6-bin vs 7-bin channel subsets, not a bug). Normalise per bin FIRST (P/F), sum
after. The same applies to anything downstream that reads `incoh`/`coh` across bins.

Result on 09-05 and 09-06, every chain: peak 4–8° off axis at +15–21 dB over the >15° median;
the far field is real spillover (PRN 28 at 52° off axis is +7 dB in a single centre bin), not
floor. Timing on cf06: ~15 s/chain from rung12, 80 s for a full 13 h day of 8 chains.

Coherence check: `|coh|²/(w·incoh)` ≈ 1/w (0.010) for noise; near boresight at +9.6 dB it
measures 0.006 — **the per-window coherent sum WINDS** (residual carrier > 1 turn/s within the
~1 s window), so the cube's coherent arc is currently incoherent and `incoh` is the beam. A
usable arc needs a per-record (10 ms) rotation with the tracked carrier before the window sum.

Viewer: `gnss_beam_cube.py export <masters> --nside 32` → `fixtures/beamcube/web/`, served by
`scripts/gnss/beamview_up.sh` on cf06 (http://cf06:8877/, port 877 is privileged). Every day in
one `index.json` must share `units` and `pointing`; the page refuses to sum a day that differs
and names it. The 08-25..09-02 elem-archive export (raw power units) lives in `web_elem/`.
