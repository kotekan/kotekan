# GuoWang per element: sensitivity, and the nav-bit wipe

Why the beam cube shows no GuoWang satellite per element, what the per-element signal really
is, and the plan that makes GuoWang usable for hemisphere beam mapping. Measured 2026-11-04 on
the 11-04 cube rows (cx19 gnss0 sender), the cf05 window reductions and the 10-30 baseband.

## 1. The question

The cube session reported per-element |coh|^2/incoh at noise for every GuoWang row (R <= 1.7
against 50-88 for GPS L5 in the main lobe) and read it as deep sidelobes or low per-element
sensitivity. GuoWang is a LEO constellation with polar planes, so it would close the northern
hole in the GNSS beam maps if it mapped per element. Does it?

## 2. What the cube rows contain

The rows come from the gw_1202 chain, which tracks only codes the library knows. On 11-04 every
pass within 1.4 deg of the boresight (nine satellites, 0.15-1.4 deg) had zero rows within 150 s
of closest approach; each was that satellite's first library pass, confirmed by the very window
that observed it. Shared-code satellites were tracked only while their known code-mate held the
table record, even with the library's satellite 44 deg below the horizon. So every GuoWang row
is a far-sidelobe satellite, and per-element power at noise is expected there.

The one known satellite that did cross close in, DIGUI-124 (PRN 1158) at 9 deg on 00:53Z, gave a
per-element excess over the dark probes of at most +1.2 dB (best element +2.3 dB), 0 dB beyond
30 deg. GPS L5 shows +7 dB per element at 52 deg off axis in the same cube
(CHORD_CUBE_ARCHIVE.md).

## 3. The two reasons

**GuoWang is not brighter per element.** The 10-30 baseband (DIGUI-94, el 32, 43 deg off axis,
`bb1030/csk_bb_test.py`) gives steering eigen-ratios of 89-117 per main-lobe channel, i.e.
128 rho + 1 with the Marchenko-Pastur median near 0.9: rho ~ 0.7 per element per 1 ms per
195 kHz channel, ~39 dB-Hz per element over the eleven main-lobe channels. GPS L5's +7 dB at
52 deg, with a 10 ms coherent record and ~2 % of a BPSK(10) signal per channel, is ~43 dB-Hz.
Both carry +-3 dB; the sign is clear. The ITU filing's maximum pfd (-129.3 dB(W/m2/MHz) at
56 deg elevation) would put GuoWang 8 dB above L5; the satellites radiate well below it.

**The record despread is matched to pilots.** A cube record is a 10.5 ms coherent despread of
the ranging half. GuoWang's ranging half carries a sign bit every 2 ms frame (the low-rate
channel, [PAYLOAD.md](PAYLOAD.md)) and its data half is a different code, so a record keeps
about rho/2 instead of a pilot's 10 rho: a 13 dB penalty, 16 dB counting the unused data half.
Net GuoWang sits 17-20 dB below L5 in the cube's units for equal sky power. At 43 deg off axis
rho/2 predicts +1.3 dB, 0 dB deeper: what the rows show.

GuoWang's advantage for mapping is coverage (about a hundred known satellites crossing the
hemisphere in minutes each, polar planes included), not brightness.

## 4. The eigen-beam windows are not weak

The cf05 reductions of 24 windows (eigen beam of 128 inputs, lobe channels stitched) show
(amp/noise)^2 per 2 ms frame between 31 and 35 dB from 12 deg out to 0.3 deg: that is the
2046-chip Gold code's own sidelobe floor (~33 dB), so the metric saturates. The beam signal is
far above noise across a window, which is why the decodes are clean.

## 5. The plan: wipe the code-half signs by prediction

The ranging-half sign stream is the low-rate channel: K=7 rate-1/2 convolutional (171/133), 3 s
= 375-bit subframes whose content is almost fully determined in advance (SOW = the subframe's own
start second, satellite ID from the library, constant common words, slow per-satellite words,
CRC-24Q over the rest; pages 2-4 filler). A predictor is the encoder run forward. It has zero
latency, which the alternative (decoding the bits) does not.

1. **Offline prototype (done 11-04, `sign_predict.py` on cf05 in `/var/tmp/kvand-trackbeam`):**
   each subframe is built from time (page = (SOW/3) mod 4 + 1, SOW = the subframe's BDT start
   second) and the satellite's four page payloads learned from any clean decode (they were
   identical across every repeat of both decoded satellites), the CRC-24Q computed, the block
   encoded (K=7, 171/133, G2 not inverted) at 250 symbols/s, one symbol per two 2 ms frames.
   Against the symbols measured by `slow_bits2.py` from the window reductions
   (`/var/tmp/kvand-gw1202/reduce/w078` DIGUI-H24-4, `w090` DIGUI-H84-1), the clean subframes
   agree at 98.3 % median (minimum 97.6 %) with the 12 trailer bits unknown, 98.8 % with their
   most common value. The trailer is the last 96 ms of each 3 s period, after the CRC, and
   varies; it costs ~1.5 % of the signs. A satellite sometimes switches to all-zero subframes
   (seen at the end of H24-4's pass): their symbols are constant, so they cost coherence only
   while the predictor still assumes page content. Subframes the decoder could not read still
   agree at ~83 %, i.e. the predictor is right where the decode failed.
2. **Node side (design):** the despread kernel reads the replica chip as
   `code[code_offset + (chip mod code_len)]` (`chip_gather3`, `cudaGnssReplicaDevice.cuh`) with
   the job's `cp_ref` reduced mod `code_len` on the host and the unwrapped phase `cp0 + n*cps`
   kept there (`DespreadJob`, `cudaGnssDespreadKernel.hpp`). Add to the job an absolute frame
   index at the reference sample, `frame_ref = nav_frame_offset + floor((cp0 + n0*cps) /
   code_len)`, and a per-slot sign table (one bit per 4 ms symbol, 2^19 symbols = 35 min,
   64 KB); the kernel multiplies each gathered chip by
   `sign[(frame_ref + floor((cp_ref + dn*cps - d) / code_len)) >> 1]`. The symbol parity is
   absolute because a subframe starts on a BDT second that is 0 mod 3, so frame 0 of a subframe
   is even. The host fills the table from the templates with a C++ twin of the prototype beside
   the code generator (CRC-24Q, K=7 encoder, SOW from `Telescope::to_time_ns` with
   BDT = UTC + 4 s) and refreshes it ahead of the records it serves. The broker's seed for the
   slot carries `nav_frame_offset` (the satellite's frame count at the seed's code phase, from
   its TLE range, theta and UTC) and the four page templates, which the library stores per
   satellite once a window decode has read them (`parse_slow.py` output; pages 2-4 are the same
   filler on every satellite, page 1 differs in the ID and a few words).
3. **Expected gain:** each 10.5 ms record becomes a coherent sum of five correctly signed
   halves: 2.5 rho instead of rho/2, +7 dB, 3 dB short of a full pilot. Include the 10 ms
   per-frame phase-step pattern the payload study found, or it costs 1-2 dB.
4. **Wrong bits are a level, not a gain error.** A sign is common to all 128 elements, so a
   wrong one scales a record down (one wrong half of five leaves 3/5 of the amplitude; the
   phase flips only with three wrong). For an error fraction f the map level drops by
   (1 - 2f)^2 in power, 0.3 dB at 2 %, identical on every element; relative gains and phases are
   untouched. The track beam decodes the true signs of the strong satellites, so f is measured
   live. Code tracking is unaffected (the DLL discriminator is sign-blind).

**Built 11-05 00:30Z (branch kv/gw-navbits, 881e86f28):** `guowang1202Nav` (generator + test),
the tracker's seed fields `nav_frame_at_ref` / `nav_pages`, the record frame index from the
unwrapped phase, `PsiJob::nav` applied in `psi_hop` by the wrap count, the broker's exact frame
arithmetic at the seed birth (`gw_nav_frame`, moved with the DLL trim) and the table's
`nav_pages` from the library. GPU-verified: `cuda_gnss_psi_bench tiled` shows synthesis = sign x
plain on every hop with alternating signs (24576 hops, exact); the despread test and the fused
bench are unchanged with the signs off.

**Finding that changes the template supply (11-05 00:20Z):** the two satellites whose pages
were constant are the inferred-orbit H ones, which the table never writes records for. Three
catalogued satellites decoded from their 11-04 main-lobe windows (DIGUI-121, -139, -142) carry
CHANGING content in every page: 20-36 of 316 bits per 12 s cycle, a union of 30-120 positions
per page over a pass, and 25-35 % of a page differs between satellites. So for catalogued
satellites a page template must come from the satellite's own recent decode, and ages: ~92 %
of the bits hold over a minute, less over hours. Expected wipe gain with a live template +5 dB,
with a stale one +3 to +4 dB, +7 dB for the H satellites. Next: decode the low-rate pages of the
track beam's targets in the cf05 loop every cycle (it has the code-half correlations) and feed
them to the library/table, so the strongest satellites always carry a fresh template.

Fallback that needs no bits: accumulate element x conj(reference element) per 1 ms code half.
The common sign cancels and the relative response adds coherently, but it is a quadratic
estimator, efficient only while rho per half is above ~0.5 (inside ~45 deg); in the far
sidelobes the predicted-sign linear sum wins by 10 dB or more.

Later, the data half: the CSK symbols are the constellation-wide stream, known for every
satellite once decoded from the strongest one (with each satellite's own 9-115 s uplink lag).
Using it per element needs a despread that lags the ring head by the decode latency and a
posted symbol table. It buys the last 3 dB; leave it until the sign wipe is in.

Across records the cube's window sums already wind with the residual carrier for every
constellation (CHORD_CUBE_ARCHIVE.md); fixing that per-record rotation is independent of the bits.

Sources: memory chord-guowang-per-element-sensitivity-20261104 (scripts cube_pass.py,
lam_series.py in the session scratchpad), chord-guowang-payload-20261104, chord-handoff-20261103-cube.
