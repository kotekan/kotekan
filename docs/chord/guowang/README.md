# GuoWang (SATNET LEO) at CHORD

China's SATNET LEO augmentation navigation system transmits a BPSK ranging signal at
1202.025 MHz (and others) from a growing constellation of polar and inclined planes. CHORD sees
it in the 1202 lobe, tracks it with a broker chain like any GNSS, and reads its messages. The
work started 2026-10-28; everything below is dated.

| Document | What it holds |
|---|---|
| [SIGNAL.md](SIGNAL.md) | Identity, modulation (2 ms frames, Gold codes, MCSK), code sharing, timing (theta), what was ruled out |
| [PAYLOAD.md](PAYLOAD.md) | The low-rate BPSK channel (decoded) and the high-rate constellation-wide CSK stream; what is predictable |
| [PIPELINE.md](PIPELINE.md) | Library, ephemeris table, the gw_1202 chain, confirmation windows and the cf05 loop, theta sweep, track beam, where the data is |
| [PER_ELEMENT.md](PER_ELEMENT.md) | Per-element sensitivity (not brighter than GPS L5), why the cube sees nothing, and the nav-bit wipe plan |
| [../../CHORD_GUOWANG_SIGNAL_20261103.md](../../CHORD_GUOWANG_SIGNAL_20261103.md) | The 11-03 signal note (historical, kept) |
| [../../CHORD_GUOWANG_BEAMDUMP_PLAN.md](../../CHORD_GUOWANG_BEAMDUMP_PLAN.md) | The beam-dump plan (historical, kept) |
| [../../CHORD_GUOWANG_TRACK_BEAM.md](../../CHORD_GUOWANG_TRACK_BEAM.md) | The continuous stream: steered beams per tracked satellite, frame format v2, arming, decode |
| [../../CHORD_TAUA_GUOWANG_20261029.md](../../CHORD_TAUA_GUOWANG_20261029.md) | GuoWang in the Tau A field (10-29) |

Code lives in `lib/stages/gnss` (descriptor `GW_1202`, code generator), `lib/cuda`
(beam dump, track beam), `python/scripts/gnss/guowang` (library, codes, passes),
`scripts/gnss/site` (ephemeris table, scheduler, sync, hidden TLEs) and the broker
(`gnss-broker`, chain `gw_1202`). Non-git tools and state are under `~/gnss/builds/gw1202`.

Owner: the `gnss GuoWang` Claude session; session state (what is running, what is next) stays
in that session's memory, these documents hold what is established.
