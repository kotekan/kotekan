# GuoWang 1202: the pipeline

What runs where for the GuoWang (SATNET LEO) 1202.025 MHz work, from the satellite library to
the message archive. The signal itself is in [SIGNAL.md](SIGNAL.md), the message content in
[PAYLOAD.md](PAYLOAD.md), the continuous stream in
[../../CHORD_GUOWANG_TRACK_BEAM.md](../../CHORD_GUOWANG_TRACK_BEAM.md). Dates are 2026.

## 1. The library (code <-> satellite)

`python/scripts/gnss/guowang/guowang_1202_db.json` (kv/gw1202, auto-committed by the sync cron).
One entry per satellite: `catnum`, `g2_phase` (PRN = phase + 1), `theta_us` (code start after
the UTC 2 ms boundary, measured; drifts ~-0.7 us/day), `status` (transmitting / no_transmission),
`passes` [utc, separation, detected], `last_observed_utc`, `last_checked_*`. 99 satellites on
11-04, 46 with a code and theta ("known"); 29 of 57 codes are shared by two satellites.
`guowang_codes.py` generates the code (G1 at 1206 x G2' phase); `guowang_passes.py` holds the
site, the boresight (az 180, el 62.70) and the TLE cache (`~/.cache/guowang/hulianwang_*.tle`,
plus `hidden_digui.tle` for the inferred-orbit "H" satellites from `gw_hidden_tle.py`).

The library grows only through confirmation windows (section 4): a satellite the chain has never
locked has no theta, and the chain cannot track it. The scheduler targets unconfirmed passes near
the boresight on purpose, so on a growth night the main-lobe passes are mostly satellites the
chain does not yet know (see [PER_ELEMENT.md](PER_ELEMENT.md) for the consequence).

## 2. The ephemeris table (what the broker tracks)

`scripts/gnss/site/gw_ephem_table.py`, cron `*/10` on the VM
(`--hours 3 --sep-deg 118 --el-min 0 --sweep builds/gw1202/sweep/sweep_sats.json`), writes
`/var/tmp/gnss-guowang/gw_ephem.json`, which exists only on the VM:

- `records`: per known satellite pass segment, ITRS position and velocity on a 10 s grid, `prn`,
  `clk_s = -theta`. Where two satellites share a code, each grid point goes to the one closer to
  the boresight and a handover leaves one empty step so the broker re-seeds.
- `transits`: every satellite's passes within 10 deg of the boresight, code known or not; every
  broker chain adds them to its transit sky (`--gw-transits`: the D3 veto, clock freeze, holds).
- `sweep`: 30 s theta-sweep steps for the implied-code satellites (section 5).

The broker (`gnss_ephemeris.load_gw_table`) treats a record as a satellite with health 0 and a
constant clock, so the code-phase model carries theta as a clock offset.

## 3. The chain: gw_1202

`config/gnss_chains_chord.yaml` block `gw_1202`: 24 slots per instance (21 satellites + dark
probes PRN 1, 2, 3), `prn-reconfig apply` every 120 s, `prn-rank boresight` (admit when predicted
within 5 deg in the next 300 s, evict when beyond 10 deg for the whole look-ahead), the remaining
slots filled by the least predicted separation. A moved slot acquires cold: dark for 1-2 min, and
a slot swap reaches FLEET-COH in 8-12 s once locked. The nodes' replica is one 2 ms frame with
the data half zeroed (`GW_1202` in `lib/stages/gnss/gnssSignal.hpp`); the record despread is
coherent over the 10.5 ms record, so the code-half sign bits cost ~13 dB per element (the reason
for the nav-bit wipe in [PER_ELEMENT.md](PER_ELEMENT.md)). Probes pass the presence floor in
~1.6 % of cycles.

Each GPU's `gnss<g>_1202_n2assemble` publishes per-PRN element directions on the in-process
ProjBoard (held 1500 s through bright passes); these are the lanes the track beam steers with.

## 4. Confirmation windows and the cf05 loop

- **Scheduler:** `scripts/gnss/site/beam_dump_sched.py run` on the VM (worktree gw1202-wt):
  80 s windows on passes near the boresight, 72 h revisit, every armed window logged to
  `builds/gw1202/armed_windows.jsonl` (gates, seq range, targets). It arms every GPU's
  `gnss<g>_beamdump_gate`; the beam dump writes `/tmp/gnss/beamdump` on the node
  (`cudaGnssBeamDump` eigen beams, 16 channels per GPU, 13 frames of 8192 samples per file).
- **cf05 loop:** `builds/gw1202/gw_loop_cf05.sh` = pull (`beamdump_copy.py`, manifest
  `/var/tmp/kvand-beamdump/manifest.json`, tombstones) -> `gw_reduce.py` (requires the armed GPUs;
  stitches the lobe channels, TLE carrier, code track, per 2 ms frame `amp`, `noise_amp`,
  `car_ph`, `code_err`, `data_bits`; output `/var/tmp/kvand-gw1202/reduce/wNNN/{summary.json,
  <sat>.npz}`, `reduce_log.jsonl`, `db_proposals.jsonl`) -> `csk_extract.py` (CSK symbols + LDPC
  messages into `builds/gw1202/csk_streams/` and `csk_streams/msgs/`) -> `gw_retain_cf05.py
  --delete` (raw copies deleted after reduce, `keep_raw.txt` exempts windows).
- **Library sync:** `scripts/gnss/site/guowang_db_sync.sh`, cron `7,37`: merges the proposals
  into the library (keeps no-transmission satellites, last-observed), commits on kv/gw1202, and
  runs `gw_frame0_sync.py`.
- **frame0:** every F-engine start must reach `builds/gw1202/frame0.json`
  (`gw_frame0_sync.py` appends the broker's newest time anchor); a window from an unknown epoch is
  held out, never renumbered.
- **No-sky watcher:** `builds/gw1202/gw_nosky_watch.py` opens a `no_sky` period in the library
  when gps_l5 shows nothing for 6 lines (or is silent 5 min) and closes it at >= 3 present, so
  misses inside an outage are not counted against a satellite.
- **Retention:** node copies under `/tmp/gnss/{beamdump,trackbeam,...}` are deletable by kvand
  since node_up.sh makes the sink directories as the user (24dfff274); `/tmp` is wiped at boot.

## 5. Implied codes: the theta sweep

`code_coverage.py` infers codes for uncatalogued planes (128/204 determined on 11-04, 45 agree,
0 conflict). `gw_ephem_table.py --sweep` cuts each implied satellite's pass into 30 s steps
(20 s record + 10 s gap) over a theta grid of -18..+12 us in 0.375 us steps;
`builds/gw1202/gw_sweep_harvest.py` (cron `5-59/10`, state in `builds/gw1202/sweep/`) calls a
step a HIT on >= 3 FLEET-COH present cycles with no probe present; two hit steps confirm a
satellite at the first hit's theta (a single fixed-theta pass fails on the ~10 ns/s drift).
First confirmation 11-04 13:32Z: DIGUI-78 = PRN 2028, 42 deg off the boresight.

## 6. The continuous stream (track beam)

Spec: [../../CHORD_GUOWANG_TRACK_BEAM.md](../../CHORD_GUOWANG_TRACK_BEAM.md). Per GPU a second
`cudaGnssBeamDump` writes a beam per tracked target (the two strongest lanes, re-steered every
frame) on its main-lobe channel under `/tmp/gnss/trackbeam`; `scripts/gnss/track_beam_arm.py`
keeps the gates armed from cron on the VM every 5 min and disarms when the cf05 loop's heartbeat
is stale. `builds/gw1202/track_beam_loop.py` on cf05 pulls complete files, decodes 10 s blocks per
PRN (16-channel stitch, TLE carrier, per-frame lag search, CSK readout, sync-word alignment,
LDPC + CRC per UTC second) into `/var/tmp/kvand-trackbeam/msgs/<day>.jsonl`, and deletes node
copies two blocks back. It fetches the VM's ephemeris table itself.

## 7. Where the data is

| What | Where | Notes |
|---|---|---|
| Raw beam dumps | nodes `/tmp/gnss/beamdump`, cf05 `/var/tmp/kvand-beamdump/raw/<node>` | cf05 copies deleted after reduce |
| Reduce products | cf05 `/var/tmp/kvand-gw1202/reduce/wNNN` | per-frame amp, phase, bits |
| CSK streams, messages | `builds/gw1202/csk_streams/` | from the windows |
| Continuous stream | cf05 `/var/tmp/kvand-trackbeam/{msgs,syms}` | from the track beam |
| 10-30 baseband | cf05 `/var/tmp/bb1030` (112 GB) | 128 inputs, lobe channels 6122-6185; `bb1030/csk_bb_test.py` |
| Beam-cube rows | L0 archive; snapshot cf05 `/var/tmp/kvand_gw_l0/` | per-element records of the chain |
| Map page | artifact EBrpVnCdLG7eK8s3WuBxEZ | `builds/gw1202/map/gw_map_data.py` on cf06 |
| ITU, patent, LDPC tables | `builds/gw1202/itu_refs/` | |

Sources: memory chord-guowang-handoff-20261103/20261104, chord-guowang-tracking-design-20261102,
chord-guowang-lane-task-20261103, chord-guowang-csk-lane-plan-20261104,
chord-guowang-track-beam-build-20261104, guowang-site-work-20261104, guowang-fengine-restart-frame0.
