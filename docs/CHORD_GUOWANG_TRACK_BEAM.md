# GuoWang track beam: the continuous XW-B2b message stream

Status: 2026-11-04, KV's call. Branch kv/track-beam. Built on the beam dump of
CHORD_GUOWANG_BEAMDUMP_PLAN.md and the signal of CHORD_GUOWANG_SIGNAL_20261103.md.

## Why

Every GuoWang satellite sends the same high-rate (CSK, 1464 bit/s) message stream: after the
49-bit header the payload is identical across satellites at the same data epoch (the 11-04
payload study). So one continuous readout of the constellation's stream needs only one tracked
satellite at a time, and the 10-30 baseband showed a steered beam reads a tracked satellite 43
and 70 degrees off the boresight with no symbol errors and ~30 dB to spare. The beam dump's
eigen beam cannot do this: it follows whatever dominates the channel, which off the boresight is
the brightest GNSS or GuoWang satellite, not the one we want.

## What

`cudaGnssBeamDump` gains target beams (`beam_lane_owner`, `beam_targets`): per frame and
selected channel, a beam per tracked satellite of one chain, steered by the element direction
that chain's assembler publishes on the ProjBoard for it (gnssLaneDirection.hpp: the averaged
despread row, held through a bright pass by `elem_dir_hold_s` and re-steered along the
satellite's track). The gw_1202 chain's assemblers publish those directions already
(`elem-dir-publish all`, 24 slots, `n_elements` 128 = the ring's inputs), so nothing new is
estimated: the weights the tracker locks with are the weights the beam uses.

The targets are the `beam_targets` (2) strongest lanes by row power, i.e. the two GuoWang
satellites closest to the boresight. Two, because each satellite lags the data epoch by 9-115 s
and the two streams overlap across a handover, and because one may be in a transit. A target
is kept while its lane is on the board and replaced when another lane out-powers the weakest
target by `beam_target_gain` (1.5) for `beam_target_hold_s` (2 s), so a handover does not flap.
Each GPU chooses on its own channels; the frame records the PRN per beam, and the stitch keys
on PRN, so the GPUs need not agree on the order, only on the set.

The direction is re-steered to the frame's mid-time (gnssFleetLane.hpp fleet_lane_direction
with the lane's own channel as the source): a LEO's fringe rate on the longest baselines is ~2
Hz at 1202 MHz, so a direction published 60 ms earlier is 40 degrees off at the array's edge.
The frame time comes from the telescope's GPS time0, or the `frame0_utc` the generator pins on
the track state, so it shares the lanes' axis.

Weights are the lane direction itself. The despread forms `conj(x) conj(replica)`, so the row
is the element response conjugated in the ring's own decode, and `sum_i q_i x_i` is the
coherent sum; `beam_lane_conj` flips that if the sky disagrees, which `beam_dump_read.py info`
shows as the overlap between the eigen weights and the target weights of the same frame
(~1 when right and the lane owns the channel, ~1/128 when conjugated wrong).

No prewhitening: the closest GuoWang is the brightest source in the lobe, and a GuoWang at the
boresight is the target. Bright GNSS transits through the lobe are eaten (the hold keeps the
weights on target; the transit itself is a few per cent of the time).

## Frame (version 2)

The version 1 frame unchanged (header, freq_ids, eigenvalues, powers, eigen weights, eigen
beam), then with `n_targets` > 0: per target and channel a `BeamDumpTarget` (prn, held, coh,
power, held_s, age_s), the target weights `[n_targets][n_chan][128]` and the target beams
`[n_targets][n_chan][n_times]` as half2. A frame with `n_targets` 0 is byte-identical to
version 1 but for the version number; the existing readers do not check it.

## Plumbing

- Generator: `--track-beam-freq-ids 6149-6159` (the main lobe, +-1.1 MHz, ~77 % of the power),
  `--track-beam-chain gw_1202`, `--track-beam-targets 2`, `--track-beam-frames-per-file 240`.
  A second `cudaGnssBeamDump` per GPU (`gnss<g>_trackbeam`) on that GPU's mod-16 share of the
  range (0 or 1 channel), its `FrameWindowGate` `gnss<g>_trackbeam_gate` and a `rawFileWrite`
  under `<record_dir>/trackbeam`, files of 240 frames (~10 s, ~24 MB).
- Rate: 1 channel x 3 beams x 4 B x 195312.5 /s = 2.4 MB/s per GPU, 17 MB/s fleet-wide, ~200 GB
  per GPU per day. The node disk (1.4 TB free) holds days; the cf05 loop deletes what it has
  decoded. `/tmp/gnss` is kvand-owned on the nodes, so `<record_dir>/trackbeam` is created by
  kvand before the restart and the files are deletable without root.
- Gate: `scripts/gnss/track_beam_arm.py` from cron on the gnss VM every 5 min extends every
  gate's window to 12 min past the fleet's last seq (start_seq 0, so no frame is dropped on a
  re-arm). If the cron stops, the gates close within 12 min and nothing fills the disk.
- Reader: `beam_dump_read.py info` lists the target PRNs and the overlap check;
  `beam_dump_read.py stitch --target PRN` gives that PRN's beam across the fleet's GPUs.
- Decode (cf05, next): pull the files, stitch per PRN, code-track with the library theta (a
  small search: theta moves ~0.5 us/day), 64-shift CSK readout per 2 ms frame, LDPC per UTC
  second (builds/gw1202/gw_ldpc.py), CRC, archive keyed by (data epoch at bit 349, page = SOW
  mod 9); from bb1030/csk_bb_test.py's readout.

## Checks at sky-on

1. `beam_dump_read.py info` on the first trackbeam files: two PRNs per GPU, the overlap median
   well above 0.5 (else set `beam_lane_conj` and restart).
2. The stitched target beam of one PRN decodes with zero LDPC syndrome (csk_bb_test's readout).
3. A handover: the dropped PRN's last and the new PRN's first messages share the data epoch.
