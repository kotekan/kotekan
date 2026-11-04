# CHORD GNSS documentation

The CHORD GNSS stack: per-element GNSS tracking on the CHORD pathfinder (128 inputs, 16 GPU
instances on 8 nodes, a broker on the `gnss` VM), used for beam mapping, calibration, ionosphere
and interference subtraction. This tree holds what is established; the flat `docs/CHORD_*.md`
files are the older documents and stay where they are. Session state (what is running right now,
what each Claude session is doing next) lives in the sessions' memory, not here.

## Operating the stack

- [../CHORD_GNSS_RUNBOOK.md](../CHORD_GNSS_RUNBOOK.md): the whole stack on `gnss`, the three
  commands, the nodes, restart order.
- [../CHORD_BUGLIST.md](../CHORD_BUGLIST.md) and [../CHORD_BUGLIST_CLOSED.md](../CHORD_BUGLIST_CLOSED.md):
  the numbered issues, open and closed.
- [../CHORD_LIVE_PRN_RECONFIG.md](../CHORD_LIVE_PRN_RECONFIG.md): the broker's PRN map
  (admit/evict, boresight ranking, cold acquisition).
- [../CHORD_GPU_BUDGET.md](../CHORD_GPU_BUDGET.md): what a GPU instance can carry.
- [../CHORD_UPSTREAM_PR_PLAN.md](../CHORD_UPSTREAM_PR_PLAN.md): how the CHORD branch feeds kotekan develop.

## Reference

- [CONVENTIONS.md](CONVENTIONS.md): frequency ids, the time axes (F-engine hops, frame0, UTC vs
  GPS), code phase as a unit, signs and geometry, record and cube rules.
- [ANALYSIS_NOTES.md](ANALYSIS_NOTES.md): C/N0 as a significance, correlation and loop health,
  GPU cost model, BRDC supply.
- [../CHORD_ARRAY_EPOCHS.md](../CHORD_ARRAY_EPOCHS.md): element-to-feed mapping is time-varying.
- [../CHORD_MULTIBAND.md](../CHORD_MULTIBAND.md), [../CHORD_JOINT_TRACKING.md](../CHORD_JOINT_TRACKING.md),
  [../CHORD_GNSS_SHARED_DLL.md](../CHORD_GNSS_SHARED_DLL.md), [../CHORD_FAST_TRIM.md](../CHORD_FAST_TRIM.md):
  the tracking design.

## Beam mapping and the cube

- [../CHORD_CUBE_ARCHIVE.md](../CHORD_CUBE_ARCHIVE.md): the per-element record archive (L0, rungs),
  units, coherence.
- [../CHORD_BEAM_MAPS.md](../CHORD_BEAM_MAPS.md), [../CHORD_BEAMCUBE_RUNBOOK.md](../CHORD_BEAMCUBE_RUNBOOK.md):
  building and reading the maps.
- [../CHORD_VIS_CAPTURE.md](../CHORD_VIS_CAPTURE.md): the visibility capture.

## Projection and subtraction

- [../CHORD_FLEET_LANES.md](../CHORD_FLEET_LANES.md): tracking lanes, the N2 projection.
- [../CHORD_DTV_SUBTRACTION_20261102.md](../CHORD_DTV_SUBTRACTION_20261102.md): ATSC subtraction.
- [../CHORD_PEER_COMPARISON_PURGE.md](../CHORD_PEER_COMPARISON_PURGE.md).

## GuoWang (SATNET LEO)

- [guowang/README.md](guowang/README.md): the index; signal, payload, pipeline, per-element work,
  the track beam.

## Where to add things

A new established result goes into the document of its area (or a new file under the area's
directory, linked from here). A dated investigation note that is superseded stays as history and
is linked as such. Keep session state out of this tree.
