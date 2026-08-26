# Live PRN reconfiguration — design (KV-directed 2026-08-26)

**Status: DESIGN ONLY, nothing built.** The content gate (`gen_fleet.py --check-prns`) is
committed and live; everything below is the node-side work it argues for.

## Why

The node PRN lists are hand-written strings in `config/gnss_fleet_chord.yaml`, and their
comments are human snapshots of which slots carry an active satellite. The broker reads **live
BRDC every cycle**. Nothing reconciles the two, so they drift apart silently.

Measured 2026-08-26 (`gen_fleet.py --check-prns`):

| signal | slots | dead slots | active+capable+visible but EXCLUDED |
|---|---|---|---|
| GAL_E5A_Q_CS / E5B | 32 | **5** (1, 14, 18, 22, 24) | **E36**, peak el **83°** |
| BDS_B2A_P_CS / B2B | 24 | 0 | **C58**, peak el 31° (all 24 slots in use) |

E36 transits essentially overhead — through the main beam — and we cannot see it. The list
was, as KV put it, at most a month old. **A month is already too long, and future campaigns
will run longer.** A startup-only list cannot track a constellation that changes under it.

⚠️ It also has a second-order cost that is easy to miss: the broker's noise-probe selector
picks the **deepest below-horizon PRN**, which is exactly the kind of PRN an excluded slot
tends to be. It picked E36, which the node cannot represent, so the probe was seeded, never
reported, and both Galileo chains ran on 2 probes instead of 3 — silently dropping from the
q+p presence gate to **brightness-only**. See [[chord-prn-lists-diverge]].

## Shape

Broker is the authority (it has BRDC, visibility and the capability model); nodes expose
`GET`/`POST` for their active list; the broker pushes it.

### 1. Fix the slot COUNT, vary only membership

`n_prn = _prns.size()` propagates into every buffer and GPU allocation and into the frame
sizes negotiated on the wire (`frame_bytes(n_prn, n_chan)`, and the epl/ctl byte formulas in
`gen_chord_gnss_config.py`). **Changing the count live is a fleet-wide re-plumb including
`bufferRecv`.** Changing which PRN occupies each slot at constant count leaves every size
byte-identical.

This is sufficient: Galileo needs 28 of 32 slots, so the budget was never the binding
constraint — we simply had the wrong 32. (BeiDou at 24/24 *is* capacity-bound, and swapping
C58 in means choosing something to drop: a judgement call, not an automation.)

### 2. The records are already self-describing

`GnssGpuRecordAssemble` writes `rec[0] = (float)_prns[p]` — the PRN travels **per slot in
every record**. So the broker learns the mapping from the data, a frame straddling a swap is
unambiguous, and no downstream protocol changes. `GET` is a diagnostic convenience; `POST` is
the only path that must be right.

### 3. ⚠️ The hard part is per-slot state, not the list

When a slot changes PRN, everything keyed to that slot must be dropped **together**: replica
codes, detections, doppler hints, elemcal warmth, standing trim, ADR arc, nav decoder. Any
survivor becomes the *old* satellite's state attributed to the new one — the accumulator
identity trap, the same class that produced the ±26-chip epoch bug and the arc-break rules.

Make this one explicit `reset_slot(i)` with a test that asserts every per-slot container is
cleared, rather than each stage remembering separately. **This is where the bugs will be.**

### 4. ⚠️ The constraint that shapes the implementation

`ChannelizedReplicaBank` is constructed with the PRN list, is non-copyable, documented *"not
thread-safe (one bank per worker thread)"*, and `GnssCudaDespread` takes it **by reference**
(`GnssCudaDespread(*_replica, n_prn, ...)`). So the bank pointer cannot be swapped under a
live stage. It needs:

* an in-place `set_prn(slot, prn)` regenerating that slot's codes, and
* a device-side refresh of that slot's replica, applied at a frame boundary.

Cost looks tolerable: ~4 MB/PRN (126 MB for 32 PRNs × 79 ch × 3125 hops) and all 32 build in
0.08 s from the disk cache, so one slot should be milliseconds.

Stages holding a PRN list, all of which must agree: `GnssChannelizedTracker`,
`GnssGpuRecordAssemble`, `GnssChannelizedSearch`, `cudaGnssChordTrack`,
`GnssCoherentCombiner`, and the `n2dual/commands` block.

⚠️ **Deploying this needs a node restart** — KV's domain, and worth batching.

### 5. ⚠️ Hysteresis, designed in from the start

The broker sees BRDC every cycle. Recomputing the list directly from it would churn slots
whenever a satellite flickers in or out of the ephemeris, and **a slot swap costs a full
re-acquisition** — that is the #92 disease one level up. Require a satellite to be
consistently absent for hours before its slot is reclaimed, rate-limit to one swap per
interval, and log every swap with its reason.

## Rejected

**Discovery-based probe filtering** ("a PRN that never appears in combiner status is not
trackable"). KV's objection is decisive: it cannot distinguish *untrackable* from *not yet
chosen* until a full sidereal day has elapsed, so it cannot bootstrap.

**`noise-probes: 4`** as a workaround. Brute force — if two dead slots revive, we are straight
back to the same failure with no warning.

## Order of work

1. ✅ `gen_fleet.py --check-prns` — committed. Catches the class, stays useful as a
   cross-check after the live mechanism exists, and would have flagged E36 the day it was
   commissioned.
2. Manifest: swap a dead Galileo slot for 36 (node restart). Immediate, recovers an
   overhead satellite on both gal chains and restores the third probe.
3. The live mechanism above.
