# Live PRN reconfiguration

**Status: BUILT, and OFF.** The node half is in place on every producer, the record assembler
follows the frame, the search is driven in step, and the broker owner exists behind
`--prn-reconfig`, which defaults to `off`. Nothing in production changes until that flag moves.

⚠️ **Deploying the node half needs a node restart** (KV's domain) — the endpoints are
registered at stage construction. The configs are regenerated and committed.

## Why

The node PRN lists are hand-written strings in `config/gnss_fleet_chord.yaml`, and their
comments are human snapshots of which slots carry an active satellite. The broker reads **live
BRDC every cycle**. Nothing reconciled the two, so they drifted apart silently.

Measured 2026-08-26 (`gen_fleet.py --check-prns`):

| signal | slots | dead slots | active+capable+visible but EXCLUDED |
|---|---|---|---|
| GAL_E5A_Q_CS / E5B | 32 | **5** (1, 14, 18, 22, 24) | **E36**, peak el **83°** |
| BDS_B2A_P_CS / B2B | 24 | 0 | **C58**, peak el 31° (all 24 slots in use) |

E36 transits essentially overhead — through the main beam — and we could not see it. The list
was, as KV put it, at most a month old. **A month is already too long, and future campaigns
will run longer.** A startup-only list cannot track a constellation that changes under it.

⚠️ It also had a second-order cost that is easy to miss: the broker's noise-probe selector
picks the **deepest below-horizon PRN**, which is exactly the kind of PRN an excluded slot
tends to be. It picked E36, which the node could not represent, so the probe was seeded, never
reported, and both Galileo chains ran on 2 probes instead of 3 — silently dropping from the
q+p presence gate to **brightness-only**. See [[chord-prn-lists-diverge]].

## Shape as built

Broker is the authority (it has BRDC, visibility and the capability model); the producer owns
membership on the node; the identity then rides the data.

### 1. The slot COUNT is fixed; only membership moves

`n_prn = _prns.size()` propagates into every buffer and GPU allocation and into the frame sizes
negotiated on the wire (`frame_bytes(n_prn, n_chan)`, and the epl/ctl byte formulas in
`gen_chord_gnss_config.py`). **Changing the count live is a fleet-wide re-plumb including
`bufferRecv`.** Changing which PRN occupies each slot at constant count leaves every size
byte-identical. Both `set_prns` endpoints therefore **refuse** a list of the wrong length
rather than padding it — a resize presented as an ordinary swap is the failure to avoid.

This is sufficient: Galileo needs 28 of 32 slots, so the budget was never the binding
constraint — we simply had the wrong 32. (BeiDou at 24/24 *is* capacity-bound, and swapping
C58 in means choosing something to drop. The broker says so and changes nothing: see the
`prnmap-full` log line.)

### 2. The frame carries the identity — so the assembler needs no endpoint

`gnss_gpu::PrnCtl` grew a `uint16_t prn`, taken from padding so `sizeof(PrnCtl)` stays 80 and
no generated buffer changes size. Every producer stamps it **for every slot**, including slots
with no seed, and `GnssGpuRecordAssemble::follow_frame_prns` adopts it at the top of each
frame.

This is the load-bearing decision. The alternative — a second `/set_prns` on the assembler,
pushed in step — would have been **two copies of slot→PRN with no interlock**, which is the
same shape as the config-vs-sky divergence this whole mechanism exists to end. Instead the
producer owns membership, the identity travels in the data, and a frame that straddles a swap
labels itself correctly with no coordination at all. Record slot 0 is now read from the
record's own control word, falling back to the config list when the producer does not stamp it
(`prn == 0`).

### 3. ⚠️ The hard part is per-slot state, not the list

When a slot changes PRN, everything keyed to that slot must be dropped **together**. Any
survivor is the *old* satellite's state attributed to the new one — the accumulator-identity
trap, the same class that produced the ±26-chip epoch bug and the arc-break rules — and every
one of them is silent, because a warm cal is a plausible cal and a phase history is a
plausible history.

Reset, in one place per owner:

| owner | reset |
|---|---|
| `ChannelizedReplicaBank::set_prn` | the slot's expanded code table |
| `GnssCudaDespread::set_prn` | device code row, Phi cache, carrier-NCO accumulator, `last_ang0`/`last_phi_ddop` |
| `cudaGnssChordTrackState::apply_prn_swaps` | seed, trim, trim diagnostics, power EMAs |
| each command instance | its own per-slot Doppler history (via `slot_gen`) |
| `GnssGpuRecordAssemble::follow_frame_prns` | element cal, NCO phase, arc continuity, open spectrum windows |
| `GnssChannelizedSearch::apply_prn_swaps` | replica cache, refine engines' code tables, detection, Doppler + overlay hints |

### 4. Where a swap lands

* **Producers** (`cudaGnssInject`, `cudaGnssChordTrack`): at the **frame boundary**, before a
  single job is built, so no record is assembled from half a map.
* **Search**: **between passes**, in the worker thread — the only instant at which the replica
  cache, the refine engines and the hint tables are not half built.

`GnssCudaDespread::set_prn` re-uploads the slot's code **stream-ordered**, deliberately: the
previous frame's waveform kernel may still be reading `d_code`, and a plain `cudaMemcpy` is not
ordered against it. The tail of that kernel would then synthesize a chimera of two satellites'
codes — one bad record on a slot that was already being reset, i.e. the most invisible failure
available.

⚠️ **Several engines can share one bank.** The search builds one `GnssCudaDespread` per
≤64-channel refine group over a single `ChannelizedReplicaBank`. Each keeps its **own** device
code table, so `set_prn` compares against the engine's view, not the bank's. Short-circuiting
on the bank would let the first engine update it and every other engine conclude it had nothing
to do — correlating the new satellite's model against the old satellite's code, on the GPU,
with no symptom. `scripts/gnss/prnswap` pins this, and **is proven to fail on the pre-fix code**
(measured: the shared-bank check goes 1.000 → 0.009).

### 5. ⚠️ Hysteresis, because a swap costs a re-acquisition

The broker sees BRDC every cycle. Recomputing the list directly from it would churn slots
whenever a satellite flickers in or out of the ephemeris — that is #92's disease one level up.
So `gnss_broker/prnmap.py`:

* an incumbent must be **below `--prn-reconfig-evict-deg` continuously for 3 h**, or **absent
  from BRDC for 2 h**, before its slot may be reclaimed — and coming back above the mask
  **restarts the clock**;
* a candidate must be **above `--prn-reconfig-admit-deg` right now**. "Rises eventually" is not
  enough: the swap is only worth paying for a satellite we can start using, and one that is up
  now is up again tomorrow;
* at most **one slot per `--prn-reconfig-interval-s`** (15 min), fleet-wide;
* nodes that **disagree** stop the stage dead. Nothing here is per-node, so a split map is a
  fault to fix, not a state to drive out of;
* an **empty prediction changes nothing** — a BRDC outage is not a constellation outage
  (the 2026-08-19 stale-EOP lesson).

The map is **read back from the nodes**, never remembered, so what the broker diffs against is
what the nodes actually hold — including after a restart reverted them to the config list.
**This mechanism is deliberately not persistent**: the config remains the boot state and a
restart is a clean slate rather than a silently-inherited history.

### 6. Arming

`--prn-reconfig off | report | apply`, default **off**.

`report` is the interesting middle: it GETs each node's map, logs the swap it *would* make, and
posts nothing — a live version of `gen_fleet.py --check-prns`, with the same finding and no
risk. The unit test pins "report NEVER posts", which is what makes it safe to arm first.

## Verification

* `scripts/gnss/prnswap` (19 checks, run on cf06) — the bank's swapped slot is **byte-identical**
  to one built that way; a refused PRN changes nothing; and, decisively, the **device** follows:
  slot 0 correlates PRN A at 1.000 and PRN C at 0.009, and after the swap exactly the reverse.
  Proven able to fail.
* `gnss_broker/test_prnmap.py` (17 checks) — the hysteresis, the rate limit, the modes, the
  split-map refusal, and that the search is driven with the same payload.
* `scripts/gnss/gate.sh` — 7/7 EQUIVALENT with the flag off, i.e. the broker change is inert
  until armed.

## Order of work

1. ✅ `gen_fleet.py --check-prns` (`31fa39e37`) — catches the class offline, stays useful as a
   cross-check, and would have flagged E36 the day it was commissioned.
2. ✅ The live mechanism above, **disarmed**.
3. **Next: a node restart** to register the endpoints, then `--prn-reconfig report` on one
   Galileo chain and read what it says about E36 before arming `apply`.

## Known gaps

* **`--prn-reconfig` is per chain, and the two bands of one constellation are separate chains.**
  Nothing couples e5a's map to e5b's, so they can diverge — and a divergent pair destroys
  `tau_band`, which is only observable as the SAME satellite on TWO carriers
  (see the manifest's note beside the E5B list). Until they are coupled, arm the pair together
  and check both maps.
* **FDMA is refused, not handled.** Under GLONASS L1OF/L2OF the satellite identity lives in the
  carrier, and `set_prn` has no way to know the new satellite's frequency offset. Carrying the
  old offset despreads at the wrong frequency; zeroing it parks the replica on band centre,
  which "looks exactly like a broken code". Both are silent, so the bank refuses the swap and
  an FDMA-aware path is left to whoever needs one.

## Rejected

**Discovery-based probe filtering** ("a PRN that never appears in combiner status is not
trackable"). KV's objection is decisive: it cannot distinguish *untrackable* from *not yet
chosen* until a full sidereal day has elapsed, so it cannot bootstrap.

**`noise-probes: 4`** as a workaround. Brute force — if two dead slots revive, we are straight
back to the same failure with no warning.

**A `/set_prns` on the record assembler.** See §2: it would have been a second uninterlocked
copy of the map. The frame carries the identity instead.
