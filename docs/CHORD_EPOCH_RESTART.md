# Surviving an F-engine restart

What each layer does when the F-engine re-bases its sequence counter, what that costs, and
which parts are fixed. Written 2026-08-26 after an F-engine restart at ~21:43 UTC put the
fleet blind for hours and needed a full manual restart chain.

## The anatomy

A restart does **two** things, and they break different layers:

1. **The sequence counter re-zeroes.** Measured: node capture windows sat at seq
   36,373,889,024 while packets arrived at 163,780,816 — **backwards by 36.2 billion**. Every
   packet lands outside the active frame range and is dropped: 195,314/s per worker, 100% loss,
   GPUs at 0%, with the NIC and the DPDK ring both perfectly healthy.
2. **frame0 moves.** The UTC of absolute sample 0 is a different instant afterwards, so every
   record's absolute time is wrong by however long the previous run lasted plus the downtime.

⚠️ **The second is the dangerous one**, because it is invisible: every cross-record estimator
works in differences, so a uniformly-wrong epoch stays uniform and everything downstream looks
healthy. The first announces itself in the node log immediately.

## ⚠️ The pattern that hid it: peer-relative checks cannot see common mode

Four separate defects found on 2026-08-26, all the same shape — a check that compares the
population **to itself**, which by construction cannot detect the population moving together:

| where | the check | what it could not see |
|---|---|---|
| `gnssFleetDll::integrate` | `p_pow < 3 × the window's own median` | the weaker half of the array, always (the E3 fix) |
| `fleet.py` presence, pre-2026-08-14 | `p_floor` from the tracked population | everything unbright; fixed by probe anchoring |
| `fleet.py` `fleet_dll` currency | `newest = max(this poll's own rows)` | **twelve instances frozen at the same hop agree perfectly** |
| `--fe-axis-stale-s` | `now - fe_axis[1]`, restamped every cycle | anything at all — see below |

The cure is always the same: **anchor to something outside the population** — the probes for
power, the wall clock for time.

## Layer by layer

### 1. Node DPDK capture — ⚠️ THE BLOCKER, not fixed

`crs16BoardCaptureWorker` has an **ahead-only** resync (`--dpdk-resync-max-advances 32`,
added for the cx19 wedge): it walks the window forward to follow a stream that ran ahead, and
comments "Only when purely AHEAD (never rewind)". A backwards re-base is therefore a permanent
drop, and **only a node restart clears it**.

The rewind restriction is correct as stated — `FramePrefetchService` hands frames downstream
as it advances, so it cannot un-hand them. But recovery does not need a rewind. It needs a
**re-seed**: the service already exposes `stop()` / `start(start_seq, stream_ids)`, which is
exactly what startup does. The missing piece is a detector and a policy:

* **detect**: behind-drops at ~100% for N consecutive seconds *while the incoming seq itself
  advances monotonically*. That is unambiguous — a reorder is small and transient, a re-base is
  total and sustained. The counters already exist (`range_drop_count`, and the seq is in hand).
* **act**: `stop()`, `start(seq_now, ids)`, and log it as loudly as a restart, because
  downstream metadata epochs are now stale (§4).

⚠️ **This is `lib/dpdk/`, shared with the N² correlator and every other CHORD pipeline** — not
GNSS-local like everything else in this document. It is the one change here that needs the
wider CHORD team's agreement, and it should be armed per-config (default off) before it is
default-on.

### 2. Gather / fast DLL — ✅ already done

`gnss::FleetDll::fold` carries the **A1 epoch reset**: consecutive frames landing far behind
`newest` are counted, and after `_epoch_strikes` the chain re-anchors, so one corrupt header
cannot trigger it and any in-order frame clears the run. **The trims survive** — only
`open`/`newest` are epoch-scoped — so a re-base costs no pull-in.

### 3. Broker — partly fixed

* ✅ **`--fe-axis-stale-s` now works.** It could never fire: `fe_axis[0]` was rewritten every
  cycle including its timestamp, so the staleness it measured was always one cycle. Today the
  broker logged `AXIS INST: lag median -6975 s ... spread 0.00` **807 times** while this guard,
  whose entire job is "has the whole time base frozen?", printed nothing. It now keeps the wall
  time of the last **advance**, as `instance_stall_verdict` already did.
  ⚠️ The two guards **partition** the space on purpose — the per-instance one deliberately
  accuses nobody when most of the fleet is stalled, trusting this one for the global case — so
  one half being inert left the worst case uncovered. `test_axis_freeze` pins both halves.
* ❌ **Presence currency is still peer-relative** (`newest = max(...)` over the poll's own
  rows). Twelve frozen combiners agree and pass, which is how "7 present" was published for
  forty minutes against a dead sky. The fix is an absolute bar: the broker already computes
  `fe_off`/`utc0_sample0`, so `newest` can be compared to the wall clock and the chain can
  publish *nothing* rather than a frozen answer. Not built.

### 4. Node record epoch (frame0) — ❌ not refreshable

`CHORDTelescope` reads `time0_ns` **once**, at construction, from the FPGA controller
(`query_gps`) or the config. There is no refresh path and no REST endpoint, so
`cudaGnssChordTrackState`'s `frame0_utc` — and every record UTC built from it — is fixed for
the life of the process. **A node that outlives an F-engine restart is stamping every record
with a stale epoch and nothing downstream can tell.**

What it would take: a `POST /telescope/set_time0` (or a re-query on demand), plus a rule for
what to do with in-flight state. The rule matters more than the plumbing — records straddling
the change carry two epochs, so the honest move is to treat it exactly like the slot swap in
`CHORD_LIVE_PRN_RECONFIG.md`: apply at a frame boundary and reset the per-slot accumulators
whose history is now meaningless.

### ⚠️ How to check an epoch in one line, and why "the recording looks normal" proves nothing

    arriving_seq / 195312.5 + frame0_ctime   ==   wall clock

Measured 2026-08-27 00:35: it gave **2026-08-24 19:57** — 52.6 hours out.

The stream itself is healthy: its seq advances at **195,314/s**, exactly the hop rate, and its
implied epoch is **stable to 0.0 s across samples 45 s apart** — the F-engine is fine, it is
simply running on an epoch that began **2026-08-26 22:37:09 UTC** (and it re-based at least
twice, ~22:21 and ~22:37, before settling).

That the OLD value was right for the OLD epoch is provable: the frozen capture window at seq
36,373,889,024 maps under it to **21:43:14** — exactly when the fleet went dark. The mapping
was working; the epoch moved out from under it.

⚠️⚠️ **A WRONG frame0 IS INVISIBLE INSIDE THE DATA.** It shifts every sample's absolute time
UNIFORMLY, so no internal consistency check can see it — only a comparison against an outside
clock can. A pipeline whose recording "looks normal" may be recording perfectly good samples
under a 52-hour-wrong timestamp, and for anything that never consumes frame0 (raw voltage
dumps, correlator products judged on their own axis) that costs nothing at all. It costs US
because GNSS seeds are evaluated against BRDC at absolute UTC.

⚠️ **And the capture window is PER PROCESS.** A pipeline that started or restarted after the
re-base latched the new epoch and receives data normally; one that started before it drops
100%. Two pipelines on the same node, in the same second, can therefore disagree completely
about whether the instrument is working — which is exactly what happened here.

⚠️ **And chive must be fixed first regardless.** After today's restart it still served the
*old* `frame0_ctime`, so even a node restart re-caches a stale epoch. Sanity check before
trusting any absolute time: `arriving_seq / 195312.5 + frame0_ctime` must land on the wall
clock. Today it landed two days earlier.

### 5. Diagnostics — ✅ fixed today

`fengine_status.sh` polled **port 12049**; the nodes bind **12048**. Every poll returned
nothing, `awk ... s+0` turned nothing into a confident `0`, and it printed *"NO DATA anywhere.
The F-engine is not delivering"* while 780,000 packets/s were arriving. "Could not poll" is now
a third outcome rather than being folded into "no data".

## The restart chain, and how much of it is really needed

Today's order — chive refresh → nodes → gather → aggregator → broker → EOP push — is six
steps because layers 1 and 4 cannot recover. With §1 and §4 done it collapses to: the F-engine
restarts, the nodes re-seed their capture windows and re-read frame0, and the gather re-anchors
by itself. The broker and aggregator already tolerate nodes coming and going (they are polled,
and a missing instance shows as `n_src` falling), so **neither should ever need restarting for
this** — they only did today because the broker's stale-detector was inert and its presence
gate believed frozen endpoints.

⚠️ **The EOP push stays manual either way**, and stays *necessary*: the table is REST-updatable
in the running process, so **a node restart silently reverts it to the config's copy**. Run
`eop_push.sh` after every node bring-up.
