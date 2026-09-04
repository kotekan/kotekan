# Startup deadlock: lockstep seq consumption of the bad-feed mask across independent GPU pipelines

*CHORD pathfinder GNSS deployment, 2026-09-02 — for Jim & Andre*

## Summary

On dual-GPU nodes, `bufferBadInputs` (`set_bf_mask`) is **one producer** whose output
buffer is consumed, peek-hold and **seq-matched**, by **both** GPU pipelines. If one
GPU's capture starts later than the other's (in our case: port-1's expected-stream-id
set completes last, plus the capture anchor's ~30 s future start), that pipeline's
first consume wants a mask frame whose seq no longer exists — the ring is full of
pre-capture frames the late consumer never released, so the producer is blocked in
`wait_for_empty_frame` and can never write the seqs the consumer is waiting for.
Producer waits on consumer; consumer waits on seqs that can never arrive. The GPU
half goes dark with **every instrument green**: full packet rates, zero drops, all
threads alive, GPU ~75 % busy. The packet-loss mask chain (`count_PL` /
expander / compactor) deadlocks identically.

We hit this on roughly **one GPU half in three per process start** across a six-node
fleet, with one node (cx19) failing five starts out of five — its port-1 stream set is
systematically the last to complete, so its dice are loaded.

## Evidence

Four gdb snapshots (two nodes × two snapshots, 10 s apart, archived): the entire
late half sits in `wait_for_full_frame`; the only `wait_for_empty_frame` holders are
`set_bf_mask`, the PL-mask stages, the port transpose, and the frame prefetcher —
identical on both nodes and both snapshots. Downstream, the correlation-vs-counts
seq check in `N2Accumulate` converts the stalled half into a whole-node `FATAL_ERROR`
(~26 k frames of divergence later), so the design flaw's end state is a dead node.

## The design point we'd like reconsidered

The two GPU pipelines are independent: nothing in the science requires them to
consume validity masks in lockstep sequence order. The mask is (a) slowly-varying
per-element *state* (bad-feed mask: one flag per pol×dish, REST-updated on
human timescales) or (b) full-cadence but *droppable-under-degradation* bookkeeping
(packet-loss counts). Coupling both GPUs to one ring with exact-seq blocking
consumption turns any start-order skew — or any transient consumer lag — into a
permanent stall, and the N2Accumulate check then turns the stall into a node death.

## Suggested fixes (in preference order)

1. **Producer paced by its own lifetime clock, not by consumer backpressure**, with
   per-consumer (or per-GPU) buffers; a consumer that lags drops *its own* oldest
   frames. Consumers tolerate seq gaps by skipping forward (for the bad-feed mask,
   "newest available" is always correct; for the PL mask, a gap means those
   integrations mark their loss-counts unknown rather than the node dying).
2. At minimum: the seq-match on mask streams should **resync by discarding the
   stale side** (with a counter/log) instead of blocking forever, and
   `N2Accumulate`'s desync check should resync-and-count rather than `FATAL_ERROR`
   on streams that are bookkeeping rather than data.

## Interim workaround now deployed on our branch (commit `91a0155e4`)

`host_bf_mask_buffer → Valve → host_bf_mask_buffer_v{0,1} → GPU{0,1}` per node: the
valves always drain the shared buffer (the producer can never block) and a stalled
half sheds only its own frames, counted in `kotekan_valve_dropped_frames_total`.
Verified on the deterministic-failure node: first fully-clean start in six attempts.

**Known wart of the workaround, relevant to fix #1:** `bufferBadInputs` relies on
consumer backpressure for pacing, so behind valves it free-runs (~34 k frames/s of
128-byte frames; harmless because the downstream ring copy re-times frames on its
own cursor, but it burns a core and stamps seqs far ahead of real time). That
pacing-by-backpressure is itself part of the design being questioned here.

## Pointers

* Backtraces: `logs/bt_20260902/bt{1,2}_{cx19,cx43}.txt` (kvand's tree)
* Workaround: commit `91a0155e4` (generator-emitted valve split, per-node configs)
* The stall's fleet-level symptom, for recognition: one GPU half's host voltage
  buffer pinned at N−1/N, `element_power` all-zero on that half, no errors anywhere.
