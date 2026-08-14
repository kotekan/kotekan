# The frame-synced tracker → broker transport (task #59)

**Status 2026-08-14: BUILT, GATED OFFLINE, NOT YET ON SKY.** The node configs carry the sender
legs (regenerated, so they arrive on the next tracker restart); the gather instance config is
committed; the broker connects only when told to and consumes only when told to. Nothing about
the existing REST path has changed.

---

## 1. Why

The broker builds its fleet products from ~60 REST round trips per cycle — 12 instances × 5
chains — each landing at a different wall time, and then works out *afterwards* which instance
and which window each reply described. Every one of those inferences went wrong somewhere in a
single week:

| | what was inferred | what it cost |
|---|---|---|
| **#53** | which records an instance summed (`/get_spectrum` windows were "whatever accumulated since your last GET") | no two instances ever summed the same records, and there was no way to ask for the past |
| **#52** | the cross-instance phase | the delay fit absorbed the misalignment into a free phase per instance: it **fitted what it should have derived** |
| **#33** | which instance a "served row" came from | `res_cycles` differenced across a row that silently changed instance read **4.92 Hz where one instance reads 0.82** — 6×, from addressing alone, and it looked exactly like physics |
| **#46** | that instances share a record time | 0.105 s of spread that nothing in the design buffers |

None of those are physics. They are one defect: **the address was inferred.** So the address now
travels *with* the data, on the F-engine's own sample counter.

This is also the bandwidth argument. The airspy prototype's real advantage was never the
ephemeris — it was that its carrier loop ran at 1 kHz against our ~0.5 Hz poll cycle. A pushed
frame-synced stream closes most of that gap **without moving the loop off the broker**, which is
where it must stay: an instance sees 7 of ~105 channels, one node's elements, one satellite, and
a local loop throws away the full-band, multi-element, inter-satellite knowledge that is the
entire reason the vector state exists (KV, 2026-08-14 — do not revisit).

## 2. Shape

```
  each node, each GPU, each chain                       the broker host (cf06)
  ────────────────────────────────                      ──────────────────────
   n2rec_buf ──┬─► GnssCoherentCombiner  (unchanged)
               │
               └─► GnssTelemPack ─► bufferSend ══════►  bufferRecv ─► GnssTelemGather
                   strip to the      :11060                              │
                   record header,                                        │ length-prefixed
                   batch 4 records                                       ▼ frames, :11061
                   on an ABSOLUTE                                   gps_distributed_broker
                   window boundary                                  (PYTHON, unchanged shape)
```

The packer is a **second consumer** of the record buffer, not a change to the chain: the
combiner reads `n2rec_buf` exactly as before.

**The gather collates nothing, deliberately.** The loops live in the broker, so policy belongs
there; and collation needs no C++ help, because every frame already carries the absolute window
index. What the gather provides is *identity and delivery* — which sender, which window, with no
inference.

## 3. The wire format (`lib/stages/gnss/gnssTelem.hpp`)

```
[ TelemHeader, 96 B ][ float32 rows[n_rec][n_prn][gnss::RECORD_FLOATS] ]
```

A row is the **tracker record header of `gnssRecord.hpp`, verbatim** — byte-for-byte the airspy
single-antenna layout (`record_stride(0) == RECORD_FLOATS`). Not a new schema: a translation
table between record slots and a wire struct is one more thing to get subtly wrong, and slots
keep being added (SKY, RES, TRIM_INC all arrived this month).

The **element blocks are dropped**: 32 antennas × 12 floats is 94 % of a CHORD record, the broker
closes no loop on them, and the per-element beam product is accumulated node-side and written to
disk there (`CMB_ELEM_*`). A broker-side per-element product would be a second stream, not a
fatter row.

Header fields that matter:

- `win` — **the collation key.** `wstart0 / (records_per_frame · hops_per_record · fft_len)`, an
  exact integer division on the global sample counter. Every sender computes it from the same
  three configured integers, so equal `win` *is* the same sky, with no tolerance — unlike UTC,
  which each instance derives independently.
- `chain`, `inst` — names, not enum ids. An integer id needs identical tables in the generator,
  the C++ and the broker, and a drifted one attributes a constellation's data to another
  constellation with nothing anywhere to say so.
- `seq` — the sender's own frame counter. Gaps mean loss. A *rate* that looks right can still be
  missing every fourth frame; only this can say so.
- `present` — bitmask of filled record slots. **Slot `r` always holds the record at
  `wstart0 + r·hops_per_record·fft_len`**, so a dropped record is a hole at a known index rather
  than a shift of everything after it.

Rate: 4 records/frame × 40 PRN rows × 26 floats = 16 736 B at 23.84 Hz per (chain, instance) =
399 kB/s; **24 MB/s** for 12 instances × 5 chains.

### Two contracts

1. **⚠️ SHIP INCREMENTS, NEVER ACCUMULATIONS.** Every float is per-record, and the two phase
   slots that could have been accumulators (`REC_CPHASE`, `REC_TRIM_INC`) are already defined as
   increments. Accumulate **once**, broker-side, over a run certified contiguous. An accumulator
   on the wire cannot be repaired downstream by any amount of transport synchronisation, because
   its *origin* is not on the wire — that is #33 exactly.

2. **⚠️ KEY ON `win`, NEVER ON THE RECEIVER'S FRAME ORDER.** `bufferRecv` makes no ordering
   promise across connections and its own `frame_id` is a local ring index
   (`current_frame_id = (current_frame_id + 1) % num_frames`); 60 senders share one listener.
   Collating on arrival order silently mis-pairs instances the moment one drops a frame or
   restarts — precisely the bug class this exists to remove.

**One frame size for every sender**, so one `bufferRecv`, one port, one buffer, and no per-chain
plumbing that can be wired up crooked. `telem_max_prn` (40) and `telem_records_per_frame` (4) are
therefore *on the wire*: change either and the gather config must be regenerated in the same
commit and both ends restarted. `bufferRecv` closes any connection whose `frame_size` disagrees,
so a mismatch delivers **no** data rather than wrong data — verified as a negative control.

## 4. Gates

| gate | what it covers |
|---|---|
| `python3 python/scripts/gnss/test_telem.py` (19 tests, 0.4 s) | **compiles and runs `scripts/gnss/telemfmt.cpp`**, so the Python parser is checked against `gnssTelem.hpp` itself, not against a transcription of it — plus the window ring, the lag rule, hole-vs-shift, and cross-instance grouping |
| `python3 scripts/gnss/telem_e2e.py` (~20 s) | the **shipped code**: real `GnssTelemPack`, real `bufferSend`/`bufferRecv`, real `GnssTelemGather`, real client. Only the record source is synthetic (`rawFileRead` replays frames this script writes, with the `sample_seq` metadata the packer keys on) |

The e2e asserts three things, each written against a specific way the REST path failed:

1. **Provenance** — every row out the far end is byte-identical to the row written for that
   (instance, hop, PRN). The payload encodes its own address, so a mis-paired frame is caught
   rather than averaged. It also checks no element block leaked through (a stride bug).
2. **Grouping by absolute window, not arrival order** — one simulated instance starts *late*, so
   its Nth frame is not its peers' Nth frame. Grouped by arrival order it would pair with the
   wrong window and the fleet sum would look fine and be wrong.
3. **A dropped record is a hole, not a shift** — one instance omits one record slot; its
   neighbours keep their own hops.

Negative controls run: a `telem_max_prn` mismatch produces
`Frame size does not match between server: 3164 and client: 3424` and **zero** frames (loud, and
it names both numbers); and the e2e's own marks are integers below 2^24 with a fixed instance
index, because a `hash()`-derived mark made the gate pass or fail with `PYTHONHASHSEED`.

Also confirmed: `broker_equiv.py check` on the fake-L5 fixture is **EQUIVALENT**, and the on-sky
L5 digest moves to the *same* hash with and without this change (it was already stale from
today's earlier work) — so the POST stream is untouched.

## 5. Deploying

**Order matters: gather first, then the nodes.** 60 senders retrying a dead listener is ~2 log
lines a second across the fleet even at `reconnect_time 30`.

```bash
ssh cf06 '/home/kvand/gnss/kotekan/scripts/gnss/gather_up.sh'   # :11060 in, :11061 out, rest :12051
# then, per node (KV — these need sudo):
scripts/gnss/node_up.sh <node> restart
```

`gather_up.sh` prints a health block after start: per chain, the number of senders and
**`win min..max (spread)`**. Spread 0 or 1 is the transport working. Anything larger is the
misalignment this whole change exists to make visible — and it is invisible in any throughput
number, which is why the health check reports it and not a frame rate. `bad_frames` non-zero
means a sender is on a different build (`RECORD_FLOATS` or the header layout).

Same view over REST: `curl -s localhost:12051/telem_gather/get_stats`.

### Consuming it

Two independent broker flags, both off by default:

- `--telem-gather cf06:11061` — connect and fill the store. **Changes nothing on its own**; the
  broker logs a rate-limited `TELEM` line with frames / gaps / bad and the per-chain window
  spread. This is worth running alone first: it makes the transport's health visible without
  putting anything downstream of it.
- `--telem-coherent` — feed `fleet_coherent` from the gather instead of `/get_records`.

`fleet_coherent` gained one parameter, `source=(got, fleet_now)`. **One estimator, two
transports, on purpose**: the whole claim is that gathered records are addressed rather than
inferred, and the way to test that claim is to run the identical math on both feeds. A second
estimator would make the difference unattributable. If the gather is down, absent, or not
carrying this chain, `source` is `None` and the REST poll runs exactly as before.

⚠️ **Enable `--telem-coherent` on ONE CHAIN at a time**, leaving the others on REST as a
same-poll control. A before/after across restarts cannot resolve it — the sky churns faster
(2026-08-13: `deep_snr` max swung 52–197 in four minutes).

## 6. What is deliberately not done yet

- **`/get_status` still polls.** The combiner computes the deep folds, the ladder, S4, ADR and
  the DLL powers; moving those is a separate, larger change. This phase carries the *record
  stream*, which is what `fleet_coherent` needs and what the #33 carrier work needs.
- **`fleet_dll` and `fleet_spectrum` still poll.** Both become straightforward once the record
  stream is trusted: E/P/L energies are already in every row, and the per-channel spectrum is a
  second stream of the same shape.
- **`record_stream()` exists but nothing consumes it.** It is the #33 fix in waiting: one
  instance's records for one PRN, in hop order, **with the gaps marked**, so carrier phase can be
  accumulated over a run that is provably contiguous and provably from one instance. What to do
  at a break is a loop decision, so it is left to the caller.

Related: `chord-frame-synced-gather`, `chord-rrate-physics`, `chord-instance-time-divergence`
(#46), `docs/CHORD_GNSS_SHARED_DLL.md`, `docs/CHORD_JOINT_TRACKING.md`.
