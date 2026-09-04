# The offline work plan — 2026-08-18 onward

Written while the nodes are down for site work, and expected to hold for a while.

**THE CONSTRAINT THAT SHAPES EVERYTHING BELOW (KV):** do not pile up hundreds of RF-path
changes against a restart we cannot test. Every item here is judged first on *how much
un-tested node-side change it adds*, and only then on how much we want it. An offline window
is a good time to write code and a terrible time to accumulate unverifiable deployment risk.

The corollary, and it is the whole ranking: **prefer work that needs no node restart at all.**
Broker-side Python deploys the moment the fleet is back and is reversible in seconds. Node-side
C++ costs a sudo restart and an observing window, and is only testable on sky.

---

## ⚠️ REVISION: #46 IS DEMOTED, AND KV'S QUESTION IS WHY

I ranked #46 first and described it as "the nodes should serve a clock." KV asked the obvious
question — *why are the nodes serving clocks, doesn't that flow through on the data already?* —
and the answer is that it does, and my framing was wrong. Recorded here because the reasoning
matters more than the conclusion:

1. **The seed path already needs no "now" at all.** `--dr-forecast-lead-s` (KV's own reframing,
   which dissolved a problem two earlier fixes were engineering around) puts a seed on the hop
   axis: choose the hop H we are forecasting TO, and the pipeline lag never enters the
   arithmetic — it only sets how large the lead must be. The comment at
   `gps_distributed_broker.py:5674` says it outright: *producing a seed needs no notion of "now",
   and asking for one is what dragged the wall clock, the NTP offset and the telemetry lag into
   the seed path in the first place.*
2. **The ephemeris epoch is not broken.** It deliberately rides WALL time, because orbit
   evaluation is the one consumer that needs absolute UTC to be TRUE rather than merely
   self-consistent. cf06's wall carries ~1.45 ms of NTP error (chrony root dispersion, measured)
   = 0.44 m = ~0.015 chips at L5. That is fine. The 65x-worse experiment was moving OFF wall
   ONTO the stale F-engine axis — the substitution was the error, and reverting it was the fix.
3. **The lag is already measured, every cycle.** `_dax = (utc0 + pow_hop/hps) - wall_at_fetch`
   at `:5725`, watched by `--clock-step-guard-s`. We are not blind to it.

So a serve-time stamp would buy **decomposition** — separating node pipeline depth from
transport — not a new capability, and nothing is currently blocked on that decomposition. The
tripwire cannot tell a wall step from a lag jump, but `chronyc tracking` can, and it is right
there in the log line.

**What I claimed and now retract:** that #46 unblocks two retracted fixes. It unblocks at most
one. The ephemeris fix is not waiting on it (it was tried, was worse, and today's wall-based
answer is correct). `--innov-dr-seeds` failed on a **215 us** residue — and 215 us is not the
~100 ms pipeline lag, so attributing it to this lag was an assumption, not a measurement.

**#46 therefore leaves the restart queue** and becomes: *find out what the 215 us actually is,
offline, from the archived logs.* If it turns out to be lag after all, the node-side change
earns its place then — measured in, not assumed in.

---

## THE QUEUE AS IT NOW STANDS

| | item | where | needs a restart? |
|---|---|---|---|
| **1** | #8 clip fraction + band power | node C++ | **yes** — the only one |
| **2** | #70 instance liveness guard | broker Python | no |
| **3** | #54 re-verify or retire | offline bench | no |
| **4** | #46 what is the 215 us? | offline, archived logs | no |
| — | #88 j2 restructure | offline | **HELD for Jim** |

One node-side item in the queue, not two. That is the shape KV asked for.

---

## 1. #8 — clip fraction + band power  [the only node-side change]

Wanted, and safe in the sense that matters: it **reads** the RF path and steers nothing.

### The stage shape — KV's proposal, and it is the right one

> *"Possibly do the valve->RF power stage, where the stage grabs a copy of a buffer every ~10s,
> and just releases immediately otherwise?"*

Yes. A consumer stage that registers on the voltage buffer, and on all but one frame in ~10 s
does nothing but mark the frame done. Every ~10 s it memcpys ONE frame to a small host buffer,
releases the frame, and computes off the critical path.

⚠️ **"Releases immediately" is load-bearing, not a nicety.** A consumer that holds frames
backpressures the GPU chain, and we have already paid for that lesson: #81 was an ABSENT feed
costing a full timeout every frame, which starved the search 356x and put it 48.9 s behind sky.
A monitor that costs real time is worse than no monitor. Mark the frame done FIRST, compute
after — and the ~10 s duty means even a slow computation cannot converge on the frame rate.

### What it measures

* **clip fraction** — what fraction of 4+4b samples sit at the rail, per instance. This is the
  direct question "are we saturating the quantiser", which no existing number answers.
* **band power** — low cadence, per band. The discriminator for #56 and for RFI events
  generally: the 08-18 event was band-selective (+16 dB on 1176 only), and a per-band series is
  what makes that visible as it happens rather than reconstructable afterwards.

### The display already exists, and speaks this language

> *"The viewer still shows two RF bands as if they're airspy-derived..."*

Correct, and it is better than it sounds. `panels/airspy_stats.js` is a stream-health strip
carrying **ADC rms, rail %, and drop rate, with a per-band ring** (`this._rf`: band ->
{ah, vh}) and thresholds already chosen: rms amber < 20, **rail% amber > 0.1, red > 1**, drop
amber > 0 in-window / red > 1e-4 cumulative.

"rail %" IS clip fraction. So the panel we need was designed, thresholded, and shipped for the
prototype — serving the CHORD numbers in the same shape lights it up rather than building a new
one. Worth reading its header comment before designing the endpoint: it also documents the two
independent places data is lost (ADC FIFO vs Valve output-full) and why the second is the silent
one — the same distinction we will want per-instance here.

### Gate

Offline: clip fraction against synthetic input at known rail occupancy. Band power against a
synthetic tone of known amplitude. Neither needs sky. ⚠️ Measure the statistic on-noise AND
on-signal before trusting either, and check it is stable — the standing rule.

---

## 2. #70 — the instance liveness guard  [broker-side, no restart]

> *"Not sure what this is... something to scream in the broker when an instance freezes or
> stops altogether?"*

Exactly that. Concretely: the broker polls `pow_hop` on every instance, keeps the previous
value, and says so loudly when a stage's own output has not advanced while its input has.

**The signature is measured, not hypothesised** — twice, on 2026-08-18:

* cx42/port 0 after the 14:05 full-fleet restart: capture window frozen at
  `[151567122432, 151567138816]`, the stream 52.5 million samples ahead, and the **entire**
  195,313 pkt/s being dropped. `pow_hop` advanced **0 hops in 30 s**; healthy is ~5.9M.
* cx19 earlier: the same wedge held for **25 hours** with 18.7 billion packets dropped, and was
  found by hand.

⚠️ **THE HEADLINE IS THE TRAP: a REST endpoint answering 200 is not a live instance.** All four
wedged instances served plausible, well-formed rows the whole time. "All 12 respond" told us
nothing, and that is why polling twice is the check.

This closes the open half of #87 (the frozen-instance case; the degraded-chain case is already
caught by the q stall guard). It would also have caught #60 and #69.

Related instruments that exist and are NOT this: `--fe-axis-stale-s` (names a frozen time base,
never verified against a real stall), `--clock-step-guard-s` (log-only), and the q stall guard
(a degraded chain, not a frozen instance).

---

## 3. #54 — re-verify or retire  [offline, KV added it]

*"Replicas differ PER-SAMPLE at CHORD scale (9.5%)"*; `C_P` was referenced and did not fix it.

The oldest unverified claim on the list: marked START HERE when written, carried through two
reconciles without re-testing, and predating essentially every replica-path change since — the
baked-CS descriptors, `chip_gather3`, the sparse-comb work, path B's injector.

Replica generation needs no sky, so this competes with nothing.
⚠️ Validate against a **noiseless synthetic** (the sparse-comb rule: replicas are built at DC).
Either it reproduces on today's path and becomes a real bug, or it does not and gets retired
explicitly. Do not carry it a third time.

---

## 4. #46 — what is the 215 us?  [offline, from archived logs]

Reduced from "build a thing" to "answer a question", per the revision above. `--innov-dr-seeds`
read **+2201 chips = 215 us x 10.23 Mcps**. That was attributed to the telemetry lag's sub-ms
residue, but the lag's median is ~100 ms — three orders out — so the attribution is unsupported.
215 us is suspiciously close to a record/window quantum; check that first. The archived broker
logs in `fixtures/` are the instrument, and they are already on disk.

---

## HELD: #88 j2 restructure

KV: *"Let's wait to hear back from Jim before proceeding."* Held deliberately. The template is
proven equivalent (137/137 blocks on all six nodes) and deployment has NOT moved, so holding
costs nothing and avoids restructuring twice if his review changes the shape.

---

## Sky-gated, for when the fleet returns — unchanged, in order

1. **#86** — `grep RATE-TEACH /tmp/gnss_broker.log` FIRST, before anything else runs. Free.
   The 14:51 restart was itself a ~2/3 trigger with forensics already armed. Blocks #33.
   ⚠️ Do NOT restart-loop to hunt it.
2. **#79** — widen the search bar one notch, 100 -> 50. Unadmitted sats are the in-poll controls.
3. **#49** — arm gal_e5b's fast code loop. One yaml line; the per-PRN trim handover it was
   waiting on is implemented and live (`gps_distributed_broker.py:7887`).
4. **#85 / #50** — correlate SPEC-FIT tau against the model-held offsets on the same sats.
   ⚠️ spec_tau is PLAUSIBLE, not STRONG (p/f median ~1.0): blocked on significance, not plumbing.
5. **#33 GAP 1** — arm `--rrate-command` on gal_e5a. Judge on KCOH duty and `rate_resid_hz`
   against unarmed chains in the SAME polls, never on sig. This is where #71's NCO is re-judged.

---

## ⚠️ CORRECTION: #55's "growing debt"

I said the `carrier_phase_from_ref` scaffolding had grown to 104 references against 22 on
record. **That number was inflated and the growth claim is not supported.** It counted raw
matching lines across 15 files, 7 of which are `config/generated/` outputs that regenerate from
the manifest — those are not debt.

The honest figure is **27 lines across 8 non-generated files**, concentrated in two:
`GnssCudaDespread.cpp` (12) and `gen_chord_gnss_config.py` (8); the rest are singletons.

The substance survives at a smaller size: the A/B is **over** — pinned to arm 1 fleet-wide,
result recorded (smoother per-record phase in 9 of 10, sign test p = 0.011, cross-instance
agreement FLAT between arms) — while the machinery to run both arms is still in the despread
path. It is a bounded cleanup blocked on #52, not a spreading tax, and I should not have
described it as one without breaking the count down first.
