# Purging peer comparisons from the tracking / seeding / control loop

**KV, 2026-08-27, and this is the rule the rest of the document serves:**

> I don't think peer comparisons are *ever* justified in the tracking / seeding / control loop
> code. They can never tell us about a given signal, they only ever tell us fleet-relative
> properties, which aren't relevant.

This is the same argument as *nothing is per-node* ([[chord-nothing-is-per-node]]): an instance
cannot represent physics, and neither can a neighbour. A satellite's discriminator is
informative or it is not, and no fact about the other satellites in the sky changes that.

## The test

For every statistic used in a per-satellite decision, ask: **what is the population being
reduced over?**

| population | verdict |
|---|---|
| other **satellites** | ❌ **PEER COMPARISON** — purge |
| **probes** (deepest below-horizon PRNs) | ✅ noise by construction, the correct anchor |
| **instances / channels / elements** of the *same* satellite | ✅ estimate of ONE value |
| **time** samples of the same satellite | ✅ |
| **frequency bins** of a spectrum | ✅ mostly noise by construction |
| an **absolute physical bound** | ✅ |

⚠️ **A THIRD CATEGORY EXISTS AND MUST NOT BE CONFUSED WITH THE FIRST.** Some estimators combine
satellites to measure a quantity that is *genuinely common to all of them* — the receiver clock,
the NH overlay offset, the band group delay. That is not judging one satellite against another;
it is averaging repeated measurements of one physical thing, and removing it would remove the
measurement. But it carries its own pathology: **the estimate steps when the POPULATION changes**
(chord-clock-median-churn — "a clock that moves because the population moved"). These need a
*churn-immune formulation*, not deletion.

So: **Category 1 purge. Category 2 harden. Category 3 leave alone.**

## Category 1 — PEER COMPARISONS (purge)

| # | site | the comparison | replacement | state |
|---|---|---|---|---|
| P1 | `lib/stages/gnss/gnssFleetDll.hpp:766-780` (`_sig_k` = 3.0) | fast-loop window gate: `p_pow < 3 × median(this window's p_pow over armed PRNs)` → leak-only | `TrimPolicy::p_floor_abs`, the presence gate's probe-anchored absolute floor | **fix exists**, `--fleet-trim-floor-from-probes`, armed on 2 of 5 chains |
| P2 | `gnss_broker/fleet.py:413` | presence **q floor** fallback: `_floor([v["q"] for v in out.values()])` when probes < 2 | refuse (`present_gate = "UNANCHORED"`) | **fix exists**, `--presence-require-probes`, armed on 3 of 5 chains |
| P3 | `gnss_broker/fleet.py:446` | presence **p floor** fallback: `_floor([v["p_pow"] for v in out.values()])` | as P2 | as P2 |

All three are *fallbacks* — they run exactly when the probe anchor is missing, i.e. when
conditions are already degraded ([[chord-peer-relative-blindness]], seven instances and counting).

**Measured cost of P1, on sky 2026-08-27**, chains still on the peer median, satellites
unambiguously **on the peak** (q ≥ 2.5) losing windows to a gate that is supposed to ask only
"is this discriminator informative":

```
gps_l5 PRN 18  q 2.85   45.7% of windows discarded
gps_l5 PRN 20  q 3.25   34.5%
gps_l5 PRN 27  q 3.28   24.4%
```

## Category 2 — SHARED-PARAMETER ESTIMATES (keep the measurement, kill the churn)

**Parked as buglist #94, 2026-08-27, on KV's call** ("this seems like it could be a huge
change in behaviour"). The recommendation -- S2 first, alone, replacing the `mean(b)=0`
gauge with an absolute prior, with a one-cycle falsifier -- lives there.

| # | site | what it estimates | the churn pathology |
|---|---|---|---|
| S1 | `gnss_broker/deadreckon.py:42` | receiver clock, circular median over per-sat code offsets | **THE DECAY ROOT** — steps 1-2 chips on membership change, ~600 s timescale |
| S2 | `gnss_broker/state_filter.py:149` | joint-filter gauge, `mean(b) = 0` over ACTIVE sats | defines `b_i` as *deviation from the fleet mean*, so `b_i` is peer-relative **by construction**; a join/leave steps `clk` ~1 chip at 6 sats |
| S3 | `lib/stages/gnss/GnssCoherentCombiner.cpp:2067` | NH overlay offset, median of sats already agreeing ±3 of a pivot | robust, but the membership still moves |
| S4 | `gps_distributed_broker.py:1867` | F-engine axis, `max(pow_hop)` over `status.values()` | a MAX over a churning set — already max-filtered + snap-guarded, population still churns |
| S5 | `gnss_broker/receiver.py` carrier / code bias | receiver clock-frequency and per-band group delay | fleet aggregates, weighted by sat count |
| S6 | clock-bias solve, `median(det_dop - pred_dop)` over satellites | receiver clock-frequency bias | ⚠️ **and it is corruptible by LAG, not just churn**: `pred` is evaluated at NOW, so a stale detection makes `lag x dop_rate` a pure fabricated bias — measured +44 Hz at 48.9 s of search starvation (task #81), +68 Hz at 90 s, enough to drag a tracker off a 55-sigma satellite |

⚠️ **S2 is the one that is genuinely half Category 1.** The common mode (add `c` to `clk`,
subtract `c` from every `b_i`) is structurally unobservable, so *something* must fix the gauge —
but `mean(b)=0` fixes it **with the population**, which is why membership churn moves `clk`. The
principled replacement is an **absolute prior**: `b_i ~ N(0, σ_b)` with σ_b from physics (per-sat
ephemeris + group-delay error is bounded and small). A prior pins the common mode without any
reference to who is currently up, and a satellite joining or leaving moves nothing.

## The aggregator: audited, and clean

The aggregator (`chord_gnss_agg6_cuda.yaml`, pid 4139879) runs exactly four stages —
`bufferRecv`, `GnssChanAlignMerge`, `GnssChannelizedSearch`, `GnssChordDequantize`. **None
contains a live reduction over a population of satellites.** The single `median` in
`GnssChanAlignMerge.cpp:108` is prose, describing S6 above.

⚠️ It does carry the same disease on the **instance** axis: the merge "advances every input to
the MAXIMUM sequence currently held", a max over a set that changes as feeds join and leave —
which is why an F-engine restart pins `target` at a value post-reset feeds can never reach. That
is guarded (the controller-reset guard), and it is the instance-axis analogue of S4. Worth
knowing when hardening Category 2: *max/median over a churning set* is one bug with two axes.

## Category 3 — verified NOT peer comparisons (no action)

`gnssChannelizedDespread.cpp:89,166` and `GnssCoherentCombiner.cpp:1448,2602` — median of `dt`,
the record period (**time**). `GnssCoherentCombiner.cpp:1177` — `median(mag)` over `nb` rate-search
**bins**. `gnssElemCal.hpp:337,379` — median over live **elements** (imputation of one value).
`cudaGnssTrack.cpp:1040`, `gnssBroker.cpp:29,67`, `gnssBandPlan.cpp:37`,
`gnssChannelizedReplica.cpp:427` — **sorting for iteration order**. `combdll.prompt_cn0`,
the kcoh floor, and `fleet.py:405,442` — **probe-anchored**.

## The plan, in order

**DONE 2026-08-27 (`c4512de93`): P1, P2 and P3 are deleted, both flags burned from
cli.py and the yaml, and `test_epl_admit` now asserts the expressions are absent from
the source. The trackers were swept and carry zero cross-PRN coupling.**

~~**1. P1 → default, then delete the branch.**~~ Make the probe-anchored absolute floor
unconditional; when the broker has no probe anchor it must ship *refusal*, not 0 (0 currently
means "fall back to the peer median"). Then `_sig_k` and the `nth_element` block are dead code
and go. ⚠️ Needs a gather ≥ `97f6f258f`, and it fails **silently** on an older one.

**2. P2/P3 → default.** `--presence-require-probes` becomes the behaviour, and the peer branch is
deleted rather than left reachable. This is KV's own call from this morning restated: *"noisily
failing is better than accepting a bad number."*

**3. S2 → replace the gauge with a prior.** Biggest single win in Category 2, and the only one
where the peer-relativity is in the *state definition* rather than in a threshold. Verify against
the existing `coast_error` selftest plus a membership-churn test: adding/removing a satellite must
move `clk` by **zero**, where today it moves ~1 chip at 6 sats.

**4. S1 → churn-immune clock.** Once S2 lands, the joint filter's `clk` is the better clock and
the circular median becomes a cross-check rather than the source.

**5. S4, S3, S5 → churn audit.** Each already has partial treatment; the work is to state the
membership-invariance property and test it, not to rewrite.

**6. Enforcement.** `scripts/gnss/peer_audit.py` re-runs the classifier. Wire it into
`scripts/gnss/gate.sh` as a static leg once Categories 1 and 2 are closed, so a new peer
comparison cannot land silently — the same reasoning as the pyflakes leg
([[chord-shadow-was-dead-static-gate]]).

## Method, and what it does NOT cover

Two passes: an AST sweep for reductions (`median`/`mean`/`percentile`/`sorted`/`max`/`min`) whose
iterable is a per-PRN collection, over `python/scripts/gnss/**`; and a hand read of all 22
`nth_element`/`std::sort`/`median` sites in `lib/stages/gnss/*.{cpp,hpp}`. Every Category 1 and 2
entry above was then read in context and classified by hand.

⚠️ **LIMITS, stated so the next reader does not over-trust this.** The sweep finds reductions with
a *recognisable* reduce call. It will miss: a threshold computed in one function and used in
another; a peer statistic assembled by hand in a loop without a named reducer; anything reached by the aggregator
outside `lib/stages/gnss` (its four stages were checked by hand and are clean); and any comparison expressed as a ratio between two
satellites' quantities without a reduction at all. The C++ pass covered `lib/stages/gnss` only.
