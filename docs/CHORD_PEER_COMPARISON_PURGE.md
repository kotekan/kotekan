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

**6. Enforcement.** `scripts/gnss/site/peer_audit.py` re-runs the classifier. Wire it into
`scripts/gnss/site/gate.sh` as a static leg once Categories 1 and 2 are closed, so a new peer
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

## Appendix: the threshold and fallback audit (2026-08-27)

The systematic pass that preceded this purge, after two peer-relative thresholds turned up in
one afternoon.

### The test

A median is not the problem. It is the *right* tool as a robust centre and as a noise level —
when the population really is noise. The dangerous shape is narrower:

> **A population statistic used as a THRESHOLD against the same population it came from,
> where the fault being detected can move the whole population.**

Ask of every one: **could what I am trying to catch move the bar with it?** If yes, the bar
must be anchored somewhere the fault cannot reach — the **probes** for power, the **wall
clock** for time, an **absolute physical bound**, or the **historical** best.

### Method

`ast`-walk every broker module: bind names assigned from `median`/`percentile`/`mean`/
`sorted(...)[len//2]`, then find where those names reach a `Compare` or a multiplicative
scale in the same function. 19 flows. C++ scanned by hand for `nth_element`/`median`/`sort`
feeding a bar: 17 sites, 1 flow.

⚠️ Grep alone was useless — 99 textual hits across 20 files, which is how the two got missed.
The dataflow narrowing is what made it reviewable.

### Verdict: two broken, both now addressed

| site | reference | verdict |
|---|---|---|
| `gnssFleetDll::integrate` (C++) | 3× **the window's own median** | ❌ **BROKEN** — the weaker half of the array can never win a competition against its own median. Fixed by `TrimPolicy::p_floor_abs` (`--fleet-trim-floor-from-probes`). |
| `fleet.py apply_presence` fallback | `_floor(**the tracked population**)` | ❌ **BROKEN** — passes ~half by construction; measured 21/48 present and a q floor of 4.72 against the q≈4 ceiling. Fixed by `--presence-require-probes` (refuse, don't guess). |
| `apply_presence` primary q/p floors | the **probes** | ✅ different population, and probes are noise by construction |
| `combdll.prompt_cn0` q gate | `probe_q` median | ✅ probe-anchored |
| `combdll.coh_cn0` floors | `s2_inc` from probes | ✅ probe-anchored |
| `deadreckon.dr_clock_solve` | MAD vs an **absolute 100-chip bound** | ✅ dispersion test against a fixed physical bound |
| `fits.q_stall_verdict` | the **historical best** | ✅ time-anchored — its own comment: *"a degrading chain must not be allowed to redefine normal downward"* |
| `almanac` / broker clock bias | median of residuals → EMA | ✅ robust **estimate**, never a threshold |
| `clsibling` k-scan | best vs 2nd-best | ✅ within-scan significance |
| `GnssCoherentCombiner` rate search | peak / median of the **spectrum** | ✅ the spectrum genuinely is mostly noise bins |
| `GnssCoherentCombiner` NH consensus | median of offsets already agreeing ±3 | ✅ robust consensus |
| `gnssElemCal` self-reference | median of live elements | ✅ imputation of one value |
| `*Despread` / combiner `dt` medians | median record period | ✅ robust centre of a cadence |

**So the whole codebase contained exactly two, and both are now fixed.** That is a bounded
answer, which is what the audit was for — "we found two more, who knows how many remain" is
not a state to leave this in.

### The four classes, for future review

1. **Robust centre** — de-meaning, gauge reconciliation, imputation, consensus. Safe.
2. **Noise level over a genuinely-noise population** — periodogram significance, probe
   anchors. Safe *because of the population*, so the safety is an assumption worth writing
   down next to the code.
3. **Absolute or historical anchor** — a physical bound, the best ever seen. Safe.
4. **Noise level over a population that is mostly signal.** ❌ This is the bug. It always
   arrives as a *fallback*, written for a regime that no longer holds — both of ours were
   correct on the airspy prototype, where `--noise-probes` put real noise rows into the
   population, and became wrong on CHORD without anyone editing the line.

### ⚠️ THE BIGGER PATTERN: FALLBACKS DROP SAFEGUARDS (seven now)

The audit above was scoped to peer-relative *thresholds*. By the end of the same night the
count had grown past that scope, and the common factor was not the median at all — it was the
**fallback**:

| fallback | taken when | safeguard it silently dropped |
|---|---|---|
| `gnssFleetDll::integrate` window gate | always | probe anchoring → peer median |
| `apply_presence` floor | probes < 3 | probe anchoring → peer median |
| ephemeris "bridging on the last good" | `PREDICTION COLLAPSE` | **the constellation itself.** Root-caused 2026-08-27: `last_good` and `peak_n` were stored in the *shared* BRDC dict (`receiver.brdc()` hands all five chains one object), so `peak_n` became a max over G/E/C — BeiDou's ~13 sats judged against GPS's ~24, permanently "collapsed" — and the bridge served BeiDou **Galileo's and GPS's satellites, with their elevations**. The `min_prn` reading in `fixtures/open_20260827_bds_prn7_path.txt` was the right shape and too small. Fixed by scoping on `(sysc, min_prn)`. |
| clock-bias **stale rescue** | no multi-sat solve for 842 s | **averaging** — ONE sample (sd 12.7 Hz) became the permanent warm-start reference, logged as "hardware news (GPSDO re-settled?)" |
| hourly-station merge `len(bodies) >= 4` | always | **coverage.** It counted *sources that answered*; NRC1 and STJO carry no BeiDou and still filled the quota, so BRUX — sixth in an all-Canadian-first list, worth 15 in-slot BDS alone — was never reached. 15 BDS/11 in-slot where the union gives 37/23. |
| CDDIS daily mirror | BKG unreachable | **existence.** It asked CDDIS for BKG's product name under `/daily/YYYY/brdc/`, a directory that has only ever held legacy GPS/GLONASS short-name files. Added 2026-07-21 *for a BKG outage*; 404'd silently on every call until the next one, five weeks later. |
| `_src_failed` negative cache | any exception | **the path/host distinction.** A 404 on a path CDDIS never publishes (the current day) blacklisted the whole host for 300 s — taking out the yesterday fetch that is the actual fallback. |

Every one reproduces the primary path's **output** while dropping one of its **safeguards**, and
every one runs precisely when conditions are already degraded — so the moment the safeguard
matters most is the moment it is not there. Seven independent instances is past coincidence;
treat "what does the fallback drop?" as a standing review question, not a per-bug discovery.

⚠️ **Two of these could not succeed at all, and that is its own class.** The CDDIS mirror
returned 404 on every call it ever made; a search that cannot return a positive is a gate that
cannot fail ([[chord-broker-refactor]]). When a fallback exists *for* an outage, the only proof
it works is exercising it against the real remote — which is why `test_skyscope.py` asserts the
URL shape rather than trusting the code to be reachable.

⚠️ **And the digest gate is blind to every one of them.** Replays pin the sky through
`GNSS_BRDC_DIR`, so neither the fetch path nor the collapse bridge is ever executed: all seven
fixtures stayed EQUIVALENT across all four fixes. `gnss_broker/test_skyscope.py` is where these
live, and each assertion was proven red against the code it replaces before being trusted.

⚠️ The clock-bias one was harmless only by luck: `--clock-bias-file` is unset, so the poisoned
calibration died with the process. With it set, one starved re-solve would have persisted a
wrong warm-start across runs.

⚠️ **Both threshold bugs were fallbacks, and that is the lesson to generalise.** The primary paths were
right and probe-anchored; the fallbacks preserved 2019-era behaviour for a fleet that no
longer resembles it. A fallback is code that runs exactly when the assumptions are already
violated, so it deserves *more* scrutiny than the primary path, not less — and, per KV, should
usually **refuse** rather than approximate.
