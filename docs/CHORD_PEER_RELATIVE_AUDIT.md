# Peer-relative threshold audit

Prompted by KV, 2026-08-27, after a third and fourth instance turned up in one afternoon:
*"That's two we didn't know about this afternoon, after we thought we'd fixed it by getting the
first one days ago."* Opportunistic fixing was clearly not converging, so this is the
systematic pass.

## The test

A median is not the problem. It is the *right* tool as a robust centre and as a noise level —
when the population really is noise. The dangerous shape is narrower:

> **A population statistic used as a THRESHOLD against the same population it came from,
> where the fault being detected can move the whole population.**

Ask of every one: **could what I am trying to catch move the bar with it?** If yes, the bar
must be anchored somewhere the fault cannot reach — the **probes** for power, the **wall
clock** for time, an **absolute physical bound**, or the **historical** best.

## Method

`ast`-walk every broker module: bind names assigned from `median`/`percentile`/`mean`/
`sorted(...)[len//2]`, then find where those names reach a `Compare` or a multiplicative
scale in the same function. 19 flows. C++ scanned by hand for `nth_element`/`median`/`sort`
feeding a bar: 17 sites, 1 flow.

⚠️ Grep alone was useless — 99 textual hits across 20 files, which is how the two got missed.
The dataflow narrowing is what made it reviewable.

## Verdict: two broken, both now addressed

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

## The four classes, for future review

1. **Robust centre** — de-meaning, gauge reconciliation, imputation, consensus. Safe.
2. **Noise level over a genuinely-noise population** — periodogram significance, probe
   anchors. Safe *because of the population*, so the safety is an assumption worth writing
   down next to the code.
3. **Absolute or historical anchor** — a physical bound, the best ever seen. Safe.
4. **Noise level over a population that is mostly signal.** ❌ This is the bug. It always
   arrives as a *fallback*, written for a regime that no longer holds — both of ours were
   correct on the airspy prototype, where `--noise-probes` put real noise rows into the
   population, and became wrong on CHORD without anyone editing the line.

## ⚠️ THE BIGGER PATTERN: FALLBACKS DROP SAFEGUARDS (seven now)

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
