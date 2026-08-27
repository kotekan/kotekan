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

⚠️ **Both bugs were fallbacks, and that is the lesson to generalise.** The primary paths were
right and probe-anchored; the fallbacks preserved 2019-era behaviour for a fleet that no
longer resembles it. A fallback is code that runs exactly when the assumptions are already
violated, so it deserves *more* scrutiny than the primary path, not less — and, per KV, should
usually **refuse** rather than approximate.
