# Fleet-combined DLL — design (2026-08-03)

## 1. The problem, in one number

A commercial receiver correlates the whole 20.46 MHz L5 lobe. One CHORD tracker instance
correlates **7 × 195.3 kHz = 1.37 MHz — 6.7%**, because the F-engine comb spreads L5 across all
eight nodes and each GPU holds seven channels of it. That is **−11.8 dB**, and no single
instance can ever do better: the bandwidth is not on the machine.

Measured on sky: the full-band-equivalent C/N0 is **45.5 dB-Hz** (inverted from `amp_snr` ~5
over a 10.5 ms record), but what the code loop actually sees is **33.7 dB-Hz**. Trackable in
principle, marginal in practice — the correlation quality `q = 2P/(E+L)` sits at 1.0–1.7 against
a gate of 2.2, clearing it only in bursts, so the trim updates episodically and never settles.

Recovering the missing 11.8 dB is what this design is for.

## 2. Why it is cheap: the DLL discriminator is NON-COHERENT

`disc = (E − L)/(E + L)` is built from **powers**. Powers add. So combining instances needs:

* **no phase alignment** between nodes,
* **no sample-level time sync** beyond identifying the same integration window,
* **no coherent machinery** at all.

This is the whole reason the fleet DLL is tractable where cross-node *coherent* combining is
not. It is also why shipping tracker record streams between hosts (40 Mbit/s each, a combiner
hierarchy per signal and per constellation) is the wrong shape of solution: the useful content
is three numbers per PRN per window.

## 3. The minimum data set

Per PRN, per integration window, per instance:

| field | why |
|---|---|
| `prn` | key |
| `pow_hop` | **absolute hop index** of the window (see below); groups instances exactly |
| `e_pow`, `p_pow`, `l_pow` | the correlator powers — **raw, not the ratio** |
| `n_chan` | how many channels contributed (weighting, and detecting a dead instance) |

**Key on the hop index, not UTC.** `GnssChanMetadata::sample_seq` is an int64 "absolute sample
index of the frame's first hop", shared across every node off the same F-engine counter; the hop
index is `sample_seq / fft_len`. Using it instead of a capture-UTC double:

* **Exact.** UTC is derived as `frame0_utc + seq/rate` — a float re-derivation of an integer we
  already have. Grouping instances by it needs a tolerance; grouping by hop index is `==`.
* **It is already the currency everywhere else.** The seed keys on `ref_hop`. Every replica
  generator forms `C(n) = arg + n·cps` over the **absolute sample index**. The search reports
  `ref_hop`. UTC in the record is the odd one out.
* **It removes the window-alignment question entirely.** §6's "group by nearest epoch, do not
  build a synchroniser" becomes "group by equal hop index", with no tolerance to choose and
  nothing to tune.

The combined record carries capture-UTC at `RECORD_UTC_SLOT` (9–10) and does **not** carry the
hop index today. `RECORD_FLOATS` is 24 and the CMB slots run to 18, so an int64 aliased into a
free pair (19–20) adds it without disturbing the existing schema or the consumers of slot 9.
Keep UTC — `overlay_apply` and the coherence ladder's `coh_span` use it — and add the hop index
alongside rather than replacing it.

**And anchor the trackers' reported window to the same index.** Internally the despread is
already absolutely referenced to the sample index, so this is a reporting change, not a
behavioural one: the record should state the hop its correlators were accumulated over. Then
seed (`ref_hop`), despread anchor, EPL telemetry and trim all sit in one integer currency, and
the broker never converts between two clocks to decide whether two measurements are the same
measurement.

**Raw powers, not `dll_disc`.** `get_status` publishes only the ratio today, and ratios do not
sum: `(ΣE − ΣL)/(ΣE + ΣL)` is not any function of the per-instance `(E−L)/(E+L)`. Publishing
the three powers is the single enabling change, and they already exist in the combined record
(`CMB_E_POW` = 11, `CMB_L_POW` = 12, prompt via `CMB_AMP_INCOH` = 3) — they are simply not
exported.

**Bandwidth.** 6 floats × 32 PRNs = 768 B per window per instance. At the record rate that is
0.6 Mbit/s; aggregated to 1 Hz it is **6 kbit/s per instance, 0.1 Mbit/s for all sixteen**. It
fits the REST polling the broker already does — no new transport.

## 4. Where the combination happens: the broker, and it is already most of the way there

The broker polls `get_status` every `--interval` (2 s), and its existing `--dll-gain` loop
already:

* applies the trim **to the seed at POST time**, leaving the stored seed as pure fit/coast
  state — so no new field and no new path is needed to deliver a shared trim;
* uses a **leaky** integrator, so discriminator noise cannot random-walk the trim into the
  clamp (a pure integrator did exactly that on L1, 2026-07-07);
* integrates **once per fresh measurement**, detecting a changed `disc` — because the combiner
  emits at ~1 Hz while the poll runs faster, and integrating a stale value over-applies the
  gain.

The change is surgical: build `disc` from **summed** E and L across instances instead of reading
each instance's ratio. Everything downstream of `disc` stays as it is.

## 5. The part that is NOT optional: one commanded phase

E/P/L are measured **relative to the code phase the instance despread at**. Today every
`cudaGnssChordTrack` runs its **own** local trim loop (`trim_gain` 0.15, gated at
`trim_quality_min` 2.2), so instances drift to different delays — and summing E/L taken at
different delays *smears* the discriminator instead of sharpening it. The gain would be thrown
away silently.

So the local loops must go: set `trim_gain: 0` on the trackers and let the broker's shared trim
be the only code loop. That also removes a duplicated controller nobody wants to reason about.

Do **not** instead ship each instance's applied trim for correction — it works, but it keeps two
loops and makes the combination depend on state that arrives with the measurement.

## 6. Latency — the thing to worry about, and the numbers

| stage | delay |
|---|---|
| combiner integration window (100 records, rolling) | ~1.05 s |
| broker poll interval | 2.00 s |
| seed POST → tracker applies | <0.10 s |
| **round trip** | **~3.2 s** |

Against that, what the trim actually has to track: the measured residual is **0.0033 chips/s**
(trims moved ±0.15 chips over 45 s), so the lag over a full loop is **0.011 chips — 2% of the
±0.5 chip pull-in**. Latency is comfortably not the limit.

Loop bandwidth should stay ≲ 1/(10·τ_d) ≈ 0.03 Hz (time constant ~40 s); the existing
`--dll-gain 0.25` with its leak already sits near there.

**Window misalignment does not need solving, and with the hop index it cannot even arise.**
Instances free-run, but every window is labelled with an exact absolute hop, so grouping is an
integer match. Where windows genuinely differ (an instance integrating over a different span),
the mismatch is visible in the index rather than hidden in a float tolerance. Residual code
motion between nearby windows is <0.002 chips at 0.0033 chips/s and below anything the
discriminator resolves, so nearest-hop grouping is safe when exact match is unavailable.

## 7. The stability condition, and it is a hard one

The clock offset is **3.45 chips/s** (0.337 ppm, measured). It is fed forward by the seed's
`code_phase_rate`, and the trim only absorbs what is left. If that fit is ever lost, the trim
alone would face **11 chips per loop** — far outside pull-in, unrecoverable, and it would look
like the DLL diverging rather than the feed-forward failing.

So: **gate the shared trim on a live rate fit.** If `code_phase_rate` is stale or absent for a
PRN, hold the trim rather than integrate. This is the same discipline as `--carrier-min-sig`.

## 8. Expected gain — and it is NOT the gain this document first claimed

Summing K instances takes the correlated bandwidth from 1.37 MHz toward the full 20.46 MHz.
What that does to `q` is the part I had wrong, and it inverts the gate design:

**Summing does not raise `q`.** `q = 2P/(E+L)` is a ratio of means, and every tap's mean scales
by K: a fleet of eight reports the same 1.5 a single node does. The first draft of this section
predicted `q` moving from ~1.5 to clear of 2.2. It will not, ever.

**What collapses is the variance.** Each summed power has K times the mean and K times the
variance, so `q`'s spread falls as 1/√K. A signal-free PRN's `q` tightens onto 1.0 — from a
measured single-instance tail of **1.87** to roughly **1.3 at K = 8**.

So the gain is a **lower bar**, not a higher statistic. That is still exactly what is needed —
a real `q` of 1.5 that is 1.5σ inside one instance's noise is ~3σ clear of eight instances'
noise — but it means:

* **The gate must be measured, not fixed.** Any constant is correct for exactly one fleet size,
  and a constant chosen for K = 1 (2.2, correct there) rejects every real signal once the sum
  has tightened the distribution around 1.0. `fleet_dll` re-measures the floor each cycle as
  median + kσ (MAD) over the live `q` population — most tracked PRNs are signal-free at any
  moment, so the median *is* the no-peak value — and logs it every time it is used.
* **This resolves what looked like a contradiction.** On 2026-08-03 I proposed lowering
  `trim_quality_min` to 1.3 and it was 1.5σ into the noise. At K = 8 the same 1.3 is ~3σ clear.
  Same number, opposite verdict, because the *population* changed. The rule survives intact:
  never move a threshold to buy sensitivity — measure the population you are thresholding.

**Still do not lower `trim_quality_min`.** That knob gates the per-instance loop, whose noise
population really does tail to 1.87. It is the fleet floor, and only the fleet floor, that may
legitimately sit lower.

## 8a. The prerequisite this design cannot supply: a seed inside the pull-in

A DLL pulls in ±0.5 chips. Measured offline with `scripts/gnss/e2e` on 2026-08-03:

| search comb | refine | seed error handed to the trackers |
|---|---|---|
| stride 4 (**2 nodes**, today) | −13.63 | **−13.188 chips** — a grating lobe |
| stride 1 (**8 nodes**) | −0.07 | **+0.373 chips** — chain closes |

Thirteen chips is 26× outside the pull-in, so with a 2-node comb **no** code loop — local,
fleet, or otherwise — can recover it, and the fleet DLL's 11.8 dB buys nothing. The grating
lobes are 13.09 chips apart because the covering channels are strided; only the full eight-node
comb has g = 1 and no lobes. **Bring the fleet up first.** Adding `--seed-cp-rate` on top of
stride 1 holds the whole chain at 0.373 chips over the tested records; with the rate absent it
walks to 1.16 chips, which is §7's gate in miniature.

## 9. Order of work

1. ✅ Carry the absolute hop index in the combined record (int64 at `CMB_HOP_SLOT` 19–20) and
   publish `e_pow`, `p_pow`, `l_pow`, `n_chan`, `pow_hop`, `pow_fft_len` in
   `GnssCoherentCombiner::get_status`. Additive; changes nothing that reads the record today.
   The hop comes from `GnssChanMetadata::sample_seq` on the input frames, which the record
   producers already stamp — no new plumbing, and `fft_len` on the combiner is the only new
   config key.
2. ✅ Broker `fleet_dll()`: `--dll-combiners`, group by (PRN, `pow_hop`), sum, compute `disc`
   from the sums. Every existing gate kept; a PRN with fewer than `--dll-min-instances`
   agreeing instances falls back to the single-combiner path, so a partly-down fleet degrades
   instead of stalling.
3. ✅ `--local-trim-gain`, default 0, so there is exactly one code loop. (Measured 2026-08-03:
   the local loop was already inert — every tracker reported `trim +0.00` because no PRN ever
   cleared its own 2.2 gate. Nothing is being given up.)
4. ✅ Gate on a live `code_phase_rate` (§7) — hold rather than integrate when it is absent.
5. ⬜ **Bring the other six nodes up (§8a).** Until the comb is stride 1 the seed is 13 chips
   out and none of the above can matter.
6. ⬜ Measure `q` before/after on the same satellites. That is the acceptance test — not
   `coherence_s`, which is a carrier-side statistic and a separate problem. Note §8: `q` itself
   is not expected to rise, so the test is **the measured noise floor falling** and the trim
   integrating continuously rather than in bursts. Baseline captured 2026-08-03 before the
   change, 2 nodes: per-instance `q` max 1.75, median ~1.1, every trim +0.00.
