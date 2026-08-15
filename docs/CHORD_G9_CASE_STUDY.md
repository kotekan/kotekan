# G9 case study: what works, what doesn't, at search SNR 2350

**2026-08-15 ~01:00 UTC, gps_l5, PRN 9.** Taken because G9 was the strongest satellite on the
chain (`get_detections` snr **2350**) while the viewer showed only the search looking healthy.
At that SNR every stage should be unambiguous, so any stage that is *not* unambiguous is a
real defect rather than marginal signal — which is the whole reason to do a case study on the
brightest thing in the sky rather than a typical one.

---

## The verdict, up front

⚠️ **CORRECTED 2026-08-15 ~02:00.** The first version of this document claimed the cross-node
combine was losing 7.6 dB. **That was an arithmetic error of mine and there is no such loss** —
see §3. Coherent combining of N instances gives **√N**, not N. Everything else below stands.

**More is working than the viewer suggests. The combine is fine; the served products are not.**

| stage | number | verdict |
|---|---|---|
| search | snr 2350 | ✅ |
| **code tracking** | `fleet_q` **3.22** vs floor 1.10, `dll_disc` +0.087 | ✅ **on-peak** |
| per-instance detection | `inst_snr` med 7.72, hi 12.41, lo 2.46, n=10 | ✅ |
| within-instance coherent integration | 12→100 records, snr 1.63→10.86 = ×6.66 | ✅ **80% of ideal** |
| fleet deep | `deep_snr` 32.3 vs floor 2.67 = 12× | ✅ detects |
| carrier arc (ADR) | `adr_arc` 1, **3187 s locked**, 303938 records | ✅ |
| **cross-node coherent combine** | fleet/median-instance ×4.18 (12.4 dB) vs √N ideal ×3.16 (10.0 dB) | ✅ **at/above ideal** |
| served C/N0 | `cn0_coh_db` **20.5** | ❌ see below |
| cross-instance phase | median \|corr\| 0.310 = 3.5× the 1/√128 floor | ✅ what per-record SNR 0.77 predicts |

---

## 1. Code tracking is on-peak, and that is new as of tonight

`fleet_q = 2P/(E+L)` is **3.22** against a measured floor of 1.10. q = 1.0 is exactly the
no-peak value and 4.0 is a clean lock at 0.5-chip spacing, so 3.22 is a real, well-centred
peak. `dll_disc` +0.087 and `dll_trim` −0.044 — converged and holding.

Hours earlier the same satellite sat at **q 0.91–0.98 with `present: False`**, frozen by #49's
prompt-gate latch. Arming `dll-deep-gate` on gps_l5 admitted it on `deep_snr` and the loop
walked the tap onto the peak. **This is the one thing that changed, and it changed the rest of
the chain with it** (see docs/CHORD_GNSS_SHARED_DLL.md §11).

## 2. Within one instance, coherent integration works

The published `rungs` ladder is the best instance's deep at 12 / 25 / 50 / 100 records:

    12 rec  snr  1.63
    25      snr  4.38
    50      snr  8.12
    100     snr 10.86      ->  x6.66 over the 12-record rung

Ideal coherent growth over that span is ×8.33, incoherent ×2.89. **×6.66 is 80% of coherent** —
the deep fold, its rate search and the phase model are all doing their job inside an instance.
Nothing here needs fixing.

## 3. Across instances, the combine is working — ⚠️ my original claim here was wrong

    median instance   7.72
    best instance    10.86
    FLEET            32.26

    measured  fleet / median instance      x4.18  = 12.4 dB
    IDEAL coherent over 10 instances       x3.16  = 10.0 dB     <- sqrt(N), NOT N

⚠️ **THE FIRST VERSION OF THIS SECTION SAID THE IDEAL WAS ×10 (20 dB) AND CONCLUDED 7.6 dB WAS
LOST. THAT IS WRONG.** Coherent combining of N instances with independent noise adds signal as
*Na* and noise as *σ√N*, so the SNR gain is **√N = 3.16 = 10.0 dB**. I had labelled ×3.16 as
the *incoherent* value; it is the coherent one.

So the combine is running **+2.4 dB above** ideal equal-amplitude coherent gain — which is what
MRC weighting does when the instances are unequal (`inst_snr` spans 2.46 to 12.41). Checked
against `sqrt(Σ ρᵢ²)` for several spreads consistent with lo/med/hi: 24.4–26.8, versus 32.3
observed, i.e. +1.6 to +2.4 dB.

**There is no missing 8 dB.** ⚠️ If anything the fleet number is *mildly optimistic* — beating
MRC is not possible — which is a smaller, opposite-signed question about how fleet `deep_snr`
is normalised against the per-instance one, and is worth a look on its own.

⚠️ **THIS ERROR COST AN HOUR OF MECHANISM-HUNTING AND IS WHY EVERY HUNT CAME UP EMPTY.** No
delay (tilt fit residual 2.06 rad — the channel phases are not on a line), no delay drift, and
no stable per-channel constant (split-half with a leave-one-out reference: excess **−0.03 dB**
over a permuted-label null). Those nulls were correct. They were nulls because nothing was
missing.

The cross-instance |corr| of 0.310 is likewise not a symptom: at a per-record per-instance SNR
of ~0.77 (7.72/√100), the expected pair correlation is ρ²/(1+ρ²) ≈ 0.37. Measured 0.310. That
is the number agreeing with physics, not disagreeing with it.

⚠️ **Addressing was ruled out cleanly and that still stands:** 128 hops in the union, 128
shared by all 10 instances (100%), none missing.

## 4. Two published numbers that do not add up

⚠️ **RESOLVED, and it is worse than a normalisation — see §7.** Both of these are fold
products, and the fold is not measuring detection.

* **`cn0_coh_db` 20.5.** The array's full-band-equivalent C/N0 for L5 is ~45.5 dB-Hz, and #47
  measured `cn0_coh` reading ~41 dB-Hz *on noise*. 20.5 on the brightest satellite on the
  chain is below both. Worth deriving from first principles rather than trusting: the deep
  statistics themselves (`deep_snr` 32.3 over `coherence_s` 1.049 s) imply ~27 dB-Hz, so the
  coherent chain as a whole is running ~18 dB below the incoherent expectation — consistent
  with §3 rather than a separate fault.
* **`spec_tau_chips` 1.196** claims 1.2 chips of code error while q = 3.22 says the tap is on
  the peak. `spec_peak_ratio` is only 1.066, i.e. a weak fit, so the likeliest reading is that
  spec_tau is unreliable here — but it is exactly the far-regime signal #50 wants to wire into
  the loop, and wiring in a statistic that disagrees with q by 1.2 chips on a locked satellite
  would be a mistake. **Resolve before #50 proceeds.**

Minor, noted for completeness: `nh_phase` −1 while the search reports `nh` 11; `s4` 0.0 while
`s4_raw` 0.467; `snr_q` None.

## 5. What this says about the viewer

The viewer shows the search healthy and little else because the two things it leans on —
`cn0_coh_db` and the coherent amplitudes — are precisely the products of the stage that is
losing 7.6 dB. **The tracking underneath is healthy and the viewer has no way to show that.**
`fleet_q`, `dll_disc`, `prompt_lock` and the ADR arc length are all published already and all
say "locked"; none of them is on the panel. That is #57's honest-health-metric argument,
answered by a concrete case.

## 5a. ⚠️ RETRACTED: the per-instance drift hunt

This section reported a per-instance phase drift and a +2.39 dB recovery from a fitted
per-instance rate, presented as part of the (non-existent) 7.6 dB. Two things are wrong with
it and both are worth keeping visible:

1. **It was chasing a loss that was not there** (§3).
2. **It was on the wrong axis.** An instance is a bag of ~7 channels selected by
   `freq_id mod 8` — a routing decision taken after the signal path. A per-instance number is a
   per-channel effect averaged inside a bag, and reporting it at the bag level hides the
   variable that matters. KV has now corrected this twice; see the `chord-nothing-is-per-node`
   note.

Redone on the channel axis with a leave-one-out reference (`scripts/gnss/chan_phase.py`), the
answer is a clean nothing: no delay, no delay drift, and no per-channel constant that survives
a split-half (excess −0.03 dB). The channels are as aligned as their SNR allows.

## 6. What to do next, in order

1. **The served products, not the combine.** `cn0_coh_db` 20.5 on the brightest satellite on
   the chain is the real anomaly (§4) — the combine feeding it is healthy, so the fault is in
   how C/N0 is derived or normalised, which is #47's family and a much narrower search.
2. **Why the fleet beats MRC by ~2 dB** (§3). Small, but you cannot exceed MRC, so the
   per-instance and fleet `deep_snr` are probably not normalised the same way — and every
   comparison of the two (including the one that produced the phantom 7.6 dB) depends on it.
2. **Re-measure #61's fold churn now that a satellite is on-peak.** The CV ~0.6 was measured
   when every tap was off-peak; if the churn follows q it was never a fold problem.
3. **Resolve `spec_tau` vs `q` before #50.**


---

## 7. ⚠️ THE ANSWER: `deep_snr` is not a detection statistic (#66)

KV, on being shown a search SNR of 8000 against a deep of 17σ: *"that simply cannot be right."*

**The null was already running.** `--noise-probes` seeds the deepest **below-horizon** PRNs as
a pure-noise anchor. Live on gps_l5, probes at elev −65, −64, −64:

| PRN | in search? | deep_snr | ×floor | coh_frac |
|---|---|---|---|---|
| **10** | **NO — probe, elev −64** | **29.9** | **11.2×** | **0.744** |
| 24 | NO — not detected | 10.6 | 4.0× | 0.710 |
| 26 | yes, real | 1.3 | **0.6×** | 0.113 |
| 8 | yes, real | 1.9 | 0.7× | 0.158 |
| 21 | yes, real | 2.1 | 0.8× | 0.158 |

**A satellite 64° below the horizon reads 11.2× its noise floor**, while genuinely detected
satellites sit at 0.6–0.8×. And `deep_snr` does not track the search: PRN 9 (search **10997**)
→ deep 17.5, while PRN 30 (search 33, barely above the ~18 noise ceiling) → deep **76.2**.

This collapses three separate puzzles into one fault:

* §4's `cn0_coh_db` 20.5 — derived from the fold;
* §3's "the fleet beats MRC by ~2 dB" — impossible, therefore inflation;
* #61's fold churn — a statistic that fires on noise churns.

**Mechanism is a hypothesis, not a conclusion.** The deep path re-searches rate and phase, so
it is a max-over-many-cells statistic; `coherent_sum()` states that it sets *"phase = 0; no
alignment searched, hence no extreme-value bias in the floor"*, i.e. the honest path is the
unsearched one. If the searched path reuses `residual_snr()` on the already-derotated series,
the orthogonal component that *is* the noise estimate was minimised by the same fit. **Test it.**

⚠️ **OPERATIONAL HAZARD.** #49's deep gate *trims* on `deep_snr`. The three PRNs armed tonight
are all search-detected, so nothing trims on noise right now — but **widening the gate
fleet-wide would admit PRN 10, a below-horizon probe, and trim it.** Do not widen until this
is fixed. The q floor is unaffected: probe `q` reads ~1.0, correctly, because `q` is not a
fold product.

⚠️ **AND THE METHOD LESSON: check the probes first.** They cost nothing, they are already
seeded, and they are the only rows in the system that are noise *by construction*. This
document spent an evening measuring coherence against carefully-built shuffled nulls without
once consulting the null the instrument already had.

---

## 8. THE ROOT, found offline (#51): the loop is rate-limited 29× below the drift

Run `scripts/gnss/e2e` at **production search density** (the default is a sparse 27-channel,
stride-4 comb whose −13.188 chip seed error is the known 2-node grating lobe — *check the node
count first*):

    ./scripts/gnss/e2e --s-stride 1 --s-nchan 105

    SEARCH LEG ERROR: +0.373 chips          <- the seed is GOOD, sub-chip
    rec  age_s      cp cmd     phase err   P/P_true    q
     0   27.0   180139.827      -0.425      0.4336   0.899
     1   37.0   198937.464      -1.314      0.0026   0.130
     2   47.0    13135.419      -2.523      0.0629   0.181
     3   57.0    31933.697      -4.054      0.0434   0.143

**A good seed, zero noise, and the tracker still walks off it at −0.121 chips/s**, with q
collapsing 0.899 → 0.143 in 30 s — the same q ≈ 1 the whole fleet shows on sky. The drift rate
is identical in the 2-node run, so it is independent of seed quality and of the grating lobe:
it is the *propagation*, not the acquisition.

The mechanism is arithmetic. A Doppler error ΔF drives the code phase at ΔF × (f_chip/f_carrier)
= ΔF × 8.695e-3 chips/s. The e2e's default seed carries `dop_rate` −0.36893 Hz/s that its truth
does not, so ΔF grows to ~14 Hz over the window → 0.12 chips/s. (That mismatch is deliberate in
the harness; what it establishes is the **sensitivity**, not the live cause.)

### Now compare it with what the loop is allowed to do

| | |
|---|---|
| drift | 0.121 chips/s → **1.46 chips per 12 s cycle** |
| maximum correction | **0.05 chips** per event (`--dr-slew-cap`) |
| | **the loop is 29× slower than the drift** |

Even at the benign `dop_rate 0` arm (0.0138 chips/s) it is **3.3× slower**. Inverting:

> **The code loop can only hold against ~0.48 Hz of Doppler error** (0.05 chips ÷ 12 s ÷
> 8.695e-3). Beyond that, the tap walks away faster than the DLL can pull it back — *regardless
> of how good the discriminator is*.

### Why this is the root and not another symptom

It is upstream of, and sufficient to explain, everything else in this document and in the
probe audit: q ≈ 1.0 fleet-wide (the no-peak value), prompt power blind at AUC 0.55, C/N0
incoherent swinging 40 dB, the fold folding noise, `deep_snr` inflating on it. It also explains
the one thing that looked hopeful tonight — PRN 4 and 9 reached q 3.2 after #49's gate let the
loop trim, then fell back. **The loop can pull in; it cannot hold.**

And it connects to a root already on file: [[chord-clock-median-churn]] records the clock
median stepping **1–2 chips** on membership churn every ~600 s. At 0.05 chips/event that is
4–8 minutes of walking off, every churn.

⚠️ Raising the cap has been tried and reverted (2026-08-11: acquisition authority engaged 22
minutes late and degraded every Galileo and BeiDou satellite). So the fix is not simply a
bigger cap — the drift has to come down (feed-forward accuracy, reseed cadence) *and* the cap
has to be matched to whatever drift remains. The 0.48 Hz tolerance is the budget both sides
have to meet.


---

## 9. Where the session ended (2026-08-15 02:25)

**The root is §8: the code loop is rate-limited ~23× below the drift.** Everything in §3–§7 —
the phase statistics, the fold, the C/N0 — was measured on an off-peak tap and is downstream
of it.

`--fast-trim-hz 5` is **armed on gps_l5**, four chains as control. Achieved **~3.1 Hz** against
a 1.94 Hz break-even. Armed PRN 4's `dll_disc` went +0.443 → +0.0823 (5.4×), but the controls
improved in the same two minutes across a restart, so **benefit is not established**;
`/tmp/qwatch.jsonl` on cf06 is recording the paired duty for 8 h.

⚠️ **It has to move to C++.** ~3 Hz is an order of magnitude below the 23.84 Hz the #59
transport already delivers, and the remainder is Python decode plus GIL in a process that also
runs five chain loops. Note the constraint that shapes the design: **loops live in the broker**
(an instance sees 7 of ~105 channels), so the target is the *actuator and discriminator* in
C++, still answerable to the broker's policy — not moving the loop onto a node.

⚠️ And it was **1.5 Hz, below break-even, until the armed-PRN filter went in** — the loop was
decoding every PRN and using one. The offline gate could not catch that: its fixture has 3
PRNs, so it hit 4.39 Hz either way. **A gate whose fixture is smaller than production cannot
measure a cost that scales with production.**
