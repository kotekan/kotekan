# G9 case study: what works, what doesn't, at search SNR 2350

**2026-08-15 ~01:00 UTC, gps_l5, PRN 9.** Taken because G9 was the strongest satellite on the
chain (`get_detections` snr **2350**) while the viewer showed only the search looking healthy.
At that SNR every stage should be unambiguous, so any stage that is *not* unambiguous is a
real defect rather than marginal signal — which is the whole reason to do a case study on the
brightest thing in the sky rather than a typical one.

---

## The verdict, up front

**More is working than the viewer suggests, and one stage is measurably losing 7.6 dB.**

| stage | number | verdict |
|---|---|---|
| search | snr 2350 | ✅ |
| **code tracking** | `fleet_q` **3.22** vs floor 1.10, `dll_disc` +0.087 | ✅ **on-peak** |
| per-instance detection | `inst_snr` med 7.72, hi 12.41, lo 2.46, n=10 | ✅ |
| within-instance coherent integration | 12→100 records, snr 1.63→10.86 = ×6.66 | ✅ **80% of ideal** |
| fleet deep | `deep_snr` 32.3 vs floor 2.67 = 12× | ✅ detects |
| carrier arc (ADR) | `adr_arc` 1, **3187 s locked**, 303938 records | ✅ |
| **cross-node coherent combine** | fleet/median-instance = **×4.18 (12.4 dB)** | ❌ **7.6 dB lost** |
| served C/N0 | `cn0_coh_db` **20.5** | ❌ see below |
| cross-instance phase | median \|corr\| 0.310 = 3.5× the 1/√128 floor | ⚠️ marginal |

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

## 3. Across instances, the combine is barely better than incoherent

    median instance   7.72
    best instance    10.86
    FLEET            32.26

    fleet / median instance  = x4.18  = 12.4 dB
      ideal COHERENT over 10 instances  x10    = 20.0 dB
      plain INCOHERENT sum              x3.16  = 10.0 dB

**Coherent combining should buy 10.0 dB over incoherent. It is buying 2.4 dB. 7.6 dB is on the
floor.** That is task #10, and this is the cleanest measurement of it yet — on a satellite
bright enough that no one can argue it is a marginal detection.

The corroborating number is the fit-free one (`scripts/gnss/telem_align.py`, which never fits
a phase and so cannot manufacture agreement): across the 10 instances on 128 shared hops,
median per-pair |corr| is **0.310 against a 1/√128 = 0.088 chance floor** — real (3.5×) but
marginal, with a per-pair spread from 0.120 (at the floor) to 0.523. Compare a previously
measured bright satellite at **0.973**. The instances are looking at the same sky and only
partly agreeing about its phase.

⚠️ **Addressing is not the problem, and this rules it out cleanly:** 128 hops in the union,
**128 shared by all 10 instances (100%)**, no instance missing any hop. Whatever is
decohering them is not the transport and not collation.

## 4. Two published numbers that do not add up

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

## 5a. #10 followed up the same night: the per-instance offset MOVES

`scripts/gnss/inst_phase_drift.py`, on G9, 256 records, 32-record (0.336 s) blocks. It asks
the one question `|corr|` cannot answer: is the per-instance offset **constant but noisy**
(in which case 0.310 is just per-record SNR and nothing is lost) or **time-varying** (in which
case one constant rotation can never align it, and that is the 7.6 dB)?

⚠️ **The null is the constant hypothesis itself.** Permuting records destroys time structure
but PRESERVES a constant offset, so the shuffled lag-1 is the best a constant model could
achieve on this data. Measured lag-1 *below* it therefore means "less repeatable than a
constant".

    FLEET: median lag-1 0.628 against a shuffled null of 0.999
    block-phase scatter 3-6x the noise expectation on the high-SNR instances
      (cx19.1 46 deg vs 13, cx27.0 42 vs 12, cx42.0 49 vs 14)

**The offset moves.** Fitting one differential RATE per instance and rebuilding the fleet sum
on the same records:

    per-instance differential rate:  -0.290 .. +0.130 Hz   (0.42 Hz spread)
    fleet coh_frac  constant-only 0.0405 -> constant+rate 0.0533   = +2.39 dB
    shuffled-null gain                                             = +0.04 dB

**+2.39 dB is real** (the null shows the fit is not inflating it from noise) **but it is only
about a third of the 7.6 dB.** So a per-instance rate is part of the mechanism and not all of
it — do not ship a rate and declare #10 closed.

The 0.42 Hz spread between instances of the *same chain* is the physically interesting number:
these are the same PFB, the same satellite and the same sky. G9's `doppler_rate_hz_s` is
−0.318, so a 0.42 Hz spread corresponds to ~1.3 s of relative timing — an order of magnitude
larger than #46's measured 0.105 s, so #46 as it stands does not obviously account for it.

## 6. What to do next, in order

1. **#10 — find the other ~5 dB.** §5a establishes that the offset drifts and that a
   per-instance rate recovers +2.39 dB of the 7.6. What accounts for the rest is open. The
   0.42 Hz inter-instance rate spread wants a mechanism: it is far larger than #46's 0.105 s
   timing spread can explain against G9's −0.318 Hz/s Doppler rate, so either the timing
   spread is much worse than #46 measured, or the rate is not coming from timing at all.
   ⚠️ Before shipping a per-instance rate into `fleet_coherent`, note that a fitted rate is
   exactly the machinery that produced the retracted-then-restored ×38 claim; it needs the
   shuffled-null comparison every time, which `inst_phase_drift.py` does by default.
2. **Re-measure #61's fold churn now that a satellite is on-peak.** The CV ~0.6 was measured
   when every tap was off-peak; if the churn follows q it was never a fold problem.
3. **Resolve `spec_tau` vs `q` before #50.**
