# Joint tracking: one receiver state, every satellite, all the time

Plan of record for the estimator restructure (KV + Claude, 2026-08-09). Supersedes nothing —
it is the frame the existing pieces slot into. Prior art it builds on: the fleet DLL and
fleet-coherent combine (`gnss_broker/fleet.py`), the dead-reckon model, the Receiver's
shared clock/bias store (task #27 M3), and the slewed seed refresh (#30).

## 1. Why

The current loop is per-satellite, per-axis heuristics — EMAs, quality gates, trims — around
a model-predict core. Every lock loss diagnosed in the week to 2026-08-09 was the same
disease in a different coat: feedback paths that existed but were unconsumed (the la pool,
#30), per-sat loops paying per-sat noise for quantities that are common (the clock), and
gates tuned against noise that also starve real signal (the railed disc, E5a's q floor,
`carrier-gain 0`). Functional, nowhere near optimal — KV's words, and correct.

**The structural fact that changes the answer: CHORD's position is known to centimetres.**
The entire unknown state is small, and everything per-satellite is computable from it + BRDC:

    x = [ clk_code   (chips)      receiver code-clock offset          fast (seconds)
          clk_rate   (chips/s)    its rate — the (l-a)                fast
          f_carrier  (Hz)         carrier frequency offset            fast
          tau_band   (chips)      per-BAND group delay (cable+PFB)    static-ish (hours)
          b_sat[i]   (chips)      per-sat slow bias                   slow (minutes) ]

`b_sat` is physical, not a fudge: iono (3–9 m = 0.1–0.3 chips at 1176 MHz), tropo, BRDC
ephemeris error. It is why E4 sits at (E−L)/(E+L) ≈ +0.8 while E28 is balanced — and it is
per-path TEC, which the science side wants anyway.

One joint estimator of x, fed by all satellites at once, steering every replica open-loop
from the state — vector tracking. A satellite needs zero individual SNR to stay locked; it
is locked to the ensemble. Lock loss becomes "the clock broke" (obvious) or "one sat's bias
diverged" (flagged, tracked through). This is the "never lose lock, every sat always" target.

## 2. The measurement: a delay is a phase ramp across frequency

dφ/df = −2πτ. The per-channel prompt correlations contain the full-band delay at
matched-filter precision, σ_τ ≈ 1/(2π·β_rms·SNR) — milli-chips at fleet SNR. E/P/L keeps
three lag samples of that spectrum and discards the rest; the E5a disc rails because per
narrow channel E≈P≈L (0.97 replica coherence) — a slope fit is immune to that entirely.

**Sparse-comb ambiguity is the CHORD-specific hazard.** One instance's 7 channels at
3.125 MHz stride alias at 3.27 chips — the measured tracker grating lobes. The defence is
the same one the search discovered: the fleet's UNION of channels (79 unique across 12
instances, different offsets per node) fills the comb. Two consequences:
  * the fit must be FLEET-LEVEL, in the broker, where the channels converge;
  * instances carry unknown constant phases φ_i, so the fold must solve them jointly:
    alternate (φ_i | τ) — given τ, φ_i = arg Σ_{c∈i} E_c A_c e^{i2πf_cτ}; given φ_i,
    coherent fold over all channels. Rank-1 alternating LS in miniature; it converges in a
    few rounds from τ=0 (the replica placement) because a locked seed is within ~1 chip.
    ⚠️ Magnitude-per-instance summing does NOT work: |·| erases exactly the cross-instance
    phase that discriminates the true peak from the 3.27-chip lobes.

**Where the data comes from — and the shipping rule.** (KV 2026-08-09: the ~30 Gbps figure
is not a hard wall — the real constraint is that the broker is PYTHON, and it acquires more
data every time a band or a chain is added. Collapsing over elements before shipping is the
right call for that reason, not for link capacity.) Per-channel prompts do not currently
leave the GPU: path A's despread sums channels in the kernel. Path B's assembler
(`GnssGpuRecordAssemble`) consumes per-channel × per-element visibilities in HOST memory
before summing — the export is a host-side accumulation there, no kernel change. Ship
SUFFICIENT STATISTICS only: per (PRN, channel) complex sums post-element-combine,
per emit window ≈ n_prn·n_chan·16 B ≈ 4 kB/instance — 12 orders under KV's ~30 Gbps
ship-everything figure. Per-element data never crosses the network at any phase of this
plan; element calibration stays node-local (§4) and ships solved gains.

## 3. The filter (replaces the EMA collection)

Weighted least squares / Kalman over x, cycle cadence, in the broker:
  * measurements: per-sat fleet delay (slope fit, §2) + per-sat carrier rate
    (`deep_rate_hz` where `deep_rate_q` clears its measured gap) + the GPS chain's cp-fits
    (already la's source);
  * process noise sets the timescales the EMAs conflated: clock fast, b_sat slow
    (minutes), tau_band near-static;
  * every chain's seeds steer from the state; the #30 slew becomes a one-line consumer
    (converge held phase to state-predicted phase) instead of a special case;
  * per-sat innovation gating replaces the per-sat quality-gate zoo: a sat whose residual
    diverges is flagged and COASTED ON THE STATE, not dropped.

### 3a. REVISION 2026-08-09: b_sat is a first-class state, and the split is BANDWIDTH

What forced the revision (docs/gnss_gpu_search.md §11.21–11.22, task #33): the BRDC model
is per-satellite wrong by **±3–7 chips** on fresh detections — measured against the search
on GPS, quasi-static, drifting monotonically through a pass, cause still open (candidate:
per-direction instrumental delay; back-burnered until other bands make it attackable).
That error is NOT common-mode: no clock, clock-rate or band-delay value absorbs it.

Two consequences, both structural:

**(1) A state of only shared parameters is misspecified**, and the current system already
demonstrates what misspecification costs. Nothing can represent the per-sat residual, so
two actuators fight over it: the #30 slew pushes each seed toward the model while the DLL
trim pushes the replica back toward the sky. That beat IS the ~600 s limit cycle (trim
±1.1 chips, disc railed 10–50% of samples, deep 20↔1300) — the drift KV kept seeing on E4,
and the sweep #30 never actually cured. Adding more SHARED states cannot fix it. The
residual has per-satellite structure and must be representable, or something downstream
keeps absorbing it badly.

**(2) The win is FEWER FITS, and mostly because of slower timescales** (KV's framing, and
the correct one — an earlier draft of this section said "bandwidth, not parameter count",
which gestures at the same thing but hides the economics). What a parameter costs is not
its existence but the degrees of freedom it eats, and that scales with how FAST it varies:
over an observation window T, a state with correlation time τ_c consumes ~T/τ_c degrees of
freedom. So N per-satellite loops running at seconds are expensive, while N per-satellite
biases drifting over minutes-to-hours are nearly free — the same data buys far more
constraint per parameter. Measured here: b_sat moves at ≤0.05 chips/min (§3b), against the
per-sat DLL trim it replaces, which swung ±1.1 chips on a ~600 s cycle. Slower state,
fewer fits, more signal per degree of freedom.

That is also why the ensemble still carries a fading satellite: the FAST dynamics — clock,
clock rate, ephemeris geometry — are shared and estimated from everyone, while the per-sat
term is too slow to have gone anywhere while the sat was dark. It is a nuisance parameter,
not a tracking loop. This is the standard precise-positioning structure (shared clock +
per-satellite biases), and the two designs are separated by one measurable test:

> Take a satellite to no usable SNR. If its replica keeps moving with the shared clock
> state while its bias holds frozen at its last strong value, the architecture is right.
> If it drifts, it was tracking itself and the ensemble was decorative.

**Rejected alternative, recorded so it is not re-proposed:** "anchor each sat to its own
held track, propagated forward." That gives every satellite its own fast dynamics and
fails the test above — a fading sat has nothing to ride. It was proposed on 2026-08-09
and withdrawn the same day.

**b_sat and clk are degenerate at a single epoch** — N satellites, N+2 unknowns, N
measurements — and the ONLY thing separating them is that the clock moves fast and the
biases do not. That separation is process noise, i.e. a filter. **So P2 cannot be staged
into "biases first, clock later": that estimates biases with no clock removed, which is
precisely today's failure.** One joint solve, or nothing.

**The biases are a measurement campaign, not the end state.** If the model error is
directional (elevation correlates at −0.39, but same-elevation sats differ 20×, so
elevation alone is not it), the per-sat table is the data from which a shared few-parameter
direction map is fitted — which then PREDICTS the bias for a satellite never yet tracked,
at birth, and collapses N nuisance parameters back into a handful of shared ones. That is
strictly better than per-sat biases and folds into P4's calibration work.

## 4. Phases

  P1  Per-channel spectrum export from the path-B assembler + fleet-level slope fit in the
      broker. MEASUREMENT ONLY first (the #30 rule: measure the statistic before the loop):
      log τ per sat alongside the disc; verify sign/scale offline against e2e's injected
      truth and on sky against (E−L)/(E+L); characterise on-signal vs on-noise and
      stability. No seed is touched → all transcript digests stay green.
  P2  The filter: x above, seeds steered from state, slope-fit τ as the code measurement.
      The per-sat b_sat absorbs what the #30 slew institutionalizes today (holding the seed
      at a model that is biased by iono). Gate: transcripts + e2e closure + on-sky A/B
      against the #30 baseline numbers (reseeds/30 min, deep stability).
      REVISED 2026-08-09 (§3a): ONE joint solve — clk, clk_rate and b_sat[i] estimated
      together from the fleet τ, separated by process noise, never staged. b_sat's range
      is sized for the MEASURED model error (chips), not for iono (0.1–0.3), and is seeded
      at birth from the first strong τ rather than walked in at loop gain. Success
      criterion, and the reason this kills the limit cycle rather than damping it: the slew
      and the trim acquire ONE target — `model + clk + clk_rate·Δt + b_sat` — so the trim's
      residual collapses to noise. Steps:
        P2a  Joint solve in SHADOW: logged beside the five estimators it replaces (l−a EMA,
             clock EMA, per-sat cp-fits, plus the trim's and slew's implicit states),
             consumed by nothing. Compare on sky.
        P2b  Switch consumers ONE PER COMMIT: l−a EMA → clk_rate, clock EMA → clk, slew
             target → state. Each with its own transcript + A/B.
        P2c  The weak-sat test above, deliberately: mask a strong sat's measurements and
             confirm it coasts on the state rather than drifting.
  P3  Retire E/L lanes (keep P + P_HEAD): path B goes 4 → 2 lanes/PRN — DOUBLE the PRN
      budget per chain (E5a is at 128/128) or half the synthesis cost, the scaling
      limiter. Do this BEFORE #31's data channels and new bands.
  P4  Element-domain calibration TRANSFER, node-local via path B's per-element
      visibilities: instrumental per-feed gains (direction-independent) solved from any
      strong sat — bootstraps in seconds — × geometric phase from ephemeris + array
      geometry × per-dish beam amplitude. Every sat gets full array gain including ones
      too weak to self-cal. Folds into #21 (GPUs → nodes → elements) and #26 (path B
      primary), not new scope.

## 5. Non-goals and guards

  * No new per-sat loops, and no further investment in the #30 slew beyond its P2
    replacement. b_sat is NOT an exception: it is a slow bounded nuisance state inside the
    joint solve, never a per-sat loop with its own bandwidth (§3a).
  * The airspy prototype keeps its per-sat architecture untouched.
  * Every phase gates on the broker transcript machinery; P1 by construction changes no
    POST. The known gate caveat: the E5a digest also moves when the sky's ephemeris does
    (task #29) — pin before relying on it for P2.
  * fleet_coherent's self-reference lessons apply verbatim to the slope fit: the fold's
    floor must be MEASURED on a channel-shuffled null each cycle, never assumed, and any
    derotation must be leave-one-out. (See the docstrings in fleet.py; they are the spec.)

### 3b. P2a SHADOW RESULT ON SKY (2026-08-09, 25 min, GPS L5, 7 sats)

The go/no-go for the whole revision was: **do the biases hold steady while the clock stays
smooth?** They do.

    CLOCK  mean +149.51 chips, sd 0.31 over 25 min          (median clock, same data: 152.7)
    RATE   mean +0.00038 chips/s, sd 0.00102 = 1e-4 ppm     (l-a EMA scatter: 0.007 ppm)
    BIASES sd about a linear trend, per sat:
           G11 0.049   G26 0.180   G21 0.182   G18 0.184   G20 0.184   G23 0.200   G25 0.202
           drift rates -0.047 .. +0.025 chips/min; spans 0.3-1.9 chips
    3943 updates, 0 innovation rejections.

Reading it:

  * **The +-3-7 chip model error decomposes cleanly** into one smooth clock plus per-sat
    biases stable to ~0.2 chips around a drift of at most 0.05 chips/min. A bias needs 20
    minutes to move one chip; the DLL pull-in is +-1 chip and the peak is ~0.5 wide. So it
    IS a nuisance state, exactly as section 3a claims, and the fast dynamics stay shared.
  * **clk_rate is 70x quieter than the l-a EMA it replaces**, on sky (the offline
    prediction was 200x against synthetic noise; 70x against the real thing). This is the
    single clearest argument for P2b's first consumer switch.
  * **The joint clock sits ~3 chips below the median clock, and that is arithmetic, not
    disagreement**: the biases are skewed (-5.4 .. +4.1), so their median (+2.5) is not
    their mean (0 by gauge). Median-clock = joint-clock + median(b) = 149.5 + 2.5 = 152.0
    vs the 152.7 logged. The old clock was never "the clock" -- it was the clock plus
    whichever satellite happened to sit in the middle of the bias distribution, which is
    why it moved whenever the constellation changed.
  * **Zero rejections in 3943 updates**: the innovation gate is not doing hidden work, so
    these numbers are the raw behaviour.

KNOWN CAVEAT, carry into P2b: the filter is OVERCONFIDENT -- reported sigma_clk 0.038
against an actual scatter of 0.31, a factor of 8. Either q_clk is too tight or the 0.3-chip
measurement sigma is optimistic (per-sat errors are almost certainly correlated cycle to
cycle rather than white). Harmless while nothing consumes the covariance; it must be fixed
BEFORE any consumer gates on sigma, or the gate will be 8x too tight.
