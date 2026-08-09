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

**Where the data comes from — and the bandwidth rule.** Per-channel prompts do not currently
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
    replacement.
  * The airspy prototype keeps its per-sat architecture untouched.
  * Every phase gates on the broker transcript machinery; P1 by construction changes no
    POST. The known gate caveat: the E5a digest also moves when the sky's ephemeris does
    (task #29) — pin before relying on it for P2.
  * fleet_coherent's self-reference lessons apply verbatim to the slope fit: the fold's
    floor must be MEASURED on a channel-shuffled null each cycle, never assumed, and any
    derotation must be leave-one-out. (See the docstrings in fleet.py; they are the spec.)
