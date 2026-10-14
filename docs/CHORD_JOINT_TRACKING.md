# Joint tracking: one receiver state, every satellite, all the time

Plan of record for the estimator restructure (KV + Claude, 2026-08-09). Supersedes nothing —
it is the frame the existing pieces slot into. Prior art it builds on: the fleet DLL and
fleet-coherent combine (`gnss_broker/fleet.py`), the dead-reckon model, the Receiver's
shared clock/bias store (task #27 M3), and the slewed seed refresh (#30).

**Status 2026-10-14: live, fed by one chain.** The joint solve is
`python/scripts/gnss/gnss_broker/state_filter.py` and every chain consumes it through
`joint-consume` (`config/gnss_chains_chord.yaml`). It carries clk, clk_rate, f_carrier, b_sat[i]
and rrate[i]; the code side is fed by gps_l5's search detections only (the gal_e5a feed was
removed 2026-10-11, #125), so b_sat exists for GPS satellites only and the other seven chains'
per-satellite code residual lives in the gather's fleet trims. `tau_band` is not built and §6
drops it: the measurement shows no per-band receiver delay, only per-constellation and
per-signal offsets. P3 and P4 (§4) are not started. §6 is the plan to close the system out
(buglist #191). The on-sky shadow and arming record that was §3b-3d is at
`be7d46e8e:docs/CHORD_JOINT_TRACKING.md`.

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
plan; element calibration stays node-local (P4) and ships solved gains.

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

What forced the revision (be7d46e8e:docs/gnss_gpu_search.md §11.21–11.22, task #33): the BRDC model
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
constraint per parameter. Measured on sky (2026-08-09, the P2a shadow): b_sat moves at
≤0.05 chips/min, against the
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

  P1  Per-channel spectrum export from the path-B assembler + the fleet-level slope fit in
      the broker (§2), measurement only first. Built (`--spectrum-endpoints`).
  P2  The filter (§3, §3a): ONE joint solve of clk, clk_rate and b_sat[i], never staged;
      consumers switched one per commit (`joint-consume`). Built and live.
  P3  Retire the E/L lanes (keep P + P_HEAD): path B goes 4 -> 2 lanes per PRN, doubling the
      PRN budget per chain or halving synthesis. Not started.
  P4  Element-domain calibration TRANSFER, node-local via path B's per-element visibilities:
      instrumental per-feed gains solved from any strong sat x geometric phase from
      ephemeris and array geometry x per-dish beam amplitude. Not started.

## 5. Non-goals and guards

  * No new per-sat loops, and no further investment in the #30 slew beyond its P2
    replacement. b_sat is NOT an exception: it is a slow bounded nuisance state inside the
    joint solve, never a per-sat loop with its own bandwidth (§3a).
  * Every phase gates on the broker transcript machinery; P1 by construction changes no
    POST. The known gate caveat: the E5a digest also moves when the sky's ephemeris does
    (task #29) — pin before relying on it for P2.
  * fleet_coherent's self-reference lessons apply verbatim to the slope fit: the fold's
    floor must be MEASURED on a channel-shuffled null each cycle, never assumed, and any
    derotation must be leave-one-out. (See the docstrings in fleet.py; they are the spec.)

## 6. Closing the system out (2026-10-14)

What runs today, per family, and where (measured 2026-10-12 00:30-12:00Z, the last quiet
half-day before the F-engine went off; tools and outputs in `fixtures/jointstate/`):

| family | joint state (broker) | beside it | gather | node |
|---|---|---|---|---|
| receiver | clk, clk_rate, f_carrier | per-chain median clock + sibling adoption (the level the joint delta lands on); l-a EMA; clock-frequency bias (search hints); NH clock-mod fit | | |
| constellation | nothing | | | |
| band / signal | nothing | | uniform standing trims per chain | |
| satellite | b_sat (GPS only), rrate (all chains) | P1 SatBiasFilter per chain; hold state, cp fits (gps_l5); carrier command applied | trim integrator per (chain, PRN) | seed, trim (4 s TTL), ElemCal, steer |

### 6.1 What the trims and the obs archive say

Standing C++ trims (median over the window, chips of the chain) and the obs `code_resid_m`
chain medians relative to gps_l5 (metres; the same-satellite cross-chain differences cancel
the receiver clock):

| chain | trim | vs gps_l5 | same-satellite differences |
|---|---|---|---|
| gps_l5 | 0.00 (per PRN -0.13..+0.23) | 0 | |
| gps_l2c | -0.03 (L2C chips) | -9.2 | l5 - l2c +10.3 m, per PRN +5.6..+16.3 |
| gal_e5a | +0.39 | +11.8 | e5a - e5b -0.16 m (p10/p90 -0.7/+0.9) |
| gal_e5b | +0.41 | +11.5 | e5b - e6 +0.4 m median; E2 -11.0, E11 -9.1, E12 -5.7, E8 -4.0 |
| gal_e6 | +0.19 (E6 chips = +0.38 at 10.23 Mcps) | +11.4 | |
| bds_b2a | +0.23 | +6.6 | b2a - b2b +15.0 +-0.7 m uniform; b2a - b3i +13.9 +-0.6 m uniform |
| bds_b2b | -0.25 | -9.2 | b2b - b3i -1.2 m |
| bds_b3i | -0.18 | -6.5 | |

1. **There is no per-band receiver delay.** Galileo's three carriers (1176, 1207, 1278 MHz) agree
   on the same satellites to 0.2 m in the median, and the per-PRN trim pattern is identical on
   E5a, E5b and E6 (E7 +0.1, E28 +0.08 on both E5 bands). One timestream through one PFB has one
   clock; the analog chain's group-delay dispersion across 100 MHz is below 1 ns. `tau_band`
   is dropped from the state.
2. **The uniform offsets are per constellation.** Galileo sits +11.5 m (38 ns) late against the
   GPS-fed clock on every band. The broadcast Galileo-GPS time offset (RINEX 4 GAGP A0 in the
   hourly BRDC) is -1 ns, so the offset is ours: the GPS chain's clock absorbs the
   constellation-common part of -gamma TGD (~+15 ns; the CAS DCB product is zero-mean per
   constellation and the broadcast mean is not added back) and the remainder is a receiver
   inter-signal term (replica or correlator convention between the GPS and Galileo codes).
   Static, so a nuisance state per constellation (isb_E, isb_C; GPS defines zero) absorbs it.
3. **The per-signal gaps are group-delay handling, not physics.** B2a carries CAS C6I-C5P, B2b
   the broadcast TGD2 fallback and B3I is the datum, and B2a differs from both by 14-15 m with
   +-0.6 m scatter across 24 satellites, tighter than any real satellite group-delay spread:
   one side carries a uniform error. L2C and E6 get no term at all (no CAS pair reached,
   broadcast 0): L5 - L2C reads +10 m with +-4 m per satellite, E6 reads -4..-11 m on E2, E8,
   E11, E12.
4. **The per-satellite part is b_sat and it is mostly atmosphere.** The same satellite carries
   the same deviation on every band of its constellation (C41 -0.5 chips on B2b and B3I). On
   gps_l5 the per-PRN b_sat medians sit within +-0.13 chips and swing to -1.2 chips at rise. The
   range model has no troposphere and no ionosphere (gnss_ephemeris.py, sky.py), so b_sat is the
   slant atmosphere (2.3 m / sin el dry tropo; 0.29 m per TECU at L5 times the obliquity) plus
   the group-delay residual and the ephemeris error along the line of sight.
5. **The fallbacks do not fire.** JOINT-CLK adopted on 1366-1367 of 1376 lines per chain; the
   joint code rate was taken on every cycle of every chain (zero `NOT taken`); the P1
   SatBiasFilter applies a median 0.01 chips (p90 0.03). The median clock is not a fallback but
   the level the joint delta is applied to, and the owner of the long-code segment.
6. **The P1 spectrum fit cannot gate a feed.** On a closed loop its peak/floor reads 1.00 in the
   median with 0.3-2.8% of entries at >= 1.5 (|tau| median 0.016 chips): the ratio is 1 by
   construction when tau ~ 0. On-peak evidence is the fleet DLL's q.
7. **Code-carrier coupling needs no step.** #93 measured the aiding gain at -0.06 with a 95% CI
   excluding the +-1 physics requires by 10x (24 h, 8 chains); the per-satellite code ramps are
   not range-rate errors. The coupling that exists stays as built: clk_rate to the code rate,
   rrate to the carrier command.

### 6.2 The state after the close

    x = [ clk, clk_rate, f_carrier, isb_E, isb_C, b_sat[i], rrate[i] ]

One unit for the code states: 10.23-Mcps chips, the first feeder's. Every feeder converts its
chips in (chip_rate ratio) and every consumer converts out, as `dr_joint_clk` already does; the
per-feeder unit declaration and the "mixed units -> refuse" rule go. The modulus stays the
shortest contributing code (1 ms). isb_sys is born with the constellation's first feed (prior
sigma 1 chip, process noise ~1 ns per hour) and GPS defines zero. b_sat keeps its prior gauge.
A consumer that refuses the joint clock coasts on clk + clk_rate dt, never on the median.

### 6.3 Steps and gates

Offline now, on the obs archive and the replays (no sky needed):

  0a. **Group-delay audit.** Add the missing terms (L2C: CAS C1C-C2L/C2X through the C1W pivot,
      else CNAV ISC_L2C; E6: C1C-C6C; B2b: C7D), decide which side of the B2a/B3I 14 m carries
      the error (PVT per-signal clocks on a day, `gnss_pvt.py`, and the ionosphere ratio between
      the pairs), and add the constellation-mean broadcast term back when CAS is used.
      Gate: `isb_obs.py` same-satellite differences read ionosphere only: band-ratio consistent,
      |median| < 1 m, no satellite beyond 3 m.
  0b. **Atmosphere in the range model.** Saastamoinen troposphere with a mapping function from
      the site height and a standard atmosphere; Klobuchar ionosphere from the BRDC (a GIM
      later). Gate: per-satellite `code_resid_m` against elevation flattens, and gps_l5's
      b_sat p10/p90 come inside +-0.2 chips on the 10-12 replay. The transcript digests move:
      re-bless with this step's commit, on the fresh replays #184 waits for.
  0c. **The state.** One unit with conversion at the feed; isb_E, isb_C; the coast-on-refusal
      consumer; selftest cases for each (a second constellation joining, an isb birth, a refusal
      coasting, a mass birth). Gate: selftest and `broker_equiv` green, fakefleet runs.

At the F-engine's return, broker only, one step per restart:

  1.  **Code feed from every chain, shadow.** The measurement is trkresid's observable
      (`cp_phys - cp_model - clk`, every chain, detectors or not): the DLL's replica placement,
      which is the sky only while the loop is on the peak. Gates on the feed: present, fleet q
      above its floor, |spec_tau| < 0.3 chips whenever the spectrum fit is valid, outside the
      transit veto for any satellite transmitting in the chain's band (#144 sky), after the
      warm-up, and a set gate of >= 3 satellites per chain per cycle. No fallback path: a
      satellite that fails a gate contributes nothing (the #85 feed's fallback to
      `seed + trim - model` was the #152 latch pathway). Served and logged, consumed by nothing.
      Gate (the GAP 2 mirror test, #85's form): the fed chains' y trends agree with the unfed
      chains' to 0.001 chips/s; isb_E and isb_C converge to the 0a table within 1 m; Galileo and
      BeiDou b_sat within +-0.3 chips; 24 h of `joint_latch_detect.py` with zero alarms and a
      re-derived reference.
  2.  **Consume on every chain.** `joint-consume: clk,rate,slew` everywhere; the seed offset is
      clk + isb_sys + b_i; the SatBiasFilter stops feeding seeds. Gate: readback trims per chain
      median within +-0.05 chips and per PRN within +-0.1 (the constellation offsets move from
      the trims into isb); the post-restart readback decay no slower than before; same-satellite
      `code_resid_m` differences unchanged from step 0a's table.
  3.  **Retire.** SatBiasFilter; the l-a EMA as a seed rate (keep the solve as the cp-rate
      tolerance reference until that reference is the joint rate); the sibling clock adoption
      as a level (the median solve stays as the modulus owner and the counterfactual the
      JOINT-CLK line prints). Dead-code pass, digests re-blessed.
  4.  Later and separate: the carrier command on the five uncommanded chains, one per restart,
      judged against the gal_e5a/gal_e5b envelope; P3 (the lanes) after #189; the element model
      stays node-local (#154).

Risks named in advance: the mirror (the trkresid observable is the sky only on-peak, so the q
gate is load-bearing and step 1's trend test is the proof); transits (feed and adoption frozen
inside the veto, #176's step guard stays); the long-code segment (the median keeps it until the
joint carries the NH epoch); mass birth (clk_rate prior 0.05 chips/s and the birth gate, as
#86 left them).
