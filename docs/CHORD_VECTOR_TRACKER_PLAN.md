# CHORD GNSS — the vector tracker: state and the road to GAP 3

Written 2026-08-26 (KV-directed), during the analog-frontend outage. Armed state in the
tables below was **grepped from `config/gnss_chains_chord.yaml`**, which is the authority —
never from a handoff. Re-grep before trusting any of it.

Companion documents: `CHORD_JOINT_TRACKING.md` (the architecture), `CHORD_BUGLIST.md`
(#90–#93 in full), `CHORD_REWORK_PLAN.md` (the structural work, complete).

---

## 1. What the joint state is, and what already rides on it

`JointReceiverState` carries, in one solve:

* `clk`, `clk_rate` — the shared receiver clock (the fast dynamics that carry a fading sat)
* `b_sat` — per-satellite code bias, gauge-pinned by `mean(b) = 0`
* per-band `tau`, receiver `f_carrier`
* per-satellite `rrate`, `rrate_dot` — the carrier-rate states

The founding principle, and the thing to re-read before proposing any change to it: **the
vector-tracking win is BANDWIDTH, not parameter count.** A per-sat bias bounded at a chip or
two and moving on minute timescales adds no fast bandwidth — it is nuisance state, not a
loop. The effective fast-unknown count stays 3–4 however many satellites are in the solve.

The discriminating test is a selftest, not an opinion: mask a satellite's measurements; if
its replica keeps moving with the shared clock while its bias holds frozen, the structure is
right (600 s dark → 0.2 chips while the clock moved 6). If it drifts, it was tracking itself.

⚠️ It cannot be staged. `b_sat` and `clk` are degenerate within an epoch (N sats, N+2
unknowns, N measurements); the only separation is process noise. One joint solve or none.

---

## 2. The three gaps — scorecard

### GAP 1 — arm the carrier command — ✅ CLOSED AS A MECHANISM (2026-08-25)

`rrate-command: true` has been live on gal_e5a since 08-23 (commit 0eca22112, KV's re-arm
#3). Final paired read on a calm plant: H2 paired diff e5a-vs-e5b **−0.03** (parity).

**Read that verdict precisely.** Commanding is *safe* — that is what closed. The *benefit* is
plant-state dependent: real on 08-23, at or below sky variance in the 08-25 2 h windows. The
judgement metric is **per-sat q SD against in-poll controls**; `|dcmd|` is descriptive only.

What made it possible was the long-span fine observable (`--rrate-phase-span-s`, commit
6f732aa3b): the per-poll fine rate has a ~1.0–1.1 Hz per-sat floor, and the tau-scaling curve
(1.12 Hz at 2 s → 0.31 at 16 s → 0.18 at 32 s, converging as 1/tau because `res_cycles`
increments telescope) says that floor is a **tuning ceiling, not physics**. Span-16 is live on
both Galileo chains, e5a in shadow and e5b feeding, with a non-overlap throttle (consecutive
poll-cadence spans share 14/16 of their data and would make the row ~√8× overconfident).

Retired for cause: **arm 17** (`--kcoh-rate-from-row`) — the carrier mirror.
Named open lever: **the slew rail**, 3–4 of 7 commands railed.

### GAP 2 — extend code-side consumption — ✅ CLOSED (2026-08-23), differently than drawn

The plan was "add `slew` to one DR chain and keep the others as controls". What shipped was
better: all four DR chains consume `clk,rate` from the median-gauge joint state, and gal_e5a
*feeds* it via #85's spec-anchored y (`sky − model` identically: held + applied_trim +
spec_tau − cp_pred). Feed-and-consume with the loop falsifier clean — the delta trend matches
the unfed controls, 20× under the +0.065 self-reference signature. 14-sat pooled GPS+GAL
state, 94% acceptance, sigma 0.083 against the 0.09–0.19 GPS-only floor.

### GAP 3 — couple code and carrier — ⏳ OPEN, re-architected as buglist #93

The original premise — "the carrier rrate row predicts code drift" — **died F1** on the calm
plant (sign 57%, r 0.059, 151 points, both gal chains). The autopsy said the design was
wrong, not the physics:

* *Reference mismatch*: rows are gauged to `mean(rrate) ≡ 0` across sats; trims are
  referenced to the (l−a) clock. Two independent common-mode conventions. De-meaning per poll
  lifted r from 0.059 to 0.281 — a 5×, exactly as the gauge predicts.
* *Amplitude*: rows carry 15–44 cm/s equivalent; measured trim ramps are ~0.9 cm/s. The code
  loop is **today the better estimator of range-rate error**; the rows are ~97% carrier-only
  content. Usable content ≈ 1% of row value, so any hand-set consume gain was ~100× wrong.

⇒ The aiding gain must come **out of the filter**, never hand-set. That is #93.

---

## 3. #93 — the critical path, and where it actually is

**The model.** Per satellite, in m/s at the tap: `n_i` non-dispersive drift (orbit + clock +
tropo model error — THE AIDING TARGET), `i_i` dispersive/iono (code +, carrier −), plus
per-chain reference states `r_code` and `r_carr` estimated **explicitly**, never as two
independent pre-applied conventions. Observables per sat: code `y_c = n + i − r_code`
(trim ramp), carrier `y_φ = n − i − r_carr` (per-sat ADR increment). Within a band,
`(y_c − y_φ)/2 = i` is the iono observable with geometry cancelled and `(y_c + y_φ)/2 = n`
is the aiding estimate. Across the band pair (1176.45 / 1207.14) iono scales as 1/f², ratio
1.053 ≈ equality — so band-consistent divergence is real iono and band-inconsistent
divergence is junk. Four observables per sat over 2–3 states: observable.

### STEP 1 — the candidate aiding gain — E1a SATISFIED, **E1b is the kill clause**

Measured 2026-08-26 00:5x over ~2 h: de-meaned ADR-vs-trim pooled r = +0.121, slope
**+0.0060 ± 0.0044**, both chains individually positive (e5a +0.0054, e5b +0.0104). The row
beside it read r = −0.086 — **the raw ADR is the better observable, as designed**. The slope
sits on the registered ~1% calm-plant prior.

E1b has never been exercised, and it carries the pre-registered kill clause:

> de-meaned r < 0.3 **in a disturbed window too** ⇒ the carrier genuinely cannot aid the
> code, and GAP 3 closes as **NO-AID**, honestly.

That is a legitimate outcome, not a failure, and it should be called if the data says it.
What was missing was a regime marker; as of 2026-08-26 `fixtures/gap3v2_decompose.py` derives
one from the **D1/D3 detector episodes in the same log** (brownout ±60 s, qualified sawtooth
±300 s) and reports CALM and DISTURBED strata separately. Selection is on the detector,
never on the trim slope — that would be selection on the dependent variable.

### STEP 2 — is the divergence iono? — ⛔ BLOCKED on the cross-band anti-correlation

Per-sat carrier rates anti-correlate between E5a and E5b: **r ≈ −0.33 (n=57, ~2.5σ)**.
Physics cannot do this — iono is same-sign on both carriers. Four instrument hypotheses were
tested and **refuted** the same night: e5a sign flip, 16b command-noise coupling
(corr = +0.010, n=48), the cmd add-back reconstruction (corr = −0.066), and de-meaning
(raw r = −0.326; median only sharpens it to −0.442). The anti-correlation rides the **raw
fine rate itself**.

Leading physical candidate, not yet a conviction: the band-alternating excursions documented
in E32's #90 anatomy, seen at population level. If that holds, **#93's estimator needs a
per-band asymmetric term**, not only the symmetric iono state.

Remaining diagnostics, in order: (1) the neither-sat-fed stratum — does it couple to feed
mechanics? (now in the judge); (2) median-vs-mean de-meaning artefact check; (3) if both
pass, a brief 16b-off toggle window — **KV's call, it is an armed state**.

⚠️ Do not build the estimator before this resolves. A joint estimator built on a frame with
an unexplained band anti-symmetry would launder it into the states.

### STEP 3 — the estimator

Per-sat `(n_i, i_i)` + per-chain references; either extend the joint filter (couple
`b_sat`/`rrate`, add iono) or a standalone per-sat KF fed by both loops. Consume `n̂_i` into
`cp_rate` behind a default-off flag, sigma-weighted. Weights from **measured** variances, so
a 1%-reliable predictor gets ~1% weight from the filter itself.

Redo the **mirror check** for the new loop this creates: aid → code alignment → fold SNR →
arc breaks → ADR availability (value-clean but availability-coupled; the stale-row sigma
inflation clause carries over). Per-sat q SD clause is standing policy. Judge on trim-ramp
shrinkage and wipe frequency, paired against an unaided control chain, **verdicts recorded
per plant regime** — calm and disturbed separately, because 08-25 proved they differ.

### Traps carried in

`res_cycles` arcs break at re-anchor (accumulator identity; wall_dt tripwire) · never fit
across #91 collapses or restarts · #92's REBASE-ADJUST posts move e5a trims by known deltas
and a ramp estimator must subtract them (`_g3_adjcum`; sub-0.3-chip ones contaminate
silently) · the calm-plant "1% usable" number is the PRIOR, so a measured slope ~30× larger
in a calm window is a red flag, not a win.

---

## 4. Armed state (grepped 2026-08-26 from the yaml — re-grep before trusting)

| chain | joint-consume | armed extras |
|---|---|---|
| gps_l5 | `rate,slew` | `joint-p2c-rotate`, deep-gate `"4 9 27"` + from-search 100, 2 PRNs model-primary |
| gal_e5a | `clk,rate` | **`rrate-command`**, **`fleet-trim-rebase-adjust: 1`**, `joint-model-primary`, `joint-feed-spec`, `joint-shadow`, `rrate-phase-span-s: 16`, `rrate-phase-feed: false`, deep-gate `"33"` |
| gal_e5b | `clk,rate` | `rrate-phase-feed: true`, `rrate-phase-span-s: 16`, deep-gate `"33"` |
| bds_b2a | `clk,rate` | — (the paired searchless 1176 control) |
| bds_b2b | `clk,rate` | — |
| common | — | `rrate-state`, `rrate-kcoh-feed: 1`, `joint-rereference` |

**Deliberately disarmed:** `kcoh-rate-from-row` (arm 17, retired for cause — the carrier
mirror) · all three `reseed-spec-tau` sites (#90 F2/F3) · `joint-model-primary` on bds_b2a ·
`joint-consume: slew` on gal_e5a (off with the feed).

---

## 5. Beyond GAP 3 — what end-to-end vector tracking still needs

Closing GAP 3 is necessary but not sufficient. One joint state driving both loops on all five
chains additionally requires:

1. **Command rollout** — `rrate-command` is armed on 1 of 5 chains. e5b, b2a, b2b and l5 are
   still open-loop on carrier. One chain at a time, controls in the same polls.
2. **The slew rail** — 3–4 of 7 commands rail, and a railed target move is invisible
   (`chord-slew-cap-saturation`). This bounds what any coupling can deliver, so it should be
   measured before the coupling is judged.
3. **The flip (model primacy)** — live on gal_e5a and two PRNs of gps_l5. Full model-primacy
   everywhere is the actual end state: every seed generated from the joint model, the sky
   used to correct it rather than to produce it.
4. **The plant-calmers**, each of which now has an instrument it previously lacked:
   * **#92** — the seed-rebase/trim handover. Armed on gal_e5a. P2 metric exists as of
     2026-08-26 (D3 v2 REBASE-WIPE counts, e5a armed vs e5b unarmed, paired in time).
   * **#91** — the brownout trim-hold policy. **D1 already computes the trigger** (see §6).
   * **#90** — the off-peak disarm latch. D2 is measuring the base rate, unarmed; if nothing
     is reported over a week, v3 should never be armed.

---

## 6. #91 — the brownout trim-hold, and why it is the cheapest win on the board

The anatomy (E3, 2026-08-25 00:39–00:48): gal_e5a's KCOH `sig_sum` fell 43,255 → 97 → 14,
every satellite at once, for ~9 minutes, while gal_e5b held ~100k. Presence rides the fold,
so both searchless 1176 chains disarmed to ZERO, **every standing trim was released**, and
the (l−a) clock — measured from the collapsing population — swung ±96 chips and had its
garbage rate adopted into seeds. A positive feedback that sustained the outage.

Releasing the trims turns a 9-minute fold fade into a full per-satellite re-pull. The fix is
policy, not physics: **when the axis is fresh and the noise floor is flat, a chain-wide
presence loss should HOLD standing trims, not release them.**

Everything it needs now exists:

* **D1** (`BrownoutDetector`) already computes "present count collapsed against this chain's
  own 600 s peak" as a labelled episode, read-only, in production.
* The **trim store** already proves holding works across a discontinuity (adoption at arming,
  max-age 300 s refusing stale saves).

So the work is: gate the arming-hold expiry on `ctx.brown.active()` behind a default-off
flag, with a max hold age so a genuine set-and-sink cannot be held forever, and never adopt
an (l−a) rate fitted on a population below a floor. Broker-side, digest-testable, no node
restart. It directly reduces the trim-rebuild churn that chops #93's measurement windows,
which is why it is worth doing *before* the GAP 3 estimator rather than after.

---

## 7. Sequencing

**While the sky is dark** (nothing on-sky accumulates — this window is for code):
1. #91 brownout-hold policy — build, gate, leave DISARMED.
2. Scaffold the #93 estimator against the shadow data path.

**First hour after recovery:**
1. Confirm `GAP3-SHADOW` is emitting `a=` values. It was silently dead for ~30 min on 08-26
   (a swallowed `C_LIGHT` NameError, invisible to the digest gate because the shadow path
   needs a live gather and never runs in replay — hence the pyflakes static pass now in
   `gate.sh`). Every accumulation window also reset across the day's restarts.
2. Then let the clocks run. #92's P2 count, #93 Step 1's calm/disturbed strata, the
   cross-band pairs, and D2's base rate all accumulate free and in parallel off one log.

**First verdict to land:** #92 P2 (~a day of REBASE-WIPE counts), because it is a
plant-calmer and every later measurement is cleaner for it.

---

## 8. The arming discipline (unchanged, use it for every step above)

1. Default-OFF flag in code; arm in the yaml on ONE chain.
2. Digest-gate every fixture with the flag off (identity), plus an ARMED replay A/B that
   **can** fail.
3. Pre-register the expectation matrix in `fixtures/` **including the falsifier**, and give
   the falsifier a CONTROL CLAUSE ("armed exceeds its control twin in the same polls"),
   never a bare threshold. A bare threshold measures the sky.
4. Archive the broker log before every restart (`broker_restart.sh` truncates).
5. Judge paired **in time** against in-poll controls, on q — never against history, never on
   sig, and never on the `DLL:` line (it lists only presence-passing sats: use D0).
6. If the bar trips, disarm first and reason second.
