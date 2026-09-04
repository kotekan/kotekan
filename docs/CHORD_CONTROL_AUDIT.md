# The control-path audit (2026-08-16)

**Why this exists.** KV: *"things have evolved ~adiabatically to a very distant structure from
the prototype... we seem to be chasing bug after bug that are historical artifacts,
misunderstandings of long-ago tests."* This is a structural audit of what the system actually
is, as against what we believe it is. Four independent passes: tracker, gather/fast loop,
broker, aggregator. Every claim carries a file:line. Nothing here was fixed; this is a survey.

**The one-line verdict.** The estimator layer is sound and the fast loop is a correct error
loop. What is missing is a **control system**: there is no place where the forecast, the
measurements and the corrections are combined by a stated rule. The seed is a mutable dict
with ~14 writers arbitrated by ~20 boolean mode flags, and the loop that corrects it has no
path back to the estimator that owns it.

---

## 1. The architecture, as designed vs as built

KV's stated model, and where the build departs from it.

| Stage | Intended | As built |
|---|---|---|
| Trackers | synthesize, correlate, sum over elements, transmit | ✅ as intended |
| Aggregator | grabs small amounts, searches, **reports detections to broker** | ⚠️ a second **absolute** estimator: blind in code phase, and its fit **overwrites** the seed's phase, rate and epoch |
| Gather | merges, runs high-speed loops, **sends results to the broker** | ⚠️ actuates the **trackers directly**; the broker is not in the loop and never reads the result back |
| Broker | vector tracking, forecast, seeds, records | ⚠️ vector filter runs in **shadow**; a legacy path still owns the answer |

### 1.1 Three deviations that matter

**(a) The gather does not report to the broker. It actuates.**
`GnssFleetTrim` POSTs absolute `trim_chips` to each tracker's `/set_trim` at 23.84 Hz
(`GnssFleetTrim.cpp:464-508`), applied at `gnssSeedTransport.cpp:329`. The broker supplies the
target list and then steps out. **Nothing in the broker ever reads `/get_dll` or `/get_trim`.**
So there are two independent write authorities on tracker code phase that never meet.

**(b) The search is blind in code phase, by construction.**
The only seeding endpoints are `set_doppler_hints` and `set_nh_hint`
(`GnssChannelizedSearch.cpp:980-988`). There is no code-phase hint. The search re-derives an
absolute code phase over the full 16-code-period span every pass. It is therefore not a
refinement of the forecast — it is a competing absolute estimate, and
`gps_distributed_broker.py:4148-4152` lets its fit replace `code_phase_chips`, `code_phase_rate`
**and `ref_hop`** in one assignment.

**(c) Vector tracking is built and shadowed.**
`JointReceiverState` (`state_filter.py:89`) carries `clk`, `clk_rate`, `b_sat[i]`, `tau[band]`,
`f_carrier`, `rrate`. Consumption is gated per-term behind `joint_consume` ∈ {`rate`,`slew`,`clk`}
and contributes only **bounded deltas** to the legacy median-based clock
(`gps_distributed_broker.py:6325-6332`, `:6146-6154`). The legacy path still owns the answer;
the filter corrects it by at most `--joint-slew-max-chips` (5.0). This is the right shape for a
shadow deployment and the wrong shape for a primary.

---

## 2. The combination rule — there isn't one

**This is the answer to KV's question, and it is the root finding of the audit.**

The seed is a bare `dict` in `seeds[prn]`. There is no seed class, no covariance, no weights.
The whole path lives inside one ~8500-line `main()`. Approximately **14 sites write the seed
during one cycle**. Roughly half write **absolutely** (last-writer-wins); the rest apply a
**bounded correction**. They are interleaved in cycle order and arbitrated by mode booleans.

### 2.1 Writers that REPLACE (measurement overwrites forecast)

| Site | What it replaces | Cite |
|---|---|---|
| Doppler source selection | 4 candidate values, each overwriting the last, no blending | `B:3963-3998` |
| First seed / re-detection | whole tuple rebuilt from the detection | `B:4089-4090` |
| **cp-rate fit** | `code_phase_rate`, `ref_hop` **and** `code_phase_chips` together | `B:4148-4152` |
| Doppler-rate accept/reject | fitted replaces model, or model vetoes — binary | `B:4132-4141` |
| Overlay period lift | `code_phase_chips`, `code_phase_at_ref_chips` set outright | `B:4264-4268` |
| Hold-freeze | the **previous** seed replaces the fresh candidate wholesale | `B:4483-4487` |
| DR birth / re-pin | whole dict from `cp_predicted + clk + b_sat` | `B:6464-6470` |
| Noise probes | whole dict from `pred`, `ref_hop: 0` | `B:4931-4937` |
| **Code rate, global** | overwrites every non-held PRN **after** the DLL and carrier sections | `B:7998-8010` |
| Carrier command | `rrate` replaces the trim-loop value | `B:8071`, `:8096` |

The code names the hazard itself at `B:8062`: *"REPLACING the trim-loop value: two controllers
on one state is the #52 disease."* That comment is correct and applies far more widely than the
line it sits on.

### 2.2 Writers that CORRECT (forecast + bounded step)

The dead-reckon slew (`B:6364-6441`) is the canonical one — and note the direction of authority:

```python
_held  = dr_seed_phys(seeds[prn], h1, ...)          # where the TRACKER is
_model = (cp_predicted(v, t_h) + _off) % _DR_MOD    # where the MODEL says
_dcp   = wrap(_model - _held)
_step  = clamp(_DR_SLEW_K * _dcp, ±_cap)
```

The **sky-following phase is the base and the model supplies the correction** — the inverse of
KV's "BRDC forecast + small tweak". Both are defensible; having both in one system, chosen by a
flag, is not.

Others: joint→slew delta (`:6325`), joint→clock delta (`:6146`), coast currency retag (`:5104`),
DLL trim at post time (`:8067`), far-regime re-seed (`:7112`), rrate slew (`:8091`).

### 2.3 Mode switching stands in for arbitration

~20 boolean branches decide which estimator wins: `--seed-doppler det` ("last word, so it
overrides the model AND the DR", `B:3992`), `dr_untrusted`, `cp_held`, `fit_trusted`,
`cp_escape >= 5`, `amp_veto`/`integ_veto`, `_slew`, `joint_consume`, `joint_model_primary`,
`bias_available`, `bias_stale`, `car_locked`, `--fleet-trim-url`, `--fast-trim-hz`,
`period_continuity`, watchdog, `probe_set`. Each is individually reasonable. Collectively they
are the arbitration policy, and it is written nowhere.

---

## 3. #41 — the ±2 code-period flicker has a named mechanism

Two distinct period-ambiguity mechanisms exist. **They produce different step sizes, and the
measured histogram picks one.**

### 3.1 The NH argmax rails at its hint-window edge → ±1, ±2 periods ⭐ LEADING CANDIDATE

`nh_hint_span: 2` means the search scans only `pred-2 … pred+2`, **5 of 20 alignments**
(`GnssChannelizedSearch.cpp:727-728`). The prediction is the search's *own* last answer echoed
back by the broker (`:718-726`, `:1116-1119`). If the true `nh` lies outside that window, the
argmax cannot reach it and **rails at the window edge — ±2**. `best_nh` enters `lift` directly
as `(best_nh + lag_periods) * 10230` (`gnssSeedTransport.cpp:254`).

This predicts the measured signature exactly. From the 2026-08-16 seed audit:

| step size | events |
|---|---|
| **2L (20460 chips)** | **115** |
| 1L (10230 chips) | 15 |
| 100–5k | 33 |
| <100 (healthy) | 71 |

±2 dominating ±1 by 7.7:1 is what an argmax railing at a window edge looks like; it is *not*
what an unbiased ±1/±2 confusion looks like. A railed argmax carries only the sign — the same
disease as a railed discriminator, one layer up.

**The test:** log `best_nh − predicted_nh` per detection and histogram it. Piling up at exactly
±2 confirms; a spread confirms something else. Cheap, and it needs no restart.

### 3.2 The 16-vs-20 alias → ±4 periods (40920 chips)

`lag_periods` is measured mod **16** (the replica-period span) but is needed mod **20**.
`gcd(16,20) = 4`, so a ±16 error aliases to ∓4 periods. The code relies on a *measured* bound —
*"across 22 injections |folded| never exceeded 3"* (`gnssSeedTransport.cpp:233-238`) — with
nothing checking it at runtime. The fold `if (lag > span/2) lag -= span` maps 8→+8 but never
to −8, though both describe the same lag: **bistable on a strict `>` against a noisy argmax**.

`--nh-period-offset` exists in the broker *specifically* because the oracle measured a constant
+4 period error, with the comment `-4 == +16 mod 20` (`B:1736-1745`). This mechanism is
confirmed on sky and currently patched by a hand-set constant.

---

## 4. Defects found, ranked

### 4.1 ⚠️ The broker is blind to the trim it commanded

Under `--fleet-trim-url` the Python integrator that writes `dll_trim` is skipped (`B:7142-7145`),
and nothing reads the C++ value back. But `dll_trim` still feeds three consumers:

* the **ESCAPE detector** — `track_vs_fit_chips(..., dll_trim.get(prn, 0.0), ...)` (`B:4332`);
* the **joint-receiver model-primary feed** — `y = held + dll_trim − cp_predicted` (`B:5615`);
* the publisher (`B:7002`).

All three see **0** for a correction that reaches ~1 chip on sky. The escape referee is
therefore biased by the full standing trim, and the vector filter is fed a measurement missing
its own control input.

### 4.2 ⚠️ Re-anchors pop the trim in Python only

The three hard re-anchors — ESCAPE (`B:4409`), watchdog (`B:5019`), new-anchor DR seed
(`B:6462`, *"any old trim served the OLD anchor"*) — all `pop` `dll_trim`. **There is no channel
to the C++ accumulator.** `set_policy` has no field for it. On a re-anchor onto the search fit
(a phase that does not include the trim), the C++ trim re-applies on top, bleeding out only
through the ~8.3 s leak. The tracker's 4 s TTL cannot catch it because posting never stops.

### 4.3 ⚠️ The integrator's steady state exceeds its own clamp

Live on gps_l5: `gain_per_s` 2.5, `leak_per_s` 0.12, measured 23.85 Hz → `gain` 0.1048,
`leak` 0.005032 per update. Railed-disc steady state `gain×0.25/leak` = **5.2 chips against a
±3.0 clamp**. `gnssFleetDll.hpp:100-105` asserts the clamp is *"unreachable by construction"* —
true for the retired (0.25, 0.05) per-update pair, not for what ships.

Root cause is one unused variable: `gps_distributed_broker.py:2806-2811` documents
`_fleet_trim_nominal_hz = 23.84` converting the Python arm's 0.05/update into 1.19/s so the C++
loop runs *"the SAME per-update leak the Python arm did."* **It is defined and never read.**
The shipped default is a flat 0.12/s → 0.005/update, **10× weaker** than intended.

Observed live: G18 armed at `disc = −0.828`, `q = 0.279` — off-peak, discriminator carrying only
a sign, and the loop integrating it toward the clamp.

### 4.4 ⚠️ 18 of 22 GPS PRNs have no code loop at all

`--fleet-trim-url` disables the Python trim **globally for the chain** (`B:7142`), while the C++
loop trims only PRNs in `armed`. Live `armed` on gps_l5 is `[8, 18, 23, 27]`; the chain carries
22 PRNs. The other 18 have no code loop from either arm.

This is the identical disease as the gal/bds disarm of `eec1d2f12`, on the chain that is
supposedly armed. Live demonstration: **G17 at `p_pow` 3.29e-8** — comparable to locked G8's
3.09e-8 — **but `q` 1.275**: strong signal, off-peak, and nothing assigned to pull it in. G24
sits at `q` 2.147, just under the 2.2 arming gate, also unarmed.

All four other chains have `armed: []` and still show the never-overwritten constructor defaults
(gain 0.25 / leak 0.05); `policy_age_s` and `policy_armed_requested` contain only `gps_l5`.

### 4.5 ⚠️ `code_phase_at_ref_chips` is paired with the WRONG `ref_hop` on every held satellite

`propagate_seed` **ignores `code_phase_chips` entirely** when `code_phase_at_ref_chips >= 0`
(`gnssSeedTransport.cpp:325-327`, mirrored `fits.py:451`). The hold branches restore
`code_phase_chips`, `code_phase_rate`, `ref_hop` and `doppler_hz` from `prev`
(`B:4483-4487`, `:4516-4520`) but **never restore or delete `code_phase_at_ref_chips`**.

Shipped seed = frozen tuple + fresh at-ref phase + **stale `ref_hop`**. Because the tracker
prefers the phase, the entire `cp_held` hold-on-lock mechanism is bypassed for any signal that
ships the phase, and the residual is `(det_ref_hop − prev_ref_hop) × 52.4 chips/hop`.

### 4.6 ⚠️ The DLL trim is a no-op on any seed that ships a phase

Both broker trim-application sites write **only** `code_phase_chips` (`B:8069`, `B:2866`).
Neither touches `code_phase_at_ref_chips`, which is what the tracker reads when present. The
search-fed path has shipped that field all along (`B:4264`, `B:1287`).

**Not currently biting** — gps_l5 uses the C++ `/set_trim` channel (applied correctly at
`gnssSeedTransport.cpp:329`), and the DR chains have `--seed-phase-transport` OFF. But it is a
live landmine: **enabling `--seed-phase-transport` on the dead-reckon chains would silently
disable their only code loop.** #45 step 6 is a pending item to do exactly that.

Only one site handles this correctly — the far-regime re-seed, `B:7121-7123`:
`seeds[prn].pop("code_phase_at_ref_chips", None)`.

### 4.7 The slew cap rails 47–100% of the time

`--dr-slew-cap` default **0.05 chips per event** at `--dr-refresh-s` 2 s = **0.025 chips/s** of
correction authority. In-code measurements: 47% of steps exactly on the cap (`B:6389-6390`),
67–100% quoted at `B:7112`. Satellites 5–8 chips from their model take ~1 hour to converge
against a ~600 s plant oscillation, and a railed step makes any change to the slew *target*
unobservable — which is why the P2b consumer could not be measured.

### 4.8 `doppler_hz` epoch is inconsistent across its four producers

Contract (`B:5052-5057`, `gnssSeedTransport.cpp:307`): `doppler_hz` must be the value **at
`ref_hop`**.

| path | status |
|---|---|
| coast | ✅ fixed `7cb011c50` — stores back-propagated to `ref_hop` |
| **birth / re-detection** | ❌ forecast at **wall-clock now**, stored with the *detection's* `ref_hop` (`B:3965`, `:4089`) — the identical `dop_rate × age` double-count, in the path that creates every seed |
| **hold** | ❌ keeps the new now-epoch value with `prev["ref_hop"]`; `B:4521` says so explicitly |
| DR birth/slew | ✅ correct |
| **noise probes** | ❌ now-epoch Doppler with `ref_hop: 0` and a live `dop_rate` — extrapolated over the whole F-engine run age |

### 4.9 The refine grid is asymmetric and never evaluates its own input

`refine_span 512 / refine_step 103` → `n_eval = 2*512/103 + 1 = 10` (integer division), trials at
**−512 … +415 samples**: centre **−0.155 chips**, and offset 0 is not on the grid. Served cp
resolution is 103 samples = **0.329 chips**, not the sub-chip the comments imply. CPU and CUDA
paths reproduce it identically by design (`gnssSeedTransport.cpp:52-53`, `:108-111`), so an A/B
between them cannot detect it.

### 4.10 `code_phase_rate` has two same-cycle writers on model-primary chains

`B:6437` writes from the l−a EMA (or a borrowed sibling value); `B:8007` overwrites for every
non-held PRN from `cb_to_seed`, which may be the **joint** `clk_rate`. Two different clock-rate
estimators, same field, same cycle, silently.

### 4.11 Tooling that cannot have been giving right answers

* **`diag/code_phase_check.py`** — the only tool that directly compares search cp to tracker cp —
  hardcodes `d["ref_hop"] / 1e6`, the **airspy** hop rate, against CHORD's **195312.5** hops/s:
  off by **5.12×**. There is no `--hops-per-sec` flag. Its default `--code-length 1023` is also
  wrong for L5 (10230).
* **`config/generated/chord_gnss_gather_f2.yaml`** sets `fleet_trim.post_targets`, a key the stage
  does not read (targets arrive only via `/set_policy`), and declares `frame_size: 25200` against
  the live 25968. That fixture actuates nothing.
* The **hop constant** is quoted as 52.3776, 52.3744 and 52.3713 chips in three different
  comments. The true zero-Doppler hop is 52.37760; `(fft_len−1)*cps` is 52.37440; **52.3713
  matches neither** — and it is the value in `fits.py:407`.

---

## 5. ⚠️ RETRACTION: "search and tracker disagree by 134004 chips"

Recorded on 2026-08-16 as evidence that trackers sat thousands of chips off the search peak.
**The comparison was invalid.**

Those differences are not period multiples — residuals are 0.076–0.271 of a code period, i.e.
780–2770 chips off any boundary:

```
diff       /10230     residual(periods)
134004     13.099     0.099
 21240      2.076     0.076
-12323     -1.205     0.205
 38149      3.729     0.271
 -8177     -0.799     0.201
-31577     -3.087     0.087
```

`code_phase_chips` is an **argument** back-referenced to absolute sample 0. It means something
only paired with the Doppler it was published with; subtracting two that carry different
Dopplers multiplies the Doppler difference by the back-reference lever (1696–5903 chips/Hz
depending on run age). Divided out, the six differences are **+79 / +12.5 / −7.3 / +22.5 / −4.8
/ −18.6 Hz** — ordinary per-PRN search-vs-seed Doppler scatter.

This is defect **#42/#43**, already fixed and already documented in `fits.py:340-346`:
*"seven false ESCAPES in one evening against tracks healthy at 40 dB-Hz."* The rule is in the
memory index under the title *"code_phase_chips is an argument… never transport one."*

**What survives:** the trackers were genuinely off-peak. That rests on `q ≈ 1` and E≈P≈L, which
are local measurements needing no cross-system comparison. **What does not survive:** any claim
about the *magnitude* of the offset.

**The commensurable comparison exists and nothing uses it:** `cp_at_ref` vs `tracker_phase_at`,
both physical phases mod 204600 — with the mandatory first-sample↔last-sample conversion
(`fits.py:401-428`), worth 52 chips if skipped.

---

## 6. What the tracker actually holds — the parameter inventory

For reference, since "what are the tracking parameters" was the question that started this.

**Per (chain, PRN), from the broker seed** (`cudaGnssChordTrack.hpp:61-77`), wholesale-overwritten
on arrival (`:218-222` — a POST that omits a field **zeroes** it):

| field | units | abs/error | epoch |
|---|---|---|---|
| `doppler_hz` | Hz | absolute | **at `ref_hop`** |
| `cp_chips` | chips | absolute **argument** | sample 0, valid only with `doppler_hz` |
| `cp_rate` | chips/hop | **residual** — geometry is fed forward by `propagate_seed`, not this | at `ref_hop` |
| `dop_rate` | Hz/s | absolute | at `ref_hop` |
| `ctrim_hz` | Hz | **correction** on `doppler_hz`, carrier only | none carried |
| `ref_hop` | hops | absolute | *is* the epoch |
| `phase_ref_chips` | chips mod 204600 | absolute **physical phase**; **preferred over `cp_chips`** | at `ref_hop`, that hop's **LAST** sample |

**Local, separate:** `trim[p]` (chips, additive correction to the model phase, absolute on the
wire), `trim_disc`, `trim_q`, `ema_*`.

**The layering is correct.** A seed arrival overwrites the seed slot only; the trim is never in
the `Seed` struct and is applied once, at `gnssSeedTransport.cpp:329`:

```cpp
out.phase_now = out.phase_ref + (chips_per_hop + sd.cp_rate)*dh + quad + trim_chips;
```

`/set_trim` touches nothing else (`cudaGnssChordTrack.cpp:325-334`).

**There is no PLL, no FLL and no carrier-trim integrator on the CHORD tracker node.** Carrier
trim is 100% broker-owned. The only node-resident closed loop is the in-tracker code trim, and it
is **default OFF** (`cudaGnssChordTrack.cpp:50`) because the fleet controller owns that authority.

**Two hazards in this layer:**
1. `trim[]` has **two legitimate writers** — the in-tracker loop (`:575`) and `/set_trim` (`:326`)
   — separated only by deployment convention. Observable via `/get_trim`'s `enabled` vs `posts`;
   not prevented.
2. **Trim expiry is a step, seed expiry is a drop.** `trim → 0` in one frame (`:249`) at the 4 s
   TTL, while the fleet controller's own release is a leak-ramp (`gnssFleetDll.hpp:472-483`).

**Two latent arithmetic defects in `propagate_seed`:**
* The quadratic term (`:323`) uses `chip_rate/f_offset_hz` with **no** `code_doppler_sign`, while
  the linear term (`:320-322`) uses `chip_rate/carrier_hz` **with** it. Three differences in one
  term: sign, denominator, chip-rate flavour.
* `f_offset_hz` is **unguarded** in that division and its config default is `0.0`
  (`cudaGnssChordTrack.cpp:44`). A chain that omits it and has non-zero `dop_rate` propagates
  ±inf/NaN into the commanded code phase.

---

## 7. Recommendations

Ordered by ratio of structural payoff to risk.

1. **Make the model error a first-class published quantity.** The premise "BRDC + vector tracking
   gets very close, the fast loop is a tweak" is currently false — search-vs-model is per-sat
   ±1–6 chips against a documented norm of ±0.2 (docs §11.22), and nothing measures that gap as
   a served number. Until it is measured, no arbitration policy can be written.

2. **Close the trim loop back to the broker.** Poll `/get_dll` and use the C++ trim wherever
   `dll_trim` is read today (§4.1). This is a read-only change and it un-blinds the escape
   referee and the vector filter simultaneously.

3. **Give `set_policy` a `reset_trim` list** so the three re-anchors can invalidate the C++
   accumulator (§4.2). Without it, every re-anchor double-counts for ~8 s.

4. **Fix the leak conversion** (§4.3) — one line, restores the intended bandwidth and makes the
   clamp unreachable as the header claims.

5. **Build the acquisition→tracking handover** that `eec1d2f12` specifies: Python trim retains
   authority for not-yet-armed PRNs, stands down per-PRN as the C++ loop takes each. This is the
   fix for §4.4 *and* the precondition for re-arming gal/bds (#49).

6. **Histogram `best_nh − predicted_nh`** (§3.1). If it rails at ±2, #41 is solved and the fix is
   to widen or drop the hint window on a presence failure.

7. **Write the arbitration policy down** — which estimator owns which field, under what
   conditions, with what authority bound. §2 is currently that document, and it was reverse-
   engineered rather than specified.

---

## 8. Method note

Four parallel read-only audits (tracker / gather / broker / aggregator), each asked for
file:line on every claim, then cross-checked against live state polled from the gather's
`/get_dll` and `/get_stats` on cf06. Two findings (§4.3, §4.4) were confirmed live rather than
by code reading alone. One prior conclusion was **retracted** (§5).

The retraction is the point of the exercise. It was produced by comparing two numbers that a
memory note, written by me, explicitly says must never be compared — which is exactly the
failure mode KV called: *"misunderstandings of long-ago tests."* The defence against it is not
more care; it is §7.1, making the quantity we actually want into something the system serves.
