# GNSS Tri-Band Architecture Audit — 2026-07-19

Scope: the deployed 3-band / 3-constellation node (one kotekan, 7 chains, 7 brokers) as of the
07-19 unified relaunch. Successor to `kotekan/docs/gnss_architecture_audit.md` (2026-07-04),
whose central recommendation R1 ("dumb correlators, slow loops") was subsequently built. This
audit examines the layer that grew *on top of* R1 during two weeks of live firefighting: the
limits, trims, fences, gates, and rescuers. Companion: `docs/pipeline_block_diagram.md`.

Inputs: full reads of the broker (2117 lines), the C++/CUDA chain, the config/launch layer,
the clock/dead-reckoning machinery, and a live interrogation of the running node (2 h era,
pgid 611434).

---

## 0. Verdict up front

The 07-04 skeleton is still right and none of this warrants a rewrite. But the accreted
control-plane machinery is not N independent patches — almost all of it defends **one
architectural decision: the frozen seed.** Hold-on-lock freezes each satellite's seed tuple
because *updating a live seed is unsafe* (the tracker's f_ref re-pin steps the commanded
phase). Everything else follows:

- seeds go stale → per-sat **carrier trim loops** re-derive, Hz by Hz, the Doppler slew the
  ephemeris already predicts;
- trims accumulate → **clamps** (±100 Hz) and **innovation gates** (±3 Hz) bound them;
- releases step the seed → **FLL fences** protect f_ref from the steps;
- the gates create absorbing states → **refade**, **watchdog**, **escape referee**,
  **alias-escape** rescue them — five overlapping rescuers, each added to patch the previous
  one's failure mode (the arg-help strings narrate this history).

The frozen-seed scheme was correct when the receiver clock wandered at 2.6 ppm and the model
couldn't be trusted. **Both premises are gone.** The GPSDO/OCXO-disciplined clock plus BRDC
now predict carrier to a few Hz and code to 0.10 chip rms — better than the frozen seeds the
machinery defends. The live node shows the machinery near-idle where it works and actively
harmful where it doesn't (§2). One missing primitive — a **phase-continuous f_ref re-pin** —
blocks flipping to model-primary (continuous) seeding; the flip is the single change that
retires most of the accreted layer and addresses the open quarry (L2C releases, L5 churn) at
the root instead of per-band.

Secondary finding: the config layer is the drift substrate. Three hand-maintained
near-identical YAMLs plus grep-based config sniffing is how the fence fix reached only one of
four GPS/L5 trackers, how nh-assist got silently dropped (07-18), and how a dozen smaller
divergences accumulated (§4).

---

## 1. What the live node says (evidence, 2 h era, 14:39–16:40)

**The clock estimators are chasing constants.** Solved values over the whole era:

| Chain | carrier bias EMA | l−a (code-rate) |
|---|---|---|
| L1 (all 3 constellations) | −152 ± 3 Hz, flat | +0.09…+0.10 ppm, flat |
| L2C | −18 ± 3 Hz, flat | +0.00…+0.02 ppm |
| L5 (all 3) | +30 ± 10 Hz | −0.02…−0.06 ppm |

These are per-chain frac-N LO offsets plus a ~0.06 ppm dongle differential — static
calibration constants. The adaptive EMA/pooling machinery was built for a 2.6 ppm *wandering*
TCXO; it is now ~50× oversized for the disturbance it tracks.

**The trim loops are absorbing model-predictable drift until they rail.** L1-GPS per-sat
trims at one instant: PRN 10/23/32 at **+90…+93 Hz** (clamp: 100) with residuals ≈ 0;
PRN 25/28 at +4 Hz (recently re-seeded); PRN 1 slamming −92 ↔ −85 Hz with ±60 Hz residuals —
the clamp-rail state, live. The bimodal split is hold age: the trim is a *drift integrator*
for frozen seeds, and its ceiling is a scheduled failure (the L2C RELEASE-excess shape).

**The L5 chains are the frozen-seed disease at full strength.** Event counts this era:
L5-GPS **1938 ESCAPE + 2105 CARRIER REACQ** vs L1-GPS 103 + 372. L5 sats carry standing
residuals of 26–60 Hz that the ±3 Hz innovation gate will never let the loop absorb —
gate → refade → BOOTSTRAP → walk → re-gate churn. (This is Quarry 2's live face.)

---

## 2. The clock question, answered directly

> Does the improved clock that allows dead-reckoning rule out significant chunks of code, or
> are there major issues we could solve by using it more widely?

**Both.**

**Chunks it rules out (already near-idle, collapse to constants + monitors):**
- Carrier clock-frequency bias EMA (`broker:1012-1042`) + its deadlock guards — the offset is
  a per-chain constant now.
- l−a pooled slope-fit estimator + persistence (`broker:2042-2083`) — measured 0.06 ppm and
  flat; the `--code-bias-file` warm-start already substitutes for the loop.
- Wide→narrow search-margin state machine (`broker:1052-1067`, `CLK_WIDE_HZ` sizing): the
  clock term in the search half-range is 3151 Hz at tcxo but **94 Hz at gpsdo** — the wide
  machinery sizes for a regime the hardware left.
- The 30 s `coast_budget` smallness (its own comment says "raise it with a disciplined
  clock") — a trusted clock extends code prediction from tens of seconds to effectively the
  pass length, which also mostly retires the coast/re-acquire churn for fades.
- The `--bias-min-sats`/`--dr-min-sats` bootstrap dependency: with a trusted clock the
  receiver clock no longer needs ≥2 detections to solve, which is the *structural* reason
  dead-reckoning is currently an assist rather than primary.

**The major issue it could solve if used more widely:** stop freezing seeds. The model
already predicts per-sat Doppler within a few Hz continuously (`--dead-reckon` validated at
0.10 chip rms; CL and nh time-assists already run pure arithmetic off it). Model-primary
("dop-continuous") seeding was designed and tried — `broker:668-693` records that it made
things *worse* (E1 42→35 dB-Hz), and the postmortem identifies exactly why: **freezing the
seed was doing double duty as f_ref stabilization, because the tracker's re-pin is not phase-
continuous** (`GnssChannelizedTracker.cpp:339-343`; same logic in `cudaGnssTrack.cpp:366-385`).
A re-pin moves the absolutely-anchored replica and the commanded phase steps by `df·t_abs` —
thousands of cycles that `arg(A)` reads only mod 1.

**So the load-bearing recommendation is one primitive:** make the tracker's f_ref adoption
code- and phase-continuous (translate cp *and* fold the accumulated commanded-phase offset so
`arg(A)`, deep integration, and ADR are continuous across a re-pin — the slot-15/19
commanded-phase bookkeeping built for the trim export is most of the required machinery).
With that in place:

1. The broker ships **continuous model Doppler** (BRDC + per-chain constants) instead of
   frozen detections. Trims collapse from ±90 Hz integrators to ±few-Hz residual loops.
2. Hold-freeze, release steps, and the fences sized to them lose their reason to exist.
3. The rescuer complex (refade / watchdog / escape / alias) shrinks to a single lifecycle
   guard, deleted one at a time with flap-census verdicts.
4. Quarry 1 (L2C release yanks) and Quarry 2 (L5 churn) are attacked at the root rather than
   with two more band-specific tunings.

What stays regardless of the clock (not clock diseases): DLL sub-chip tracking, BOC
false-lobe escape geometry, overlay integer ambiguity (CL/nh/CS), fades, multipath, iono,
BRDC-freshness guards, and the DR integrity/capability gates (a model that can invent a
satellite still needs its backstops).

---

## 3. The rescuer complex (what overlaps, what's dead)

From the full broker read (`gps_distributed_broker.py`, line refs current as of 07-19):

**Five mechanisms that all fire on "the NCO lost the satellite":** hold-release BOOTSTRAP
demotion (1420-1425), escape-referee demotion (1318-1322), carrier refade (2006-2011), BeiDou
watchdog (1507-1535), alias-escape v2 (1931-1953, **dead in prod** — flag default 0). On
BeiDou, watchdog (45 s) and refade (10 emits) are both armed and can race on the same PRN.

**Three stacked filters guarding one escape decision** (1293-1304): 5-consecutive
sign-consistent disagreement + 9-sample median gate + amp veto — over-determined.

**Two independent coast currency updaters:** the in-seed-loop translation (1358-1400,
fence-gated) and the coast-loop translation (1561-1581) — the latter is gated by
`--trim-precomp-coast` which is **OFF in production**, so coasting falls to the legacy
raw-dop overwrite. Given `--coast-to-horizon` is ON (coasts are routine), the "long coasts
lose the code peak" fix is effectively unshipped. Worth a deliberate decision.

**Inconsistent held-sat treatment:** the DLL (silently ON everywhere — default gain 0.25,
never set by any launch script) keeps integrating on held sats, while l−a seeding skips them;
`cp_err` reads/writes the same quantity the escape referee adjudicates.

**Dead broker code:** `--drop-hits` (parsed, referenced nowhere); alias-v2 block; trim-precomp
carrier `orig`/`flip` branches (bench-rejected); `--dop-continuous` true-branch (the design
that awaits the re-pin primitive); the skyfield `visible_prns()` non-almanac path; two `del`'d
function params kept for signature stability.

**Copy-paste constants:** c = 299792458.0 ×4; BDS-3 "PRN ≥ 19" ×5 (plus the redundant TLE
name-filter encoding of the same rule); "10 s freshness" ×4; "2×acquire" presence bar ×2.

---

## 4. The data plane (tracker + combiner)

**Fence semantics are the one real cross-band incoherence.** `fll_reacq_hz` defaults to
0.1 cycle/record → **100 Hz at 1 ms records (L1-GPS, all three L5 trackers), 25 Hz E1C, 10 Hz
B1C, 5 Hz L2C (overridden to 12)**. But the disturbance the fence exists to survive — broker
seed steps — is a few-to-tens of Hz *regardless of band*. So protection varies 20× across
chains by accident of record length: L1/L5 effectively unfenced (wide = the healed
configuration), B1C at 10 Hz sits exactly in the danger zone — and B1C is the one band that
needed the watchdog armed. The l2c yaml comment ("B1C's 25 Hz fence is why B1C healed") also
misattributes: 25 is E1C; B1C is 10. Interim recommendation: set the fence explicitly, in Hz,
fleet-wide (~15-25 Hz), and re-census whether the BeiDou watchdog is still needed. (After the
re-pin primitive, the fence question dissolves entirely.)

**Dead config keys silently accepted:** `carrier_shared:` (never read; only `fll_gain==0` is
enforced), `n_prn:` on tracker blocks (decorative — `prns.size()` rules), `prns:` on combiner
blocks (never read). These document intent but enforce nothing.

**Hot-path leftovers:** with `max_anchor_age_s: 0` everywhere, the age branch of the tracker
re-pin state machine fires every record — the fence is the only live gate; the age machinery
is vestigial. `PrnCtl::energy_scale` written 1.0, never read. Combiner record slots 17-19
zeroed and unused (free export space — e.g. future dTEC fields). `gnssRecord.hpp:25` still
calls slot 19 "(reserved)" though it is `REC_TRIM_INC`.

**Overlay handling is a string ladder, not the descriptor table** (07-04's R6, still open):
eight string-compared cases with hardcoded generators and PRN counts
(`GnssCoherentCombiner.cpp:53-105`), `secondary_length` in the descriptor deliberately 0, and
two nearly-identical deep-wipe ladders (overlay 593-693 vs navwipe 694-757) that a
table-driven design would collapse. Matters when the next signal lands, not before.

**Debug leftovers shipped in prod:** `phase_dump_prns` enabled only in `live_l1_dual20.yaml`,
writing per-record text into /tmp/gpswipe (part of the flagged 30 GB). Decide: keep as
standing forensics or gate behind an env.

**Execution-model divergence between siblings:** L1 runs its three trackers parallel on CUDA
streams 1/2/3 (with the 07-19 multi-stream join fix); L5 serializes all three on stream 1 —
it predates the join fix. Harmless at 43% GPU, but the two files claim to be the same design.

---

## 5. The config/launch substrate

**Divergences that look like drift, not physics** (full table in the audit transcript):
`doppler_step` 100 (L1) vs 250 (L5) on identical record geometry; `acquire_windows` 20/40/100;
`acquire_snr` 10/8/10 in configs while all brokers hardcode `--acquire-snr 6` (the config
value never reaches the broker — two gates, one imaginary); L5 gains 8/8/8 vs 11/11/11
elsewhere; `hint_ttl_s` 8 vs 20 within one file; L2C's 4 s integration window vs ~1 s
everywhere else; the fence non-propagation (§4).

**Plumbing hazards:** every col-0 grep in `run_live.sh` works only because `CFG` is always
the unprefixed per-band file — pointing it at `live_3band.yaml` would silently empty `TRK`
and friends (only the nh-assist grep was hardened, after the 07-18 bug). `--cl-assist`'s
trigger greps `signal: GPS_L2C_CL` which no deployed config contains — the CL assist can
never arm on the current tree. `WS_PORT` is consumed by nothing (viewer ports are baked into
`spawn_pyviewer` lines) yet `run_3band.sh` dutifully sets it. `gen_3band_config.py` takes
SHARED keys from band 0 silently, `eval`-collapses arithmetic with exceptions swallowed
(failure surfaces as "buffer doesn't exist" at stage construction), and prefixes words inside
comments; validation checks only YAML-parses + unique top-level keys.

**Stale files:** the `replay_l1anchor44*` trio (old split-cudaProcess layout, superseded by
`replay_l1_1737.yaml`), `live_intgn*.yaml` (referenced by nothing), `replay_l2c_cpu.yaml`,
`offline_l1ca_replay.yaml` / `airspy_gps_l2c_offline.yaml`, `gnssChannelizedAcquire.cpp/.hpp`
(zero config references), `run_mono.sh`'s `build_mac` path. The CPU-chain `live_l1/l2c/l5.yaml`
are reachable fallbacks (`run_band.sh *_cpu`) — keep, but label.

---

## 6. Recommendations, ranked

**A. The re-pin primitive → model-primary seeding.** Make the tracker's f_ref adoption code-
and phase-continuous, then re-run the dop-continuous experiment (its 07-1x failure is
explained and expected to invert). Stage it: (1) primitive + replay-bench leg, (2) enable on
L5-GPS (the sickest chain — its churn is the cleanest before/after), (3) fleet, (4) delete
rescuers one at a time, each with a flap-census verdict. This is the successor to 07-04's R1
and the root fix for Quarries 1 and 2. Everything in B–D below shrinks or disappears behind it.

**B. Interim (cheap, now): fence in Hz fleet-wide** (~15-25 Hz explicit on every tracker,
esp. B1C's 10), then census whether the BeiDou watchdog still earns its keep. Also decide
`--trim-precomp-coast` (the unshipped coast-currency fix) deliberately.

**C. Config generation (07-04's R2, now overdue).** One per-signal table → generate the three
band yamls (or all 7 chains); broker flags read from config rather than hardcoded; kill the
grep-sniffing in `run_live.sh` by having the generator emit a small env manifest per band.
This closes the entire fence-didn't-propagate / nh-assist-dropped bug class.

**D. Freeze clock machinery into calibration + monitors.** Seed bias and l−a from persisted
constants at startup (files already exist); keep the estimators running only as *alarms*
(flag if l−a moves >0.05 ppm or bias >10 Hz from calibration). Raise `coast_budget`. Narrow
the cold search margins to the gpsdo profile.

**E. Subtractive sweep** (half a day, safe now): dead flags (`--drop-hits`, alias-v2,
trim-precomp on-paths), dead config keys (`carrier_shared`, tracker `n_prn`, combiner `prns`
— or make the tracker actually validate them), `WS_PORT`, `gnssChannelizedAcquire`, the
stale replay/intgn configs, the slot-19 comment, the misattributed fence comment, the
`--cl-assist` dead trigger (fix or remove), constants dedup (c, PRN≥19, freshness windows),
phase-dump gating.

**F. Table-driven overlays (R6)** — before the next constellation/signal, not before.

### Explicit non-recommendations
- No rewrite; no moving loops back into C++ (the slow-loop shape is right — the loops just
  need a model-primary input instead of a frozen one).
- Don't delete the DR integrity/capability gates or BRDC-freshness guards when promoting
  dead-reckoning — those are the model's honest backstops, not clock-era scar tissue.
- Don't tune more per-band rescuer parameters (watchdog thresholds, refade counts, escape
  filters) before A/B — that's the tree-level work this audit argues against.
