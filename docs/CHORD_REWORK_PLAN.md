# CHORD GNSS — the rework plan (KV-directed 2026-08-26)

Two staged programmes, to be started **after the outstanding verdicts land** (#92 handover,
#93 shadow v2, #90 v3) **and after a compaction**. Written down now so the reasoning survives
the context boundary rather than being re-derived.

Item (1) of the same plan — **A1, gather robustness** — is DONE in code (`95ce4b712`) and is
waiting only for a deploy window; see the last section.

---

## (2) BROKER REFACTOR — starting with the digest fixtures

### Why now rather than after stable locks

Measured 2026-08-26, not estimated:

| | |
|---|---|
| `gps_distributed_broker.py` | **11,073 lines** |
| inside ONE function (`_decoded_entries`) | **7,135 lines (64%)** |
| inside two (`+ main`, 2,059) | **83%** |
| max nesting depth | **15 levels** |
| live argparse flags | **220** (plus the `_FROZEN` dict from #89) |
| settings in the production yaml | 135 |

The deepest-nested line in the file is the #92 handover POST added on 2026-08-25. That is the
mechanism, not a coincidence: **every new guard becomes another nesting level, and its state
gets hoisted thousands of lines from its use** (`_g3_hist`, `_adm_pop`, `_reseed_was_present`
are declared near line 2900 and used near 7900–10500).

THE COST, PAID IN SKY TIME on 2026-08-25: #90 needed four guard iterations in one evening
(was-present, startup hold-off, brownout floor, plus the F3 finding). Each is ~10 lines of
pure decision logic. **Each required a broker restart on the live fleet to exercise**, because
there is no way to run that logic offline. That is the argument for going sooner: the
instability is partly *caused* by how hard the loop logic is to test, so "wait for stable
locks first" is backwards.

### Step 0 — refresh the digest fixtures  ✅✅ COMPLETE 2026-08-26 01:2x (all three captures)

`broker_onsky_e5a_20260826` is blessed and **GATE GOOD** (`3cba60e49`): 113 cycles, 1474
posts, determinism holds, digest moves under a 1e-12 perturbation of `carrier_hz` and
`dll_gain`. It is the model-primary chain with span + command + handover live, i.e. every
path added in the last week. Fixtures now carry their own ephemeris in a
`<transcript>.brdc/` sidecar (`broker_equiv._pin_for`) so a new capture cannot move the
fingerprint of the existing four.

✅ The remaining two landed 2026-08-26 01:1x–01:2x (`ea6a69170`): `broker_onsky_l5_20260826`
(search-fed, 164 cycles, moves `bias_alpha` too) and `broker_onsky_b2a_20260826` (the first
BeiDou capture, 172 cycles). Full sweep **7/7 EQUIVALENT** at `ea6a69170`. Coverage ledger:
`carrier_gain` unreachable by design (production gain 0.0); **`code_bias_alpha` is covered
ONLY by the 2026-08-08 l5 fixture** — keep it in the gate.

⚡ THE FIRST REPLAY ALREADY PAID FOR THE PROGRAMME: it crashed on an
`UnboundLocalError` (`t_now_abs` read by six consumers before the first ephemeris refresh
assigns it; five swallow it in a `try`, the sixth killed the chain thread). Fixed in
`44b976023` with all four existing digests unmoved. A startup-ordering race production
hides — exactly the class the gate exists to find, found before any refactoring began.

### Step 0 (original note, kept for the reasoning)

`scripts/gnss/broker_equiv.py` over four fixtures is the safety net that makes mechanical
extraction safe, and **it is currently partly blind**: the fixtures are blessed at older
commits (`f2524ab04`, `26211b9a8 DIRTY`) and predate the current axis/span/handover regime.
A refactor gated on a stale fixture proves the old paths still work.

* Capture a fresh on-sky fixture per chain class (one search-fed: gps_l5; one model-primary
  Galileo with span + command armed; one BeiDou), covering a re-base event and a brownout if
  the sky offers one.
* Bless at the current HEAD; keep the old fixtures too (they are the only regression against
  the pre-axis regime).
* Outstanding since the p9 handoff — this is its third mention and it now blocks work.

### Step 1 — DONE 2026-08-26 (all three, plus more than was planned)

| | before | after |
|---|---|---|
| `gps_distributed_broker.py` | 11,084 | **8,598** |
| the cycle loop body | 6,940 | **4,834** |
| max nesting depth | 15 | **10** |
| named stages in the cycle | 0 | **10** |
| unit-tested decision modules | 0 | **4** (66 checks) |

⚠️ **THE HEADLINE NUMBER IN THE ORIGINAL PLAN WAS WRONG.** "7,135 lines in `_decoded_entries`
(64%)" was a grep artifact — it measured def-to-next-def. `_decoded_entries` is **30 lines**.
The monolith was always `main()` itself (10,287 lines, 93% of the file), of which the cycle
loop was 6,940 and argparse 1,938. Re-measure with `scripts/gnss/broker_iface.py map`.

Extracted, each a pure move gated at 7/7 EQUIVALENT:
* **`gnss_broker/admission.py`** — the #90 gate as an `AdmissionGate` with its six rules named
  and ordered, plus the #50 `reseed_step` policy. `test_admission.py` replays all eight of
  2026-08-25's fires offline, including PRN 34's sidelobe-stable tau growing THROUGH two fires.
  The F3 threshold-flicker gap is asserted as **still open** rather than papered over.
* **`gnss_broker/handover.py`** — #92's `TrimHandover`, which owned the deepest-nested line in
  the file. Tests pin the bound (a 400-chip shared-clock birth is not a handover) and the rule
  that a FAILED post must not be credited into `adjcum`.
* **`gnss_broker/rampfit.py`** — #93's shadow `RampTracker`. The test that matters asserts a
  1.0-chip re-anchor inside a 400 s window reads as FLAT, not as a rate.
* **`gnss_broker/cli.py`** — 2,505 lines of CLI surface and #89 frozen tuning. `--help` verified
  byte-identical at 2,551 lines.
* **Ten named cycle stages** promoted to nested routines: detections→seeds (765 lines),
  dead-reckon is NOT among them (see below), DLL is not either.

**THE GATE IS NOW ONE COMMAND AND 18 SECONDS**: `scripts/gnss/gate.sh` runs all seven fixtures
in parallel plus the unit tests, and both halves were **proven able to go red** before being
trusted. The unit half exists because the digest half is blind by construction to the #90 gate
(it strikes on the wall clock) and to #92 (it posts to a gather that never answers in a replay).

**A LATENT BUG FELL OUT, and it is the best argument for the whole programme.** The carrier
loop assigned `fleet = sorted(car_trim.values())` — a LIST — over the DLL's per-PRN state
DICT for the rest of the cycle. Every later consumer, including the FLEET-TRIM arming block
that decides which PRNs the C++ trim loop actuates, then indexed a list of floats by PRN. It
has never fired because `--carrier-gain` is 0.0 in production; arming the carrier loop (exactly
what #52 wants) would have taken the **C++ trim loop** down, with a traceback pointing at the
trim code. Found by interface analysis, not by reading. See the buglist's A-FIXED entry.

### Step 1b — DONE 2026-08-26: the topology, and the two elephants

| | start of day | after step 1 | now |
|---|---|---|---|
| the cycle loop body | 6,940 | 4,834 | **1,045** |
| largest block inline in the loop | 1,883 | 1,846 | **67** |
| named routines | 0 | 10 | **29** |
| max nesting | 15 | 10 | **9** |

**THE KEY MOVE WAS NOT EXTRACTION, IT WAS OWNERSHIP.** Both elephants resisted for the same
reason: a name assigned only inside a block has no binding in `main()`, so `nonlocal` is a
SyntaxError and the block cannot become a routine at any price. The fix is an OWNER OBJECT --
`_DllProducts` and `_DrProducts` -- because **attribute assignment needs no `nonlocal`
declaration**. Once `fcoh` became `_dllp.fcoh` and `clk_now` became `_drp.clk_now`, everything
downstream was mechanical.

* **The DLL was two things bolted together**: ~490 lines of independent endpoint polls with a
  231-line control loop in the middle. Now eight `_instr_*` routines and `_stage_dll_control`,
  the only part that actuates.
* **Dead-reckon was a pipeline all along**: predict → collect offsets → solve clock → adopt →
  seed. Now five `_dr_*` routines, and its working set (`pd`, `offs`, `la`, `tag`, the
  propagation constants) is named on `_drp`, so each arrow of that pipeline is a field rather
  than an implicit shared local.
* **`fleet` is now `_dllp.fleet`.** That dict is the cycle's central state and spent its life
  as a bare local, which is exactly what let the carrier loop assign a sorted LIST over it (the
  A-FIXED buglist entry). As an attribute of its producer, that collision is no longer a bug
  you must catch — it is one you cannot write.

**`broker_iface.py` WAS WRONG THREE TIMES, AND THE GATE CAUGHT IT EACH TIME.** Worth recording,
because the tool is the thing future work will lean on:
1. it judged live outputs by LINE ORDER, which stops meaning execution order the moment a stage
   is promoted (a routine's body sits before the loop but runs later). `up` was called dead,
   became a local, and three fixtures moved.
2. it counted a Store inside an already-promoted routine as a main-level binding. It is not.
3. it treated function PARAMETERS and import aliases as reads of enclosing state.
The rule it now uses is order-independent: a name is shared only if some other scope READS it
without first assigning it. All 29 routines were re-audited under the corrected rule; the two
flags that survive are both provable false positives (a nested closure reading its parent's own
local, and an `import ... as _rs`).

### Step 2 — DONE 2026-08-26 PM: out of `main()` entirely, interface named

| | start of day | after step 1 | now |
|---|---|---|---|
| `gps_distributed_broker.py` | 11,084 | 8,598 | **7,411** |
| the cycle loop body | 6,940 | 4,834 | **1,042** |
| stages living OUTSIDE `main()` | 0 | 0 | **16 of 30** |
| modules in `gnss_broker/` | 15 | 19 | **30** |

**THE MOVE THAT MADE IT POSSIBLE, again, was ownership.** A nested routine costs nothing to
create because a closure supplies every free name — which is exactly why the interface stays
invisible. Measured: the 29 stages read **210 distinct free names** out of `main()` between
them. A stage cannot leave until its set is written down, and a bare local has no home to
reference from another module.

* **`ChainContext`** (`context.py`) names that interface. Its split is load-bearing: STABLE
  (bound once at startup, never rebound in the loop — 39 of the 45 shared names, measured) vs
  PER-CYCLE (`t0`, `best`, `status`, `probe_set`, refreshed where each is actually rebound).
* **Owner objects for each loop's memory** (`loopstate.py`, `clockbias.py`): CarrierState,
  WatchdogState, NhOverlay, DllLoopState, HoldState, CpTracking, ClockBias — 43 bare per-PRN
  tables, ~290 renames. The membership test is *"would these reset together if the loop
  restarted"*, not "are they about satellites".
* **The nonlocal wall fell** when `pred`/`up` became context attributes: a module-level
  function can assign `ctx.pred`, where it could never declare `nonlocal pred`. That is the
  general escape, and it is what let the almanac stage — the first that WRITES shared state —
  move out.
* Modules now: `instruments` (8 measurements), `deadreckon` (the clock pipeline),
  `almanac`, `codeloop` (the DLL control loop + watchdog), `statepub`, plus `context`,
  `clockbias`, `loopstate`.

**`broker_extract.py` REFUSES what it cannot map** — a free name with no ChainContext slot, or
a stage that writes shared state. Refusing is the point: the argument for the whole programme
is that an interface should be impossible to leave unnamed.

**FOUR DEFECTS THIS STEP SURFACED**, none of which reading would have found:
1. `_rr_railed += 1` — an augmented assignment READS before it writes, and `broker_iface`
   recorded only the Store. Killed gal_e5a 20 s into the live swap. **The fixture gate was
   green before and after**: the counter only increments when a carrier command RAILS, and no
   command railed in 113 recorded cycles. A digest gate covers executed paths, not reachable
   ones.
2. `receiver_state` and `dr_eph_mod` are imported CONDITIONALLY, so on configurations without
   those flags the names do not exist and anything mentioning them raises. The **synthetic**
   fixture caught the second — every on-sky fixture runs dead-reckon, so the fake fleet covers
   a configuration none of the real captures do.
3. The `_fleet_trim_stat` key set is a contract (`rb`/`rb_fail`); dropping keys gives readers
   a KeyError, not a zero.
4. `car_locked` is a SET, `nh_off_hist` a pooled LIST, `nh_offset` a one-cell list whose
   purpose is shared identity. Initialising them as dicts would have been a silent behaviour
   change the digest might well have missed, since those tables are dead in production.

### Step 3 — what is left

14 stages still nested, and the blockers are now small and named: `utc0_sample0`, `_xb_pred`,
`coast_polls`, `have_sig`, `_rr_cons`/`_rr_resid`/`_rr2_resid` (the rate-feed cluster — it
wants a `RateFeedState` owner), the nav-decoder set (11 objects — a `NavDecoders` owner), and
the helper functions the big stages call (`cp_predicted`, `_joint_state`, `_p2c_*`,
`_decoded_entries`). The pattern is established; what remains is applying it.

### Step 1c (superseded) — what was left after step 1

The loop has one block over 60 lines. What remains is not decomposition but **promotion out of
`main()` entirely**: the 29 routines are still nested, so they close over ~200 shared names
instead of declaring an interface. Turning them into module-level functions or methods needs
that interface named — which is what the owner objects have started. Do `_dllp`/`_drp` first
(they are already coherent), then the seed table and the per-chain config.

### Step 1b (superseded) — where step 1 had stopped

`broker_iface.py map` now reports **zero** lines promotable without `nonlocal`. What remains:

* **dead-reckon (1,846 lines)** — carries `clk_now` across cycles with **no binding in `main()`
  at all**, so `nonlocal clk_now` is a SyntaxError. Promoting it was tried, the gate caught it
  immediately, and it was reverted. This is the epoch-critical stage — every walkoff this
  project has chased lives in it — so its state must be made explicit FIRST, in its own commit
  with its own reasoning. Not a move.
* **the DLL (1,209 lines)** — same shape: `fcoh` and `_kcoh` are assigned only inside it and
  read outside via `(fcoh or {})`, which guards None but not `NameError`. Give them an explicit
  home, then promote.
* **nav-bits (323)** — twelve carry-over names, one of which is an `except ... as e` binding
  that Python deletes at handler exit. Off in production, so the gate is blind to it. Leave it.

The pattern for all three is the same and it is a DESIGN change, not a move: give the block's
cross-cycle state an explicit owner (an object, as `AdmissionGate` and `TrimHandover` now are),
and the promotion becomes mechanical.

### Step 1 — the original plan, kept for the reasoning

Ordered by (decision density × how often it changed this week), highest first. All three are
pure functions of state plus inputs, which is why they are extractable at all.

1. **The admission / re-seed gate** (#90). Inputs: per-PRN `present`, `q`, `spec_tau`,
   `spec_ratio`, wall time, chain present-count history. Output: fire / hold / strike, with a
   reason. Currently ~80 lines at nesting depth 6–8 inside the DLL loop, holding four pieces
   of closure state. Target: `AdmissionGate` with the v3 rules (min-absence, recent-lock) and
   a table-driven test — every one of 2026-08-25's eight fires becomes a test case.
2. **The trim / handover logic** (#92, #77, #78). The birth re-base delta, the bound, the
   armed gate, and the `_ft_armed_last` handover. Target: pure `handover_delta(...)`, plus
   wiring the OTHER re-anchor sites to `/adjust_trim` (that is the remainder of #77 — the
   transport now exists).
3. **The shadow instruments** (GAP3-SHADOW, JRRP). Ramp-slope estimation with window resets,
   #92-delta subtraction, de-meaning, the ADR capture. These are measurement code whose bugs
   produce *wrong verdicts* rather than crashes, which is the worst failure mode we have; they
   deserve tests most and have none.

### Step 2 — the loop skeleton

Only after 1–3 are out: split `_decoded_entries` along the seams it already has (the numbered
"3d''"-style comment blocks are de facto section markers). Target under 500 lines per section
and no new behaviour in the same commit as any move. Every step gated on 4/4 EQUIVALENT.

### Rules for the whole programme

* **Never mix a move with a behaviour change** in one commit — the digest is only informative
  if a red digest means "you changed behaviour", never "you also reformatted".
* Flag surface: #89's method (freeze, prove equivalence, delete) applies again. 220 live flags
  against 135 used in production is the next `_FROZEN` sweep.
* Keep the file's comment culture. The long ⚠️ notes are the project's memory of what already
  went wrong; a refactor that drops them re-opens those bugs.

---

## (3) AUTOMATIC PATHOLOGY DETECTORS

### Why

Every mechanism found this week came from a human noticing a shape in the viewer (E3 → #91 and
#92; E32 → #90; E4 → the population-bias catch). That works and should continue — but the same
pathologies now **recur**, and on 2026-08-25 the recurrence actively corrupted an experiment:
four #90 fires rode an unnoticed band brownout, and a q-SD "no harm" verdict was computed on a
population that silently drops the sick satellites.

So the goal is NOT to replace case studies. It is to make the *already-understood* pathologies
announce themselves, so a case study starts from a labelled event and an arm is never judged
across an unnoticed one.

### D0 (first, and it is small) — a population-honest q series

THE LESSON OF 2026-08-25: the broker's `DLL:` line lists only satellites passing the
presence/arming gate, so **a satellite whose q craters leaves the sample**. Any statistic over
that population measures survivors. E4 was absent from the e5b line for 85 minutes — exactly
its sick interval — while my q-SD comparison reported "no harm".

Deliverable: a per-PRN per-chain q/state series that keeps absent and disarmed satellites with
an explicit state label (`present` / `absent-seeded` / `dropped`), served and logged. Every
future arm judges on this, never on the DLL line. Small, and it retro-fixes a whole class of
verdicts.

### D1 — brownout detector (#91)

Signal: chain present-count against its own 600 s peak, plus band-vs-band asymmetry (e5a
steady while e5b collapses is the discriminator that identified the 2026-08-25 23:00 event).
Already implemented *inside* #90's admission gate as a suppression rule; **promote it to a
first-class chain-level signal** that logs an episode with start/end/depth. Then: arms
auto-annotate "this window contained a brownout", and #91's own root cause gets a population
of labelled episodes instead of one case study.

### D2 — latch detector (#90 v3's targeting, reused)

Signal: continuously absent ≥ 300 s AND healthy (q ≥ 2.0) within the 15 min before absence
began AND the chain is NOT browned out. That is exactly the v3 admission rule — so **build it
as a detector first and let it run unarmed**. It answers the question the flight could not:
*how often does the deep latch actually happen?* One night of armed flying produced zero
genuine targets, so the base rate is the missing number, and a detector measures it at zero
risk before any re-arm.

### D3 — the sawtooth / handover detector (#92)

Signal: standing trim ramping monotonically past ~0.5 chips followed by a q drop and a trim
wipe within one cycle. Gives #92 its P2 population without waiting to notice one by eye, and
distinguishes handover sawtooth from brownout collapse — the two that superposed on E3 and made
per-PRN structure look heterogeneous.

### Shape

All four are broker-side, read-only, and default-on (they change no control path). Each emits
one line per *episode*, never per poll — a detector that logs continuously is a detector nobody
reads. Episodes carry `{chain, prn, t_start, t_end, depth, class}` so the judge scripts can
exclude or stratify by them automatically. That last property is the real prize: **verdicts
that are automatically regime-stratified**, which is what both GAP 3 and #92 needed and neither
had.

---

## (1) A1 — ✅ DEPLOYED 2026-08-26 01:11 (checklist steps 1–4 green; live proof = next F-engine restart)

`95ce4b712`. `FleetDll` now re-anchors on an F-engine frame0 move (size + persistence
discriminator; trims/arming survive), `epoch_resets` is served in `get_stats`, and a WARN fires
when a chain drops frames as LATE at a rate comparable to its close rate.
`scripts/gnss/epochreset.cpp` is the regression, verified to FAIL against the pre-fix header.

**Deploy checklist, when the soak wraps:**
1. `scripts/gnss/build_tool.sh epochreset && scripts/gnss/epochreset` → 13/13 PASS.
2. Archive `/tmp/gnss_gather.log`, run `scripts/gnss/gather_up.sh` on cf06 (binary already
   built in `build_nodpdk`).
3. Confirm 60 senders, spreads ≤ 1, and `epoch_resets: 0` in
   `/fleet_trim/get_stats` on a healthy stream.
4. Confirm the trim store re-adopted (`offered > 0` with `adopted == 0` is the alarm).
5. The real proof only arrives at the next F-engine restart — until then this is a
   regression-tested fix with the live path unexercised. Note the next restart's
   `epoch_resets` and `late_frames`.
