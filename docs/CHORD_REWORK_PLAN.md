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

### Step 1 — extract, in this order, each with unit tests

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
