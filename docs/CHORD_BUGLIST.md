# CHORD GNSS — buglist / open items

Open issues, unknowns to revisit, and watch items for the **CHORD** side (branch `kv/chord-gnss`).
Newest/active at top within each group.

**Scope, and why this file exists.** `airspy_docs/buglist.md` is the PROTOTYPE's list — a
different repo (`lwlab@gx10.vdl:/home/lwlab/airspy_gps/docs`), a different instrument, and its
newest entry is 2026-07-28. It has never tracked CHORD. Before this file the CHORD open items
lived in two prose sections that only a reader of the whole document would find
(`CHORD_GNSS_STATE.md` §8.22.3/§8.22.4 and `gnss_gpu_search.md` §9.13), so "what is open" had no
single answer. This is that answer. The two sections above remain the narrative; this is the
index, and it is the one to keep current.

Last reconciled: **2026-08-22 (code/config pass at HEAD `6f82e101d`)**, against session tasks #6–#89.
Previous reconcile 2026-08-18 (offline review) against #6–#88; before that 2026-08-17 ~00:45
against #6–#83, and 2026-08-06 against #6–#20.

⚠️ **THIS PASS READ THE TREE, NOT THE SKY.** Every `[verified 08-22]` below means *checked
against the working tree and `config/gnss_chains_chord.yaml` at HEAD `6f82e101d`* — a grep, the
commit that landed it, or the yaml line that arms it — and the entry says WHICH. The fleet was
not reachable from the host this pass ran on (`curl cx19:12048/...` → 404, no
`/tmp/gnss_broker.log`, no numpy to run the regression tests), so **nothing here is a live
measurement taken today**. Numbers quoted from commit messages and config comments are labelled
with the date they were measured, never today's. Claims I could not settle either way are marked
⚠️ **UNVERIFIED** in place rather than left looking fresh.

**How to read the status marks.** `[verified 08-22]` = checked against the tree this pass, with
the check named. `[carried]` = believed true, last checked when the entry was written, NOT
re-verified — treat as a claim, not a fact. `[withdrawn]` = tried, measured, backed out.

---

## The instrument as CONFIGURED, 2026-08-22 — read from the tree, not from the sky

The 08-17 prompt-health table that stood here has been **deleted**: it was one live poll, five
days old, and re-reading it as current is exactly the trap this file exists to prevent. What can
be stated from the tree is the control authority each chain actually holds, and that has changed
structurally since 08-18. Line numbers are `config/gnss_chains_chord.yaml` @ `6f82e101d`.

| chain | search | C++ code loop | joint state | deep gate | carrier |
|---|---|---|---|---|---|
| **gps_l5** | yes, `detectors:` :267 | `fleet-trim-url` :451 | consumes `rate,slew` :527 | from-search 100 :392 | `rrate-command` OFF |
| **gal_e5a** | no | `fleet-trim-url` :671 | consumes `clk,rate` :1271 | hand list `"33"` :1319 | `rrate-command` OFF |
| **gal_e5b** | no | `fleet-trim-url` :1580 | consumes `clk,rate` :1612 | hand list `"33"` :1613 | `rrate-command` OFF |
| bds_b2a | no | **none** | **none** | none | — |
| bds_b2b | no | **none** | **none** | none | — |

**"One chain works and four do not" is out of date, and the replacement fact is a three-tier
split.** gal_e5b was brought to gal_e5a's baseline (`0c0a08d85`, R7/#49) and both Galileo chains
now consume the joint clock and rate, so THREE chains have a closed code loop and TWO have
nothing at all. The two BeiDou chain blocks contain only transport keys — endpoints, `signal`,
`seed-phase-transport`, `dr-clock-adopt` — and not one control flag between them.
[verified 08-22: chain blocks start at gps_l5 :220, gal_e5a :562, bds_b2a :1389, gal_e5b :1460,
bds_b2b :1654; b2a/b2b read end-to-end, nothing else is set.]

**And the pair is finally twinned.** As of `6f82e101d` gal_e5a and gal_e5b differ on no
non-endpoint flag: `dr-forecast-lead-s`, `rrate-coarse-deweight`, `joint-consume`,
`rrate-command` and `rrate-phase-feed` were all matched, the last two OFF on both. Until 08-22
they differed on four axes at once, which had already confounded one attribution (see A2).
Any future A/B across this pair now changes exactly one variable. [verified 08-22 against the
yaml; the commit claims it was also checked by resolving both chains through `broker_multi.load`,
which I did not re-run.]

⚠️ Still true and still the frame: **only gps_l5 has a search** (`detectors:` on that chain
alone, :267), so #79's "the SEARCH admits" mechanism cannot transfer to the other four, and only
gps_l5 runs THE FLIP (`model-primacy-max: 2`, :541). Everything else is dead-reckon.

---

## ✅ SHIPPED 08-17 early AM — #83 build-out steps 1–3 (commits eb30892b3..68a8b8eea)

1. **#76 the trim readback** (A4 below): `get_dll` serves the integrator; broker reads it
   back each cycle. First data: trims settle sub-chip on gps_l5; the −3.0 rails seen at
   integrator birth were re-convergence transients.
2. **#83 2(d) the INNOVATION, served**: measurement − forecast per accepted detection,
   forecast by `tracker_phase_at` (what the tracker RUNS). ⚠️ The first deploy forecast
   with `dr_seed_phys` (the cp0 argument) and read wrap-uniform thousands of chips on
   healthy satellites — **that same cp0 fiction is the resolved mechanism behind the
   ±700–4500-chip CP_ERR reports**: the escape referee had been drawing accusations from
   a uniform distribution, neutered only by its sign/median gates. The referee now
   consumes the innovation (one number, two consumers). Live: gps_l5 p95 |innov| 1.2–2.8
   chips, n≈90/10 min — Phase 3's gate statistic exists on sky. DR chains have NO search
   (`detectors=0`) so no detection-innovation is possible there — their far-regime signal
   is #84/#50's spec_tau or a fold-based innovation, still open.
3. **#57 the residual-rate fold**: coh_cn0 fits a one-parameter within-integration
   residual (50 ms segment phasors, single-lag + lag-8 refine; probes ride the identical
   fit so the floor stays calibrated) and serves `rate_resid_hz` — the per-satellite
   carrier-rate innovation Phase 3 will consume. Synthetic gate extended: injected
   +2.00 Hz off → sky err +0.06 dB, rate_resid −2.01 Hz. This is the fix for the
   fleet-wide 60 dB KCOH sig flap (the fold duty-cycling on a one-cycle-old rate).

All three landed with the 4-fixture digest gate EQUIVALENT throughout (the readback and
innovation are flag-gated/serve-only; holds golden 9b2fab5c intact).

---

## ✅ SHIPPED 08-17 midday — #83 PHASE 3, first arm (commits 5d339f2fc..44ce01e76)

The one-controller build-out reached its first closed loop, each step measured before the
next was allowed:

1. **P3-1** `--rrate-kcoh-feed`: #57's probe-calibrated remaining rate into the joint
   rrate state, all five chains — the DR chains' first credible carrier-rate measurement
   (live: 1-2 sats/cycle/chain at σ 0.03-0.30 Hz).
2. **P3-2** `--joint-p2c-rotate` on gps_l5: the rotating coast test. First results: G28
   withheld 600 s → banded residual −0.13/−0.19/−0.31 chips. The shared state carries a
   satellite it is not told about.
3. **P3-3a** MINNOV served: the MODEL innovation per established satellite. Measured:
   **p95 1.4-1.9 chips vs INNOV p95 2.2-2.5 on the same satellites** — the model forecasts
   the sky better than the search-anchored seed (each re-anchor injects the search's own
   1-2 chip per-fix scatter).
4. **P3-3b THE FLIP** (armed gps_l5, cap 2, KV-approved): flipped PRNs seed from the model
   (dr-slew: BRDC + joint clk/b_sat, 'slew' consumer); detections feed only the filter,
   the innovations and the referee. Enter on measured minnov_p95 < 2.0 (n≥30); exit loud
   on p95 > 3.0 / referee starvation (600 s) / dr-integrity flag. Eligible-unflipped PRNs
   are the in-poll controls. Three would-be orphaning bugs found and fixed pre-arm (det
   eviction of dr ownership each cycle; the slew's no-detectors guard; dr_untrusted).

All four fixtures EQUIVALENT throughout (every new consumer default-off, yaml-armed).

---

## ✅ SHIPPED 08-17 afternoon — the axis fix soak PASSED + THE FLIP HOLDS (3c43d0ace..10286ffa6)

1. **2(b) fleet-wide soak PASSED** (30 min, all five chains under seed-phase-transport +
   dr-fengine-axis): SEEDAUDIT |step| med ≤0.9 / p90 2-6 chips **mixed-sign** (pre-fix:
   +6..+14 chain-common), KCOH folding real signal everywhere (bds_b2a sig up to 5937),
   zero tracebacks. No chain reverted. The gal_e5a trim sawtooth to the 3.0 rail
   (~1 chip/min) is the pre-existing #78/plant-oscillation behavior — same rate with the
   OPPOSITE sign at 02:06 before any of the day's changes, both signs in-window — not the
   walk.
2. **The flip integrity veto fixed** (`3c43d0ace`, deployed 15:47): dr_untrusted's a0
   residual judges the LEGACY model (EMA clock, no b_sat, ~1-chip bar) — a model the
   flipped sats don't run; on sky it flapped PRNs in/out every ~30 s and killed every flip
   in minutes. Flipped sats are now exempt at BOTH sites (flip exit + dr-loop seeding
   guard; either alone orphans the sat seedless) with MINNOV as their referee (p95/starve
   exits untouched) and a rate-limited MODEL-PRIMACY NOTE keeping the disagreement loud.
   **ON SKY: G23 flipped 16:04 (minnov p95 0.84) and HELD past 13 min at q 3.0, prompt
   68-80×** (record under the old veto: 16-18 s); G10 flipped 16:15, legacy-flagged +1.59
   five seconds later, overridden, q 3.6 at 177-197× — both as good as or better than
   every search-anchored sat in the same poll. Steady-state slew targets read CHIPS
   (±0.05..2.2) — the axis-fix prediction confirmed on the flip path. ⚠️ New cosmetic
   wart: the FIRST slew after ENTER can log a cross-currency model-held (G10 +101463.4
   chips ≈ 19.9 Hz × the 5095 chips/Hz lever); 0.05-chip actual motion, self-clears.
3. **`--innov-dr-seeds` staged** (`caa1e92c3` + yaml `10286ffa6`, rides the next restart):
   re-admits DR-owned seeds to INNOV now that dr stamps ride the F-engine axis; hard-tied
   to dr-fengine-axis in code. Validation on sky: served p95s vs MINNOV, few-chip scale
   expected.

All four fixtures EQUIVALENT for both commits.

---

## ✅ SHIPPED 08-17 night / 08-18 early — the wedge, the lag, and three retractions

1. **THE FLIP HOLDS** (`3c43d0ace`): dr_untrusted judged the LEGACY model (EMA clock, no
   b_sat) and killed every flip in minutes. Flipped sats exempt at BOTH sites, MINNOV
   referees. PRN 8 held 65 min, PRN 10 70+ min against a 16-18 s record; zero dr_untrusted
   exits; all exits honest.
2. **#84 CLOSED** (`461cebddc`): the aligned gather anchored on the slowest instance, so one
   frozen bench instance pinned SPEC-WINDOW fleet-wide. `--spectrum-stale-margin 8` →
   1/12 → 12/12 served, n_inst 1 → 10, and **b_sat finally accepts** (n≈600).
3. **THE cx19 WEDGE, root-caused** (`478797a84` + `4ddb8aca6`): the DPDK capture window's
   `advance()` is reachable only from the in-range path, so once the stream runs past it
   NOTHING brings it back. First stray packet was 5472 samples — under one frame — and 25 h
   later the window was unchanged with 18.7 BILLION packets dropped. Hit gnss1/port 1, then
   gnss0/port 0 after the recovery restart, which froze the broker's time base and took
   every clock-adopting chain down (#75). Ahead-resync armed on cx19 as canary.
4. **A1, the telemetry lag, decomposed**: 217 ms = 105 ms window quantisation + ~100 ms
   pipeline + 5 ms HTTP.
5. **`--fe-axis-stale-s`** (`9ffa65c06`): names a frozen time base. Not yet verified against
   a real stall.
6. **`--dr-forecast-lead-s`** (`d4c679f2b`, armed gal_e5a): seeds built for a CHOSEN future
   hop instead of "now" — KV's reframing, which dissolved the problem two failed fixes were
   trying to engineer around.

⚠️ **THREE RETRACTIONS, all measured**: serving `now_hop` (reverted — a seed never needed
"now"); the ephemeris epoch on the F-engine axis (**65x worse**, reverted); and
`--innov-dr-seeds` (falsified 20 min after arming, +2201 chips = the sub-ms residue of the
lag). Plus a fourth: my own falsifier for the forecast lead had no control clause and
tripped on an innocent change. A bar with no control clause measures the sky.

---

## ✅ SHIPPED 08-18 → 08-22 — the offline queue drained, and #33 GAP 2 got a real verdict

Everything in this section was checked against the tree this pass; the check is named inline.

1. **#46 — the node SERVES a clock** (`75445d458`). Three additive JSON fields on
   `GnssCoherentCombiner::get_status`, no wire-shape change: `serve_unix` (wall clock of the
   reply), `ingest_utc` (newest capture UTC INGESTED, before processing) and `ingest_unix`.
   `serve_unix - ingest_unix` is snapshot staleness; `ingest_utc - utc` is processing lag, not
   sky. [verified 08-22: `GnssCoherentCombiner.cpp:2525-2530` serve the three fields,
   `:485-486` fill `_ingest_utc`/`_ingest_unix`, declared `GnssCoherentCombiner.hpp:405-406`.]
   ⚠️ **This shipped AFTER the 08-18 offline plan explicitly demoted it and retracted its
   justification** (`docs/CHORD_OFFLINE_PLAN.md` §"#46 IS DEMOTED"): the argument that it
   unblocks two retracted fixes was withdrawn, because the ephemeris epoch does not want it and
   `--innov-dr-seeds` failed on a **215 µs** residue that was never shown to be this lag. The
   fields are a decomposition instrument, and **nobody has yet used them to decompose the
   215 µs** — see O11.
2. **#8 — clip fraction + band power, both halves** (`b8ca0f4f7` node, `958597061` broker,
   `295f1d0dd` viewer, armed fleet-wide by `392f79455`). Node: `GnssChordVoltageTap` takes
   `band_power_chans` / `_period_s` / `_hop_stride`, absent list = OFF
   (`GnssChordVoltageTap.cpp:37-56`). Generator: `rf_monitor_channels()`
   (`gen_chord_gnss_config.py:163`, wired at `:566`/`:863`) picks channels in **two lobes** so
   1176.45 and 1207.14 MHz are watched in the same sample — which is what makes a band-selective
   source diagnosable at all. Host half shipped 08-22 as **`scripts/gnss/rail_watch.py`**
   (`f13d57cfc`), a FIXED-cadence poller writing `/home/kvand/gnss/fixtures/obs/rf_rail_*.jsonl`.
   [verified 08-22: file present, 7929 B, both subcommands; the C++ config keys read in place.]
   Its own docstring records the event that forced it — 2026-08-22 00:38–01:05 UTC, clip
   0.085% → 5.57%, G9's q RISING to 5.4 (r = +0.86 vs clip) while E8 collapsed 3.0 → 1.0 —
   reconstructed from a ~5 min log line plus warnings that only fire above 1%, i.e. **sampling
   density correlated with the value being measured**. Do not make it event-triggered.
3. **#70 — the instance liveness guard** (`6e48d0c36`). A REST endpoint answering 200 is not a
   live instance; the guard polls twice and diffs. [verified 08-22: THE INSTANCE LIVENESS GUARD
   at `gps_distributed_broker.py:8474`, decision is a pure function in `fits.py`
   (`instance_stall_verdict`, imported :81) with `test_instance_stall.py`, plus the q stall guard
   `--q-stall-*` at :649-675 and the named-stall message at :5248-5263.]
4. **#49 / R7 — the fast code loop on gal_e5b** (`0c0a08d85`). [verified 08-22:
   `fleet-trim-url` + `fleet-trim-bandwidth: 2.5` + `fleet-trim-readback: 1` at yaml
   :1580-1582.] The per-PRN Python→C++ handover that `eec1d2f12` demanded is the enabling
   condition and it is live.
5. **#86 — ROOT-CAUSED AND FIXED** (`5e3e9ab5d`): `sigma_rate0` defaulted to `1.0` — **the same
   number as `rate_max`**, the garbage bound this file describes as 0.1 ppm, 2500× the truth. A
   prior that wide makes `clk_rate` nearly free at birth, so one 0.94-chip innovation taught
   −0.1002 chips/s at K1 0.107 and `predict()` walked the clock away: live at 15:1x on 08-21,
   clk **+684 chips** against a legacy truth of 150.9 that all five chains agreed on to under a
   chip, updates frozen at 101 while rejections passed 140, and σ still reporting 0.040 — wrong
   by 13000× its own claimed precision, for ten minutes, with **zero log lines**. Prior is now
   physics (0.05, ~8× the measured 0.0006–0.0065 drift). **A PRIOR AND A CEILING MUST NEVER
   SHARE A NUMBER.** [verified 08-22: `state_filter.py:175` default `sigma_rate0=0.05`, `:179`
   `rate_max=1.0` kept separate as the garbage bound, rationale at :193-210, clamp at :1164.]
6. **#89 — 72 broker flags retired into `_FROZEN`** (`63c6f20a6`): 278 → 206 argparse flags.
   KV's measurement was that 278 flags carried ~30 actual tuning decisions. [verified 08-22:
   `_FROZEN` dict at `gps_distributed_broker.py:227`, applied at `:2579`; `add_argument` count is
   now **207**, and the extra one is `--dr-clock-wait-s` (:1361), added by `76cebb3eb` after the
   retirement. The arithmetic holds.]
   ⚠️ **#89 IS TWO DIFFERENT TASKS.** `7b1cb0094` is also "#89" — the N² eigencalc Hermitian
   check ("measure the non-real diagonal instead of dying on it"). Same number, unrelated work.
   Do not merge them; the next reconcile will be tempted to.
7. **#68 CLOSED as a side effect** (`e39166a06`): the EOP table is a rolling six-entry ~5-day
   window inherited verbatim from `live_config_20260730.json`, so any captured base is stale
   within a week and stays stale. `live_eop_table()` now polls the fleet at generation time.
   cx19's "Requesting EOP later than in table" counter FROZE at 17,286,844 after the live push
   — delta 0 over 25 s, having climbed at frame rate since the base was captured. **That flood
   had been read as log noise for weeks, and it had already blocked one diagnosis and killed
   recv1.** [verified 08-22 from the commit; the generator change is in the tree.]
8. **Warm ElemCal** (`56db6c38a`, `2122d83d2`, `13af9b5a9`): 0% warm fleet-wide because the
   trackers re-anchor nearly every record and record-assemble called `ElemCal::reset()` on every
   one. But the per-element cal is `E_e·conj(E_ref)` and the common carrier cancels out of that
   product, so a carrier re-pin does not move the element gains — the reset was over-conservative.
   Now HOLD across re-anchor with a decoupled `_elem_prev_ok`. [verified 08-22:
   `GnssGpuRecordAssemble.cpp:88,241,354,427,476`.]
9. **`/set_reference_element`** (`1ceb6f42e`): swap the header reference feed live, no restart.
   [verified 08-22: callback registered `GnssGpuRecordAssemble.cpp:189-192`, config key read at
   `:49-52`, default 0.] The open lead it exists to chase — that reference element 0 may be the
   WEAKEST feed — is A6 below.
10. **N² send armed fleet-wide** (`941206fe7`, `392f79455`, after `dd5096e14`/`bbd01ca62` disarmed
    it twice): both legs, 2 connections per node to recv1:11027, dropped frames 0 on all twelve
    legs. What had killed recv1 was FRAME METADATA — 5.007 ms of dUT1 from the stale EOP table —
    i.e. item 7 above, not the correlator. [verified 08-22 from the commits; `--n2-send-legs`
    default `both` at `gen_chord_gnss_config.py:1994`. Live status NOT re-checked.]
11. **The joint filter's structural fixes**: `--joint-rereference` wired and benched
    (`7bf48e603`, `44dda0548`, default OFF — a gauge change is a change of COORDINATES, not an
    observation); a shorter code period MIGRATES the state instead of rebuilding it
    (`ec572655e`); the joint feed uses the trim the TRACKER APPLIED, not the one this process
    happens to hold (`03486d5f0`); filter notes escape on model-primary chains (`b3820d776`).
12. **#33 GAP 1, arms 11–17: campaign closed, machinery done, DISARMED everywhere.** Arm 11
    disarmed on per-sat command runaway past the physics bound (`d29b12445`); arm 13 disarmed by
    its own falsifier — the third feedback path was the fine phase feed (`616430410`); arm 14b
    disarmed per pre-registration, "stable but useless at current row precision" (`1f6b59aaa`);
    arm 17 a measured WASH (`5dea10518`). ⚠️ **Arms 15, 16 and 16b are VOID, not failed** — they
    were armed 08-20 16:03/16:49 INTO the cf06 gather wedge with the C++ code loop dead
    (`6f82e101d` states this explicitly). A control for an experiment whose verdict is void is
    not a control, it is an uncontrolled difference — which is why the pair got twinned.
    [verified 08-22: every `rrate-command` occurrence in the yaml is commented out; `carrier-gain:
    0.0` at :43 and :266.]

---

## 🔴 Active — the next levers, ranked  [re-ranked 2026-08-22 against the tree]

State: HEAD `6f82e101d`, tree clean apart from four fixture digests. **The 08-18 ranking is
retired in full**: its organizing principle was "the node restart is the scarce resource", and
the node-side queue it was protecting is now EMPTY — #8 and #46 both shipped and are armed
fleet-wide, #70 shipped broker-side, and the only other entries on it were #54 (still never
re-judged, now O9) and #88 (still open, now A8).

**THE NEW ORGANIZING PRINCIPLE: unconfounded sky time is the scarce resource.** In the last four
days this project voided FOUR arms (GAP 2 attempt 3, and #33 arms 15, 16 and 16b) by running them
across an instrument fault nobody checked for, ran a two-chain A/B that differed on four axes at
once, and let a falsifier be breached for fifteen minutes because the monitor recorded but did
not abort. None of that cost code; all of it cost verdicts. So the ranking below is by **can this
arm produce a verdict that survives** — which puts the instrument's own silent-failure modes and
the honesty of its declared uncertainties AHEAD of new features.

### A0. THE SAME PHYSICAL QUANTITY IS ESTIMATED TWICE, PER BAND  [new 2026-08-22, KV]
E5a and E5b are the SAME SATELLITE on the SAME CLOCK. Geometric range, satellite clock and
receiver clock are identical; the only physical difference is ionospheric, and iono scales
as 1/f^2, so across 1176.45 -> 1207.14 MHz the differential is (1.0261)^2 = 1.053 -- about
**0.01 chips** on a 0.2-chip iono delay. The code phase is one number, not two.

Yet the system estimates it twice, independently, per band:
  * the **DLL trim** is per chain (measured 2026-08-22: E31 |model-held| 12.98 chips on e5a
    against 3.24 on e5b; E33 1.36 against 10.03 -- disagreements of 4-7x on a quantity that
    should agree to 0.01);
  * the **fold rate** is per chain, from each chain's own record-stream fit (`_kcoh_rates`).
    Cross-band sign agreement measured at 38-62%, i.e. COIN FLIP, and the observed b/a ratio
    scatters -2.25..+10.12 against a predicted 1.026. Two independent estimates of one
    physical residual, and they do not even agree on its sign.

This doubles the noise on every shared quantity and lets the two halves of one satellite
disagree. It is the same structural error in two places, and it is exactly what the joint
state exists to remove -- `rrate` already lands both carriers of a Galileo satellite on ONE
m/s row (1.0261 apart) by design; the code side has no equivalent.
⇒ THE LEVER: one per-satellite code-phase state shared across the bands of a constellation,
with only the (tiny, computable) iono differential applied per band -- not two free
variables. Until then, every cross-band comparison in this project is comparing two
estimators rather than two measurements, which has already produced one retracted finding
today (the "e5b is 30x less sensitive" claim, killed by a time series).
[verified 08-22: trims and fold rates read per-chain from the live fleet]

### A0b. THE SATELLITE CLOCK CARRIES NO GROUP-DELAY TERM (TGD/BGD/ISC)  [new 2026-08-22]
VERIFIED IN CODE, `gnss_ephemeris.py:504`:

    clk = af0 + af1*tc + af2*tc**2 + F_REL*ecc*sqrta*sin(ek)

Polynomial plus relativistic, and nothing else. TGD appears in the repo ONLY inside nav
decoders (`gps_navdecode.py`, `gps_lnav_encode.py`); nothing feeds it into the correction
path. Broadcast clocks are referenced to an IONOSPHERE-FREE COMBINATION -- L1/L2 for GPS
(so an L5 user owes TGD + ISC_L5), E1/E5a for Galileo F/NAV and E1/E5b for I/NAV (so a
single-frequency E5 user owes the corresponding BGD). None of it is applied.

⚠️ AND THE STRUCTURE EXPLAINS THE SHAPE OF WHAT WE MEASURE. A delay common to every
satellite of a constellation is DEGENERATE with the receiver clock, so a chain that SOLVES
its own clock absorbs it and never sees it. A chain that ADOPTS another chain's clock
inherits the reference chain's convention and shows the difference as a constant. Measured
2026-08-22, signed median trim by chain:

    gps_l5   -0.005 chips   <- solves its own clock: absorbs its own group delay
    gal_e5a  +0.848         <- adopts gps_l5's clock
    gal_e5b  +1.201         <- adopts gps_l5's clock
    (sign test on the Galileo trims: 13 of 14 positive, p = 0.0009)

⚠️⚠️ BUT THE MAGNITUDE DOES NOT CLOSE, AND THIS ENTRY MUST NOT BE READ AS THE EXPLANATION.
Broadcast group delays are 5-20 ns = 0.05-0.2 chips. The observed common mode is 85-120 ns
(+0.85..+1.20 chips) -- about 10x too large -- and e5a/e5b differ by 0.35 chips (34 ns)
where their BGDs should differ by a few ns. So this accounts for the SHAPE and roughly a
tenth of the SIZE. The remaining ~90 ns is constellation-common AND band-dependent, which
points at the replica/code-phase convention or the cross-band adoption arithmetic, not at
the ephemeris.

✅ FIXED 2026-08-23 (`gnss_ephemeris.group_delay_s`, wired into `predict_all(signal=...)`
and the three dead-reckon call sites). MEASURED EFFECT on the live BRDC, per chain:

    gps_l5   median +0.115 chips (+11.3 ns)   per-sat -0.085..+0.196   sd 0.112
    gal_e5a  median +0.006                    per-sat -0.044..+0.098   sd 0.047
    gal_e5b  median -0.001                    per-sat -0.049..+0.055   sd 0.030
    bds_b2b  median -0.024                    per-sat -0.323..+0.335   sd 0.177
    bds_b2a  0.000 -- THE TERM IS NOT BROADCAST (see below)

So the prediction above held exactly: the CONSTELLATION-COMMON part is ~0.1 chips and the
+1.3-chip BeiDou / +0.85-chip Galileo offsets are untouched. What it does buy is the
PER-SAT spread (sd 0.03-0.18 chips) landing correctly in b_sat instead of in the model.

⚠️ B2a CANNOT BE CORRECTED FROM RINEX 3, and this is a data-source limit, not an omission:
TGD_B2ap lives in B-CNAV2, while RINEX 3 BDS records carry only TGD1 (B1I/B3I) and TGD2
(B2I/B3I). `group_delay_s` returns 0.0 rather than borrowing TGD2, which is a 1207 MHz
delay applied to an 1176 MHz signal. This BLOCKS the multi-constellation clock feed (the
2026-08-23 arm died on BeiDou's trims being outside the DLL's linear range) -- so the real
lever for that is NOT here. RINEX 4 CNAV records, or decoding B-CNAV2 on sky, would supply it.

⚠️⚠️ FOUND WHILE FIXING THIS -- OUR GALILEO BRDC IS MIXED, AND `best_eph` PICKS BY FRESHNESS.
On 2026-08-23: 19 satellites carried F/NAV records (clock referenced to E5a,E1) and 11
carried I/NAV (referenced to E5b,E1). Which reference a satellite gets is therefore
arbitrary and CAN FLIP at a refresh, stepping that satellite's clock by (BGD_a - BGD_b).
Both cross-type conversions are implemented from the ICD identity t_E1 = t_IF_a - BGD_a =
t_IF_b - BGD_b, and `test_group_delay.py` asserts the invariance directly -- without them a
record-type flip is a silent per-satellite step of a few ns.
⚠️ The trims are the ONLY reason we can see this at all: they are the receiver telling us
the model is wrong by a constant. Do not "fix" the trims by widening a clamp or feeding
them into an estimator before the constant is accounted for -- three interventions have
already failed that way (the ungated feed, the gated feed, the slew arm).

### A1. THE GATHER WEDGES ON A FRAME0 RESET, silently, and it kills the code loop fleet-wide
**Rank 1 because it VOIDS EXPERIMENTS, and it has done so at least three times.** `FleetDll`
keeps a strictly monotonic per-chain high-water mark. An F-engine restart moves `frame0`, so the
senders' absolute window index restarts small and **every subsequent frame is dropped forever**:
```
gnssFleetDll.hpp:195-198   if (c.have_newest && h->win < c.newest && !c.open.count(h->win))
                               { c.n_late++; return FoldStatus::LATE; }
gnssFleetDll.hpp:281-284   if (!c.have_newest || h->win > c.newest) { c.newest = h->win; ... }
```
`win` is unsigned and `newest` is never decreased, never cleared, and `have_newest` is never
reset — four occurrences in the whole tree, all in that header. `close_oldest()` is only
reachable from the path those frames no longer take, so `c.row` freezes at its last aggregate
and `dll_callback` (`GnssFleetTrim.cpp:581-593`) serves it verbatim, 200 OK, plausible numbers,
**identical `hop` on every row — that is the tell**. Meanwhile the 60 TCP connections stay
ESTABLISHED and `GnssTelemGather` keeps broadcasting, so nothing looks down.
**The condition is counted and never acted on**: `_late_frames` increments at
`GnssFleetTrim.cpp:252-253` and is exposed read-only at `:625` — the diagnostic exists and
nothing consumes it. `gather_up.sh`'s own health check reads only `/telem_gather/get_stats`
`spread`, which is HEALTHY after a reset (all senders agree on the new small window), so the
script passes while `fleet_trim` stays wedged.
⚠️ **The Python arm already gets this right**, which is why only the C++ fold wedges:
`gnss_broker/telem.py:363-373` re-sorts its ring on a backwards `win` rather than assuming
monotone. The fix is that behaviour, or a backwards-`win` reset, plus an ALARM on `late_frames`
climbing at frame rate. Today the only mitigation is `scripts/gnss/gather_up.sh` (senders
self-reconnect in ~90 s; no node restart needed).
[verified 08-22: file:line evidence above read directly; `git log --since=2026-08-15` over
`gnssFleetDll.hpp`, `GnssFleetTrim.cpp`, `GnssTelemGather.cpp`, `gather_up.sh` returns nothing —
**no fix has been attempted in code**.]
**Pre-flight this before ANY arm**: poll `/fleet_trim/get_dll` twice — `hop` must ADVANCE and
must match the live epoch.

### A2. THE SEED CHURN — one disease behind #78, #77, the 96% slew rail, and GAP 2's "noise"
Four entries that this file has been carrying separately are the same fault, and stating them
together is what makes the lever obvious. On a dead-reckon chain the code-domain seed is
manufactured by a control loop that **rails**, and there is no channel to tell the C++ side.
Measured 2026-08-22 on gal_e5b, same satellites, same sky, one variable (yaml :1584-1591):
the model-vs-tracker gap drifted at a paired median **0.0706 chips/s** against 0.0055 on e5a,
5 of 9 satellites past the slew cap against 3 of 9, **the slew railed on 96% of steps**, and
`held` ran out to **−27..−52 chips** before a re-birth snapped it back.
That snap-back IS the churn that GAP 2 then inherits and mislabels as noise (A3).
The two structural contributors, both re-verified this pass:
* **#78 — the fleet DLL's steady state exceeds its own clamp.** `_fleet_trim_nominal_hz = 23.84`
  is defined at `gps_distributed_broker.py:3304` and **appears exactly twice in the file, the
  other being its own comment at :3298 — it is written and never read**, and
  `--fleet-trim-leak-per-s` still defaults to 0.12 (:2073), ~10× weaker than intended, giving a
  railed-disc steady state ~5.2 chips against a 3.0 clamp. ⚠️ Still NOT the free one-liner:
  restoring the leak cuts the railed steady state to ~0.52 chips but cuts DC authority equally.
  It is a bandwidth/authority trade and needs a measured pull-in test.
  [verified 08-22 by grep count and the argparse default; the C++ side converts per-second to
  per-update at `GnssFleetTrim.cpp:451-456`, so the bandwidth genuinely scales with rate.]
* **#77 — re-anchors pop the trim in Python only.** The three hard re-anchors pop `dll_trim`
  and there is still **no channel to the C++ accumulator**: `reset_trim` does not exist anywhere
  in the tree. On a re-anchor the C++ trim re-applies on top of a phase that does not include
  it, bleeding out only through the ~8 s leak — a per-re-anchor step in exactly the quantity A3
  is trying to measure. Lever unchanged: add a `reset_trim` list to the policy POST.
  [verified 08-22: `grep -rn "reset_trim\|reset-trim"` over `python/scripts/gnss/*.py` and
  `lib/stages/gnss/*.{cpp,hpp}` returns **nothing**.]
⚠️ **ONE CORRECTION TO THE STORY AS I RECEIVED IT.** The claim that matching `joint-consume` on
gal_e5b took the paired drift ratio 12.8× → 1.0× is **NOT supported by the tree**. The 12.8×
figure is recorded (yaml :1586), but `6f82e101d` states plainly that the attribution was
confounded — the pair also differed in `dr-forecast-lead-s`, `rrate-coarse-deweight`,
`rrate-command` and `rrate-phase-feed` — and describes the clean within-chain e5b before/after as
"now running". **No post-fix ratio is recorded anywhere in the repo.** Treat "1.0×" as an
in-flight measurement, not a result, and write it down when it lands.

### A3. GAP 2's MEASUREMENT IS 42× MORE PRECISE THAN IT IS — and that alone explains the failure
Tested properly 2026-08-22 01:24–02:07 and it **fails on its own**, from the best baseline of the
session (clk +149.710 ± 0.094, 16% rejection, |b| ≤ 5.3, 0 notes), over 37 min:
```
t+378    rej  88 / upd  79    |b|  6.3    clk 133.5
t+1282   rej 431 / upd 396    |b| 23.0    clk 178.4   <- |b|>20 falsifier BREACHED
t+2186   rej 737 / upd 629    |b| 46.9    clk 153.6   notes 0 -> 6
```
**And the mechanism is not (mainly) the feedback loop.** Per-satellite scatter of `y` about its
own straight line — so a real slow bias is not counted as noise — reads E8 13.3, E16 17.3,
E25 15.7, E2 10.8, E32 12.6, E3 9.0 chips, **pooled 12.5 chips, fed to the filter as
joint-sigma = 0.30. Understated 42×.** A measurement declared that precise gets ~half its samples
rejected and the surviving half is the tail nearest the state — biased sampling that drags
`b_sat` a little further every cycle while the gauge sloshes it into `clk`. It also explains why
attempt 6 looked immaculate: its spread was 1.5–3.8 chips. **Same config, quieter sky. The config
was never the variable.**
⚠️ **And the honest reading may be worse than "weight it correctly":** 12.5 chips of scatter
against a physical bias scale under 6 chips may mean `y` carries NO per-satellite information at
all — in which case honest weighting makes the filter correctly IGNORE it, which is not
"GAP 2 works now".
**THE STRUCTURAL FINDING, and it is a rule, not a bug:** on a model-primary chain
`y = held + trim − cp_predicted` is built from the very seed the consumer sets, so feeding and
consuming close a loop with no observation in it. `slew` closed it fast (+0.065 chips/s, dead in
10 min); `rate` closes it slowly, because a rate must integrate before it shows. **gps_l5 is
immune because it does not feed** — its `y` comes from search DETECTIONS, an independent look at
the sky, so the loop is open at the detector. ⇒ **A CHAIN MAY FEED THE JOINT STATE OR CONSUME AN
OUTPUT THAT REACHES ITS SEED, NEVER BOTH**, unless something outside the loop pins its input.
**Levers, in order**: (1) weight `y` at its measured scatter and see whether anything survives;
(2) kill the churn at A2, since `y` inherits the seed's own steps (r = +0.71 with p90 |seed step|,
startup excluded); (3) give a DR chain an independent anchor — a search, or #50's `spec_tau`.
⚠️ **I failed to abort on the falsifier**: it was breached at t+1282 and the run continued 15 more
minutes because the monitor only RECORDED. *Saying a falsifier is armed is not the same as arming
it.* [verified 08-22: the whole verdict is recorded in yaml :1197-1223 (`f2524ab04`); the feed
flag `joint-model-primary` is COMMENTED OUT at :1224, so the loop is currently open.]

### A4. DR chains have never consumed `slew` on a NON-FEEDING chain — and that is now legal
The one-controller rule (A3) forbids feed+consume. **The feed is currently off on every chain**,
so the prohibition does not bite, and the per-satellite `b_sat` correction — *the whole point of
#33* — has still never been tested on a dead-reckon chain with the loop open. What was measured
and disarmed (`85a185f98`) was `slew` on a chain that was ALSO feeding: joint-vs-legacy walked
+8.456 → +13.211 → +16.356 chips in 122 s, monotone, and gal_e5a's q stability regressed 5.3×
(mean |dq| per 15 s 0.103 → 0.549) against gal_e5b's 2.4× over the same window — the sky's share,
from a chain nobody touched. **That verdict does not apply to the current configuration.**
The arm is one yaml line (`joint-consume: clk,rate,slew` on gal_e5a, :1271) with gal_e5b as a
now-genuinely-matched control, judged on paired mean |dq| per 15 s at MATCHED post-restart age.
⚠️ Do not arm this and the feed together, ever. [verified 08-22: yaml :1057 still carries the
commented `joint-consume: slew` and the note tying it to the feed; the feed is off.]

### A5. #33 GAP 1 — re-fly the carrier command as a TRUE single-axis A/B
Machinery is done and consumed at one site; `rrate-command` is off on all five chains and
`carrier-gain: 0.0` (:43, :266). The 08-20 payoff blocker was cleared on two instruments (e2e
[4e] `eb38a418b`; the warm-plant re-measurement) and that clearance still stands. What is NOT
usable is the arm history: **arms 15/16/16b are void** (armed into the gather wedge) and arm 14b
was disarmed as "stable but useless at current row precision". So this is not "re-arm and
continue" — it is **fly it once, cleanly, on the twinned pair**, with A1's pre-flight run first.
Judge on KCOH duty and `rate_resid_hz` against the unarmed sibling IN THE SAME POLLS, never on
sig. This is also where #71's carrier NCO gets re-judged: it failed 3× as a bolt-on because the
seed is the reference and is re-pinned every window, and one controller owning seed and carrier
is the entire point.

### A6. Reference element 0 may be the WEAKEST feed  ⚠️ UNVERIFIED
The instrument to test it shipped 08-20 — `/set_reference_element`, live swap, no restart
(`1ceb6f42e`; callback `GnssGpuRecordAssemble.cpp:189-192`, default 0 at `:49`). The claim that
element 0 is in fact the weakest feed is carried from the ElemCal work and **I could not check it
this pass** (it needs a live per-element amplitude read). Cheap to settle, and it bounds how much
the now-warm ElemCal is actually buying. [instrument verified 08-22; the claim is NOT.]

### A7. #56 — the ~5× hourly signal swings are FINALLY instrumented
This sat blocked on #8 for weeks. Both halves now exist and are armed fleet-wide, and
`rail_watch.py` already caught one event end-to-end (clip 0.085% → 5.57% over 27 min, with q
moving in OPPOSITE directions on two satellites). The discriminator is stated in the tool's own
docstring and should be applied to the first full day of `rf_rail_*.jsonl`: **if clip rises
because total power rises, the cause is a source or an interferer; if power is flat while clip
rises, it is a gain or quantiser-scaling change.** Nothing blocks this but the reading.

### A8. #88 — finish the j2 restructure  [pure offline, still open]
Unchanged and still open: `build_n2dual_branch` still builds blocks in Python
(`gen_chord_gnss_config.py:883`, called at `:2856`/`:2879`), and deployment still rides the
captured base (`live_config_20260730.json`; the generator still injects into "the captured base's
dpdk block", `:2145`). The EOP fix (`e39166a06`) is the proof of why the second half matters: a
captured base goes stale with no event to blame. [verified 08-22 by grep.]

### A9. #85/#50 — `spec_tau` against the model-held offsets
Unchanged. #84's fix revived the measurement (`spectrum-stale-margin: 8` is live at yaml :143).
Correlate SPEC-FIT tau against the model-held offsets on the SAME sats before feeding anything.
⚠️ `spec_tau` is PLAUSIBLE, not STRONG (p/f median ~1.0) — #50 is blocked on significance, not
plumbing. This is also the most promising independent anchor for A3's lever (3).
[verified 08-22: `reseed-spec-tau` is still commented out, yaml :1342 — the only occurrence.]

### A10. #33 GAP 3 — code/carrier coupling: plumbed, and never once armed
⚠️ **A small correction: "not started" is wrong.** The coupling is implemented —
`d(b_sat)/dt = rr_bsat_chips_per_m · rrate` in the state transition
(`state_filter.py:438-459`, `:729-732`) with a bench (`test_rr_bsat_coupling.py`) — but
`--rr-bsat-chips-per-m` defaults to 0.0 (`gps_distributed_broker.py:1555`) and **is not set
anywhere in `gnss_chains_chord.yaml`**, so it has never run non-zero. Physically
|k| = f_chip/c. Blocked behind A3: coupling a per-satellite bias to a rate measurement is
pointless while the bias state is being dragged by a 42×-understated sigma.
[verified 08-22 by grep across `python/` and `config/`.]

---

## 🟠 Open — real, with a known lever

> #77, #78 and #56 used to live here. They have moved UP into the Active section — #77/#78 into
> A2 because they are one disease with the slew rail and GAP 2's inflated noise, #56 into A7
> because #8 finally unblocked it. They are not new; they are re-ranked, and A2/A7 say why.

### O1. #82 — wrap-edge label partners (residue of #41)
When the true delay sits within one code period of the circular window's edge (~12.5% of
snapshots), the adjacent alignment's wrapped peak carries ~15/16 of the power AT THE TRUE
DOPPLER, so no comparison statistic can suppress it, and the fold arithmetic that makes
unwrapped partners label-invariant breaks. Rate ~2–3/hour at any SNR. Fix belongs in fold-aware
label COMPOSITION, not the argmax. Decoding clue: PHCONT reports `m % LC_SEG`, so every report
lands in [0,20) — "+19" is −1, "+17" is −3; the large-k near-exact events are small NEGATIVE
flips wrapped by the report. Resolve that first. [carried from 08-17, NOT re-verified this pass]

### O2. #40 — the deep fold's rate search wrong-bins onto overlay sidebands
Same disease as #41, one stage down: a maximum over a nuisance parameter (rate) on a grid where
mismatch sidebands can land on-bin while the true peak scallops. #41's fix pattern (score each
hypothesis at its continuous max, via a monotone mainlobe pair-sum) should transfer. [carried]

### O3. #50 — `spec_tau` far-regime re-seed: implemented, gated, NEVER ARMED
The one code-error estimator that works beyond DLL capture range, bounded (0.75 chips, gain 0.5)
and gated on `spec_peak_ratio` — a shuffled-null significance, i.e. a gate that can fail.
`reseed-spec-tau` is commented out on every chain and the RESEED log count is 0 for the broker's
entire life. **Its urgency went UP this pass**: A3 needs an anchor that is independent of the
seed, and this is the only candidate that exists. Tracked as a lever at A9.
[verified 08-22: one occurrence in the yaml, :1342, commented.]

### O4. #63 — the purge: the deep fold is what still blocks deletion
3 of 4 done; the comb DLL is live (`--telem-dll`, `gps_distributed_broker.py:2028`, and the
discriminator is formed broker-side at `:7560`). The deep fold's re-searching estimator is what
remains, and it is entangled with #47 (served C/N0 blind to code error) and #57.
[verified 08-22 that the comb-DLL half is in the tree; the remaining scope is carried.]

### O5. #67 + #65 — what is left of the observability quartet
**#68 and #70 are CLOSED** (`e39166a06`, `6e48d0c36` — see the shipped section). What remains:
* **#67** the broker's in-process telemetry reader delivers ~25% of the stream (GIL convoy).
  ⚠️ **UNVERIFIED this pass** — I could not find the measurement or a named code site by grep,
  and it cannot be re-measured without the fleet. Treat the 25% as a claim from 08-16.
* **#65** `agg_up.sh` still TRUNCATES its log instead of rotating — `> "$LOG"` at
  `scripts/gnss/agg_up.sh:71`, and `gather_up.sh:59` does the same. This has destroyed burst
  evidence twice. One-line fix (`>>`, or rotate on start). [verified 08-22 by reading both
  scripts.] The second half of #65 — the search's refine time swinging 600× at constant
  alignment count — is unexplained and NOT re-verified.

### O6. #62 — element cal applied NON-CAUSALLY
Weights are updated with the very record they weight, for both the comb and `/get_spectrum`.
⚠️ **Partially addressed at ONE site and only there**: the per-element complex gain now forms its
reference from the RAW per-antenna prompts, upstream of the cal, explicitly "so nothing
non-causal (#62) and no element's own noise can enter its reference"
(`GnssCoherentCombiner.cpp:576`). That is the leave-one-out reference, not the combine.
**The main claim — the combine and `/get_spectrum` weights — was NOT re-verified this pass.**

### O7. #46 (the OTHER one) — instance record-time divergence, 0.15 s over 5 lag levels
Distinct from the served-clock #46 that shipped as `75445d458`; **two different items share the
number** and the 08-18 list already carried both. No buffering exists in the design, so the
spread should not exist. **Newly measurable**: `ingest_unix` and `serve_unix` now decompose
node pipeline depth from transport per instance, which is exactly the axis this needs.
[carried as a fact; the new instrument verified 08-22.]

### O8. #73 — the gather's full per-element × per-freq × EPLH product is 0.98 GB/s
Against a 1 Gbps link. A budget problem to solve before the full product can ship. [carried]

### O9. #54 — replicas differ PER-SAMPLE at CHORD scale (9.5%)
Long-standing, marked START HERE at the time and never resolved. It was item 3 on the 08-18
offline queue ("re-verify or retire") and **that never happened**. A scan harness exists —
`cudaGnssDespreadTest.cpp:78` scans the absolute anchor "if the GPU-vs-CPU disagreement grows
with n0 it is the …" — so re-judging whether it still reproduces is a bench run, not a design
task. [verified 08-22 that the harness exists and the item is untouched; the 9.5% is carried.]

### O10. THE 215 µs HAS STILL NOT BEEN DECOMPOSED — and now it can be
`--innov-dr-seeds` was falsified on sky 2026-08-17 22:30: the first flipped PRN served
INNOV +2201.31 chips / p95 2198.42 while unflipped sats in the SAME log line read 2.39–2.64.
2201 chips = **215 µs**, the sub-millisecond part of the telemetry lag (integer ms vanish —
1 ms is one L5 code period), because `t_now_abs` rides the newest `pow_hop`, which trails the sky
by −99.6 ms median / 59 ms IQR. The 08-18 plan turned #46 into exactly this question — *what IS
the 215 µs?*, answerable offline from archived logs — and noted 215 µs is suspiciously close to a
record/window quantum. **Nobody has answered it**, and `#46`'s new `serve_unix`/`ingest_unix`
fields make it a subtraction rather than an inference. Until then `--innov-dr-seeds` stays 0
(yaml :106) and MINNOV referees dr-owned sats. Re-arming needs lag COMPENSATION on `t_now_abs`,
not a flag flip. [verified 08-22: flag is 0 in the yaml; rationale at yaml :98-105 and
`gps_distributed_broker.py:1658-1669`.]

### O11. Two unrelated tasks are both numbered **#89**
`63c6f20a6` = retire 72 broker flags. `7b1cb0094` = measure the non-real diagonal instead of
dying on it (the N² eigencalc Hermitian check, cx19 only). Both call themselves #89 in the
subject line. Not a bug in the instrument, but it will corrupt the next reconcile if it is not
written down. [verified 08-22 from `git log`.]

---

## 🟡 Premise changed — re-measure before acting

### P1. #48 — SPLIT, and the open half now has a name
*"Why does the prompt tap land on noise for 25–45 min"* — that was the presence latch, root-caused
and fixed on gps_l5 (#79). *"Why is b2b blind at all times"* — **still open**, and this pass gives
it a sharper frame: bds_b2b (and bds_b2a) have **no code loop, no joint consumption and no deep
gate at all** — they are transport-only chains. A chain with no control authority is not evidence
about the sky. The config's own note records b2b as having "the SMALLEST |disc| of the five
(median 0.25, 0/12 railed) while tracking nothing" (yaml :1681) — small |disc| there means small
SIGNAL, not a good loop. **Decide deliberately**: either arm the BeiDou tier (the #49 mechanism
verdict is already paid for twice — 0.81 vs 0.42 |disc| duty, paired) and find out, or declare the
tier a control and stop reading its numbers as instrument health. Today we do neither.
[verified 08-22 from the chain blocks; the 08-17 amplitude numbers are NOT re-measured.]

### P2. #61 — "the rate is real, but only ON-PEAK"
The premise was that the fold sat downstream of the code latch. The latch is broken on gps_l5, so
this wants a straight re-measure rather than more theory. [carried; premise-change verified 08-17]

### P3. #49 — the deep-gate rollout: SUPERSEDED, and the last hold-out is BeiDou
On gps_l5 admission is auto-generated from the search. On gal_e5a and gal_e5b the fast code loop
is now armed (`0c0a08d85`) with a hand deep-gate list of `"33"` on both. **The only chains still
with no route at all are bds_b2a and bds_b2b.** [verified 08-22 from the yaml.]

### P4. #33 / #52 / #57 — the joint-tracking and carrier cluster
Substantially advanced but not closed. #33's `rrate` state is built, fed and consumed on three
chains and **uncoupled to `b_sat`** (A10: the coupling is implemented and pinned at 0.0). #52's
remaining 0.27 Hz has no confirmed mechanism. #57's coherent-fold η is answered but the fix is not
shipped. **New this pass**: the binding constraint on all three is no longer plumbing, it is that
the joint state's inputs are dishonestly weighted (A3) and its seeds churn (A2). Fix those before
re-opening any of the three. [carried, re-scoped 08-22]

### P5. #79/A5 — the deep-gate WIDENING (100 → 50): TRIED, and WITHDRAWN
This stood in the 08-18 list as R6, "ready, unchanged, stable for a day. Cheap." It was armed and
then **reverted as unproven and as an active confound** (`3dbf7c718`, 08-21 12:35). The fleet's
9 → 32 lock recovery was the GATHER FIX, not this, and the in-chain controls prove it: PRN 31
q 0.98 → 2.99 and PRN 16 0.89 → 2.64 **recovered without being admitted**. Ordinary prompt-power
presence works once the loop is alive. **Do not re-list this as ready.** If it is re-flown it must
be one variable, on a verified-live loop, with the unadmitted sats as in-poll controls.
[withdrawn; verified 08-22 from the revert commit and its diff to the yaml]

---

## 🅿️ Parked 2026-08-24 — measured, root not chased (KV: "park as a thing to follow-up")

**#90 — OFF-PEAK DISARM LATCH on chains with no search admission path (live case: E32, 2026-08-24 22:17-23:0x UTC).**
The anatomy, fully instrumented in /tmp-era logs + fixtures/expectations_20260824_gap1_gonogo.txt notes:
* E32 (gal_e5a + gal_e5b, dead-reckoned, dop +1600 falling, C/N0 33-38 dB-Hz) held a REAL
  ~0.5-0.9 chip model-vs-sky offset in its standing C++ trim (readback 32:-0.63..-0.91).
* 22:17 (e5a) / 22:25+22:29 (e5b): the SKY peak excursed ~+0.9 chips from the trimmed tap in
  <1 min (SPEC-FIT tau -0.03 -> -0.61; transient, BAND-ALTERNATING, then permanent from
  22:31). The fast loop correctly chased the railed disc (+0.86..+0.91) and drove the
  standing trim to zero — the "wipe" is the loop obeying, not a reset. First event
  recovered (trim rebuilt -0.27 -> -0.78 over 7 min at slew-class rate).
* Second event stuck: q below the trim floor -> fleet presence lost -> the 90 s
  fleet-trim-hold expired -> PRN DISARMED at ~22:32 -> trim released -> LATCHED off-peak.
  Meanwhile SPEC-FIT still measures a peak at tau ~ -0.5 with p/f 1.3-2.4x EVERY MINUTE,
  and per-instance A decays smoothly (95 -> 28-40 over 25 min) on BOTH bands.
* THE DEFECT: on gps_l5, #79's fix re-admits via the narrowed search; the Galileo/BeiDou
  chains HAVE no search (0 detections) — so once disarmed off-peak there is NO re-admission
  path, even with a per-minute spectral fix sitting in SPEC-FIT. The obvious repair: let
  SPEC-FIT (already spec-anchored, already feeding #85) be the admission/re-seed authority
  for searchless chains — one-shot re-seed at model+tau when p/f is significant and q is
  floored. UNBUILT.
* The EXCURSIONS themselves (transient, band-alternating, growing as the sat descends) are
  UNATTRIBUTED: multipath at falling elevation is the leading candidate (band-dependent
  interference phase explains the alternation); ephemeris/clock event not excluded. Judge
  on the next low-elevation repeat, not on this one sat.

**#34 — the FUTURE-HOP emitter: instances serve a hop EXACTLY 100 RECORDS ahead.**
Link 2 of `docs/`-side lock-walkoff chain. Re-measured per INSTANCE 2026-08-24 (the new
`AXIS INST` line, from `_inst_hops_now`): **13 events in 360 reports (3.6% of polls)**,
~0.24/min fleet-wide, two classes.
* *(A) startup, once*: all 12 instances on ALL FIVE chains at **+2.9..+3.1 s together**,
  spread 0.06-0.15 s, 4 s after broker start. Moving together ⇒ a SHARED term. cf06's clock
  is clean (chrony RMS 48 µs, no steps) ⇒ the `utc0_sample0` capture anchor.
* *(B) steady state, 8 events*: a SUBSET of instances (1, 3, 4, 9, 11, 11, 12 of 12) jumps to
  **+0.86..+0.98 s** while the rest hold −0.15 s. One chain at a time. A per-instance subset
  IS a publisher defect, so the original attribution stands for this class.

⚡ **THE LEAD: the class-(B) jump is +1.0486 s relative to the healthy rows, which at
`hops-per-sec: 195312.5` is 204800 hops = EXACTLY 100 RECORDS = 25 FRAMES.** A round number in
the F-engine's own units is a buffer or a ring lap, not jitter. The `WALL-vs-F-ENGINE OFFSET
JUMPED ±1.04 s` alarms (~10 per 10 min) are the same thing seen through `max()`, which is why
they never resolved. **Find what holds 100 records / 25 frames** — NOT the taps ring
(`taps_win` 32 windows = 128 records).

**#35 — ✅ RESOLVED SAME DAY (d85644b7b): links 3-7 root-caused and FIXED — the
cp_predicted epoch mixing.** `predict_all` ran at WALL, the range extrapolation baseline was
AXIS age (`t_abs − t_now_abs`); under `--dr-fengine-axis` (fleet-wide since 08-17) that
misplaces every sat by K·dop·lag chips. eps, the residual ramps, the ±4-chip clock oscillation
and the E8 walkoffs all trace here. Falsifier PASSED: trim-vs-dop slope −1.0e-3 → +5.4e-5
chips/Hz, railed 3.6% → 0.93%, bds_b2b q 1.87 → 2.71. ⚠️ the digest gate cannot see
axis-regime defects (every fixture predates 08-17) — RECORD A NEW ON-SKY FIXTURE. ⚠️ the
joint state's churn is PREDICTED to shrink (its feed was contaminated); not yet measured.
The original re-measurement task below is superseded but kept for the record.

**#35-old — re-measure links 3-7 of the walkoff chain; link 1 is GONE.**
The F-engine axis lag was **−18.0..−19.3 s (gps/e5a) and −7.8 s (b2a) at the same instant** on
08-23. Measured 08-24 per instance, n=360: **median −0.15 s, worst single instance −0.35 s,
chain-dependence ~0.03 s.** ~100× smaller; the chain-dependence that made it diagnostically
interesting is ~300× smaller.

⚠️ **Everything downstream was measured under that lag** — the epoch-frame mixing,
`eps = +420 ± 120 ms`, the GPS per-sat residual ramps (±5-10 chips over hours), the ±4-chip
clock oscillation. None of it can be reasoned about from the old numbers. The named
measurement: are per-sat residuals still walking, and do they still correlate with dop? Needs a
span of HOURS — leave an instrument running, do not soak-and-judge.

⚠️ Also unexplained: WHAT fixed link 1 between 08-23 and 08-24 (the fe_off persistence guard,
the broker cycle speedup, or an F-engine restart). Only some of those are durable.

⚠️ **Measurement trap, cost a wrong population once:** the obvious place for an axis instrument
is beside `_fh = max(pow_hop)`, but that `status` is `{prn: row}` from ONE combiner and every
PRN in a poll carries the SAME hop — it reports "spread 0.00 s across 32 rows" and says nothing
about the fleet. Use `_inst_hops_now`.

## ⚪ Deferred / distant

* **#21 / #26 — Path B (dual-input N² correlator)**: designed in full, injector endpoints exist.
  **Status changed**: `--n2-send` is now armed FLEET-WIDE on both legs (`392f79455`,
  `941206fe7`), after two disarms — the second of which (`dd5096e14`) was the stale-EOP kill of
  recv1, not the correlator. So N² runs alongside GNSS; Path B proper remains parked on chive's
  GPS-time service. [config verified 08-22; live status NOT re-checked]
* **#22 — fp16 Φ into production**: ⚠️ **the 08-17 entry was imprecise and is corrected here.**
  An fp16 Φ path DOES exist — `cudaGnssChordDespread.cu:335` (`phi16 == 6`, hop-sorted + fp16),
  with the comment "BENCH ONLY … the struct field type is a lie the caller opts into …
  **production never takes this path**". So the lever is not missing, it is deliberately not
  taken; the measured 1.3× is the reason to take it and the open question is whether the kernel
  is byte-limited or request-limited, which the bench exists to answer. [verified 08-22 in code]
* **#55 — delete the `carrier_phase_from_ref` A/B scaffolding once #52 settles**: ⚠️ the count
  went the WRONG way. The 08-17 entry said 22 references; today it is **28 in code**
  (`.py/.cpp/.hpp/.cu`) across at least 9 files, plus 72 more in docs/yaml. The scaffolding is
  more entangled than when the item was written, not less. [verified 08-22 by grep]
* **#24** noise-debias the per-element beam-map value; **#31** data channels (L5-I/E5a-I/B2a-D);
  **#36** trackers should take PRN lists from the broker at runtime; **#37** GPS block map vs sky;
  **#38** visibility-matrix export (207 GB / 5 min — a writer, not new correlation); **#25**
  per-subband accumulator (writer landed; the accumulator is the remaining half); **#11** mask
  seeded PRNs by boresight angle rather than elevation > 0; **#14** CUDA acquire declines a valid
  blind Doppler grid. [all carried, none re-verified this pass]
* **#75 — gal/bds have no independent clock**: a gps_l5 outage takes the whole instrument down.
  The structural fix is per-PRN model primacy (#83 Phase 3), so this closes as a consequence
  rather than on its own. ⚠️ **Partly mitigated**: both Galileo chains now consume `clk,rate`
  from the joint state rather than adopting gps_l5's EMA clock, and one clock now spans two
  constellations (`07b116910`). The two BeiDou chains still carry bare `dr-clock-adopt: true`.
  [verified 08-22 from the yaml]
* **cf06 reboots ~weekly at 03:00 and nothing there comes back** — looks exactly like a healthy
  idle fleet. Not fixed, only known. ⚠️ Now compounded by A1: whatever restarts on cf06 must
  include the gather, or the fleet comes back with a wedged code loop and a plausible 200.
  [carried]

---

## 🗑 Retired, and 🚫 withdrawn

**Withdrawn in this pass** — tried, measured, backed out. These are NOT open work:

* **🚫 the deep-gate widening 100 → 50 (#79/A5)** — `3dbf7c718`. Unproven and a confound; the
  in-chain controls recovered without admission. Full reasoning at P5.
* **🚫 GAP 2 feed on a model-primary chain** — `f2524ab04` and five earlier attempts. Not
  "failed and retry": the measurement is 42× understated (A3) and feed+consume is structurally
  circular. Any re-arm needs an honest sigma or an independent anchor first.
* **🚫 `joint-consume: slew` on a FEEDING chain** — `85a185f98`. Closes a loop with no
  observation in it. Note this does NOT retire `slew` on a non-feeding chain, which is A4.
* **🚫 #33 arms 11, 13, 14, 14b, 17** — disarmed on measured verdicts (`d29b12445`, `616430410`,
  `873267111`, `1f6b59aaa`, `5dea10518`).
* **⚫ #33 arms 15, 16, 16b and GAP 2 attempt 3 — VOID, not withdrawn.** They ran across an
  F-engine restart / the cf06 gather wedge with the C++ code loop dead. **A void arm is not
  evidence in either direction**, and treating one as a control is how the e5a/e5b pair stayed
  confounded for a day.
* **⚫ #46's own justification** — retracted in `docs/CHORD_OFFLINE_PLAN.md` before the fix
  shipped anyway. It unblocks at most one of the two things it was sold on, and the 215 µs it
  was supposed to explain was never shown to be this lag (O10).

**Retired earlier and still retired:**

* **B1 — split-aperture "beats the genie"**: closed by #6 (comparator error).
* **P1 — `despread_max_chips`**: closed (#17); the only config still carrying 105 is
  `chord_gnss_cx19_chips105.yaml`, an explicit A/B variant.
* **C1 / P2 — GPU chip gather**: closed (#18) — `chip_gather3` landed bit-exact.
* **C2 — two `test_gnss_channelized_acquire` failures**: closed (#20) — the fine-lag sign is a
  bank property, now explicit, 16/16.
* ~~**The per-PRN Python/C++ trim handover**: retired as a no-op.~~ ⚠️ **UN-RETIRED, and it has
  now paid off twice.** The "no-op" reading was scoped to gps_l5, where the set it serves is
  ~empty. On the dead-reckon chains it was the whole ballgame, and it is the enabling condition
  `eec1d2f12` demanded: `gal_e5b` got its code loop in `0c0a08d85` precisely because the
  handover is per-PRN (`gps_distributed_broker.py:8388`, `if prn not in _ft_armed_last`) so
  `fleet-trim-url` no longer silences the Python integrator chain-wide.

---

## ✅ Closed since the last reconcile (2026-08-06 → 08-17)

With commits, newest first. These are the narrative of the eleven days this file missed.

| # | what | commit |
|---|---|---|
| 79 | **The presence latch**: trim admission was decided by on-peak statistics, so off-peak was permanent. The SEARCH now admits (tap-independent). armed 3→8, disc +0.89→~0, amp 0.0–0.5→4.3–10.4 | `8dbdfd3aa` |
| 80 | The at-ref phase now rides the hold in both arms; the tracker prefers it, so a fresh phase beside a frozen anchor commanded the despread off-peak | `a9da911b3` |
| 83 P1 | **The seed object**: typed `Seed` with (owner, epoch) provenance; the audit's §2 writer table became data; EPOCH-SKEW + owner-trail instruments | `09dad7486`, `78a170763`, `f69d75c63`, `68795d1db` |
| 81 | An ABSENT merge feed cost a full timeout every frame — 0.067 → 24.12 fps, search epoch lag 48.9 s → 0.01 s | `61ae6be3d` |
| 41 | NH label flicker: overlay-mismatch Doppler sidebands land on the acquire's grid while the true peak scallops. Fixed by scoring alignments on a monotone mainlobe pair-sum | `203844862`, `4554fc6d1`, `56ce712bd` |
| 72 | REC_PHI0 publishes the comb's phase currency — cross-instance coherence 0.30 → 0.87–0.94 | — |
| 64 | Telemetry frame 62832 → 25200 B; broker cycle 16.5 → 13.0 s | — |
| 60/69 | The node "wedge" was a FALSE POSITIVE — a sampled FPGA-seq check compared different boards | `caba685a8` |
| 58 | The deep fold's 5 Hz rate cap was the failure mode — duty 61% → 88% | — |
| 59 | Frame-synced transport at v3 — E/P/L per channel, per record, absolute-window keyed | — |
| 53 | Aligned spectrum gather live — root was `n_reanchor` counting folded re-pins | — |
| 51 | The C++ fleet code loop tracks — confirmed per-record at 95.4 Hz | — |
| 47/66/10 | Served C/N0 is blind to code error; `deep_snr`'s blindness and its 20 dB paired scatter are one fault | `58ec8aed6` |
| 42/43/44/45 | The transport-hardening set: at-epoch `cp_err`, `propagate_seed` reference audit, coast Doppler re-tag, all 7 transport steps | `a780c9914`, `f0b24e0ea`, `4b44497d8`, `7cb011c50` |
| 39 | Search cp clean, guard restored — led to the median-churn root | — |
| 35 | The C/N0 scatter was two PAIRING bugs, not a physical mechanism | — |
| 34/30/28/29/23 | E5b tracking; E5a code-rate drift; dead-reckon clock warm start; BRDC pin; multiband descriptors | — |

Also landed as infrastructure: `docs/CHORD_CONTROL_AUDIT.md` (the four-part structural audit
that produced #76–#80), and a fourth equivalence-gate fixture — `broker_onsky_l5_holds`,
the first with **hold-path coverage**, selftested `GATE GOOD` and blessed at `928f8b8d8`.

---

## ✅ Closed 08-18 → 08-22 — the index row for the shipped section above

| # | what | commit | verified how, 08-22 |
|---|---|---|---|
| 46 (serve) | node serves `serve_unix` / `ingest_utc` / `ingest_unix` beside `utc`/`pow_hop` | `75445d458` | read `GnssCoherentCombiner.cpp:2525-2530`, `:485-486`, `.hpp:405-406` |
| 8 | ADC clip fraction + per-band power: node, broker, viewer, and a fixed-cadence host poller | `b8ca0f4f7`, `958597061`, `295f1d0dd`, `392f79455`, `f13d57cfc` | `GnssChordVoltageTap.cpp:37-56`; `rf_monitor_channels()` at `gen_chord_gnss_config.py:163`; `scripts/gnss/rail_watch.py` present |
| 70 | instance liveness guard — a 200 is not a live instance | `6e48d0c36` | `gps_distributed_broker.py:8474`; `instance_stall_verdict` in `fits.py`; `test_instance_stall.py` |
| 49 / R7 | fast C++ code loop armed on gal_e5b | `0c0a08d85` | yaml :1580-1582 |
| 86 | `sigma_rate0` defaulted to `rate_max` — a prior and a garbage ceiling shared a number | `5e3e9ab5d` | `state_filter.py:175` (0.05) vs `:179` (`rate_max=1.0`), rationale `:193-210` |
| 89 (flags) | 72 broker flags retired into `_FROZEN`; 278 → 206 | `63c6f20a6` | `_FROZEN` at `:227`, applied `:2579`; `add_argument` count 207 = 206 + `--dr-clock-wait-s` |
| 68 | the EOP table rolls; the generator now pulls it from the fleet, not the capture | `e39166a06` | generator change in tree; cx19's warning counter froze at 17,286,844 (measured 08-19) |
| — | **the unobservable tau row**: a band with ZERO dual-band satellites got a tau row, making tau exactly degenerate with clk | `9193c7c54` | `state_filter._add_band` at `:510`, refusal at `:546-553`, `tau_min_dual` default 1 at `:186`; `receiver.py:118-128` no longer latches `ref_band` by thread order; `python/scripts/gnss/test_tau_observability.py` present (⚠️ NOT run — no numpy on this host) |
| — | **seeds born before the clock bootstraps**: the first re-birth after BOOTSTRAP stepped the whole fleet by exactly one clock | `76cebb3eb` | `--dr-clock-wait-s` default 30.0 at `gps_distributed_broker.py:1361`, guard at `:5865-5876` |
| — | warm ElemCal held across a carrier re-anchor | `56db6c38a`, `2122d83d2`, `13af9b5a9` | `GnssGpuRecordAssemble.cpp:88,241,354,427,476` |
| — | `/set_reference_element` — swap the header reference feed live | `1ceb6f42e` | `GnssGpuRecordAssemble.cpp:189-192` |
| — | JFEED-TERMS printed a DIFFERENT variable than the code under test | `90594deb4` | commit read; *a diagnostic that does not read the code under test is a second bug, not evidence* |

**THE TWO NEW FAULTS WORTH READING IN FULL, because both were invisible and both had a cascade:**

1. **The tau row you cannot observe destroys the clock.** Four broker runs on 08-21, identical
   code and config, split two ways purely by which chain constructed the state first:
   `ref_band` on the measurement band → σ(clk) 0.11–0.30, converged; `ref_band` on the OTHER band
   → **σ(clk) 19.901 forever, b_sat to −1693 chips**. tau separates from clk ONLY through
   satellites seen in both bands, and no satellite ever is — so the row was unobservable BY
   CONSTRUCTION. The cascade, all four steps measured: tau degenerate with clk ⇒ σ(clk) pinned at
   σ_tau0 through 116 accepted updates ⇒ **P00 = 396, and `birth_max` is conditioned on
   `P00 < 100`, a gate whose comment says it is "open during bootstrap BY DESIGN" — but P00 never
   fell, so it was open FOREVER** ⇒ unvetted births, and `gauge_max_b=60` excludes a wild bias
   from the restoring gauge, so biases reached −1693 chips ⇒ clk and the biases traded the
   degenerate common mode (one birth stepped clk −180 while every bias jumped +160).
   ⚠️ **This is why the same arm was immaculate at 22:16 and catastrophic at 23:40.** It was read
   as slow feedback wind-up. It was a race at broker start. Two rules fall out: *do not estimate
   what you cannot observe*, and *a gate whose enabling condition can be held false forever by an
   unrelated fault is not a gate.* Also banked: **we have never measured a nonzero tau** — every
   value ever logged is +1.44 to +1.47 ± 19.901 chips, 0.07σ from zero — and the physics agrees
   (one ADC, one PFB, so 1176.45 and 1207.14 MHz are different BINS OF THE SAME FFT; the real
   differential effect is ionospheric, hence per-ray, hence already carried by `b_sat`).
2. **Do not seed on a guessed clock.** A seed born before the receiver clock bootstraps carries
   no clock, so the first re-birth after BOOTSTRAP anchors `cp0` on `model + _off` and jumps by
   exactly the clock — and because `_off` is fleet-common, **every satellite jumps by the same
   amount in the same cycle**: E5 +148.35 / E27 +148.63 / E38 +147.42 against `_off` +148.353,
   40 events, step/off = 0.9958, **all within 57 s of broker start and none after 60 s**. This is
   the ~150-chip class of the seed churn at A2, and it is fixed; the ordinary 5–30 chip churn is
   not.

---

## Method note for the next reconcile

The coverage census added in #83 P1c prints, at the end of every transcript replay, exactly
which seed writers that fixture exercised. Use it: the Aug-8 fixtures drive 8 of 18 writers and
**no** hold path, which is why a "gated" refactor could still have broken holds. A gate vouches
for what its fixture runs and nothing else — and now it says so out loud.

**Added 2026-08-22, from what this reconcile cost.** Four days made the previous ranking wrong in
its *organizing principle*, not just in its contents — it ranked by "the node restart is scarce",
and by 08-22 the node queue was empty. So the first question at the next reconcile is not "what
changed" but **"is the axis this list is sorted on still the binding constraint?"** Three more
habits this pass had to correct for, all of them cheap to avoid and expensive to inherit:

* **Delete stale measurements; do not date them and move on.** The 08-17 prompt-health table read
  as current for four days. A table with a date on it still gets read as the answer.
* **Verify the tree, not the summary.** Two claims handed to this pass did not survive: "GAP 3
  not started" (it is implemented and pinned at 0.0) and "the drift ratio went 12.8× → 1.0×" (no
  post-fix number exists in the repo, and the 12.8× attribution is recorded as confounded). Both
  were honest summaries; neither was in the tree.
* **Say which check you ran.** `[verified]` with no named check is indistinguishable from
  `[carried]` a week later. Every mark in this pass names its grep, its file:line, or its commit —
  and where the check was impossible (no fleet, no numpy), the entry says so instead.

## 📏 Measured 2026-08-22 after the twinning, not yet reflected above

The buglist above records the 12.8× e5a/e5b drift ratio as **measured but confounded** — the
pair also differed on `dr-forecast-lead-s`, `rrate-coarse-deweight`, `rrate-command` and
`rrate-phase-feed`. It was right to flag it. Here is the follow-up it was waiting for.

**The chains were twinned on every non-endpoint flag** (verified by resolving both through
`broker_multi.load` — identical flag sets), and the model-vs-tracker drift re-measured:

| | before | after |
|---|---|---|
| gal_e5a median \|drift\| | 0.0055 chips/s | 0.0033 |
| gal_e5b median \|drift\| | 0.0706 chips/s | 0.0033 |
| satellites over the slew cap | 9 of 36 | **0 of 36** |
| **paired e5a/e5b ratio** | **12.8×** | **1.0×** |

⚠️ **THE ABSOLUTE LEVELS ARE STILL CONFOUNDED AND THE RATIO IS NOT.** The BeiDou chains
improved over the same window (b2a 0.0084 → 0.0023, b2b 0.0025 → 0.0015) **with no config
change at all**, so a fleet-wide common improvement of order 2–4× is in the data and is not
ours. A common factor cannot change a within-pair ratio, so 12.8× → 1.0× is attributable to
the twinning; the 21× headline on e5b is not.

⚠️ AND IT KILLED A PREDICTION. E13/E15/E23 were fast on BOTH chains, which was read as
per-satellite and therefore expected to SURVIVE a chain-level fix. E23 went 0.0920/0.0884 →
0.0043/0.0033 — it improved on e5a too, which a change to e5b's rate source cannot explain.
So that drift was time-varying, not satellite-intrinsic, and the ephemeris check it motivated
is no longer motivated by evidence.

**Still open from this measurement:** the fleet-wide common improvement is unexplained. If the
BeiDou chains drift back up while the Galileo pair stays low, that separates the two cleanly.
