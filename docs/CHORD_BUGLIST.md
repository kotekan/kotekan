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

Last reconciled: **2026-08-18 (offline review)**, against session tasks #6–#88.
Previous reconcile 2026-08-17 ~00:45 against #6–#83.
Previous reconcile was 2026-08-06 against #6–#20 — eleven days and sixty items stale, which is
why this pass rewrote rather than appended.

**How to read the status marks.** `[verified 08-17]` = checked against the tree or the live
fleet during this pass. `[carried]` = believed true, last checked when the entry was written,
NOT re-verified — treat as a claim, not a fact.

---

## State of the instrument, measured during this pass (2026-08-17 00:40)

Per-chain prompt health, one poll of `/<chain>/get_status`:

| chain | sats | amp ≥ 5 | median amp | median deep |
|---|---|---|---|---|
| **gps_l5** | 15 | **6** | 1.5 | 15.3 |
| gal_e5a | 12 | 0 | 0.4 | 3.0 |
| gal_e5b | 12 | 2 | 0.4 | 3.2 |
| bds_b2a | 12 | 2 | 0.3 | 3.2 |
| bds_b2b | 12 | 0 | 0.0 | 2.8 |

**One chain works and four do not**, and the gap opened tonight: gps_l5 is the only chain with
the #79 fix armed. The deep-fold detection bar is ~3× a floor of ~2.7 (so ~8.1); four chains sit
at a median deep of ~3, i.e. **the majority of their satellites are not detected even by the
deep fold**. That is the single biggest open fact about this instrument and it frames most of
the list below.

⚠️ **And the obvious fix does not transfer.** #79 admits satellites to correction on SEARCH
detection, and **only gps_l5 has a search** (`detectors:` is set on that chain alone;
config:210). The other four are dead-reckon only. Whatever fixes them is NOT "arm #79 there".

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

## 🔴 Active — the next levers, ranked  [rewritten 2026-08-18 offline review]

State: HEAD `15b5e1bb6`, tree clean. **Nodes are POWERED DOWN** (site work); broker up on cf06
carrying the q stall guard. The C++ binary (`build/kotekan/kotekan`, 18 Aug 02:27) is current
with HEAD and is AHEAD of every node that was running — so the next restart deploys the DPDK
resync fix as well as whatever else lands in the binary before then.

**THE ORGANIZING PRINCIPLE FOR THIS RANKING: the node restart is the scarce resource.**
It needs sudo, it costs an observing window, and one is already owed (`n2-send`, the fleet-wide
`resync_max_advances: 32`). Anything node-side that is not in the binary when that restart
happens waits for the one after. So work that is (a) buildable offline and (b) node-side ranks
above work that is merely important, and the offline window is exactly the time to write it.

### R1. #46 — SERVE A CLOCK, don't infer one  [node-side, build offline, rides the restart]
Verified in code this pass: `GnssCoherentCombiner::get_status_callback` serves `pow_hop` and
`utc` **from the last completed window and carries no serve-time timestamp at all**
(`GnssCoherentCombiner.cpp:2389,2497`). So one number is doing two jobs, and the broker has to
infer record-staleness and serve-latency from it — which is precisely why the axis is
drift-free but stale, right for LABELS and wrong for EPOCHS.
The minimum fix is **additive JSON fields in a REST callback**: a serve-time wall clock, and
the newest hop the node has INGESTED beside the newest it has PROCESSED. That converts the
inference into a measurement, and it is cheap and low-risk (no wire shape, no pipeline change).
**Why it ranks first: it unblocks two things already retracted for want of it** — the ephemeris
epoch on the F-engine axis (measured 65x worse, reverted) and `--innov-dr-seeds` (falsified in
20 min by the lag's sub-ms residue). Both are real fixes waiting on one number.
⚠️ The full version — the CURRENT F-engine hop from the capture stage — is more plumbing than
the additive one. Do the additive version first and measure with it before designing the rest.

### R2. #8 — the clip-fraction + band-power monitor  [node-side, build offline, same restart]
Unchanged in substance and now more urgent, because #56 has stopped being a curiosity: the
08-18 1176 MHz event proves a **band-selective external source exists**, and the chronic 5-10x
swings contaminate every amplitude judgement we make — they already produced one false alarm
about THE FLIP, cleared only by pairing in time. We are blind to the very swing we keep
tripping over. 4+4b clip fraction + a low-cadence band-power log is the instrument.
Pairing it into the same restart as R1 is the efficient shape: two node-side items, one window.

### R3. #70 — the instance liveness guard  [broker-side, NO restart, deploy the moment nodes return]
We now know the exact signature, measured twice: **a REST endpoint answering 200 is not a live
instance.** cx42/port 0 sat with its window frozen and the entire 195,313 pkt/s stream dropped
while serving plausible rows; `pow_hop` advanced 0 hops in 30 s against a healthy ~5.9M.
The guard is a poll-twice-and-diff in the broker — cheap, Python, no restart, and it would have
caught the 25 h cx19 wedge, the four wedges on the 14:05 restart, and #87's frozen instance.
This closes the open half of #87.

### R4. #88 — finish the j2 restructure  [pure offline; Jim is reviewing right now]
Delete the block-building in `build_n2dual_branch` (the template already renders 137/137 on
every node, so gate 2 is what proves the deletion safe), then move deployment off the captured
base. The second half is what actually stops us drifting from upstream. Doing it while Jim
reviews is free parallelism, and it makes his review of the structure a review of the thing we
will actually deploy.

---

**Then, when the nodes return — sky-gated, in this order:**

### R5. #86 — harvest RATE-TEACH, first thing, before anything else runs
Costs nothing: `grep RATE-TEACH /tmp/gnss_broker.log`. The 14:51 broker restart was itself a
~2/3 trigger and the forensics were armed before it, so the log may already name the teacher.
Three offline reproductions all HEAL, so the mechanism is live-only and this log is the only
instrument that will ever see it. **Do NOT restart-loop to hunt it.** #33 is blocked here.

### R6. #79/A5 — widen the search bar one notch, 100 -> 50
Ready, unchanged, stable for a day. Unadmitted sats are the in-poll controls. Cheap.

### R7. #49/A3 — arm the fast code loop on gal_e5b
⚠️ **Status corrected this pass: the prerequisite is SATISFIED.** eec1d2f12 made any re-arm
conditional on a per-PRN trim handover, and that handover is implemented and live —
`gps_distributed_broker.py:7887`, `if prn not in _ft_armed_last`: Python keeps acquisition
authority for PRNs the C++ loop is not touching and stands down per-PRN as it takes them.
It is not a flag, which is why searching for one finds nothing. So this is one yaml line plus
a broker restart, and the verdict it confirms is already paid for (0.81 vs 0.42 duty, paired).

### R8. #85/#50 — spec_tau against the model-held offsets
#84's fix revived the measurement (BSAT accepts, n up to 51). Correlate SPEC-FIT tau against
the model-held offsets on the SAME sats before feeding anything. ⚠️ spec_tau is PLAUSIBLE, not
STRONG (p/f median ~1.0) — #50 is blocked on significance, not on plumbing, and `reseed-spec-tau`
has a RESEED count of 0 for the broker's entire life.

### R9. #33 GAP 1 — arm the carrier command on one chain  [BLOCKER CLEARED 2026-08-20]
`carrier_correction_hz()` is implemented and consumed at one site and armed by nobody
(`carrier-gain: 0.0` on all five, `rrate-command` off everywhere). gal_e5a is the canary.
**2026-08-20 03:1x: the 398f31de5 payoff blocker is CLEARED, two-instrument verdict.**
(1) e2e [4e] (eb38a418b): the full offline arithmetic — shipped propagate + shipped despread
kernel at live t_abs + assembler-verbatim dcyc/f_nco fold — is CLEAN under constant ctrim, a
live-like command staircase, a 4x-aggressive staircase, and noise; only the known ~mcyc-scale
step-pairing drip (f_nco*dt uses the new ctrim over the old gap) remains. (2) The live payoff
re-measurement on the WARM plant (fixtures/expectations_20260820_gap1_payoff_remeasure.txt,
pre-registered, /set_carrier_trim +2 Hz hold + ±2 Hz toggle stream, b2a + e5b same-poll
controls): the commanded chain sat INSIDE its controls at every poll on both convicting
statistics, and PRN 28 — the 08-14 66x case — ran coh 0.72-0.83 while commanded. The 08-14
negative was a property of the 08-14 plant (never-warm ElemCal: records rode the single
weakest feed), not of commanding.
RE-ARM: one yaml line (`rrate-command: true` on gal_e5a) + broker restart. Acceptance gate =
fixtures/payoff_split.py, same controls, plus the frozen-command feedback discriminator.
Judge on KCOH duty and `rate_resid_hz`, against the unarmed chains IN THE SAME POLLS — never on
sig. This is also where #71's carrier NCO gets re-judged: it failed 3x as a bolt-on because the
seed is the reference and is re-pinned every window, and one controller owning seed and carrier
is the entire point.
BEHIND IT: GAP 2 (three chains consume nothing from the joint state) and GAP 3 (code and
carrier are not coupled — the unwritten physics). #86 RATE-TEACH: harvested clean 08-20
(0 events, live + archived logs) — no longer blocking.

---

## 🟠 Open — real, with a known lever

### O1. #82 — wrap-edge label partners (residue of #41)
When the true delay sits within one code period of the circular window's edge (~12.5% of
snapshots), the adjacent alignment's wrapped peak carries ~15/16 of the power AT THE TRUE
DOPPLER, so no comparison statistic can suppress it, and the fold arithmetic that makes
unwrapped partners label-invariant breaks. Rate ~2–3/hour at any SNR. Fix belongs in fold-aware
label COMPOSITION, not the argmax. **Decoding clue found 08-17**: PHCONT reports `m % LC_SEG`
(B:4236), so every report lands in [0,20) — "+19" is −1, "+17" is −3. The large-k near-exact
events are small NEGATIVE flips wrapped by the report, which also likely explains the
±2-derived vs ±1-observed discrepancy. Resolve that first. [verified 08-17 in code]

### O2. #40 — the deep fold's rate search wrong-bins onto overlay sidebands
Same disease as #41, one stage down: a maximum over a nuisance parameter (rate) on a grid where
mismatch sidebands can land on-bin while the true peak scallops. #41's fix pattern (score each
hypothesis at its continuous max, via a monotone mainlobe pair-sum) should transfer. [carried]

### O3. #77 — re-anchors pop the trim in Python only
The three hard re-anchors (ESCAPE, watchdog, DR re-seed) `pop` `dll_trim`, but there is no
channel to the C++ accumulator: `set_policy` has no field for it. On a re-anchor the C++ trim
re-applies on top of a phase that does not include it, bleeding out only through the ~8 s leak.
Lever: add a `reset_trim` list to the policy POST. [carried, audit 08-16]

### O4. #78 — the fleet DLL's steady state exceeds its own clamp
Live: per-update gain 0.1048, leak 0.005033 → railed-disc steady state 5.2 chips against a 3.0
clamp, because `_fleet_trim_nominal_hz = 23.84` is defined and never read and the shipped
`--fleet-trim-leak-per-s 0.12` is ~10× weaker than intended. ⚠️ **Re-scoped 08-17: this is NOT
the free one-liner the audit implied.** Restoring the intended leak cuts the railed steady state
to ~0.52 chips but cuts DC authority equally — holding a real 1-chip standing offset would then
require a near-railed discriminator. It is a bandwidth/authority trade and needs a measured
pull-in test, not a constant restoration. [verified 08-17 against live policy]

### O5. #50 — `spec_tau` far-regime re-seed: implemented, gated, NEVER ARMED
The one code-error estimator that works where the discriminator cannot (beyond DLL capture
range), bounded (0.75 chips, gain 0.5), and gated on `spec_peak_ratio` — a shuffled-null
significance, i.e. a gate that can fail. `reseed-spec-tau` is commented out on every chain and
the RESEED log count is **0 for the broker's entire life**. Urgency on gps_l5 dropped now that
#79 admits satellites to the DLL, but it is the leading candidate for A2 (the DR chains).
[verified 08-17]

### O6. #56 — fleet-wide signal strength swings ~5× per hour at the INSTANCES
Upstream of tracking and of the combine; still unexplained. Blocked on **#8** (no 4+4b
clip-fraction monitor and no low-cadence band-power log — we are blind to the very swing).
Do #8 first. [carried]

### O7. #63 — the purge: the deep fold is what still blocks deletion
3 of 4 done; the comb DLL is live on gps_l5. The deep fold's re-searching estimator is what
remains, and it is entangled with #47 (served C/N0 blind to code error) and #57. [carried]

### O8. #67/#68/#65/#70 — the observability quartet
#67 broker's in-process telemetry reader delivers ~25% of the stream (GIL convoy);
#68 node log flooded by `/telescope` EOP warnings at frame rate — drowns real messages and has
already blocked one diagnosis; #65 `agg_up.sh` truncates its log instead of rotating (destroyed
burst evidence twice, most recently 08-16) **and** the search's refine time swings 600× at
constant alignment count, unexplained; #70 no generic stall guard (a stage producing nothing
while its input advances should say so — would have caught #60, #69, and the chain death).
Individually small, collectively the reason several diagnoses cost hours. [carried]

### O9. #62 — element cal applied NON-CAUSALLY
Weights are updated with the very record they weight, for both the comb and `/get_spectrum`.
[carried]

### O10. #46 — instance record-time divergence (0.15 s, 5 lag levels)
No buffering in the design, so the spread should not exist. [carried]

### O11. #73 — the gather's full per-element × per-freq × EPLH product is 0.98 GB/s
Against a 1 Gbps link. A budget problem to solve before the full product can ship. [carried]

### O12. #54 — replicas differ PER-SAMPLE at CHORD scale (9.5%)
Long-standing, marked START HERE at the time and never resolved; `C_P` was referenced but did
not fix it. Worth re-judging whether it still reproduces. [carried, stale — re-verify]

---

## 🟡 Premise changed — re-measure before acting

### R1. #48 — SPLIT, and half of it is now explained
*"Why does the prompt tap land on noise for 25–45 min"* — **that is the presence latch**, root-
caused and fixed on gps_l5 tonight (#79): prompt power is suppressed exactly when the tap is
off-peak, so an off-peak satellite failed presence, was never trimmed, and stayed off-peak. The
duration was the hold duration. Prompt amps on admitted satellites went 0.0–0.5 → 4.3–10.4.
*"Why is b2b blind at all times"* — **still open** and still true (bds_b2b median amp 0.0
tonight); it belongs to A2, not to the latch. [verified 08-17]

### R2. #61 — "the rate is real, but only ON-PEAK"
The premise was that the fold sat downstream of the code latch. The latch is now broken on
gps_l5, so this wants a straight re-measure rather than more theory. [verified 08-17 that the
premise changed]

### R3. #49 — the deep-gate rollout
On gps_l5 its mechanism is superseded: admission is now auto-generated from the search rather
than hand-listed. On the four DR chains the hand list is still the ONLY route, so #49 survives
there — but re-scoped as "the interim measure until A2 lands", not as the plan. [verified 08-17]

### R4. #33 / #52 / #57 — the joint-tracking and carrier cluster
All three were written before tonight's control-path work and all three depend on the seed
layer that #83 is rebuilding. #33's rrate state is built but unconsumed and uncoupled to
`b_sat`; #52's remaining 0.27 Hz has no confirmed mechanism; #57's coherent-fold η is answered
but the fix (shorten the fold / fix the injected rate) is not shipped. Re-read them against the
seed object before picking one up. [carried]

---

## ⚪ Deferred / distant

* **#21 / #26 — Path B (dual-input N² correlator)**: designed in full (plan in
  `docs/gnss_gpu_search.md` §11 and the session plan file), injector endpoints exist and are
  seeded on cx19. Not the critical path while one chain in five is tracking. [carried]
* **#22 — fp16 Φ into production**: the only synthesis lever that measured (1.3×). Not present
  in the tracker kernel today. [verified 08-17: no fp16/half2 in the fused kernel]
* **#24** noise-debias the per-element beam-map value; **#31** data channels (L5-I/E5a-I/B2a-D);
  **#36** trackers should take PRN lists from the broker at runtime; **#37** GPS block map vs
  sky; **#38** visibility-matrix export (207 GB / 5 min — a writer, not new correlation);
  **#25** per-subband accumulator (the writer landed; the accumulator is the remaining half);
  **#11** mask seeded PRNs by boresight angle rather than elevation > 0; **#14** CUDA acquire
  declines a valid blind Doppler grid; **#55** delete the `carrier_phase_from_ref` A/B
  scaffolding once #52 settles (22 references still in tree). [mixed; #22/#55 verified 08-17]
* **#75 — gal/bds have no independent clock**: a gps_l5 outage takes the whole instrument down.
  The structural fix is per-PRN model primacy (#83 Phase 3), so this closes as a consequence
  rather than on its own. [carried]
* **cf06 reboots ~weekly at 03:00 and nothing there comes back** — looks exactly like a healthy
  idle fleet. Not fixed, only known. [carried]

---

## 🗑 Retired in this pass

* **B1 — split-aperture "beats the genie"**: closed by #6 (the genie anomaly was a comparator
  error; sky_deep re-judged and enabled). The 2026-08-06 diagnosis stands as narrative.
* **P1 — `despread_max_chips`**: closed (#17). Verified 08-17 that the only config still
  carrying 105 is `chord_gnss_cx19_chips105.yaml`, an explicit A/B variant, not a regression.
* **C1 / P2 — GPU chip gather**: closed (#18) — `chip_gather3` landed bit-exact, kept for
  consistency rather than speed.
* **C2 — two `test_gnss_channelized_acquire` failures**: closed (#20) — not pre-existing; the
  fine-lag sign is a bank property, now explicit, 16/16.
* ~~**The per-PRN Python/C++ trim handover**: retired as a no-op.~~ ⚠️ **UN-RETIRED 01:15 —
  that conclusion was scoped to gps_l5 and does not generalise.** It is true that on gps_l5
  the set it would serve is ~empty (armed ⊇ present, and the search hands presence to the C++
  side). But on the four DEAD-RECKON chains the handover is the whole ballgame: `eec1d2f12`
  disarmed #49 there precisely because setting `fleet-trim-url` silences the Python integrator
  chain-wide while the C++ loop trims only already-present PRNs — measured result, **0 armed on
  three chains, 1 on the fourth, and no code loop at all**. That commit's own condition for any
  re-arm is this handover. Implemented 01:15; see A5. My "no-op" reading came from testing the
  idea only against the chain where it happens to be inert.

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

## Method note for the next reconcile

The coverage census added in #83 P1c prints, at the end of every transcript replay, exactly
which seed writers that fixture exercised. Use it: the Aug-8 fixtures drive 8 of 18 writers and
**no** hold path, which is why a "gated" refactor could still have broken holds. A gate vouches
for what its fixture runs and nothing else — and now it says so out loud.
