# CHORD GNSS — open items

What is open on the CHORD side (branch `kv/chord-gnss`). Closed items are in
[`CHORD_BUGLIST_CLOSED.md`](CHORD_BUGLIST_CLOSED.md); their narratives are in git history.

**Last reconciled: 2026-10-13 after the dead-code merge** (previous passes 2026-10-11 at `0011cf3b1`, 2026-09-12 at
`b45949144`). The 10-13 pass re-checked every open entry's anchor against the tree after `d505f3943`: all still
present except the 09-14 mitigations of #131 (gone with the launchers) and the generator's profiling switch #114 relied
on. Closed since 10-11: **#55, #62, #63, #113, #115, #121, #125, #144, #146 residual, #150, #159, #161, #172, #173,
#174**, the #1750 latent lock (PR #1763), and the 10-11 batch (#31, #34, #88, #102, #107 residual, #112, #123,
#130/#153, #133, #149, #160, #162). Opened: **#175–#189**. The same evening the ready-now set ran in two reviewed lanes and merged: **#186, #183** closed; **#176, #175** have their guards built (live at the next broker restart); **#119, #152** are instrumented; **#94**'s gauge test is in; **#131**'s shutdown race is fixed and its death not reproduced. The 10-11 and 10-13 memory notes record every check.

**Fleet state (10-13): the F-engine is OFF** (since 10-12 17:05Z, for about a week); the six nodes are down, their hosts
up; the broker, obs writers and viewer run the merged tree on the gnss VM, the gather and aggregator the 10-12 binaries.
The last sky ran on `d30be5f66` (telemetry v8, the L5 code fix). The dead-code removal and the j2 deletion are merged and
pushed; `fleet-d505f3943` and `gather-d505f3943` are built and not staged. The F-engine last ran the 10-07 gain table
(#167) and did not send rows C/D (#166). `[live]` marks dated before 10-10 sit across a re-base and a binary change;
treat them as `[carried]` until re-measured.

**How to read the marks.** `[tree]` = checked against the working tree, with the check named.
`[live]` = measured on the running fleet, with the date and the number. `[bench]` / `[archive]` =
measured offline, which still works with the fleet down. `[carried]` = believed, not re-checked —
treat as a claim. Every entry names its check or says it has none.

**This file holds only what is outstanding.** An item LEAVES it the moment it closes — it is not
ticked off in place — and a partial result leaves too: the finding moves to
`CHORD_BUGLIST_CLOSED.md` and what stays here is the remaining question, in as few lines as it
takes to state. A list where closed and open items sit side by side stops answering "what is
left", which is the only question it exists to answer.

---

## The axis this list is sorted on, and why it changed

This list has now been ranked three times on whichever resource was scarce — node restarts, then
unconfounded sky time, then analyst time — and each of those constraints evaporated before the
ranking did. The 2026-09-10 reconcile found the real pattern:

**Nine of thirteen entries in the old Active section were written from numbers that a later
discovered *pipeline* fault had generated.** No amount of experimental hygiene catches that,
because both arms of any A/B carry the same defect. Every one of them was eventually found the
same way: **regress a per-(satellite, arc) rate against a physical variable. A physical cause
scales with a physical variable — Doppler, Doppler rate, elevation. A pipeline fault scales with
a COMMAND — a trim, a forecast lead, a configured cap.**

So the first question about any entry here is *is the observable trustworthy*, and the list is
sorted on the only thing that gates the survivors: **where the fix lives.** A fix in the broker
ships today; one in the node binary waits for a cycle; one in the wire format needs a version
bump; one on a bench needs no deployment at all.

---

## The instrument as configured, 2026-09-10 (pair columns updated 2026-10-11)

Resolved through the broker's own loader (`scripts/gnss/broker_multi.load` on
`config/gnss_chains_chord.yaml`) rather than by reading the yaml by eye — the 08-22 pass recorded
that check as not run, and doing it is what found the two errors below. **Eight chains, not five.**

| chain | search | C++ code loop | joint | deep gate | carrier command | notable |
|---|---|---|---|---|---|---|
| gps_l5 | **yes** (only chain) | yes | `rate,slew` | hand list `4 9 27` | — | `model-primacy-max: 32` |
| gal_e5a | no | yes | `clk,rate` | `33` | **on**, cap 10 Hz | the formal pair with e5b (10-11): `rrate-phase-feed` on both, neither feeds the joint state |
| gal_e5b | no | yes | `clk,rate` | `33` | **on**, cap 10 Hz | twin of e5a |
| gal_e6 | no | yes | `clk,rate` | — | — | |
| bds_b2a | no | yes | `clk,rate` | — | **on**, cap 10 Hz (matched 10-11) | |
| bds_b2b | no | yes | `clk,rate` | — | — | twin of b3i |
| bds_b3i | no | yes | `clk,rate` | — | — | twin of b2b |
| gps_l2c | no | yes | `clk,rate` | — | — | `dll-spacing 0.4` (lobe fix) |

**All eight chains now hold a closed C++ code loop and consume the joint state.** The old
three-tier split ("three chains have a loop, two have nothing") is dead. Only gps_l5 has a
search, which is still the frame: every `search admits` mechanism is gps_l5-only, and the other
seven are dead-reckon.

**The gal_e5a/gal_e5b pair is the formal comparison pair since 2026-10-11 (#125):** matched on every non-endpoint
key, so any A/B on it changes exactly one. bds_b2b / bds_b3i remain a second twinned pair (same constellation, same
PRNs, `joint-consume` and nothing else). The joint state is GPS-fed only again.

**State at 2026-10-12 17:05Z, when the F-engine went off for about a week.** Nodes, gather and aggregator run d30be5f66
(telemetry v8, #158; the L5 code fix, #177, verified on the sky); the broker runs the fleet-ADR silence gate (#155
candidate 2) and the projection fields since 13:26Z. Our bad-inputs cron is held for the outage (#164). At the return:
the broker re-anchors to the new F-engine epoch before the nodes are cycled, the cron is restored once bffs posts a
list from valid data, the aggregator's L5 search is judged (#177), and the local commits on kv/chord-gnss are pushed
on KV's word. The memory note `chord-handoff-20261012-fengine-off` has the rollback binaries and the build dirs to keep.

**10-13: the dead-code removal is merged (`d505f3943`, 339 files, −39k lines).** The gather, aggregator and cube-archiver
configs are regenerated; the six node configs, the unit reinstall and the broker/obs/viewer restarts are KV's to run
(auto mode refuses them); the node binary `fleet-d505f3943` joins the canary at the return, with `gather-d505f3943` for
the aggregator and gather. #184–#187 below are what the removal left for later. 221 local commits, unpushed.

---

## Open — the fix is in the BROKER or a script (ships today, no node cycle)

### #184 — four finished A/B arms stay in the broker because retiring them moves the replay digests (2026-10-13)
**What [tree]:** the removal pass retired every finished A/B except the ones whose live arm changes the `broker_equiv`
digest of the seven recorded replays: `--n2-combiners` (the path-B population), `--seed-bias-source` ema/slow with
`--seed-bias-alpha`, state_filter's median gauge, and fits' optional `fft_len` precision argument. Each keeps a dead arm
and its flag. The replays predate the L5 code fix (#177) and telemetry v8, so re-blessing their digests under the new
arms would bless a run nobody can reproduce.
**Fix:** once the F-engine is back, record fresh replays from the live chains, retire the four arms and re-bless the
digests in one commit. Check: `broker_equiv` identical on the new replays before and after the retire.

### #185 — the #98 relative integrity veto never ran: `_FROZEN` pinned `integ-veto-baseline-s` to 0 over the yaml's 600 (2026-10-13)
**What [tree]:** the chains yaml asked for the relative veto (`integ-veto-baseline-s: 600`, #98) but the broker's
`_FROZEN` table overrode it to 0 on every chain, so only the absolute veto ever ran. Found by the removal pass, which
deleted the dead relative arm and kept the absolute veto as it ran; the yaml key is gone.
**Question:** is the relative veto wanted? If so it is new work (from git history, without the override), judged on a
night's veto counts per chain against the absolute one; if not, close.

### #187 — clean-up that waits on the new binaries and the restarted broker (2026-10-13)
**What [tree]:** (a) the generator still emits keys the C++ now reads only to refuse other values (cudaGnssInject
`conjugate`, `phi_fp16`, `despread_max_chips`, `despread_chips_centered`, `trim_endpoint`; cudaCorrelatorDual
`gnss_freq_map`, `gnss_synth_compact`, `rfi_all_pass`), so a regenerated config still starts on d30be5f66; (b) the
REST replies carry `addressable` (GnssGpuRecordAssemble) and `pow_fft_len` (GnssCoherentCombiner) for the old broker.
**Fix:** the broker already runs the merged tree (restarted 10-13 12:57Z), so (b) can go with the first fleet binary after
`d505f3943`; (a) waits for that binary too: stop emitting, then drop the reads. Check after each step: `gen_fleet
--check` and `--check-config` on all nine configs.

### #189 — `epl_from_prompt` (E/L lanes as the rotated prompt): wire test at the F-engine's return (2026-10-13)
**What [tree, branch kv/epl-from-prompt, worktree ~/gnss/elp-wt, 5bd27ec7f on 961c4d9c9]:** the waveform kernel walks
the prompt once and writes the E and L lanes as the prompt rotated by +-2 pi (f_c - f_car) ds/chip_rate per channel,
behind the config key `epl_from_prompt` (generator `--epl-from-prompt`, every chain but L2C). Offline: identity and
DLL verified on the 10-12 telemetry capture, `cuda_gnss_despread_test` keeps P/P_HEAD bit-exact and pins the sign,
cx52 A/B gnss_waveform 6.40 -> 3.24 ms per frame per GPU (docs/CHORD_GPU_BUDGET.md, fixtures/gpu_budget/README.md).
Not merged (after the 10-13 14:10 ET session rollover, KV), not pushed, not armed.
**Wire test:** one node first (cx51), `--epl-from-prompt` in its manifest entry at a node cycle after the broker has
re-anchored; judge against a sibling node, same sky same time: the lobe DLL's readback decay and `q`, the trim-quality
and `present` gates, per-channel E/P/L in the telemetry (E_c/P_c must read exp(-i k0 df) as on 10-12), arc survival
through one transit. Then the manifest's common flags. L2C stays on three trials by construction (one channel per
instance, its rotated discriminator is identically zero).

### #176 — the joint clock adopts a one-off 0.8-chip outlier inside a transit and every 10.23-Mcps chain loses its prompts (2026-10-12)
- **What:** C33 (0.61°, CA 12:54:15Z 10-12). Through the pass the joint solve's sigma grew 0.42 → 0.68 and was REFUSED
  (bound 0.50); at 12:58:17Z, C33 at ~1.9°, a 4-satellite solve 0.806 chips from legacy with sigma 0.228 passed the
  bounds (5.0 chips / 0.50 sigma) and was ADOPTED on every chain (×0.5 on E6, ×0.05 on L2C); 30 s later the delta
  was −0.07 again. ~29 s after the adoption every satellite on E5a, E5b, B2a, B2b, B3I and L5 lost its prompt power
  for 1.3–1.9 s on all 12 instances at once (`FADR BREAK ... dark130-190/silent`, telemetry gaps 0); E6 (0.4 of its
  chips) and L2C (0.04) did not. Under the old fold those records would have been folded as phase: a C19-scale slip
  on every satellite of six chains.
- **How often:** 51 adoptions with |delta| > 0.5 chips on gal_e5a from 10-11 16:22Z to 10-12 13:00Z (172 > 0.3).
  A slow run (+0.56 → +0.98 → +0.58 over 8 min through C13 at 3.1°, 12:00Z 10-12) broke no arc: the DLLs follow a
  drift. The damage is the one-off jump.
- **Not the C19 cause:** through C19 on 10-11 every adopted delta stayed within ±0.35 chips.
- **Maybe the same as #175:** an E6 re-seed ~0.45 chips off the peak after a null crossing is what a ~0.9-chip joint
  step scaled ×0.5 would leave.
- **Mechanism:** the joint state is fed by the gps_l5 search's detections (`SEED-OFFSET 1176.45MHz: joint-vs-legacy`),
  and C33 transmits B2a at 1176.45 MHz, inside that band: L5 presence fell 6 → 3 at closest approach, and the joint
  clock walked from −0.2 to −1.4 chips off the legacy clock between 12:57:46 and 12:58:08 (`JOINT[shadow]`, `SEED-OFFSET
  PRN 10 ... diff -1.390`) while its satellite count fell 6 → 4 and its sigma read smaller, not larger. The adoption
  then carried the walk to every chain. The C13 run at 12:00Z is the same walk, slower. The dead-reckon clock is held
  inside a transit (#142); the joint clock is not.
- **Fix candidates (not built):** hold the joint clock (refuse adoption) while the transit sky (#144) has a satellite
  within the transit bound that transmits in the joint's feeder band, as #142 holds the dead-reckon clock; or refuse a
  delta that moves more than ~0.3 chips from the previous adopted one unless it persists for two solves.
- **Check:** `grep 'JOINT-CLK' broker.log | grep ADOPTED` and the |delta| census taken at the 10-12 telemetry v8 rollout.

- **Fix [built 10-13, `6e73de015`; live at the next broker restart]:** `dr_joint_clk` keeps the joint clock it last
  adopted (`dr_state["jclk_last"]`) and refuses a joint clock more than 0.3 chips from it until the new level has held,
  to 0.3, for 30 s (`joint_clk_max_step_chips` / `joint_clk_step_hold_s`, frozen in `cli.py`); a bounds refusal
  restarts the hold; the refusal reads `JOINT-CLK: ... -> REFUSED (step -0.664 chips from the last adopted joint clock;
  the new level has held 0 of 30 s)`. The 5-chip / 0.50-sigma bounds are unchanged. It keys on the joint clock LEVEL,
  not the delta this entry named: over 10-11 16:22Z → 10-12 13:00Z the legacy median's churn steps the delta > 0.3
  chips 90 times while the joint steps 21, so a delta-keyed guard would park each consumer on the churned legacy clock
  ~67 times a day. The hold is in seconds because the cycle is 2 s and the JOINT-CLK line is 30-s rate-limited: "two
  consecutive solves" at the cycle cadence would have confirmed the 40-s C33 excursion.
- **Offline proof (gal_e5a JOINT-CLK lines, 30-s cadence):** 12:58:17Z (−0.806, 0.664 below the last adopted 149.450)
  REFUSED, 12:58:53Z (−0.066) ADOPTED directly; the C13 run at 12:00Z is followed (two lines wait one line and are
  confirmed on the next; at the 2-s cadence its steps are 0.02 chips and never meet the guard). Over the span the guard
  refuses 5 of the 51 |delta| > 0.5 adoptions and 13 lines in all; 9 of the 13 are levels that held and were adopted one
  line later. No persistent level is refused for longer than the hold. At the live cadence the guard caps a C33-type
  excursion at ~0.3 chips plus the legacy offset rather than refusing it outright; the #142-style alternative (hold the
  joint while the transit sky has a feeder-band satellite inside the bound) is the stronger fix if this proves soft.
- **Check:** `gnss_broker.test_jointstep`; on the sky, `grep 'JOINT-CLK' broker.log | grep 'REFUSED (step'` through the
  next transit of a 1176.45-MHz emitter, and no `FADR BREAK ... dark` on every chain after a JOINT-CLK adoption.
  Digests unchanged (no replay carries a refused adoption).

### #175 — after a transit's null crossing every E6 tracker is re-seeded ~0.45 chips off its peak and rails there for an hour (2026-10-11)
**What [live 22:34–23:35Z]:** E06 passed 0.24° from boresight at 22:27Z (E6 prompt 3.5 million × the floor). As it
crossed the dish's first null (2.8–3.5°, 22:34–22:36Z) the chain browned out (PROMPT-CN0 of every PRN down 15–30 dB,
`BROWNOUT open: 5 present vs 10`), and at 22:34:52Z PRESENCE admitted a "displaced" detection for every PRN at once,
all at +0.41..+0.47 chips (E31, E4, E23, E5, E16, …) through `presence-admit-displaced`. From 22:37Z the DLL disc of
E4/E5/E9/E16/E28 sat at the +0.5 clamp with q 2.0–2.3 and prompt power 10 dB down (E4: 310× → 34–126×), on all 12
instances alike; their SPEC-FIT tau read −0.1..−0.2 chips while the bright ones (E6, E31, E23, back on-peak within
minutes) read +0.01. E4 returned only at 23:24:35Z through another displaced admission (+0.589 chips); E16 was still
railed at 23:40Z. The rotated log shows the same railed state for hours before #163 (by the log's day count: 10-06
04–14Z at 21–89% of DLL readings, 09-28 05–08Z and 14–15Z, 10-01 21Z–10-02 00Z, 10-09 21Z), so it is not new; with
#163's lower baseline q it now reads "weak" on the viewer.
**Why it sticks:** a railed disc is leak-bounded (`gnssFleetDll::dll_tau` clamps at ±1, at most ±0.25 chips of trim)
and the model holds the displaced position (`dead-reckon SLEW … model-held −0.013 chips`), so nothing pulls the
tracker the half chip back; only a fresh displaced-presence admission does.
**Fix (broker):** a simultaneous displacement of every PRN by the same offset is a chain event (clock or brownout),
not N detections: refuse `presence-admit-displaced` while a BROWNOUT is open or when ≥3 PRNs report the same offset
within 0.1 chips in one cycle; and a PRN whose disc has sat at the clamp for >60 s with SPEC-FIT tau disagreeing
should re-run presence (the escape path the bright ones took).

**Fix, 10-13 (`9f193b74e`; live at the next broker restart):** `displaced_chain_event` (fleet.py) refuses a cycle's
displaced admissions while D1's BROWNOUT is open on the chain, or when three or more of them lie within 0.1 chips of
one another; refused rows fall back to the q/prompt verdict, one `PRESENCE -displaced: <reason> -- N PRN(s) at off
a..b REFUSED` line per cycle and reason (30-s rate limit). Rows within 0.1 chips of their peak are exempt from both
rules: the deep path admits dim on-peak satellites by the dozen on BeiDou, and refusing them emptied the chain in the
replay (the exemption removes about half the refusals the bare rules would make). The brownout read is the previous
cycle's. **Offline:** the 10-11 22:34:52Z admission (E31/E4/E23/E5/E16 at +0.41..+0.47) is refused as one offset — the
brownout opened ~20 s later, so the offset rule is the one that catches it; E23 at 22:38:56Z (−0.200, deep) is refused
inside the 22:37–22:39Z brownout; E4 at 23:24:35Z (+0.589) and the other fourteen single admissions are admitted. Over
10-11 16:22Z → 10-13 00:22Z, 18 same-offset lines (every one a chain-wide step, including the #176 adoption's ×0.5
echo on gal_e6 at 12:58:49Z) and 38 brownout lines of ~1500. A persistent chain-wide step is refused every cycle by
design: fix the clock, do not re-seed N satellites. Digests unchanged (no transcript carries the flag).
**Still open:** the escape for a PRN whose disc has sat at the clamp > 60 s while SPEC-FIT tau disagrees — it needs a
per-PRN 60-s history (a detector beside D3, fed from the fleet row's disc, the readback trim and `spec_fit[prn].tau`) and
an actuator that releases the C++ loop's standing trim or re-seeds the PRN; not a presence verdict. The `PRESENCE
+displaced` line is rate-limited to one per 120 s, so the log under-reports admissions. **Check live:** `grep 'PRESENCE
-displaced' broker.log` through the next E06/E31 transit; the dim E6 PRNs' DLL disc should not rail after the null.

### #171 — the ALERT files reach no one (2026-10-09)
**What [tree]:** `chain_health_cron.sh` (cf06, */15) and `elem_ref_cron.sh` (gnss, */5) write ALERT files that
nothing reads or forwards; both faults they are built for (#159's lost sky, #154's mis-pins) were found by a person
reading logs. **Fix:** tie them into CHORD's monitoring and alerting (KV: "ultimately"); until then a one-line summary
in the viewer would do.

### #158 — projection phase 2 leftovers: `b` reaches only the lobe fold, `k`/source are not exported, the dB judge never ran (2026-10-02)
**⚡ 10-12: two of the three leftovers done, rolling out (KV: "let's roll that out while things are quiet").**
- **Export, LIVE 10-12:** telemetry v8 (3f381df37, nodes cycled 12:28–12:29Z, gather + broker 12:20Z): `REC_PROJ_K`
  (the record's rank, 0 none) and `REC_PROJ_SRC` (OR of the sources over the instance's channels: 1 own row, 2 sibling,
  4 probe stack) beside `REC_PROJ_COST`. The fleet ADR walk sums them per satellite per cycle and rows carry `proj_b`
  (mean cost where a basis was in force, −1 none), `proj_k`, `proj_src` (9dba98d35, broker 13:26Z; the first cut put
  them in the Python lobe fold, which the C++ taps arm bypasses, so through C33 every row read −1/0/0).
- **The dB judge ran** (`fixtures/tec_wander/db_judge.py`, out/db_judge_1012.txt): other-satellite coherent C/N0 at
  closest approach (±90 s) minus the 10 min before, Galileo bands, against each census's quiet controls. Freeze alone
  (09-28/29): mean loss −2.6/−1.9/−1.5 dB (E5a/E5b/E6) with controls at 0, 45/33/46% of victims beyond 3 dB (controls
  21/26/34%). Projection live, every census 09-30 → 10-12: mean loss at the control level (0 ± 1 dB), 13–19% beyond 3 dB
  (controls 12–29%). Presence losses: 21/15/11 victims in 12 pre-freeze windows, 0 in every freeze+projection window
  except the 10-11 evening (C19: 5 on E5a and 5 on E5b; E06: 4 on E6). The plan's criterion is met on the mean; the
  row-level C/N0 median is blind to the C19-type brownout (+0.6 dB at CA while presence fell), so judge close passes by
  presence and the grid-table C/N0, not rows. Metric noise: the controls' worst victim sits at −3 to −7 dB.
- **Still open:** b into the broker's solves (down-weight by b instead of the 5° veto); phase 1c. Config note since
  #170: the formation freeze (8°) is now wider than the projection window (6°), so the shared model no longer learns
  through passes on projected data; safe, but not the plan's design.
**Phase 2 is closed on its judge** (KV, 10-02): E5a × E6 arcs through transits 39% (freeze) → 74% (09-30) → 82% (10-01)
→ 97% (28/29, night of 10-01/02, freeze 2°); lock held 149/149; closure median 0.57 TECU = quiet sky. Three items the plan
(`fixtures/projtest/PROJECTION_PLAN.md` phase 2) listed were not built; none blocks phase 3.
- **The export is `b` only.** `REC_PROJ_COST` (record slot 29, telem v7) carries b = cos² of the victim's live weights in the
  projected subspace per record row. Per-record `k` and the source kind (own row / sibling board / probe stack) exist only
  in `/get_elem_cal` (the 10-s poller). Adding them is one header slot each = a WIRE bump (v8): queue for the next flag day.
- **`b` is consumed by the fleet DLL lobe fold only** (`gnssFleetDll.hpp` fold + `combdll._lobe_fold`, weight 1−b). Not by
  JFEED's `joint_sigma` (moot since 10-11: no chain feeds the joint state, #125) (0.3 for every satellite, `gnss_broker/cli.py:897`; the #152 item "freeze it inside the transit
  veto" → scale σ by 1/(1−b), or drop b > 0.3), and not in `/get_dll`'s `FleetDllRow`, so no obs row records a transit's cost.
- **The plan's dB judge never ran:** other-satellite C/N0 loss at closest approach against the freeze-era twins
  (`freeze0928/work/pairs.md` method) and lock-loss victim-minutes. We judged by arcs, locks and closure
  (`gnss_transit_arcs.py`). One run over 09-30..10-02 against 09-25..27 closes it.
- Phase 1c (`GnssN2Eigen`, the science-N² a-source for row-less emitters) was never built; the gated probe stack stands in.
  It moves to phase 3, where the N×k block of the extended triangle seeds it.
- Standing caveat, not a leftover: the search tap is single-element and unprojected, so #142/#152 stay necessary.
Check: `[tree]` `grep -rn REC_PROJ_COST lib python` (one writer, one reader = the fold); `grep -n joint_sigma
python/scripts/gnss/gnss_broker/*.py` (constant). `[live]` census `fixtures/tec_wander/out/transit_arcs_1002`.

### #142 residual — two observations from the pre-fix L2C losses are still unexplained
The fix (`ea9129b60`, closed) has held through seven transits. Two things seen during the old losses
have no explanation, and with the losses gone they can be studied only in the archive: a repeatable
+1510 m L2C code-residual plateau while the trackers were being dragged, and gps_l5's legacy clock
~10,014 chips (≡ −216 mod 10230) off the joint clock at 09-27 21:03Z. Evidence
`fixtures/projtest/forensics/l2c/timeline.json`.

### #144 residual — objects in no BRDC are still invisible to the transit sky
The health and age gates are fixed (`aef331c10`, closed): the freezes and the railing veto read `sky.transit_sky`.
What is left: NORAD 40748 "BEIDOU-3S M2S" (Celestrak C58; 0.66° on 09-27, +10 dB) and GLONASS (L3OC at 1202 MHz,
no propagator, #129) are in no BRDC at all. A TLE supplement needs a path that is not Celestrak-from-cf06 and keeps
capability on the IGS SINEX. Evidence `fixtures/projtest/unk/`.
### #151 residual — the seeds and the sky still keep two ephemerides
The reload cadence is fixed (`b2223cc7a`, closed). EPH-REBASE (#101), off on every chain since 10-11 (#125), was
removed 2026-10-13 with its "no step is computed" half (the census, the shared-store flag and the popped step are at
`be7d46e8e`); what remains is the two-copies design note:
- **Two copies, two phases.** The seeds' copy (`dr_state["eph"]`) reloads on the broker's start phase and
  the sky's (the shared `receiver.brdc()` store) on its own, so a seed model step can land up to 15 min from
  the sky's.
- **Fix:** one ephemeris for the seeds and the sky. The geometry-only transit sky moved to #144.

### #152 — the joint clock latches chips off the search clock and nothing notices or re-latches it (2026-09-28)
**Twice now a transit has left the shared joint clock confidently wrong, adopted fleet-wide.** 09-23
08:00–10:39Z, after E31 at 0.3°: joint − legacy +2.0 chips. 09-28 from 13:09Z, after C13 at 3.0°: −1.8 to −2.4
chips until C33's pass began at 13:38 (the pass then reshuffled it to −0.7 by 14:03). Both sit inside the
5-chip JOINT-CLK bound, which limits a per-cycle delta and nothing persistent, so every 10.23-Mcps consumer
adopted them. Presence looked normal, but gal_e5a and bds_b2a `code_resid_m` moved −94 / −92 m (3.1 chips)
against their pre-transit level while gal_e5b, gal_e6 and bds_b3i did not [live 09-28, obs 12:25–12:45 vs
13:13–13:30]. The 09-23 latch took ~3 h to decay; the 09-28 one showed no decay in the 30 min before C33
disturbed it.
**Mechanism (JOINT[shadow] summaries, 09-28).** JFEED ingests y = seed + trim − model from model-primary
trackers whether or not they are on the peak, so a runaway seed (#142) comes back as a measurement. Before
12:58 the per-satellite biases were GPS −0.2..−0.7, Galileo +0.1..+0.4 chips; after the runaway E8/E15 read
−27..−33, E3/E26/E13 −4..−8. The gauge (median b = 0) moved clk −2.2 and every search-anchored GPS b went
+2.0 to compensate: a self-consistent state at sigma 0.048 that nothing pulls back.
**Detector — built and replayed on 41 h of broker.log** (`fixtures/obs/joint_latch_detect.py`, offline,
exit 1 on a latch). d = median(joint − legacy) per minute over the 10.23-Mcps consumers (the JOINT-CLK
lines; gal_e6/gps_l2c excluded until #143). A minute counts only when gps_l5's own search solve is healthy
(≥ 3 integrity residuals with |r| < 1 chip, none BAD) and no BORESIGHT TRANSIT veto fired in the previous
5 min, because the legacy clock itself goes bad in passes (#142). LATCH = |d − 0.3| > 1 chip for 10 counted
minutes. Healthy d: p1 / p50 / p99 = −0.02 / +0.22 / +0.44 chips. It fires on exactly the two known latches
(09-23 08:00Z, d +2.02; 09-28 13:08Z, d −2.19) with no false alarm over 09-23 00:45–11:55Z and 09-27
08:00Z – 09-28 13:55Z, 21 transits among them [archive]. ⚠️ A ROLLING baseline fails: it absorbs a slowly
decaying latch and then fires on the recovery (09-23 11:13Z), so ref is a fixed band. ref is a gauge
convention (median b over the current membership): re-derive it when the membership changes, e.g. after #150.
**Independent product-side check,** no broker internals: the gal_e5a − gal_e5b difference of median
`code_resid_m` (same Galileo satellites) went −0.3 → −87 m, which as geometry-free code is ~600 TECU;
bds_b2a − bds_b3i moved −90 m. It can run from the obs files in the cf06 health cron.
**10-11: the latch pathway is gone.** e5a's model-primary feed into the joint state was removed (#125), so JFEED now
carries only gps_l5's search solve and no runaway Galileo seed can reach the joint clock. The detector stays as the
backstop; the kick below is PARKED unless d leaves ±1 chip again. Check: run `joint_latch_detect.py` over the next
week's log — zero alarms closes this.
**Kick (parked) — none exists today** (the broker serves no joint-clock lever; its `/set_carrier_trim` and
`/set_nh_prn_offset` levers were removed 2026-10-13). In order
of preference: (1) in-broker RE-ANCHOR on the alarm: move the joint's gauge until its clock agrees with the
healthy search clock + ref (clk by +δ and every b by −δ, so no prediction inside the joint moves — only
the clock level the consumers adopt),
re-birth the model-primary satellites whose b sits > 3 chips from the median, and log it as loudly as TIME
ANCHOR; (2) the fallback that already works: a strike counter → `os._exit` → systemd restart, the 9c56c1682
pattern (costs ~10 min of L5 settling and one arc break). **Prevention** is upstream of both: take JFEED only
from on-peak trackers (present, q above floor), freeze it inside the transit veto, and land #142 — the
detector stays as the backstop.
**Check:** replaying the 09-23 and 09-28 windows alarms within 10 min of onset and the re-anchor brings d back
inside ±1 chip of ref; zero alarms on a quiet day; e5a/b2a `code_resid_m` returns to its pre-transit level.

**10-13: the detector replayed over the whole log, 09-18 22:33Z → 10-13 12:56Z (20,849 scored minutes; healthy d
p1/p50/p99 −0.15/+0.21/+0.46 chips).** Three alarms: the two known (09-23 +2.02, 09-28 −2.19) and a THIRD, **10-06
10:27–12:15Z, d −1.96**: gps_l5's search was dead 07:10–10:23Z (0 integrity residuals, dead-reckon clock frozen at
133.74 chips, the joint at ~137, unscored), and when it re-acquired at ~150.0 the joint followed only to 148.2 and sat
2.0 chips low for 1 h 45 min, recovering by itself 12:11–12:18Z. Before #125, so the e5a-feed mechanism; zero alarms
10-07 → 10-12 17:07Z (the last gps_l5 solve before the F-engine went off). Two things to carry: (1) the healthy-d
reference moved with the membership, as warned — p50 +0.22 through 10-10, +0.13 on 10-11, −0.06 on 10-12 — so
`--ref 0.3` is now asymmetric; re-derive it on the first full day back; (2) the detector is blind exactly when the
joint is worst: the joint sat ~13 chips off for 3 h on 10-06 while gps_l5 had no solve, unscored by design (a second
metric that needs no legacy clock, e.g. the gal_e5a − gal_e5b `code_resid_m` difference, would see it).
**Check stays open:** a week of zero alarms after #125 (only ~1.7 scored days exist, 10-11 16:22 → 10-12 17:07Z). Run:
the `JOINT-CLK | gps_l5 dead-reckon clock | BORESIGHT TRANSIT` lines of each rotated log through
`fixtures/obs/joint_latch_detect.py`, per file and concatenated in order (its >1 h backward-jump rule assigns days;
verify the rollover count against the rotation dates). Helpers and the per-day table: `~/gnss/builds/readynow/work152/`.

### #155 — dTEC arcs die in transits of 1176-MHz emitters: the fleet ADR holds the arc while the victim band's residual is dark (2026-09-29)
**⚡ 10-12 status (four nights 10-08 19:00Z to 10-12 00:40Z, `out/transit_arcs_1012`, events_1012.txt; closure per day
by `triple_closure.py`, out/gal_<day>_closure.log).** The instrument floor is unchanged since 09-30 and the locks hold; the
product's remaining losses are the close passes, and one of them on 10-11 was the #155 mechanism on every Galileo satellite.
- **Floor (Galileo triple closure, clean segments, TECU rms; divide by 1.31 for E5a × E6):** quiet days 09-30 / 10-03 /
  10-07 / 10-11 give 0.10 at 1 s, 0.34–0.41 at 60 s, 1.26–1.37 at 600 s, 2.0–2.4 at 1800 s, 2.8–3.3 at 3600 s. Disturbed
  days are 1.5–3× worse at 600 s: 10-01 2.6, 10-09 4.2 (F-engine work, cx51 out, 54868 steps > 5 cm against ~200–1200 on
  other days), 10-10 1.9 (re-base, rows C/D outage). Arcs on a quiet day: ~65 per band, median 167 min, p90 405 min;
  coherent C/N0 at el ≥ 30 median 32 (E5a) / 35 (E6) dB-Hz, 100% above the product's 25 dB-Hz gate.
- **E5a × E6 arcs surviving a transit, by night (quiet controls in brackets):** 10-08 91% (32/35) [92%], 10-09 76% (45/59)
  [52%: disturbed], 10-10 92% (22/24) [96%], 10-11 evening 38% (9/24) [95%]. Locks 99% through every window. Trajectory
  39% (09-29) → 74% (09-30) → 86–89% (10-03/04) → 91–92% on quiet nights now.
- **The 10-11 evening (first close passes after the 17:50Z broker restart with #144/#161; before #170):** C19 (BDS-3,
  B2a/B2b exactly on E5a/E5b) at 0.94°: every Galileo satellite's E5a and E5b browned out (C/N0 p10 24–27 dB-Hz, 6–7% of
  samples below 20; E6 untouched at 37+), the fleet-ADR arcs HELD and integrated through, 9/9 SLIPPED by 10–170 cycles
  (−150 to −1400 TECU) plus one GONE. E06 at 0.24° (all three bands dark, the F-engine clipping product): 2 slipped, 1 gone,
  3 no arc. The equally close passes the night before showed no brownout at all (C38 0.72°, G19 0.64°, C22 1.68°: C/N0 p10
  36–39, 22/24 survived). C19 was tracked on bds_b2a throughout (fleet_present, 36 dB-Hz), so the nodes had its direction;
  whether the projection ran on it cannot be checked (that node-log lifetime is gone). One case, not a verdict.
- **Candidate 2 is still not in place:** `fleetadr.fold_record` only treats energy-less records (S = 0) as dark; a vouching
  set at 3–19 dB-Hz folds noise and the arc keeps its key. C19's slips are what it would have caught. **Candidate 3 would
  have saved all nine:** E6 stayed ≥ 31 dB-Hz on every satellite while E5a/E5b were dark.
- **Census caveat:** the BRDC store had no daily file for 10-11 at run time, so 10-11 daytime transits are absent (and
  10-09 afternoon is thin); re-run once BRDC 2026-284 lands. B2a × B3I (0–33%) and L5 × L2C (5–29%) still slip in quiet
  windows: thin bands, unchanged, not usable for dTEC.
- **Next:** the first nights under #170 (8° freeze) and #163: G09 ~21:21Z daily (1.27°, an L5 emitter: 5/5 on 10-10,
  4+1 slipped on 10-11); BeiDou MEO close passes repeat weekly (C38 10-17 ~05:00Z, C19 10-18 ~20:00Z).
- **Not the quantiser (10-12):** the rail recorder (rf_chan rows, hi+lo fractions in the lobe channels) shows the E5a
  lobe railing ≤ 0.2% of samples at the worst instance/channel during both C19 (10-11) and C38 (10-10), with the same
  band-power rise (median +40%, p99 2.5×); a quiet 10-11 19:00Z window railed 40× more with no transit. E06 at 0.24°
  did rail E6 at 5% (the self case). So the C19 brownout was not rail-driven, and dynamic F-engine gains would not
  remove it: it sits in the array's response to the bright satellite (freeze/projection path), ours to explain.
- **Candidate 2 LIVE 10-12 13:26:29Z (eebea6a1a, 8a4afd29b, c69b07655):** `fleetadr.fold_record` treats a record as
  silent after 5 records in a row whose prompt power (median over instances) is 10 dB below the instances' own level
  AND whose phase steps disagree across instances (R, the mean resultant of the unit steps, below 0.5 over at least
  six instances; ~1 while the phase is measured, ~0.25 for noise); silent records cannot vouch, so the arc ends and a
  new one starts when the signal returns. `FADR BREAK` lines print `silent<n>`, rows carry `fadr_n_silent`. Two
  earlier cuts were live briefly and failed on live data: power alone with one record enough (12:20–12:37: 625 arcs
  in 7 min, a seed re-pin dips every instance for one record), then power alone over 5 records (12:37–13:26: L5
  broke ~7 arcs/min, G23 at 42 dB-Hz swinging 11–14 dB with R 0.78–0.94). Replays of live gps_l5, gal_e5a and gal_e6
  captures (fixtures/tec_wander/replay_silence.py, silence_anatomy.py) leave every satellite bit-identical to the
  fold before the gate. Old-fold baseline: 0–1 `FADR BREAK` lines per hour.
- **C33 (BDS-3, 0.61°, CA 12:54:15Z 10-12, under the power-only gate):** no Galileo brownout through closest approach
  (one arc ended on each of E6 and E5b); projection rank 1 from the sibling BeiDou row on E5a/E5b, cost 0–0.5. Then at
  12:58:46Z every arc on the six 10.23-Mcps chains ended together: a joint-clock jump, filed as #176. The census
  (`out/transit_arcs_c33`, split 12:20:55) reads that as LOCK LOST 7 of 8 E5a × E6 arcs, honest where the old fold
  would have slipped them.
- **G32 (GPS IIF, 3.0°, CA 14:31:45Z 10-12, combined rule + rows):** E5a × E6 7/7 SURVIVED, locks held on Galileo
  and B2a, 2 arcs ended on B2b, no joint-clock step above 0.3 chips. Rows carried the projection on ~90% of present
  rows 14:20–14:34 (sources 2/4/6 on Galileo, 1 on L5; b median 0.02–0.03, max 0.69 on E6). Three survivors jumped in
  the closure (E15 +8.5, E08 +5.1, E32 −3.3 TECU); E15's projection cost on E6 was 0.50–0.57 at the time, so a
  projection-induced band phase step is the lead to test. G09 (21:21Z) fell inside the F-engine outage.
- **Watch at the return:** after the post-restart transient (56 L5 arcs ended in 13:28–13:34Z) the gate ended about one
  arc per 3 min, all on gps_l5 and BeiDou single satellites in real fades (C38: cross-instance coherence 0.74 → 0.08),
  against 0–1 per hour under the old fold. Galileo: none outside transits. Recount over a quiet day once the sky is
  back; the L5 fix (#177) raised every L5 satellite's prompt, which should lower L5's rate.
**⚡ 09-30, with projection live fleet-wide (from 09-29 17:00Z): candidate 1 largely holds.** Seven transits had data,
09-30 03:17Z to 15:10Z, between two F-engine outages.
- **E5a × E6 arcs surviving a transit:** 74% (39/53), against 39% (26/67) with the freeze alone.
  - Through 1176-MHz emitters: 25/38, against 15/54.
  - Through close (< 1.5°) 1176 passes: 9/16, against 5/36.
- **Closure (ionosphere-free) median jump per transit:** 0.85 TECU, against 10.2 with the freeze alone. Quiet windows
  0.77–0.81, so the typical damage is now at baseline.
- **Not fixed:** C31 (0.44°) slipped 4 of 9 and C11 (0.72°) 2 of 7. Suspects: the band-edge channels' own-row direction, or
  victims outside the probe stack.
- **Candidates 2 and 3 (an honest FleetAdr; sibling-band bridging) still stand.**
- **Page v3:** https://claude.ai/artifact/15dmeuuxeESKk2WNsANhmU
- **Census:** `--t-lo 2026-09-28T20:30:00 --t-hi 2026-09-30T15:10:00 --split 2026-09-29T17:00:00 --events events_0930.txt
  --out out/transit_arcs_proj`.
- **Page build:** `fixtures/tec_wander/page/make_page.py OUT transit_arcs_proj page_template_proj.html`.
- **Next:** rerun over more nights, and after the freeze is relaxed from 6° to about 2°.

- **What:** since #142 (live 09-28 20:18:48Z) the tracking lock holds through every transit: 362 of
  363 fleet-ADR arcs on all eight chains, against 92% (525/573) the night before. The dTEC arcs do
  not. Only 39% of E5a × E6 product arcs survive a transit (26/67), the same as the night before
  (37/94), against 99% and 96% in matched quiet windows. 58% of the arcs that entered came out
  SLIPPED: arc key held, level displaced.
- **Mechanism (measured on one case):** E16 through C39, 09-29 04:24–04:47Z. E5a's C/N0 was 6–19
  dB-Hz for about five minutes while E6 stayed near 40. One `(fadr_arc, fadr_hop0)` held on both
  bands, with all 12 instances, throughout. The E5a × E6 level went from +0.5 to −2062 TECU, which
  is −362 E5a cycles. The likely reading: the fold keeps integrating the commanded Doppler while
  the residual is unmeasurable, so the error grows with the dropout. The product's C/N0 gate and
  5-cm step rule cut the arc there, so nothing wrong is published, but the arc does not continue.
- **By emitter:** arcs survive transits of satellites with nothing at 1176 MHz (G19 IIR-B, G12
  IIR-M per IGS SINEX: 11/13). They mostly die in transits of 1176 emitters (GPS IIF, Galileo,
  BDS-3: 15/54).
- **Damage:** the median jump of the Galileo triple-frequency closure (ionosphere-free) across a
  transit fell from 170 TECU before #142 to 10 after, against 0.8 in quiet windows. #142 cut it
  about 17×, but it is still many cycles.
- **Not judged:** B2a × B3I and L5 × L2C slip even in quiet windows (0–21% survive; B3I and L2C
  are thin). Projection, live fleet-wide since 09-29 17:00Z, had no transit in this census.
- **Fix candidates:**
  1. Projection may remove the dropout itself. Judge it on the first transits under it (G09
     09-29 22:14Z, then the rest of that night).
  2. FleetAdr ends or flags an arc while its vouching instances' residual is unmeasurable (C/N0
     below roughly 20 dB-Hz for more than a few seconds). That gives every consumer honest
     continuity, not only the TEC product.
  3. Bridge the dark band's residual rate from a locked sibling band of the same satellite. The
     non-dispersive part scales with frequency. Untested.
- **Check:** `python/scripts/gnss/gnss_transit_arcs.py`, run on cf06 from `fixtures/tec_wander`
  (~2 min; the exact command is in the page footer). Census
  `fixtures/tec_wander/out/transit_arcs_0929_census.json`. Page
  https://claude.ai/artifact/15dmeuuxeESKk2WNsANhmU (built by `fixtures/tec_wander/page/make_page.py`).
- **Traps:**
  - Order grid hops by (F-engine epoch, hop), never by a row's `t`. `fadr_g_hist` times inherit
    the poll time and swap neighbours; the first pass faked 63% NO ARC that way.
  - A row's `carr_resid_m` is dominated by the ~100-ms record-count stamp (±40 m). Judge a
    band's ADR only at exact grid hops.

### #119 — `--fit-flush-on-reject`'s own revert trigger is tripped, and unread
Pre-registered as "revert if flushes happen on healthy sats outside events". **[live]** 69
`cp-fit history FLUSHED` in 57 minutes on a healthy fleet, all on gps_l5, concentrated on five
PRNs (24, 8, 18, 32, 26).

**Chased 2026-09-10, and the obvious hypothesis is falsified.** The flushes do NOT track #97's
source period defect: **0 of 69** fall within 30 s of a `SOURCE PERIOD DISAGREES` on the same PRN
(70 of those occurred in the same window); only 14 of 69 coincide with a `period ADOPTED`. So it
is not an #97 instrument. But the guard is not over-sensitive either — the rejected rates are
physically impossible (`+0.56`, `-23.75`, `+8.25`, `+24.08` chips/s against a clock of ±0.01), so
it is catching real garbage. **The pre-registered revert trigger should therefore NOT fire: it
was written on the assumption that a healthy fleet cannot produce a poisoned history, and that
assumption is wrong.** Rewrite the trigger; keep the guard.
**What is still open is where the garbage comes from.** Not an unwrap gap: a third of the flushed
PRNs were seen <60 s earlier, and the gaps that do appear are 115–710 s, inside the unambiguous
range. Next: dump the fit history itself for PRN 24 at a flush and find the sample that drags the
slope.

✅ 10-11: the `--fit-gap-s 3600` half is fixed — 1200 s in `common` (inside the unwrap's 21–25 min unambiguous
range), live at the 12:58Z broker restart. What is left is the garbage's origin, above.

**10-13: the fit's INPUT samples are not in the log** — the cp-fit line carries only the fit's output (`cp_ref @ h0`,
slope, pts) at a 10-s rate limit — so the dragging sample cannot be named offline from any existing log. `0a43f3557`
(live at the next broker restart) logs the last 12 (hop, cp in seed currency, det dop Hz) at every FLUSHED. What the
10-12 log does show for PRN 24 (03:26–03:32Z, el 54°): the first flush (03:28:54) was a 256-point history whose slope
went 0.015 → 0.082 → 0.741 → 1.383 chips/s in 44 s while the search Doppler scattered ±50–100 Hz (measured
alternating +62/+188 against a predicted ~+140) and the long-code residuals ran 1000–5000 chips at snr 30–41, just
above `--fit-min-snr 30`; every flush after it was a 3–5-point rebuild at |slope| 18–1238 chips/s, one per ~20 s. Over
the 10-11/10-12 logs 534 of 583 flushes that pair with a cp-fit line are on ≤ 8-point histories: the garbage is the
FRESH weak detections, and the flush cascade is the symptom. The replay gate cannot see this guard at all — no fixture
enables `--cp-rate-model-tol`; one transcript recorded with the live knobs would make it real.
**Check:** on the next sky day take one `history at flush` line for PRN 24/8/18/32/26, unwrap the cp column by nearest
wrap and see whether the dragging step rides a det-dop jump (~1560 chips per Hz of Doppler error at that run age) or a
low-snr sample; the fix is then a higher `--fit-min-snr` or a seed-currency step gate, not the flush.

### #120 — the stack's boot path: the VM's units are untested across a reboot, and cf06's cube leg has no unit
**[live 10-11]** The broker, gather, aggregator, eight obs writers, viewer and rail run on the gnss VM as user units
(`~/.config/systemd/user`; `gnss-stack.target` in `default.target.wants`; `Linger=yes`; every unit
`ConditionHost=gnss`, `After=remote-fs.target network-online.target`, `RequiresMountsFor=/home/kvand`; broker after
gather, viewer and obs after broker). A VM reboot has never been tried and needs root. cf06's leftover copies of the
units are inert (`ConditionHost`), but its cube leg — the archiver (`build_nodpdk` kotekan, pid 5407, parent init,
up 8.6 d) and `cubecompact_loop.sh` — is two bare processes with no unit, on the host that reboots weekly.
**Validation short of a reboot:** (1) `systemctl --user stop gnss-stack.target && systemctl --user start
default.target` proves the wiring and the ordering on the VM (~20 s; the nodes keep tracking; the L5 fast loop rails
~10 min after the broker returns); (2) `loginctl show-user kvand -p Linger` = yes, `systemctl --user is-enabled
gnss-stack.target` = enabled; (3) the boot-ordering question (the user manager starts after
`systemd-user-sessions.service`, itself after `remote-fs.target`, so the NFS home should be up first) is only proven
by a real reboot: ask the VM's admin for one at a quiet hour. (4) Write a unit for the cube leg (cf06, or the VM if
it moves) and the compactor loop.
**10-11 12:52Z, the test (KV's go):** `systemctl --user stop gnss-stack.target` stopped the aggregator, viewer, rail and
the eight obs writers but NOT the broker or the gather — same PIDs, no state change — although both load
`PartOf=gnss-stack.target` and the target's `ConsistsOf` lists them; the cause is invisible without the user journal
(kvand is not in `systemd-journal`). `start default.target` then pulled the stopped units back in order. The broker and
gather were restarted explicitly at 12:58:24Z (`systemctl --user restart gnss-gather gnss-broker`): broker after gather,
"starting 8 chain(s)" at +1 s, telemetry frames and trim posts flowing at +40 s, no CHAIN DIED, and the viewer (PartOf the
broker) followed. **Still open:** why the target's stop does not propagate to those two (journal access), the real reboot,
and the cube leg's unit. Until this is understood, stop the broker and gather by name.

### #129 — we cannot place a GLONASS satellite at all (BeiDou turned out to be fine)
**[tree, 2026-09-11]** Found chasing #56's residual.

**THE REAL GAP: no GLONASS positions anywhere.** `gnss_ephemeris.parse_rinex_nav` is `G/E/C only`
by its own docstring. The daily BRDC we already download carries **273 GLONASS records** (plus 22
QZSS, 40 NavIC, 1601 SBAS) and we discard every one. `glonass_freq_channels()` reads those same R
records for FDMA channel numbers, so **the data is in hand and only the propagator is missing** —
PZ-90 state vector plus integration, structurally different from the Keplerian G/E/C path, which
is presumably why it was never written. This matters because we run `GLO_L3OC_P`/`_D` at
**1202.025 MHz** and #56's secondary feature peaks at **1202.5 MHz**: a signal at a frequency we
monitor, from the one constellation we cannot geometrically reason about at all.
`BRDM00DLR_S` (already coded as a CDDIS fallback in `gnss_brdc_supply.py`) carries 26 GLONASS, 5
QZSS and 3 NavIC — so the ephemeris supply is solved; it is purely the propagator.

⚠️ **BEIDOU IS NOT A GAP — a first pass here claimed it was, and was wrong.** `BRDC00WRD` omits
C05, C15–C18, C43–C46 on every recent day, and BRDM00DLR omits them too. That is CORRECT: the IGS
SINEX `SATELLITE/PRN` block lists the **currently assigned** BeiDou PRNs as C01–C04, C06–C14,
C19–C42, C56–C58 — **40 of them, and BRDC carries 39.** C43–C46 are *SVN* numbers in the SINEX's
left column, not PRNs; reading them as PRNs is what produced the false gap. The one real absence
is **C57**, assigned per SINEX and missing from BRDC — worth one look, not a constellation hole.

⚠️ **THE METHOD POINT, which cost two wrong answers in one afternoon:** "absent from BRDC" does
not mean "missing". Ask the authority which PRNs are *assigned* before concluding anything is
absent — `igs_satellite_metadata.snx`, per the standing rule that capability comes from IGS SINEX
and never from Celestrak. A TLE catalogue lists an object in orbit, which is not the same claim.

**Also still true:** the B3I chain's PRN list is 19–42 (`/gnss{0,1}_b3i_n2assemble`,
`_n2dual/commands`), so C56–C58 are outside it even though SINEX has them assigned and BRDC
carries C56/C58. Small, and separate from the above.

## Open — the fix is in the NODE BINARY (queue for the next cycle)



**Top priority: #177.** The others below it came from the same 10-12 comment review and are
unconfirmed or dormant.

### #177 — the GPS L5 code generator wraps XB at 10230, not 8191: every L5 PRN loses up to 10.5 dB (2026-10-12)

**⚡ LIVE AND VERIFIED 10-12 (a25e36345, fleet binary d30be5f66):** canary cx44 16:47:50Z, the other five 16:56–16:57Z.
Paired in time (fixtures/tec_wander/l5_canary_judge.py, the broker's per-instance deep SNR, cx44 against the median of
the other nodes, 20 log10; noise ±0.6 dB): G25 +8.2 dB (predicted +10.4), G03 +4.6 (+6.4), G04 +2.7 (+1.9), G32 +2.5
(+1.4), G01 +1.8 (+1.5), control G28 +1.5 (+0.2), about +1 dB common to all. After the fleet cycle cx44's lead fell back
to within ±0.8 dB on every satellite (G25 +7.7 → +0.4): the five gained the same. Fleet: G25 prompt C/N0 +12 dB; G26,
the weakest, went from never present to present 93% (+6.9 dB coherent). 0 chain deaths. The aggregator (its L5 search uses the code)
was restarted onto the fixed binary at 17:04:44Z, minutes before the F-engine went off for about a week. **Still to do
at the return:** judge the aggregator's L5 search SNR per satellite, and re-solve the per-satellite L5 gain.
Branches: kv/l5-xb-wrap is in (cherry-picked), so `~/gnss/l5fix-wt` can go once its branch is deleted; kv/comment-pass
(the review agent's, with uncommitted work in `~/gnss/comments-wt`) conflicts with the fix in `gpsL5Code.{cpp,hpp}` and
must rebase onto kv/chord-gnss before it merges.
**What [tree + archive]:** `gpsL5Code.cpp` ran the 13-stage XB register (period 8191) for 10230 chips and indexed it
at `(i + advance) mod 10230`. IS-GPS-705 clocks XB on from the PRN's initial state, i.e. `(i + advance) mod 8191`, so
the last `advance` chips of every code (up to ~7200) come from the wrong XB phase. Predicted loss: Q5 (the live
`GPS_L5_Q_NH`) mean ~4.4 dB, worst PRN 6 −10.5, 25 −10.4, 18 −10.1, 17 −7.9; least PRN 10, 28, 2 (≤0.3 dB). It is the
per-satellite L5 gain of 09-22 (`fixtures/beamcube/work/jt_l5_3d.npz`): predicted vs solved G r = 0.93, slope 1.01 dB/dB,
21 PRNs, 1.3 dB rms residual against DRAO's 0.64 dB spread; 09-08 solve r 0.90. The 09-22 hunt checked the XB tables,
not the wrap; `l5_band_decode.py` copied the construction, and the test pinned this generator's own balances.
**Fix: `e36372bee` on `kv/l5-xb-wrap`** (generator, header, test, `l5_band_decode.py`). New boost case `xb_wraps_at_8191`
(I5·Q5 cancels XA and must repeat every 8191 chips) fails on all 32 PRNs before the fix; the first-24-chip PocketSDR pins
are unchanged. The code phase is unchanged, so seeds and locks carry across. **Check after the cycle:** one canary node
paired in time against the others: PRNs 6/18/25 up ~10 dB in `cn0_inc`, PRNs 10/28/2 flat; then the fleet, and re-solve
the per-satellite G (it should collapse to DRAO's spread). Element models, thresholds and the #154 reference see only a
higher SNR. recv1's ConfigTracker restarts once on the binary change (#165).

### #178 — B3I's G1 register repeats every 8189 chips, not 8190 (2026-10-12, unconfirmed)
**What [tree]:** `beidouB3ICode.cpp` resets G1 to all-ones when it reaches `0011111111111`, checked after the shift, so
that state is never output; simulated, it is reached after 8189 shifts (natural period 8191). If the ICD's truncated G1 is
8190 chips, every chip from 8189 on uses a G1 one step off: ~1000 of 10230 chips wrong on every PRN, a uniform ~−1.9 dB,
invisible to a per-satellite comparison. gnss-sdr does the same, which is where the table and convention came from.
**Check:** the BDS-SIS-ICD-B3I G1 truncation text; if still ambiguous, a same-time A/B on one node (8189 vs 8190) on the
live `bds_b3i` chain.

### #179 — the search covers only half the L2C main lobe (2026-10-12)
**What [tree]:** `gnssBandPlan::half_bandwidth_hz` returns `chip_rate_hz` for BPSK, which for time-multiplexed L2C CM is
the 511.5 kHz component rate. The bank models the 1.023 Mcps combined stream, whose first null is at 1.023 MHz.
`covering_bins_union` → `GnssChannelizedSearch` therefore picks channels for ±511.5 kHz + Doppler, about half the lobe
(~0.7 dB of the search's SNR). Node chains take their channels from the generator, not from this function.
**Fix:** use the combined rate (`chip_rate_hz * 2` when `time_multiplexed`). Check: the search's L2C channel count
before/after, and the aggregator's L2C `search_snr` on the same satellites.

### #180 — Galileo E1/E6 memory codes index the table before range-checking the PRN (2026-10-12)
**What [tree]:** `generate_e1b/e1c/e6b/e6c_code` evaluate `*_HEX[prn - 1]` before `decode()` throws for PRN outside
1..50, so an out-of-range PRN reads past the array first. `ChannelizedReplicaBank::set_prn` relies on the throw to refuse
a PRN with no code, so a swap to PRN > 50 on a live E6 or E1 chain is undefined behaviour. E5a/E5b check first.
**Fix:** range-check in the four generators before indexing. Check: a boost case calling each with 0 and 51.

### #181 — the replica bank applies no shared overlay for B2I, L3OC and L2OC (2026-10-12, dormant)
**What [tree]:** the shared-overlay ladder in the `ChannelizedReplicaBank` constructor omits `BDS_B2I`, `GLO_L3OC_P/D`
and `GLO_L2OC_P`, although each has `secondary_length > 0`; `channels()` / `hoprate_stream` apply no overlay for them
and the NH search reports "no secondary code". No live chain uses them. Only `BDS_B2I` (`B2I_NH20`) and `GLO_L3OC_D`
(`L3OC_BC5`) keep a shared `gnssOverlay` registry row; the pilots' rows (`L3OC_NH10`, `L2OC_OC2`) were removed
2026-10-13, though their arrays stay in the code generators.
**Fix:** add them to the ladder (for B2I and L3OC_D the shared overlay can come from the registry). Check: a replica
test with `nh_phase >= 0`.

### #182 — the combiner's overlay + navwipe path wipes suffix rungs at the full window's overlay phase (2026-10-12, dormant)
**What [tree]:** `GnssCoherentCombiner.cpp` (the `!_secondary.empty() && _navwipe_bit_records > 0` branch) passes the
full window's `best_ph` to `overlay_apply` for every rung, but `overlay_apply` counts the chip index from the first record
of the window it is given; a suffix rung starting d records later needs `best_ph + d` (the navwipe pin beside it is
shifted). No live config sets `secondary_overlay` or `navwipe_bit_records`, so the path does not run.
**Fix:** shift the phase by the rung's record offset, as `pin_here` does.

### #170 — a model formed inside a near transit (5–8°) stays mis-shaped for over an hour (2026-10-09)
**What [live 10-09 19:12–20:30Z]:** L5/E5a/E5b pol-1 models that formed on the 19:12Z restart during the C21 pass
(19:40–19:48Z at 4.9°, outside the 2° freeze) stayed at shape similarity 0.07–0.33 to the fleet for over an hour while
the pol-0 models and the other bands were fine. #154 corrects only the global phase; nothing corrects a shape, and the
adaptation EMA is the only recovery. (The cable work that evening is a confound; the pass is the candidate.)
**Fix:** freeze model FORMATION (not steering) inside a wider cone, ~8°, or re-seed a model whose similarity to the
fleet reference's shape stays below 0.5 for N minutes from that reference. Check: restart one node during the next
5–8° pass and watch `elem_shared_ref status` similarity recover within minutes.
**10-11: both built (`11ab0ee25`), awaiting the next node cycle.** Formation cone 8.0° (no first model inside the
cone; a warm one keeps adapting), re-seed after 600 s (a warm half below the floor that long takes the
reference's shape, once per interval) gated on the sibling pol still matching at similarity 0.8 (all three fixed in
the stage since 2026-10-13; they were the keys `elem_sum_shared_form_deg`, `_ref_reseed_s`, `_ref_reseed_sib_sim`) —
both pols off at once means the reference is stale and the cron's re-take must win. `/get_elem_cal` reports `form_frozen`,
`form_deg`, `fleet_ref.reseeds`. Boost test 12 cases. Judge after the cycle: restart one node during a 5–8° pass
(`form_frozen: true` until it clears, then sim ≥ 0.8 within minutes); after the next F-engine gain change the re-seed
counters must not move while the cron re-takes.
**10-11 23:00Z: DEPLOYED on all six; the freeze half JUDGED, passed.** cx44 restarted 23:15Z inside E31's pass
(8° at 23:11Z, closest 3.3° at 23:29Z): `form_frozen` true on all 16 assemblers and no model for 32 min; at 23:47:48Z
(bore 8.1°) every assembler formed from 10–17 shadows, pinned to the fleet reference, sim 0.85–0.98, reseeds 0, fleet
R 1.00/1.00 on every band. Open: the re-seed half, judged at the next F-engine gain change (`fleet_ref.reseeds` stays 0).

### #168 — our 32-element code assumes the n/2 pol split (2026-10-10)
**What [tree, from the rows-C/D survey]:** the shared model splits pols at h = n/2 (`GnssGpuRecordAssemble.cpp`, the
pol-1 anchor and both learners), so 64 elements = [0,31]+[64,95] is right and 48 = [0,31]+[64,79] silently mixes
pols; nothing refuses it. Growing the GNSS array to rows C/D must go to 64 inputs, never 48.
**Fix:** derive the split from the dish table, or refuse `n_elements` not in {32, 64} in the assembler and the
generator. Check: a generator refusal on a 48-element live_element_ranges.

### #156 — `get_chord_metadata()` reads an object's `parent_pool` while `deepCopy` can be assigning it (object-content race; found 2026-09-30)
- **What:** `chordMetadata::deepCopy` does `*this = *chord_other` under both objects' locks, and that assigns the weak_ptr
  `parent_pool`. `get_chord_metadata()` reads `mc->parent_pool.lock()` with no lock (chordMetadata.cpp:447, 455, 464, 477).
  So a thread that holds no frame and inspects a slot `copy_metadata` is filling races the `deepCopy` into it.
- **Found by** the adversarial TSan review of #1714: a lock-free observer, get_chord_metadata() on a copy_metadata target.
  The slot atomics don't cover it, because it's a race on the object's contents, not the slot.
- **Impact:** believed rare. It needs copy_metadata plus a reader outside the frame protocol. No crash is attributed to it,
  but a weak_ptr read racing its assignment is UB.
- **Fix candidates:**
  - Read `parent_pool` under the object's lock (one accessor).
  - Or treat it as immutable after construction: stop `deepCopy` copying it. NDArrayRingBuffer::set_metadata already sets
    it explicitly before deepCopy.
  - Upstream.
- **Check:** the #1714 frame-cycle test with B observed through get_chord_metadata() (the template in the TSan review
  avoided exactly this), run under TSan (cf06:/var/tmp/kvand-tsan-review, `setarch -R`).

### #145 residual — ElemCal drops any element with rho² > 0.99
The reference-weight defect is fixed (`978561481`, closed). Left from the same review: an element whose
rho² exceeds 0.99 is dropped, so a very bright satellite's own cal can end with zero weights. Unmeasured.

### #154 — the shared element model latches its phase pin from its FIRST model, which is one shadow (2026-09-28)
`shared_consensus` (`GnssGpuRecordAssemble.cpp`) forms the first shared model from a single warm shadow
(the one with the most live elements), latches `_g_pin_ref` from it, and phase-aligns every later model
to that pin. Only a process restart, a collapse onto one element, or a reference-element change clears
it. The model's shape keeps learning, but its per-pol phase stays referenced to whatever sky the node
started into. Twice on 09-28:
- cx44 restarted 18:12Z inside a BeiDou dropout (#151) and pinned its BDS models on weak shadows (B2A
  +85/+155° off the fleet). It stayed there until a second restart.
- All six restarted 20:15:55Z before the broker had re-anchored after the F-engine re-base, and pinned
  on noise. Cross-instance R pol0 read E6 0.15, E5b 0.66, B2B 0.69 and E5a 0.70, and E6's fleet lobe sum
  cancelled (xcoh −0.066). It lasted until KV cycled them again at 20:39Z, after which R read 0.94–1.00
  and xcoh 0.86–0.99.

With #153's silent relaunches, every node is a candidate at any hour.

**2026-10-04, twice more, and the first fix idea is not enough.**
- The F-engine came back re-based at 01:22Z and every node's unit relaunched itself within seconds, three
  minutes before the broker's anchor: R pol0/pol1 read E5a 0.69/0.56, B3I 0.61/0.74, E6 0.67/0.52 and L2C
  0.33/0.74, while L5, which re-acquires from its own search, read 1.00. KV cycled all six at 02:16Z.
  Because the units now relaunch themselves, this is the default outcome of every re-base.
- After that clean restart, cx43 GPU 0 (E6, +87/+120°) and cx42 GPU 1 (B2b, +83/+21°) pinned wrong at their
  FIRST model: no collapse anywhere since. The offsets wander as the shape matures: cx42's B2b drifted back
  to +16/+20° within half an hour, while cx43's E6 read +52/+133°.
- Re-latching from the node's own model cannot repair a GLOBAL phase offset: the current model already
  carries it. The node has no local truth either: reference element 0's phase in the pinned models scatters
  over ~150° across HEALTHY instances, so "anchor real positive" would flag healthy instances.

**Fix (staged for the week of 10-05, KV):** (1) ROOT: pin `<F, G>` real positive with F ONE reference vector per
band and pol shared by every instance: a committed snapshot of the fleet consensus per array epoch (like the
element positions file), loaded from config and settable over REST. Apply a correction as a rate-limited
slew, ~1°/s, so it never steps the carrier phase; fall back to the own-first-model pin only when no F exists.
This covers startup mis-pins, collapses and the post-re-base case with no restart. (2) SAFETY NET:
`snap_xinst.py`'s math as a 5-min cron with gauges and an ALERT file like `chain_health_cron`. An instance
> 45° from the fleet for > 15 min outside a freeze gets the fleet consensus POSTed as its reference.
Canary first in a log-only mode. **Check:** after an F-engine re-base the fleet reaches R ≥ 0.95 in every
band within ~10 min, with no node restart (`fixtures/fix0928/canary/snap_xinst.py`).

**2026-10-06: built, not yet deployed.** The 10-05 16:55Z F-engine outage (back re-based 03:36Z) did not
noise-pin the fleet: every relaunch died on the configs' EOP table (ended 10-06 00:00Z) until the hourly
push reached it, so the nodes came up 03:54–05:17Z, after the broker's 03:40Z anchor. Startup mis-pins
remained: cx27 GPU 0 L5 at −155/+57° (shape similarity 0.82/0.88, so purely a global phase) and five B3I
halves 31–46° off.
- Node (`GnssGpuRecordAssemble`, `gnssSharedPin.hpp`, boost test `test_gnss_shared_pin`): config
  `elem_sum_shared_ref` / `_mode` (off|log|live) / `_slew_deg_s` (1.0) / `_min_sim` (0.5), REST
  `set_elem_sum_shared_ref`, and a `fleet_ref` block in `/get_elem_cal`.
- Generator `--elem-shared-ref FILE --elem-shared-ref-mode`, refused on an epoch mismatch. Manifest:
  `elem-shared-ref` (now `config/elem_shared_ref_20261010.json`), mode `live`.
- Tool `python/scripts/gnss/elem_shared_ref.py` (snapshot | post | status | watch). The 10-06 snapshot is
  R 0.94–0.99 in every band, and healthy instances match its shape at a median similarity of 0.88–0.99
  (a noise model scores ~0.2).
- Safety net `scripts/gnss/site/elem_ref_cron.sh` (log-only unless `ELEM_REF_ACT=1`), in the gnss crontab
  every 5 min with `ELEM_REF_ACT=1`.
- Binary `~/gnss/builds/fleet-pin154-20261006` (md5 08949ff5). Rollback: `build/kotekan/kotekan.prev_27ea60023`.

**2026-10-06 LIVE fleet-wide.** Restarted 15:46–15:47Z on the new binary in log mode. Every node's own
offset and similarity matched the tool exactly on all 89 assemblers. The restart left R ≥ 0.98 except
E6 pol1 at 0.85: two clusters 58° apart, cx19/cx42/cx27 GPU 0 at about −23° and the rest at about +36°.
cx43 went live by REST at 16:00:33Z: 21 slews of up to 36°, all done by 16:01:13Z (about 40 s at 1°/s),
and nothing in the broker log. The other five went live at 16:01:43Z. By 16:02:21Z all 89 were within
1° of F, and R read 1.00 in every band and pol. No restarts, no FATAL, no half below min_sim. The
manifest is now mode live (regenerated 16:02:50Z), so a relaunch comes back live. **Still to confirm:** the
next F-engine re-base reaches R ≥ 0.95 within ~10 min with no node restart.

**2026-10-09: the 10-06 reference went stale and the fleet fell back to its own pins.** The F-engine came
back on 10-08 (~01Z) with new gains. The reference holds each element's absolute complex response, so every
model's shape moved: median similarity to it fell to 0.17, and the nodes applied it on 5 of 178 halves
(min_sim 0.5). So the 10-08 20:12Z re-base did not heal, and cx42/1 B2A pol0 (177° off) and three B3I pol1
halves (~70° off) stayed put. The cron listed all of it ("take a new snapshot") but only reports. A new
snapshot, `config/elem_shared_ref_20261009.json` (R 0.976–0.999), went live by REST at 01:24Z (cx42) and
01:30Z (the rest): 178 of 178 halves applied and R 1.00 in every band and pol by 01:33Z, B2A broker xcoh
0.62 → 0.98, no restarts. The reference must be re-taken whenever the F-engine's gains change. The re-base
check above is still open: it has not yet run against a current reference.
**2026-10-10:** the F-engine came back re-based at 01:31Z with new per-board phases; the acting cron re-took all
eight bands by 02:10Z (committed as `config/elem_shared_ref_20261010.json`, 2bad4afda) and every instance applied
it. A node restarted after a re-take and before the commit runs up to 30 min on the stale file (`ACT_EVERY_S` in elem_shared_ref.py).
**Still open:** the no-restart re-base check above, and the file is re-taken by the cron, not by the re-base itself.
The alerting TODO is #171.

### #131 — the gather dies when a telemetry client flaps
**[live]** 2026-09-14 20:57:45: the cf06 gather (`build_nodpdk`, 09-09) exited with no FATAL, no
core (no `coredumpctl` on cf06) and a log that ends mid-burst: six `dropped client fd N -- it
could not take a frame within 200 ms` lines in its final second, and **20 loopback connects /
20 drops in its last 40 lines** on fds cycling 201–210. That client was the live broker's telemetry
reader reconnecting after each drop. The broker had been flat (34 drops, 19:12→20:48) until 21
`broker_equiv` replays ran on the same host (three `gate.sh` runs, 20:50–20:58) and starved it.
**Consequence:** every `set_policy` and trim post `Connection refused` for 50 min, the fleet ran
on the gather's *standing* policy (so seven chains looked fine), and L2C — never armed in that
policy — was the visible casualty. Trim store aged past 300 s, so recovery cost a pull-in.
**Two defects:** the gather must survive a client that connects and is dropped 20 times a minute
(reaping path, 241c0e4a1 lineage; suspect a use-after-close in the poster/reaper race); and the
broker must fail LOUDLY when its gather link is refused -- it logged `TELEM DOWN frames 0` every
30 s and `0 posts / 1349 failed` while the viewer showed a working fleet. **Mitigations shipped
09-14:** `gate.sh` refuses on a host with a live gather/broker; the gather runs under
`segvtrace.so` (separate log `/tmp/gnss_gather_segv.log`) so the next death leaves a backtrace.
**How to see it:** `ss -ltn | grep -E "11060|11061|12051"` empty; broker log `frames 0` with
`nothing yet` on every chain; the 2-line `stack_up.sh` health table.
**10-13:** both mitigations are gone with the dead-code removal (`gate.sh` and the `segvtrace.so` launcher went with the
pre-systemd launchers), so the next death leaves no backtrace; the gather now runs under its unit with
`Restart=on-failure`, 30 s. The fix is in the gather binary on the VM (the reaper path), deployable without a node
cycle; a flap harness (connect, take nothing, repeat 20×/min) is the test.

**10-13: read, sanitised, flapped — not reproduced.** The one defect found by reading and confirmed by ThreadSanitizer
is the stage-7 one: `~GnssTelemGather` closed `_listen_fd` while the accept thread polled it — fixed (`c955796ff`:
shutdown, join, then close; `client_drops` under its own mutex, `bad_frames` atomic). A shutdown-time race, so it cannot
explain a death mid-run. The flap harness now exists: `scripts/gnss/telem_flap.py --kotekan <bin> --config
config/generated/chord_gnss_gather.yaml --capture fixtures/tec_wander/telem/telem_l5_20261012_1252.bin [--clients N
--flaps-per-min R --read --no-sender]` (ports moved +1000; exits non-zero if the gather dies; the e5a_e6 capture is v7
and the v8 gather rejects it). The UNFIXED build survived 20/min, 200/min × 10 clients, reading clients, no-sender and a
starved host (a reading client dropped every 3 s, as the broker was on 09-14) with no ASAN/UBSAN report; the fixed build
survived 10 min at 20/min (203 drops) and 10 min at 10 reading clients × 200/min (17,045 drops, 858 reaps).
**What is left:** the 09-14 exit itself — no FATAL, no `terminate called`, no core, under 21 parallel replays on the
same host — fits a SIGKILL (OOM) better than a stage bug. cf06's kernel log for 2026-09-14 20:50–21:05 settles it
(`sudo journalctl -k --since "2026-09-14 20:50" --until "2026-09-14 21:05"`, or `zgrep -a "Sep 14 20:5"
/var/log/kern.log*.gz | grep -i -E "oom|Killed process"`): an OOM kill closes this as host starvation; nothing there
leaves it open with the harness as the standing check. The fix deploys with a gather restart on the VM, no node cycle.

### #114 — `cudaRFISKtilde` is 6× slower on gpu0 than gpu1 on the same node
**[live]** 12.89 ms vs 2.12 ms per frame on cx43 with identical per-GPU configuration, and it is
the largest non-inject kernel on the busier GPU. Per the standing rule that **nothing is
per-node or per-instance**, an asymmetry like this is a bug until proven otherwise. Investigate
before optimising anything else on the GPU: it is a bigger lever than anything left in the GPU
queue.
**10-13:** `/gpu_profile/<name>` still exists in `gpuProcess`, but the generator's `--n2-debug` (which set
`log_profiling`) was removed as dead, so the check now needs `log_profiling: true` hand-set on one node's config for one
run at the F-engine's return.

### #126 — the F-engine axis is drift-free and ~217 ms stale, and only labels can use it
Filed for years as "the 215 µs has never been decomposed"; **it was decomposed on 2026-08-17**,
a day before the question was written down, and this list never absorbed it: total serve lag
217 ms = ~105 ms window quantisation (`pow_hop` advances in 40960-hop steps) + ~100 ms pipeline
+ 5 ms HTTP, and the 215 µs is that number's sub-millisecond residue. What is open is the fix,
not the measurement. The axis is right for LABELS (a label and its phase are stamped at the same
t, so staleness cancels) and wrong for an EPOCH (where staleness is the whole error) — which is
why substituting it into the ephemeris measured 65× worse and was reverted.
**Fix: serve the CURRENT F-engine hop beside `pow_hop` in the combiner status.** That removes the
quantisation sawtooth and the pipeline term together, and unblocks both `--innov-dr-seeds` and
the ephemeris epoch.

---

## Open — bench or offline, no deployment at all

### #188 — comment pass, stage 8: the Python, scripts and hand-maintained configs still narrate their history (2026-10-13, ON HOLD)
**What [tree]:** stages 5–7 of the comment pass (`800e1e17c..be7d46e8e`, 10-12) covered the C++ only: `lib/stages`,
`lib/cuda`, `tests/boost`, `external/n2k_dual`, `lib/metadata`, `lib/utils` — 128 files; stage 7 took 8772 → 6072 comment
lines and corrected 374 stale ones. Stage 8 was planned and never started (the dead-code audit took its slot). Counted
10-13 after the dead-code merge (`git grep -c '^\s*#'`; "dated" = lines carrying a 2026-MM-DD date, which undercounts:
most history reads "used to", "armed", "reverted", "found by"):

| family | files | comment lines | dated |
|---|---|---|---|
| `config/chord_gnss_node.yaml`, `gnss_fleet_chord.yaml`, `gnss_chains_chord.yaml` | 3 | 3002 | 220 (the chains yaml 166, the worst file in the tree) |
| `config/gen_chord_gnss_config.py` | 1 | 873 | 48 |
| `python/scripts/gnss/gnss_broker/` | 49 | 5730 | 394 |
| `python/scripts/gnss/*.py` (the tools) | 67 | 2372 | 197 |
| `python/scripts/js_viewer/` | 10 | 298 | 37 |
| `scripts/gnss/*.{py,sh}`, `scripts/gnss/site/` | 53 | 1253 | 92 |
| `docs/*.md` | 17 | — | 191 — out of scope unless KV says otherwise: the buglist, runbook and handoffs carry dated state by design |

**How, when it resumes (KV 10-13: on hold — "these audits really chewed through usage limits"):** the stage 5–7 method,
one or two agents at a time. The brief is `~/gnss/builds/commentpass/STYLE.md` with `exemplar.diff` (a comment states an
invariant or a why; dates, "used to", armed/reverted, debugging steps and found-by notes go — git has them; a wrong
comment is corrected, never deleted; a doubt is a finding, not an edit, in the `findings_s7.json` shape). Work on a
branch off `kv/chord-gnss` in its own worktree, never the main checkout (the stack runs from it). Every agent edit is
proved comment-only before it is committed, per language: yaml — `yaml.safe_load` of old and new identical, and
`gen_fleet.py --preview-dir` plus the three instance recipes byte-identical against the pre-edit generator; Python — the
`tokenize` stream minus COMMENT/NL tokens identical, docstrings diffed and reviewed separately (they are the functional
text), then `venv-ft/bin/python -m pyflakes` and `/opt/kotekan_env/bin/black --check` (CI's black); shell —
comment-stripped text identical plus `bash -n`. Order: the three configs and the generator (what an operator reads
first), then `gnss_broker/`, then the tools, scripts and viewer. Exclude files owned by open PRs (#1675, #1750, #1751,
#1752). Afterwards: the cf05 `vpy.sh`-style checks (broker tests, the seven `broker_equiv` replay digests identical,
`gen_fleet --check`, obs/viewer argv parse), `stack_contract_gate.sh`, then KV restarts the Python services (the obs
writers wait for the anchor since `a74c3c75e`, so a restart during an outage is safe).

### #157 — `test_buffer_peek` fails its `peeks_won > 0` check under ThreadSanitizer (flaky under slowdown; found 2026-09-30)
- **What:** under TSan's slowdown the worker thread wins every iteration, so `BOOST_CHECK(peeks_won > 0)`
  (tests/boost/test_buffer_peek.cpp:430, and the similar check near :490) fails. It failed 6/11 runs on #1714's commit and
  5/11 on develop. It passes without TSan, and TSan reports no race.
- **Why it matters:** it blocks running the buffer tests under TSan in CI. #153 showed that TSan is the tool that catches
  this class of bug.
- **Fix:** make the contention deterministic (a barrier so the peek goes first at least once), or check "not vacuous" some
  other way. Upstream test code.

### #56 — transits confirmed; the residual is very likely GNSS we cannot see (see #129)
**[archive + BRDC, 08-22/08-23]** Near-boresight transits are the mechanism, and the amplitude
tracks how much of the tap's band the satellite actually lights. ⚠️ **The tap is NOT L5-only**:
`gnss{0,1}_srch_tap/rf_stats` monitors 22 channels over **1166.8–1282.4 MHz** (L5/E5a/B2a ×7,
E5b/B2b ×7, L2C ×1, B3I ×7, E6 ×6), and it publishes absolute `freq_ids`, so a clipping channel
can be named. Every pass inside 2° of boresight, both days:

| sat | sep | power vs day median | channels it lights |
|---|---|---|---|
| E6 | 0.26° | 4.9× | 20/22 |
| C34 | 1.12° | 4.3× | 21/22 |
| G9 | 1.21° | 3.5× | 8/22 |
| C38 | 1.74° | 3.3× | 21/22 |
| E3 | 1.99° | 2.4° → 2.4× | 20/22 |
| G19 | 0.93° | 2.2× | **1/22** |
| C19 | 1.15° | 2.0× | 21/22 |
| G19 | 1.11° | **1.0× (none)** | **1/22** |

**Seven of eight close passes raised the power 2.0–4.9×. The one that did not is a GPS Block IIR**
— PRN 19 predates L5 and L2C, so of the 22 monitored channels it lights exactly one, legacy L2
P(Y) at 1227.6 MHz, ~6.6 dB below L5. So "a satellite near boresight rails the array" holds for
every satellite with real in-band content (7/7); the exception is the one class with almost none.
⚠️ A first pass at this filtered satellites by *L5* capability and so wrongly excluded G19
altogether. The correct model is "how many of the 22 monitored channels does it transmit in".

**What is left is the residual: bursts of comparable amplitude with no satellite within 5°**
(08-23 15:00 is the day's largest at 58.7 with the closest satellite 10.3° away; also 16:30,
22:20; 08-22 17:40, 11:10, 15:00, 19:30, 16:30). Cross-correlating consecutive days cannot
separate a solar from a sidereal repeat — the events are 12–20 min wide and the daily shift is
only 3.93 min, so lag 0 (r 0.51) and lag −4 min (r 0.47) are indistinguishable.
**The hypothesis under test is aeronautical.** 1166–1282 MHz sits inside the ARNS band
(960–1215 MHz) shared with DME/TACAN and Link-16, whose emitters are orders of magnitude
stronger than a GNSS satellite and need no boresight transit to swamp a sidelobe; aircraft
schedules repeat near-daily with tens of minutes of jitter, which fits the archive better than
either a solar or a sidereal repeat. **The discriminator is spectral, not geometric**: a GNSS
transit spreads clip across the whole comb, DME occupies a single 1 MHz channel, and the tap
gives per-channel clip with named frequencies. `fixtures/rail_aircraft.py` records taps +
OpenSky aircraft + BRDC illuminators on one clock (it is also `rail_watch`'s replacement).
⚠️ Concentration is only meaningful during a burst: at the ~5e-4 baseline clip a single channel
holding the maximum is Poisson noise on a handful of samples, not a narrowband source.

**[measured 2026-09-10, 180 samples over 3 h]** The recorder ran and the discriminator answered —
**it is not DME, and it is not a transit.** Six bursts, clip median 1.83e-04 → max 6.38e-02.
Mapped to real frequencies the repeated feature is a **flat-topped plateau ~12–15 MHz wide centred
~1266–1268 MHz** (19:13:41, cx19.0: 1260.2→1275.8 MHz at 0.017/0.054/0.047/0.046/0.043/0.011); one
event also lit 1199–1206 MHz. **L5/E5a (1166–1185) and L2C (1227.5) sat at the noise floor
throughout.** DME is excluded twice over: ~1 MHz wide, and confined to 960–1215 MHz. The width and
flat top match **B3I (1268.52 MHz)** — but the nearest BeiDou at the three worst bursts is
**18.7°, 21.3°, 23.8°** against a 2.48° FWHM, and that separation *grows* while the clip stays
high. So it is a broadband, band-selective emitter with no satellite within 18°.

⚠️ **TWO ERRORS TO LEARN FROM, both the ORIGINAL #56 error repeated.** (1) The first read of this
data called it DME because adjacent `clip_vec` entries were treated as adjacent in frequency —
**they are 3.1 MHz apart**, the node holding every 16th `freq_id`, so the tap's 22 channels are
four sparse per-band groups, not a comb. `rf_stats` publishes `freq_ids` (absolute) beside `chans`
(local); **always map through it.** (2) `fixtures/rail_aircraft.py`'s `illuminates()` hard-codes
**1176.45 MHz** capability, so its logged "nearest GNSS" scores a band that was dark in every one
of these bursts — scoring by *nearest satellite* rather than *what transmits in the band that lit
up*, which is precisely what produced the first wrong answer for this item.

**[2026-09-11] THE ANSWER: WE CANNOT SEE THE BIRDS THAT WOULD DO THIS.** Both lit features land
exactly on constellations the catalogue is blind to, confirmed against our own signal table:

| observed | `gnssSignal.hpp` | why we could not see the source |
|---|---|---|
| peak **1202.5 MHz** | `GLO_L3OC` **1202.025** MHz, 10.23 Mcps | **no GLONASS anywhere** |
| plateau **1260.2–1275.8 MHz** | `BDS_B3I` **1268.52** MHz → null-to-null **1258.3–1278.8** | **BDS-3 MEO C43–C46 invisible** |

Three independent blind spots, any one of which is sufficient:
1. **`gnss_ephemeris.parse_rinex_nav` is "G/E/C only"** by its own docstring. Today's
   `BRDC00WRD` carries **273 GLONASS records, 22 QZSS, 40 NavIC, 1601 SBAS** — all discarded.
   `glonass_freq_channels()` reads the R records for FDMA channel numbers and never for position,
   so **there is no GLONASS propagator in the tree**: a GLONASS satellite cannot be placed at all.
2. **The BRDC product itself omits BeiDou C05, C15–C18, C43–C46** — not "missing today", absent
   from **every one of the last 12 daily files**. C43–C46 are BDS-3 **MEO**, which do pass
   overhead at 49°N. (Raw-vs-parsed compared: the parser is not dropping them, they are not there.)
3. **The B3I chain searches PRNs 19–42**, so C43–C46 are excluded by configuration even if the
   ephemeris arrived — `/gnss{0,1}_b3i_n2assemble` and `_n2dual/commands`, 24 PRNs, on every node.

⚠️ **CORRECTION, same day: only the GLONASS half survives.** A first pass claimed BDS-3 MEO
C43–C46 were missing birds that could explain the 1268 MHz plateau. They are **SVN numbers, not
PRNs** — IGS SINEX lists 40 currently-assigned BeiDou PRNs and BRDC carries 39 of them, so BeiDou
coverage is essentially complete and no untracked BeiDou explains the plateau. What stands is the
**1202.5 MHz** feature against `GLO_L3OC` at **1202.025 MHz**, where we have no positions at all.

**So the 1268 MHz plateau is still unexplained**, and with the catalogue ruled out the **sidelobe
hypothesis moves to the front**: the B3I chain was tracking strong satellites throughout both
bursts (PRN 42 at A 222 during the 18:26 burst, PRN 34 at A 225 during 19:13) with the nearest
BeiDou at 18.7–23.8°, which is sidelobe territory for these dishes. **Next test:** does burst
amplitude track a strong B3I satellite's *sidelobe angle* rather than its boresight separation?
Catalogue work is tracked as #129.

⚠️ Limits: OpenSky serves live only without credentials, so the archived bins cannot be
attributed to specific flights; Celestrak is unreachable from cf06; and GEO is geometrically
impossible at el 81° from 49°N, which rules out the whole geostationary belt.

### #94 — the shared-parameter estimators: 1 of 6 sites done
S2 (the prior gauge) is built, armed, and its falsifier now passes on sky (closed file). The
programme around it did not move: **S1** (the circular-median clock, whose docstring still names
itself "THE DECAY ROOT") was supposed to be demoted to a cross-check once S2 landed and was not —
legacy still owns the segment — and **S3–S6**, which are "state the membership-invariance property
and write the test", are not started. There is no gauge/membership test among the 20 broker test
files. Separately and not evidence against the gauge: the standing `SEED-OFFSET joint-vs-legacy`
control reports **39% refusals** (936 of 2371) and nobody reads it.

**10-13 (`ff13934c8`): S1 demoted; the S2 gauge's membership invariance is tested; S3–S6 are not.** S2 is what runs:
every `joint_receiver()` constructs the prior gauge and seven of eight chains run `joint-consume: clk`, so
`dr_joint_clk` sets the seeds' clock level from the joint state; the circular median keeps the long-code period, is the
fallback on a refused joint clock, and gps_l5 (`rate,slew`) alone still flies it by manifest choice. `deadreckon.py`
says so, and the 30-s `dead-reckon clock` line carries `(median: a cross-check, the adopted joint clock is the level;
...)` or `(median: the level on this chain; ...)` per chain — prefix and the `drift ... integrity:` tail unchanged for
`joint_latch_detect.py`. `gnss_broker/test_gauge_churn.py` states the property with the median gauge as control: a
join or leave moves the prior gauge's clk < 0.06 chips in the following 90 s (median gauge 0.82 / 1.22), clk + b_i holds
to 0.17 for every stayer, clk_rate < 0.001, and `gauge_mode="prior"` at all seven call sites. The S3 (combiner NH
median), S4 (F-engine axis max over rows) and S5/S6 sites are population statistics with no level to pin; a churn test
for them needs a different property (a bounded step per membership change) and is still open. The "39% refusals" above
is not the SEED-OFFSET line (that line has no refusal in it; JOINT-CLK REFUSED ran ~31% on 10-12): re-attribute or drop.

### #106 — the establishment excursion, and the one instrument that was never built
The withdrawn freeze is correctly fenced (default 0.0, absent from the yaml). The cause is still
unexplained and **[live]** still reproduces: after today's 13:36 broker restart the gps_l5
readback walked 0.470 → 1.780 → 2.272 → 2.502 → 2.789 → **3.000 (the clamp) at t+5.5 min**, and
recovered by t+7 min. What is missing is a number: the **pre-clamp** common-mode trim demand.
Nothing exposes it — the clamp is applied inside the loop and only a rail counter escapes.
**Fix: expose the pre-clamp demand per PRN in the FleetDll stat line, then read one restart.**
Minutes, not a soak.

⚠️ **Judge it on the right disturbance.** A *fleet-wide* node roll is the hardest reproduction
(~30 min, clamp-pinned); a *single* node roll costs nothing measurable and cannot be used to
provoke it — see the closed file for the measurement, and the standing traps for the operational
rule. The other staleness note stands: the lobe-coherent combine made the discriminator ~1.5×
steeper without the loop constants being retuned, which is the most likely reason the excursion
still reaches the clamp at all.

### #97 residual — the source defect fires at its original rate
**[live]** 44 `SOURCE PERIOD DISAGREES` in 57 minutes (~46/h, against the pre-fix 37–49/h), at
snr 531 with a 0.0-chip within-period residual — so the search's period label is wrong on strong,
clean detections. The command side is fixed (the label is now a fleet consensus, so no single
satellite moves its own seed) and the node-side consensus cut the ±1 class, but not ±3/±6.
**Next: the per-detection `(nh, cp_long, snr)` stream against injection.** A node-side dig.

## Open — blocked upstream



### #165 — recv1's ConfigTracker exits when a sender reconnects with a different binary (2026-10-10)
**What [live 10-10/11]:** `ConfigInfo` carries the sender's version, commit and cmake settings; when a node
reconnects on a new binary the receiver FATALs on the mismatch, so every binary rollout costs recv1 a restart and
about a minute of N² gap — and every recv1 restart is a #164 exposure.
**Fix (chord):** compare the fields that describe the stream, not the build. Ours: none.

### #164 — recv1 locks itself out through the bad-feed mask after a restart (2026-10-10)
**What [live 10-10 15:46Z, ~4.5 h lost; 10-11 00:57Z, cleared 01:15Z]:** right after a recv1 restart the newest subset
file holds no valid rows; bffs's power-outlier test sees `n_measured 0`, flags all 32 inputs, the bad list grows to
113, the RFI mask goes to 100% and no valid row is written again: self-sustaining. Clearing it is a manual procedure
(hold the cron, reset the list, restart; the 10-10 chord-merge memory note has it).
**Standing check after any recv1 restart:** subset files ~3.1 GB = valid, ~350 MB = empty; `/var/lib/choco/bffs/run.json`
power-outlier `n_measured` 32. **Fix (Jim):** bffs must keep the previous list when `n_measured == 0`. Ours: a line
in `chain_health_cron` that reads the newest subset file's size.
**10-12, the F-engine outage:** a dark F-engine is the same trigger. Our cron is held from 16:41Z (`#HOLD-fengine-off-
20261012` in the crontab, backup `~/gnss/builds/crontab_20261012_before_fengine_off_hold.bak`), so our six stay on the
85-input list; restore it at the return once bffs posts a list from valid data. cx47/cx52 still take bffs's list
through choco and may lock out until Jim's fix is in.

## Open — measured, real, and nobody's lever



### #169 — node control-plane errors since the 10-11 00:55Z restart (2026-10-11)
**What [live cx43, 00:56–02:46Z]:** 20 `Could not read FPGA controller … GET …/config failed`, 20 `Request failed with
socket error`, 42 `trim EXPIRED with no /set_trim`, 134 `NIC dropped N packets … rx ring was full` (29671/16850 at
start-up, ~1 packet/s since; `/metrics` `nic_rx_missed` +421/+171 in 6.5 min). The previous 13-h lifetime on
04122d179 had 0/0/3/17. The DPDK lcores (5, 6, 7, 21, 22, 23) share no core with any stage. Confounds: the window-8
binary (0011cf3b1), the #164 lockout and recv1 restart at 00:57–01:15Z, the F-engine's state.
**Next:** re-count at the next quiet hour; if the expiries and the trickle persist, A/B one node on 04122d179.
Also one corrupt-sequence packet on cx27 at 10-10 20:18:45Z from raw source ID 6, which is not an A, B or RFI board
(formerly #174; one event, noted because #1750 stops kotekan on a capture fault).

### #167 — the 10-07 digital-gain table quantizes pol 1 to ~0.1–0.3 LSB in the GNSS bands (2026-10-10)
**What [archive, recv1 subset autos 10-10 01:59Z vs 10-07]:** table `digitalgain_20261007T213936Z` (live since the
10-08 F-engine restart) replaced the 09-25 GNSS-band cuts: at 1170–1182 / 1200–1214 / 1260–1285 MHz the 4-bit data
sit at X ~0.5, Y 0.12–0.27 LSB rms per component (10-07: 0.7–0.9 both); L2 (1227) and 700 MHz are normal (~1.6–1.9).
Y mostly quantizes to zero: a heavy pol-1 quantization loss in exactly the bands we track. Candidate cause of #163.
Not a recv1 fault: the GNSS nodes see the same voltages. **Lever:** site — restore the 09-25 GNSS-band settings or
raise Y ~3–4×. Ours: none. Memory note `chord-fengine-gains-20261007-gnss-bands` has the per-band numbers.

### #166 — the F-engine is not sending rows C and D (2026-10-10)
**What [live 10-10 11:48–12:0xZ]:** with C/D enabled in the X-engine config (chord 13e442ac6) every frame reported
100% packet loss: packet rate 1.73 M/s against 1.82 with C/D marked missing (ten boards would be ~1.7×); recv1 wrote
fill, GNSS rail power 0 on every channel. Rolled back (e635d6619). Blocks the 32-dish / 64-input GNSS plan.
**Lesson:** before enabling dishes in the X-engine config, check the F-engine sends their boards (packet rate vs board
count). **Lever:** site (fpga_master).

### #163 — E6 peak sharpness fell 3.65 → 2.4 at the 10-07 F-engine restart while every other chain's rose (2026-10-08)
Broker DLL q (2P/(E+L)) on strong satellites: gal_e6 3.5–4.5 for three weeks, 2.3–2.5 from the first post-restart data
(10-08 00:27Z); over the same step gal_e5a 4.1 → 5.5, e5b 3.4 → 4.5, b3i 4.3 → 5.8, l5 4.0 → 5.3. E6 stays locked
(disc ~0, prompt power 100–800× the floor on all 12 instances, hours-long ADR arcs), but sits near the viewer's
2.2 lock line. Every node and GPU shows it in its own combiner. Not our code or config: cx19 on the pre-#1750
binary (15:28Z) read the same. The F-engine came back with new gains (rail fraction 2.5× lower in every band),
and dish work was under way that week, so the suspects are per-channel gain phases across 1278–1284 MHz or the
dishes. A separate E6 cross-instance coherence loss from ~12Z 10-08 was #154's stale reference (fixed 10-09).
**10-10: candidate cause = the 10-07 gain table, #167** (pol 1 at 0.12–0.27 LSB rms in the GNSS bands). Check the
table before anything else. Next, when it settles: per-channel prompt phase across the E6 lobe
from the gather stream (ragged = gains, smooth = delay), or the new gain table.
**10-11: CAUSE FOUND — the channel weights ARE the gain table.** The combiner weights each channel by the
F-engine digital gain: per-channel prompt amplitude against |g| has log-log slope 0.9–1.1 (r 0.91–0.98) on e6,
e5a and e5b over all 12 instances, and nothing in the node normalises a channel by its noise. The sharpness of a
3-channel (E6) or 5–7-channel (10.23 Mcps) comb correlation is set by the outer channels' weight, so q is a
property of the gain profile across the lobe. `digitalgain_20261007T213936Z` (still live at 23:12Z) turned the
E6 lobe from centre-notched to centre-peaked (centre/edge gain 0.60 → 1.59; the low edge at 1273.8 MHz now
0.46 LSB rms on X and 0.18 on Y, i.e. mostly zeros) and deepened the centre notch everywhere else (e5a 0.57 →
0.46, e5b 0.68 → 0.50, b3i 0.59 → 0.38). Less outer weight on E6: q 3.6–3.7 for the ten days before, 2.47–2.49
since (hourly history from the rotated log); more outer weight elsewhere: their q rose. GPU1's E6 instances read
0.1–0.5 above GPU0's because their comb phases land on higher-gain channels. Not phases: within an instance the
channels agree to ~10°. It bites: the broker's 2.2 gates (`dll_quality_min`, `q_stall_bar`) now sit inside E6's
working range. **Levers:** site — a flat or 09-25-style E6 profile and ≥3× on Y; ours — normalise each channel by
its noise rms ahead of the despread so q no longer depends on the table (node change, every chain at once,
judged in minutes); display — a per-band line, or one relative to the chain's noise-q floor (E6: floor 1.36,
bright 2.5). Recipe in memory `chord-e6-q-drop-20261008`.
**10-12: NOISE NORMALISATION LIVE (a79daf533 + af94d62db, cycles 00:10Z and 00:23Z).** `chan_noise_norm`: the
assembler weights each covering channel by 1/rms of its own measured power before every cross-channel sum (record
E/P/L, element blocks, the comb block), mean 1 over the frame's channels; the power is GnssChordVoltageTap's 1-s
band-power pass, per (channel, input), averaged over the beam's own 32 inputs (`chan_noise_norm_elems`, the
correlator's live tiles; the first cut averaged all 128 and the RFI antennas diluted it to ±10%). E6 weights now
1.1–1.35 / 0.76–0.90 / ~1.0 across each instance's three channels. Per-instance q medians before → after: e6 2.40 →
3.03 (fleet DLL 2.8–3.1 on every on-peak PRN), e5a 5.26 → 4.17, e5b 4.29 → 3.75, b3i 6.19 → 4.84, l5 4.0–5.0; R
1.00/1.00 every band, 12 live, 0 CHAIN DIED, C/N0 unchanged. The amplitude-vs-gain slope does NOT move (the spectrum
and taps endpoints are energy-normalised per channel, blind to a per-channel scalar): judge by q, not by it. The rest
of the gap to 3.65 is the table's LEVEL, not its shape: Y at 0.18 lsb rms on the low edge is quantised away and no
weight brings it back — that part waits on the F-engine gains KV has asked for. `/get_elem_cal` `chan_norm`
reports on/applied/miss/w.

### #147 — 2026-09-26 22:01Z: a fleet-wide ~3 ppb reference-frequency step
All 8 chains and every instance: a carrier offset that scales with carrier frequency (E6/E5a 1.09 vs 1.087
expected), a −37 m (−127 ns) code step, carrier residual ±2.5 Hz for >14 min and beam-cube 1-s coherence
0.90 → 0.03 for ~2 h, with no lock loss and no broker event; gps_l5's code-rate clock estimate moved
−0.001 → +0.002..+0.003 ppm (also ~3 ppb). Maser/GPSDO or F-engine reference? Not ours to fix, but it
must be vetoed from any coherence statistic. Evidence `fixtures/projtest/forensics/l5/ev2201*.py`.

### #122 — the readback carries a chain-common offset on the non-primary bands
**[live]** per-PRN means: bds_b2b −0.42, bds_b3i −0.40, gal_e6 +0.29, gal_e5a +0.20, gal_e5b
+0.16, against gps_l2c −0.01 and bds_b2a +0.08. A per-chain constant the loop is not closing.
b3i and e6 are, for the first time, two fresh bands to test the standing BeiDou offset against.

### #124 — the joint state is band-scoped, but its safety labels are per-chain
**[tree]** `band_id = "%.2fMHz" % (carrier_hz/1e6)`, so **gps_l5, gal_e5a and bds_b2a share one
`1176.45MHz` joint state** — the one gal_e5a feeds. The withdrawal "never `joint-consume: slew`
on a feeding chain" is enforced per chain, so adding a feed to gps_l5 or bds_b2a would silently
close exactly the loop that withdrawal forbids. Absorbs the old A4 (DR chains have never consumed
`slew`): the right experiment is on the bds_b2b/b3i twin, which shares no state with a feeder.

### Smaller, still true
- **#14** acquire declines a blind Doppler grid: only a boost test; unverified.
- **#37** GPS block map vs sky: the registry exists; the join to the beam maps does not.
- **#40** the deep fold's rate search wrong-bins onto overlay sidebands — **demoted**: the coarse
  feed is now only the re-acquisition path. The fix is to retire the consumer (use phase
  accumulation for re-acquisition too), which needs a slip-recovery path first.
- **#75** gal/bds have no independent clock: **[tree]** seven chains adopt (both Galileo chains
  too, not just BeiDou — the old status line was wrong), bounded at 5.0 chips. PVT now solves 8
  independent per-chain clocks and none of them is consumed.
- **Record-format hygiene**: UTC is recomputed into every PRN row instead of the frame header;
  the combiner reads the prompt by raw index (`rec[3]`, `rec[4]`…) in C++ *and* Python. Do the
  index normalisation before anyone reorders a slot.

---

## Standing traps — the ones that have cost the most

- **Verify the tree, not the summary.** Every mark here names a grep, a commit or a live number.
  `[verified]` with no named check is indistinguishable from `[carried]` a week later.
- **Delete stale measurements; do not date them and move on.** A table with a date on it still
  gets read as the answer. This pass deleted several rather than re-dating them.
- **A fix deployed onto a restart transient will look like it worked** — and one reverted on a
  restart transient will look like it was wrong. Judge a broker restart at t+10 min, and separate
  the sky (setting satellites) first.
- **Stopping the broker stops the viewer, and starting the broker does not bring it back.** The viewer unit is
  `PartOf=gnss-broker.service`; on 10-12 a flag-day stop left it down for two hours until KV hit dead ports. After any
  broker stop, start `gnss-viewer` by name (runbook).
- **A running process hides a deleted library until its next restart.** Every deployed binary
  here has a RUNPATH into an NFS build dir (`readelf -d <binary> | grep RUNPATH`); the gather's
  pointed at `~/gnss/builds/gather-2d852b79e`, deleted as cruft on 10-12 after checking only process
  argv, and the gather died at its next restart with the fleet blind at the broker for 6 min.
  Before deleting a build dir, list what every live kotekan has mapped (`/proc/<pid>/maps`).
- **Stagger node rolls: one at a time costs nothing, all six costs ~30 minutes.** The standing
  C++ trim lives on each node, so a single roll discards a twelfth of the fleet's trim state and
  ten combiners carry the measurement — nothing moves. Roll them together and every satellite's
  trim restarts from zero, the loop pins at the clamp for ~6 min and takes ~30 to settle.
- **A pipeline fault scales with a command; a physical cause scales with a physical variable.**
  Regress per-satellite rates against Doppler, Doppler rate and elevation before believing any
  new observable.
- **Nothing is per-node or per-instance.** A per-instance quantity is a bug (this cost the
  frame-boundary carrier phase five weeks), and a per-GPU asymmetry is a bug until proven
  otherwise (#114).
- **An artifact nobody regenerates is a souvenir**, and an instrument nobody owns is dead within
  a fortnight (#56's `rail_watch`, #120's missing units).
- **A gate vouches for what its fixture runs and nothing else.** #108's fix rode the fleet for
  12 hours before anything tested it; the only prior evidence was an offline harness.
