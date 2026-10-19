# CHORD GNSS — closed items

The durable record of what was fixed, so `CHORD_BUGLIST.md` can hold only what is open.
One line per item: **what it was, what closed it, how you would check.** Commit hashes are the
addressable half and are kept; **line numbers are deliberately stripped** — at the 2026-09-10
reconcile 6 of 12 citations in the old closed index pointed at code that had moved, and a wrong
address is worse than none. The full narratives, including every retired hypothesis, are in this
file's git history (`git log -p docs/CHORD_BUGLIST.md`).

Reconciled 2026-09-12 against HEAD `b45949144`, 57 commits after the 2026-09-10 pass (which was
itself 496 commits after 2026-08-22, by seven parallel section audits that each read the tree AND
the live fleet). Added since: #54, #65, #93/GAP 3, #111, #116, #117, #118, #128. **2026-10-11 pass** (HEAD `0011cf3b1`):
#34, #55, #102, #107 residual, #112, #113, #121, #125, #130/#153, #133, #144, #146 residual, #149, #150, #159, #160, #161, #162, #172, #173; moot #11, #31, #88, #123, #174.

⚠️ **Three of those closed on a MEASUREMENT rather than a fix**, and their write-ups below carry
the sensitivity as well as the result — #93 in particular is a *null*, and a null is only worth
what its lever check is worth. Do not cite them as "no effect" without the confidence interval
that goes with it.

---

## Closed by a fix, verified

| # | what it was | closed by | check |
|---|---|---|---|
| #33 | the vector tracker | closed as built 2026-08-30 | `rrate-state`, `rrate-kcoh-feed` armed fleet-wide; `rrate-command` on 3 chains |
| #35 | walkoff/trim links 3–7 | `d85644b7b` | railed fraction 3.6% → 0.03%, max \|trim\| 0.61 of a 3.0 clamp |
| #36 | runtime PRN lists | — | `POST <ep>/set_prns`, scheduled `at_seq`/`at_hop` in `prnmap.py` |
| #38 | visibility-matrix export | `b9cefdc26` | `scripts/gnss/viscap*.py`; 103 GB captured 09-07, solved the element positions |
| #46 | instance record-time divergence | frame-synced transport | live `TELEM … spread 0` on 7 of 8 chains, 0 gaps on 7.3 M frames |
| #57 | C/N0 estimators | — | `PROMPT-CN0` / `KCOH` lines live on every chain |
| #79 | the presence latch / deep gate from search | — | `dll-deep-gate-from-search: 100` on gps_l5 |
| #83 | the seed object | `eb30892b3..68a8b8eea` | — |
| #84 | spec window pin | — | — |
| #92 | the two wipe classes | — | D3 wired; 0 REBASE-WIPE / 0 BARE-WIPE in an hour of healthy fleet |
| #96 | gps_l5 trim integrates a code-rate error | its own RESOLUTION, then `e843a5082` | `cp-rate-model-primary: 1` — the fitted rate is no longer a command on any chain |
| #97 | one detection rewrites a period (command side) | `ffb5ce600`+`1041061f1`+`94b67887d` | `nh-joint: apply`; whole-period steps 74/h → 0 |
| #98 | a log-only monitor ate the hold chain for 5 weeks | — | monitor/escape/hold/release verified at one indentation depth, pinned by an ast test; on sky, `hold_age` mean 97.8 s with integrity live |
| #99 | dead-reckon integrity residuals ±5 chips | `2766f9bf8`→`c6f67c885`→`54829c5a0`→`82dcaeedb` | station position was 155 m off; live residuals +1.3/+1.4 chips, PVT resid_rms 2.2 m |
| #100 | wrap-poisoned cp-fit history | `3266ff905`, armed `ee77f3dea` | `--fit-flush-on-reject 3` — but see #119, its revert trigger is tripped |
| #103 | GPS 21× lock churn onto the 3.27-chip comb lobe | `model-primacy-max: 32` + comb geometry pinned | TRACK-vs-MODEL 1027/13 h → 38/h |
| #104 | cross-chain clock adoption had no plausibility bound | — | `dr-clock-adopt-max-chips: 5.0`; live 2 adoptions, 0 refusals |
| #105 | fleet-common q-crash bursts | 3 fixes, 2026-08-31 | `seed-bias-source: slow`; the drift-EMA monitor line is live |
| #107 | nine node deaths: a post-publication patch of ring slot-0 metadata | `ba3f69c02` + `b3666a3ad` | in the running binary; 0 DESYNC fleet-wide past every historical death point |
| #108 | every node wedged at 2^33 bf-mask frames (~15 h) | `e92573330` + `b80170e95` | **verified on sky 2026-09-10 16:11 UTC** — ALL TWELVE valve instances (6 nodes x 2 GPUs) past 2^33, leader at 110.7% (~1.4 h beyond the wedge point), 89/89 senders live and 0 stale throughout. First crossing since the fix; the fleet had wedged there every time before, and the fix had ridden the fleet 12 h untested |
| #111 | the Earthdata token failed silently and DCB had no fallback | `6fe68f1b7` | the silent part was that `fetch_dcb` checks the cache BEFORE the network, so a dead token serves an ageing product for 14 days and only then returns None. Now: a status dict (`ok`/`no-token`/`auth-rejected`/`unreachable`), product age from the filename epoch, token expiry from the JWT claim, a WARN against `--dcb-max-age-days` (default 10) and `--dcb-require` to make it fatal per chain |
| #110 | arming the beam cube segfaulted every node | `633a59a5f` + `3173ee067` | cube frames landing; no node death in 12.4 h |
| #22 | fp16 Φ tables | `f0b1ca2b6`→`477c06eca`→`87173351c` | **armed in production** — `phi_fp16: true` ×15 per node, `fp16 Phi ARMED` in the node log |
| #24 | noise-debias for beam-map values | `ca47ba74a`, `c343eadc8` | cube cells are in pedestal units; `--elem-norm` fits residual gain |
| #45 | transport hardening | — | — |
| #51 | fast-trim split | — | — |
| #59 | frame-synced gather | — | the telemetry transport this list now depends on |
| #71 | carrier NCO long lever | reverted deliberately | `carrier-gain: 0.0` |
| #86 | rate lock / E5a no-op | — | — |
| #90 | off-peak disarm latch | admission gate | 0 LATCH in an hour |
| #95 residual | "the DLL cannot arm either" | `c4512de93` | the peer-median window gate was deleted; absolute floor is the only path, `presence-admit-displaced` on all 7 DR chains |
| #127 | the observables record was filed under the day the LOGGER STARTED, not the day the rows were measured | `461eb8ce0` | `--out` is strftime-expanded per row against the row's own epoch. `test_obs_day_roll.py` (8 tests; 6 fail against the old one-shot writer). On sky: a running logger's filename changes at 00:00 UTC with no restart. Found 2026-09-10 — `gps_l5_20260905.jsonl` held four days and 2.0 GB while 09-06..09-08 had no file at all, and the same shape runs back through August. Nothing failed: the rows were right, the name lied, and a consumer opening a day by name read an empty day. Both the `--out` help and `obs_up.sh` already promised the roll. **Archive re-filed 2026-09-10**: 13.65 M rows redistributed across 09-05..09-10 (0 wrong-day rows on a full re-scan), pre-September deleted by KV's call (15 files, 5.1 GB, see `fixtures/obs/DELETED_pre_september_obs.txt`). ⚠️ The re-file cost 183 rows of 3.35 M (0.005%) in the 09-10 files: the recovered rows were appended to files a live logger held open, and **O_APPEND is not atomic on NFS**, so ~91 writes interleaved. Quarantined in `<chain>_20260910.corrupt.txt`. Stop the writer before merging into its file — the append-atomicity that makes this safe locally does not exist over NFS |
| #54 | GPU vs CPU replicas differ per sample | see below | prompt 3.48e-02 → 4.79e-05, peel per-sample 9.55e-02 → 3.31e-05 at 6.8 d; every exactness gate still exactly 0 |
| #65 | three stack scripts truncated their log on start | `f6f7afd61` | rotate-then-start in `agg_up.sh`/`gather_up.sh`/`broker_restart.sh`, 3 rotations kept; verified in tree at `agg_up.sh:70-77` |
| #128 origin | the acquire refine test was red, and had been since #105 | see the #54 section | the test gridded Doppler at 0.1 bin, outside the estimator's domain; bin-spaced now, refine recovers 0.294 of a true 0.300 bin |
| #128 | the Doppler refine's bin precondition was enforced on one path of two | see the #54 section | guard hoisted out of the `_cuda_acq` branch so the CPU path warns too; generator emits the bin-aligned 62.5 Hz unconditionally |
| #93 / GAP 3 | carrier-aided code loop: is there an aiding target at all | closed by measurement | 24 h × 8 chains, 7013 window-sat pairs: at the 0.06 chips/min that motivated it the aiding gain is **−0.062, 95% CI [−0.209, +0.063]** against the ±1 physics requires |
| #116 | the observables writers never rolled the UTC day | `461eb8ce0` | **deployed and verified on sky**: `gps_l5_20260910.jsonl` ends at Sep 11 00:00:00 and `_20260911.jsonl` takes over; 8 tests, 6 of which fail against the old writer |
| #117 | cross-chain grid pairing yielded 40%, and the loss was sampling | see below | `GRID_KEEP=4` snapshots published as `fadr_g_hist` and consumed by `gnss_tec_chord.py`; 7 tests, and **verified on sky**: per-chain coverage 50.0% → 99.8%, pair coverage 8.1% → 99.6% |
| #118 | `dop_rate_rejected` was never cleared | — | one line, mirroring `cp_rate_rejected`. **Verified live**: rejections-per-cycle now bounded and fluctuating (1–7, up and down) with fresh values each cycle, where the bug could only grow |
| #141 | gps_l5's carrier seeds rode the solved clock-frequency bias, a 2-sat median that snapped −23..+16 Hz after restarts and transits, so it was the residual carrier on every L5 PRN (fleet 1-s coherence 0.03) | `01964948a` (`seed-bias-source: zero`: seeds at 0 Hz, the solved bias steers only the search and the probes) + `987d9efcb` (the first-seed guard kept under `zero`; the first cut killed gps_l5 on cycle 1) | **live**: L5 common carrier −0.05..0.00 Hz, sinc² 0.96–1.00 at 09-28 16:26Z (was −1.14 Hz / 0.02); over 09-29 06:00–12:54Z median \|common\| 0.03 Hz on gps_l5, gal_e5a and bds_b2a alike, 0 of 415 minutes past 0.5 Hz, median sinc² 0.99; L5 against the solved bias slope +0.002, r 0.05 (was 0.975, r 0.943). `fixtures/fix0928/141/verify141.py`, `snap141.py` |
| #142 | the cross-band clock bootstrap copied gps_l5's legacy clock (a median of as few as 2 detections, which stepped 250–690 chips in passes) into five chains with no step limit; their trackers lost lock 30–110 s after each step | `ea9129b60`: ≥ 4 sats (`dr-update-min-sats`), a 5-chip step limit with a re-pin after 3 agreeing solves, the clock frozen at zero rate while the transit sky has anything inside 5°, the cross-band bound in time units, joint-clock adoption resetting #104's 300-s escape, refusals hold | **live** 09-28 20:18Z → 09-29 12:40Z: 7 transits (G9, E28, G19, C39, G12, E11, C34), each one FROZEN and one CLEAR line (742–1408 s); held clock 149.3 → 150.3 chips over the night; 0 RE-PIN, FORCING or REFUSED; legacy within 5 chips of JOINT[shadow] in 1969 of 1969 samples; thin-sky holds 98 min, longest run 17 min (revert bar 30). Consumers' present counts moved only on each transit's own bands, apart from single-sample PRESENCE UNANCHORED dips while a noise probe re-seeds, and were back at baseline by CLEAR + 15 min. `fixtures/fix0928/142/tools/verify_after_restart.sh`. Residual: open list |
| #143 | the joint-clock consumer compared the joint clock (10.23-Mcps chips) with the chain's own in the chain's chips, so gps_l2c (+142.7) and gal_e6 (+75) refused it every time | `01964948a` (the joint declared in its feeders' unit; `joint_clk_delta` converts and wraps in the consumer's chips; the band-level SEED-OFFSET line and the rate consumer convert too) | **live** 09-28 20:18Z → 09-29 12:40Z: all seven consumers ADOPTED on 1968 of 1969 JOINT-CLK lines (the one refusal each is the start-up line). ⚠️ The filing's "it alone would have protected L2C and E6" was wrong: in a #142 step the converted delta is 250–690 chips against a 5-chip bound, so they refused exactly when e5b, b2b and b3i did. #142 is what protects them. The per-PRN SEED-OFFSET line, which the first fix missed (+142.8 / +75.7 REFUSED on every l2c / e6 line, diagnostic only), followed in `8a2b4df34` (in the tree; live at the next broker restart) |
| #145 | ElemCal calibrated the reference element against the others' weighted SUM, so \|w_ref\| ≈ (n−1)·SNR × the others' median and the per-PRN combine was the reference element alone | `978561481` (a leave-it-out MEAN: divide by W_rest; shared_hold's no-model fallback no longer installs the shadow while frozen), fleet-wide in the node bundle d53b6a254 09-28 18:33–20:39Z | synthetic array-gain efficiency 0.17 → 0.97 at SNR 1; viscap replay: projection prevents capture only with it (G09 at 1–2°: −8.7 → +1.1 dB against the held model); **live**: served `sim` 0.46–0.86 (was 0.02–0.23), shared-model R 0.94–1.00 and fleet xcoh 0.86–0.99 after the 20:39Z cycle. Residual (rho² > 0.99 drop): open list |
| #146 | a FatalError message containing a brace went back to fmt as a format string, so fmt threw inside FATAL_ERROR and the process died on SIGABRT instead of shutting down (recv1 09-27; 46 of 917 sites can carry braces) | `35d7120d0` (format once with `vformat_to_n`), shipped with the #1699 gpuProcess commits `1a739bdd0` `a05fd0a5d` `eafb60f9b` | **bench**: the Boost case fails on the old code and passes on the new (`fixtures/upstream0928/`); in the node bundle since 09-28. ⚠️ That bundle's abort rate is #153 (#1699's lifetime changes are its first suspect). Upstream: open list |
| #151 | the dead-reckon (seeding) ephemeris reloaded every 7200 s while the sky took 900 s, so BeiDou records expired ~10 min before each reload (09-28 11:00Z 9 → 3 PRNs, 13:00Z 12 → 6) | `b2223cc7a` (`_DR_EPH_REFRESH_S` 900; DCB stays on 7200) | **live** 09-28 20:18Z → 09-29 12:40Z: 66 reloads per chain at 15 min, DCB every 2 h; the bds_b2a `active=` minimum over each hour's first 12 min stayed 8–13, and no hour dropped. Residual (two ephemerides; EPH-REBASE inert): open list |
| #102 | per-element steering disarmed behind a loader that could not read the measured table | `d7f6c94b5`, `f77cb41f3`: positions from the arraymap with an epoch key; live on every band since 09-27 | `elem-steer-bands: all`, `elem-steer-sign: 1.0` in `gnss_fleet_chord.yaml`; `--elem-positions-from arraymap` refuses a stale epoch |
| #107 residual | the last publish-then-mutate sites in lib/cuda | upstream, in our tree through the 10-01 develop and 10-09 chord merges | `grep -rl did_set_metadata lib/cuda \| wc -l` = 86; `git diff origin/develop -- lib/cuda/cudaPLMaskExpander.cpp` empty |
| #112 | `set_bf_mask` free-ran at 170–188k frames/s and set #108's fuse | the chord templates: `bufferBadInputs` blocks on `wait_for_empty_frame` | `/buffers` on a node: `set_bf_mask/bf_mask_0` at frame 6 after 1.75 h (cx43, 10-11) |
| #130 / #153 | node processes aborted at random with `malloc(): unaligned tcache chunk detected` | `2aac0bc53` (the applied-mask echo publishes its metadata once) + upstream #1713 (prefetcher) + #1714 (a second `set_metadata` is fatal); see the write-up below | `grep -a -c "malloc()" /tmp/gnss_node.log*` = 0 on all six nodes, one `Kotekan version` line per lifetime; set-once live from the 10-01 build, ~10 days without an abort |
| #133 | the RF-band selector leaked the group key and mislabelled its frequency | 09-17, `BAND_LABEL` shared by the server and the panels; keys `high/mid/low` | the selector reads `Mid · 1227.60–1278.75 MHz` |
| #149 | N² to recv1: our nodes could not satisfy upstream `chord`'s receiver | the stage-1 merged correlator (10-06) and the dish-rows re-base (`2949fecd2`, 10-09): bffs stream, DishInputs subset and frame descriptors from the chord templates | recv1 takes six nodes × three legs; `acq_20261010_115026` |
| #160 | `gnssN_n2dual` spent 3–4 s on its first frame and both ports fell 5–39 frames behind at every start | `bc2d18420` (phasor recurrence, pinned async upload on the inject stream) + `04122d179` (no software fp16 conversion or double copy) | first `n2dual` output 0.4–0.6 s after its port; 0 start-up catch-ups over two fleet cycles (10-10/11); #1750's window is 8 frames |
| #162 | three uncommitted leftovers in the live checkout | `8a9661d55` (node_up GNSS_ENV), `5e97adaec` (beam-cube `--prn-norm`, opt-in), `b4d744517` (telemfmt ignored) | `git status` clean |
| #121 | bds_b2a commanded carrier trim on the 40 Hz default cap and railed there (+39.99 Hz on C24, 10-11 02:35Z) | `carrier-max-hz: 10.0` on bds_b2a, matched to gal_e5a/e5b (yaml, 10-11), live at the 12:58Z broker restart | `JRR-CMD[C]` on bds_b2a stays within ±10 Hz; the commanded-step sign calibration is an experiment, not a defect |
| #159 | the nightly beam-cube build rewrote the live broker's ephemeris store with yesterday's sky | `5174a6d69`: `fetch_brdc` sends any `when` more than an hour from the wall clock to `cache_dir/offline/<day>`, so no offline caller can prune the live store | a 2026-10-09 fetch built its own 892-record store under `offline/20261009` and the live store's mtime did not move |
| #34 | the producer published a forecast `ref_hop` as `pow_hop` (FUTURE lines) | closed as bounded (10-11): both consumers clamp it (`7a4bc0089`, `babd111c7`) and the rate fell from 24/h to ~1/h; the producer half was never worth a node cycle | `grep -c FUTURE` on the broker log: ~1/h, harmless |
| #55 | the `carrier_phase_from_ref` A/B stayed in the kernels long after #52 gave it a winner | `d64a04bfe`: phase from the window reference only; the untaken arm, the switch and its `n_m` removed (−44 lines) | fleet-wide since the 2026-10-11 14:07Z cycle (`gd64a04bfe8` on all six, 0 FATAL, 0 catch-ups); `grep -rn carrier_phase_from_ref lib` empty |
| #150 | C11–C14 are BDS-3 MEOs on re-used BDS-2 PRNs, and the "PRN < 19 = BDS-2" rule in the broker, the lists and `--check-prns` hid them | `6a19f658b`: BeiDou capability from the IGS registry's block types (prnmap, the broker's implied gate, the almanac sky; the floor is the fallback), lists 11–14 + 19–42, the check judges by the registry | lists live at the 10-11 14:07Z cycle, gate at the 14:16Z broker restart: `signal-capability bds_b2a: 2 PRN(s) excluded by the IGS registry (9, 10)`; C11 (el 41°) seeded within 40 s, FLEET-INST align 0.998, 52.8 dB-Hz duty 1.00 on bds_b2a by 14:18Z; C12–C14 follow on their passes. The registry also admits C6–C8 (BDS-3 IGSO, peak 18–22° here): C08 was tracked at 37 dB-Hz through a runtime lane swap, so the start-up lists were the only place they were absent; added 10-11 (KV), 31 slots, gate GREEN on the day's sky |
| #125 | the e5a/e5b A/B pair differed on six non-endpoint flags and three documents claimed otherwise | yaml 10-11 (KV's decision): e5a takes `rrate-phase-feed: true`, loses its joint feed (`joint-model-primary`, `joint-feed-spec`, `joint-shadow`), the #92 handover and the #101 eph-rebase; current-state notes head the e5a, e5b and bds_b2a blocks; live at the 14:41Z broker restart | resolved from `common` + chain blocks the two differ only on `signal`, `trackers`, `combiner`, `spectrum-endpoints`, `dll-combiners`; e5a's commands stay within e5b's envelope (median 0.04 Hz, none over 2 Hz) and JOINT-CLK stays inside ±1 chip of the search clock |
| #113 | `eigencalc: Only 0 of 32 elements are unmasked`, 148× in 12.5 h | closed as benign (10-11): 4 in 1.75 h on the chord templates' mask path, the valve discards the frame and nothing downstream reads it | `grep -c "Only 0 of" /tmp/gnss_node.log` on a node: a handful per day |
| #172 | the VM's service logs grew without rotation (broker.log and aggregator.log 28 GB each) | `scripts/gnss/site/logrotate.conf` + the hourly `:23` crontab line (daily or past 4 GB, copytruncate, 10 gzipped generations), 10-11 | `/var/tmp/gnss-logs/logrotate.log`; `ls -la /var/tmp/gnss-logs` shows `.1.gz` generations and small live files |
| #173 | a node config on disk went stale ~6 days after regeneration and a relaunch then died on its EOP table | `eop_cron.sh` step 4 (10-11): regenerate the on-disk configs when the staged table is newer and `gen_fleet --check` passes | `eop_cron.log` prints `on-disk configs regenerated: table end A -> B` the hour after choco's daily table refresh |
| #144 | the transit sky (shared-model freeze, clock freeze, railing veto) dropped unhealthy and stale-ephemeris satellites, so E14/E18 passes never froze anything | `aef331c10`: `sky.transit_sky` (health gate off, a day of toe validity) read only by the three transit consumers; seeds keep the strict sky byte-identical; the veto line carries the health flag | on a copy of the 10-11 sky the transit sky holds E14 and E18 (unhealthy) that the strict sky drops; watch `BORESIGHT TRANSIT -- E14 ... (health 130)` and `shared_frozen 1` on their next pass inside 2° |
| #161 | gps_l2c's clock latched +10 ms off a cross-band bootstrap from gps_l5 on noise, and the mod-1-ms JOINT-CLK check held it for 8 h | `f29211e30` + `34acbcf90` (established = two residuals inside a chip and none beyond 5; the first rule withheld every donor on a healthy sky): `clockhealth.dr_solve_healthy`; no bootstrap from an unestablished donor; whole-period steps refused without a fresh healthy record; the NH hint learns nothing while unestablished; JOINT-CLK resolved across the chain's own period | `test_clockperiod.py` (16 checks: +10 ms and +1 ms REFUSED, the right millisecond ADOPTED, other chains unchanged); on a healthy sky no `clkxboot-*`/`nhjoint-unhealthy`/`nhhint-unhealthy` lines and gps_l2c's JOINT-CLK unchanged; on a collapse they appear in that order and a whole-period error reads `REFUSED (the clock-mod-20-ms record ...)` |
| #115 | live PRNs dropped past the chain's wire rows | `9bdfab8c2`: rows 24 for b2a/b2b/b3i, 18 for l2c (the 31-slot BeiDou lists ran 18 live against 14: C36/C39/C42 despread on every instance and never shipped); gather buffer regenerated 32752 → 39280 B, gather restarted, nodes cycled 2026-10-12 00:49Z | `DROPPING` 0 on every node; C36/C39/C42 on 12 instances in b2a and b2b; TELEM gaps 0, lag −0.06..−0.11 s unchanged |
| #146 residual | upstream develop still re-parsed the FatalError message | landed upstream as #1753 (merged 2026-10-09, `lib/core/kotekanLogging.*` now identical to ours); our regression test went up as PR #1762 (10-11) | `git diff origin/develop -- lib/core/kotekanLogging.cpp` empty; #1762 adds `test_error_message_keeps_braces` |
| #186 | `seeding.py` appended a detection's `cp_err` to the 9-sample median history twice, the second time unguarded, so a `None` could enter it | `a7f5744d5` (10-13; live since the 21:05Z broker restart) | one append, under `cp_err is not None`, pinned by `test_holdchain` (ast); 22 broker tests + selftest pass; the seven replay digests byte-identical, so no replay had a median-gate decision riding on the doubled sample |
| #183 | `gnss_transit_arcs.py` dropped a transit window that did not fit the span or straddled `--split` without saying so (C33 vanished twice on 10-12) | `0308a4fe5` (10-13) | every dropped window is printed after the kept list as `DROPPED (span: needs <t_lo> .. <t_hi>; split: straddles <split> +15 min)`; the C33 rerun on cf06 lists it dropped for both reasons and the 10-12 census is unchanged; control windows are still skipped silently |
| — | `test_skyscope`'s "a second call while the refresh is in flight starts no second one" check failed intermittently on cf06 (0 of 20 to 2 of 2 runs) | `6e9341aa8` (10-13) | the race was in the test: it read `calls` — appended only by the brdc-refresh daemon thread — before that thread reached the monkeypatched fetch; the test now gates the second call on an Event the fake sets; 20/20 green, `gnss_brdc_supply` unchanged (its second call found the thread alive and started nothing every time) |
| #178 | B3I's G1 cycle was 8189 chips, not the ICD's 8190: the reset was tested after the shift, so the phase 1111111111100 (the 8190th chip, per BDS-SIS-ICD-B3I 4.3) was never output and from chip 8189 every code ran against a G1 one step ahead — 1002 of 10230 chips wrong on all 63 PRNs, a uniform −1.89 dB with no per-satellite signature (gnss-sdr's reading, which the table was transcribed with) | `fdba6b530` (10-14; rides the next node binary) | the ICD text plus a simulation of its register (the phase is reached after 8189 shifts, so an 8190 cycle must output it); two independent generators, GNSS-DSP-tools `b3i.py` and PocketSDR `sdr_code.py`, agree with each other on all 63 codes and with ours only after the fix; `tests/boost/test_beidou_b3i_code` pins the 20 chips straddling the reset and five whole-code fingerprints from the references (the old generator fails from chip 8189); `b1i_b3i_code_check.py` passes with re-derived fingerprints; the sky confirmation is bds_b3i per-PRN C/N0 on the canary against the fleet at the same time, +1.9 dB expected **Sky judge 10-17 14:05–14:40Z (fleet binary 188b6e800, after #194, dishes tipped):** B3I minus its siblings on the same BDS-3 satellites vs 10-11/10-12 (`~/gnss/builds/judge1017/cn0_band_diff.py`): B2b−B3I −3.3 → −6.8 dB, B2a−B3I +4.6 → +0.9 dB (paired on 4 common PRNs: +3.3 / +2.5 dB for B3I), while the control pair B2a−B2b moved +0.75 dB; per-PRN scatter ±2 dB (C29 −8.5). B3I is up 2.5–3.5 dB against its siblings, the predicted +1.9 inside it; the per-satellite verdict waits for the realignment. |
| #188 | stage 8 of the comment pass: the GNSS Python, scripts, site scripts, systemd units, configs and our viewer files narrated their history in comments and docstrings | `c65c50ceb` `a73fc3411` `658ed943f` `6c0ac9b93` (10-15) | 212 files, +12164/−19136; ~950 stale comments corrected against the code, ~380 more in adversarial review; every file proved comment-only per language (`prove_s8.py`), the generator byte-identical for all nine configs, `run_component.sh --print` identical, and on cf06 the broker tests, selftest, seven replay goldens and 26 tool and viewer tests identical between the base and edited trees; strings were left alone (#193) and the bugs the reviewers found are #192 |
| #193 | stage 9 of the comment pass: help text, operator and log messages, C++ log text and the generated config headers narrated their history | `d898cea86` `9e7abd7f2` `c3c52629d` `7adf7d747` `8cd896a9b` `d700cd2b8`, merged `ba14cc8c3` (10-18); configs regenerated 10-19 | 627 literals in 104 files, each proved string-only per language (`prove_s9.py`: the code identical outside the edited literals, placeholders, tags, key= fields and every fragment a test, doc or site script matches kept); on cf06 the broker tests, selftest, goldens and tool tests identical base vs edit; config bodies identical; cf05 builds, boost, `--check-config`; 27 code bugs the reviewers found went to #192 |
| #197 | `cuda_gnss_despread_test GPS_L2C_CM 10` failed two E/L rows of an unchanged kernel: since #179's covering rule (a time-multiplexed component spans twice its own rate) the case covers 10 channels and the E/L channel sums sit in a near-null, 0.2% of the prompt's, where the float-tables-vs-double-replica difference of ~1e-5 per channel term read as 6e-3 of the cancelled sum | `04fdb52d5` (10-17) | the base 188b6e800 Test build and the rebased tree print the same six rows to the fourth decimal; the compare is now the vector norm over channels (no cancellation, the same ULP floor): max relative error 5.6e-5 at L1/N20 and 1.6e-4 at L2C_CM/N10 on every row |
| #189 | the waveform kernel synthesised three replicas (E, P, L) per channel per hop, 80% of the GNSS GPU cost, although on a 195 kHz channel a code offset of ds chips is a phase rotation of the prompt (exp(+i 2 pi (f_c - f_car) ds/chip_rate)) up to the within-channel ramp | `cdec5ae30` (kernel, one chip walk, config key `epl_from_prompt`, generator `--epl-from-prompt`, every chain but L2C), `dbe5c5083` + `24af79405` (manifest: cx51 canary, then the common flag; live fleet-wide 10-17 18:33-18:36Z on fleet-04fdb52d5) | offline 10-13: identity to -18..-37 dB and the DLL to 0.004 chips on the 10-12 telemetry, cx52 A/B gnss_waveform 6.40 -> 3.24 ms per frame per GPU; on sky 10-17: cx51 vs cx43 then every node vs the fleet median within +-0.16 in per-instance q and 0.02-0.08 rms in discriminator (the pre-cycle spread), telemetry on cx19/cx27/cx44/cx51's own frames kE/kL = -/+1.000 k0 at -33..-39 dB and tau 0.003 chips rms, nvidia-smi 67-71 -> 58-64 % on every GPU; judge files ~/gnss/builds/judge1017/epl/ (judge_epl.py snapshot/compare/fleet). Stage 2 `b309a150a` (10-17): two synth lanes per slot under the key, the E/L rows rotated from P in GnssN2RecordAssemble from PrnCtl::rot (PrnCtl 80 -> 112 B, FrameHdr.lanes_per_slot), tiles frames halved, the dual's used lanes 384 -> 192; pinned by the kernel test's pack-lanes check; canary cx51 19:47Z, then the fleet 20:01-20:03Z: every node within +-0.21 of the fleet median in q (the pre-cycle spread), discriminators 0.02-0.07 rms apart, the telemetry identity exact (-92..-107 dB, the rows ARE the rotated prompt), nvidia-smi 54-59 % on every GPU with data (67-71 before stage 1, 58-64 after it) |
| #198 | a node stop waited out the senders' reconnect interval: bufferSend's connect loop slept `reconnect_time` with `sleep()` (an upstream TODO), and a fleet cycle takes recv1 (#165) and the cube archiver down at its first node, so every later node's 15 cube and 15 telemetry senders (30 s) were asleep when SIGTERM came and the cycle ran at exactly 30 s per node (18:33-18:36Z, 20:01-20:03Z) | `d624d7784` (10-17; rides the next node binary, fleet-d624d7784 built) | the wait is the connection condition variable in one-second slices re-testing stop_thread (all Stage::stop sets); judge at the next fleet cycle: node_up.sh restart cadence a few seconds per node with the receivers down, not 30 |

## Handed off — faults we found whose fix is someone else's

| # | what we measured | who has it | how it closes |
|---|---|---|---|
| #166 | with rows C/D enabled in the X-engine config every frame reported 100% packet loss: the F-engine does not send those boards (10-10; rolled back `e635d6619`) | site (fpga_master); KV 10-13: "understood upstream and not ours to fix" | the 64-input GNSS plan waits on it; check the packet rate against the board count before enabling dishes again |
| #167 | the `digitalgain_20261007T213936Z` table quantizes pol 1 to 0.12–0.27 LSB rms in the GNSS bands (X ~0.5); the cause of the rest of #163's E6 sharpness gap | the F-engine group; KV 10-11: "I've told the F-engine folks, there's nothing more we can do on this" | a new table; our noise normalisation (#163, live 10-12) already removes the shape term, the level term is theirs |

## Closed because the premise died (moot)

| # | the premise | what killed it |
|---|---|---|
| #21/#26 | "Path B remains parked" | Path B **is** the production tracker: `no-path-a: true`, path A not instantiated, 17.712 → 2.512 ms/frame |
| #25 | per-subband archive | retired from the chains yaml (`809edf361`); superseded by the beam cube |
| #40 | deep fold rate wrong-bins | demoted: the coarse feed is now only the re-acquisition path (see #40 in the open list) |
| #48 | "b2b has no control authority" | b2b has a live closed code loop, 6 present PRNs and a 5 m PVT solution |
| #49 | deep-gate rollout, BeiDou hold-outs | all 8 chains carry `fleet-trim-url` + `dll-combiners`; the lobe-coherent fleet DLL is the route |
| #61 | "the rate is real only on-peak" | the latch is fixed and the consumer was replaced by an estimator 3 orders better |
| #67 | in-process telem reader at 25% | the 60-polls-per-cycle reader is gone; broker is 0.72 cores, wait-bound |
| #73 | the 0.98 GB/s full product | its consumer was retired in favour of the visibility capture |
| #82 | wrap-edge label partners | the `PHCONT` instrument its only number came from no longer exists |
| #94 A0 | the same quantity estimated twice per band | both halves rested on bugs since fixed (`d85644b7b`, `ba015109e`); live cross-band trim agreement is 0.042 chips |
| #109 | the aggregator merge holds 3 feeds 18 min out of step | last occurrence 2026-09-05 14:18; 5 days and every node cycle since with no recurrence; the merge is producing detections. Its own prediction ("one node-restart away from returning") is falsified |
| A2 | seed churn as "one disease" | root-caused elsewhere (`d85644b7b`); #77's missing channel shipped as `/adjust_trim` |
| A3 | GAP 2's measurement is 42× more precise than it is | live: 11/11 accepted, spread 0.245–0.259 chips against a declared 0.3 — σ is now ~10× conservative |
| A6 | reference element 0 may be the weakest feed | **false**: element 0 is rank 10 of 32 at 0.94× the median. The weak feeds are 14/9/24/10/12 — and 9/10/14 are the pol-swapped dishes |
| A9 | `spec_tau` against model-held offsets | model-held is ~0; live taus are ±0.02 chips at p/f ~1.0 |
| #31 | "the data channels are built, armed, and producing nothing" | not a bug, a roadmap item (an independent receiver feeding nav bits from a live local stream, lagging ~0.5–1 s). What it needs, from the 10-11 read: the only consumer (the track-state `nav_bits` parse and peel) was removed in `a2a2770a8`, so (1) restore a consumer in `cudaGnssChordTrack`; (2) a broker POST route such as `/set_nav_bits` keyed per PRN with GPS-time stamps; (3) a source in `seeding.py` that predicts forward over the lag and maps GPS time to the capture clock (the removed `navbit_brdc.py`, in git history, did that from a synced PRN), registered under a new source name; (4) `bit_export: true` on the chain so `navbit_health` can veto. The contract: per PRN `{utc0, bit_s, bits (+1/-1/0 int8), grid}` on the capture-clock UTC, forward over a horizon. The supply chain and its `{'none': N}` line were removed on 10-11 (the decoders, cross-checks and decode health stay behind `--nav-bits` and `bit_export`) |
| #88 | "finish the j2 restructure" | folded into the upstream PR plan, stage 5 (`CHORD_UPSTREAM_PR_PLAN.md`): the GNSS hook into `config/chord/pathfinder.j2` |
| #123 | "cf06's single 1 GbE is the fleet's binding constraint" | the stack moved to the gnss VM (10G) on 09-17; only the cube leg still crosses cf06, and it has shown no fault |
| #11 | "boresight seeding mask: nothing masks seeding" | the railing veto and the shared-model transit freeze (#102, #154) now veto trims and freeze the model inside a transit; what the mask was for is covered, and #144/#170 own what is not |
| #174 | one corrupt-sequence packet from raw source ID 6 on cx27 | folded into #169 (one event) |
| #63 | "the tracker's summed prompt rides every wire frame on the link that is the fleet's constraint" | #123 died with the move to the 10G VM; the header prompt now feeds the fleet ADR (whose head/tail split is header-only), the comb DLL's instance alignment and the telemetry series, so the purge would be a v8 bump plus an ADR refactor for 9 of ~93 floats a row; the broker's real lever is the comb reduction in the C++ gather (KV, 2026-10-12) |
| #62 | "the comb's taps are weighted by a cal that already absorbed the record (non-causal by one record, α ≈ 0.005)" | every assembler runs `elem_sum_shared: true` / `elem_sum_adapt: false` since #154 (09-28): the weights are installed from the 300-s consensus before the combine, the per-record update feeds the shadow learner only, and the header and the comb use the identical held weights; the hoist is still right if adapt mode ever returns (KV, 2026-10-12) |

## Withdrawn — tried, measured, backed out (verify these stay dead)

| what | why | the grep that proves it is dead |
|---|---|---|
| deep-gate widening 100 → 50 | an active confound | `dll-deep-gate` is `4 9 27` / `33` / `33`, never `all` or `50` |
| GAP 2 feed on a model-primary chain | it measured its own seeded rate (the mirror) | the spec-anchored successor is armed on gal_e5a only |
| `joint-consume: slew` on a feeding chain | closes the loop | only gps_l5 takes `slew`, and it has no feed flag — **but see #124: the joint state is band-scoped, so this is a per-chain label on a shared object** |
| freezing the fast loop through establishment | made L5 worse (57–95% → 14–38%) | `--establish-hold` default 0.0, absent from the chains yaml |
| B1 split-aperture | comparator error | recorded in `gnss_fleet_chord.yaml` |
| `despread_max_chips 105` | superseded by CENTERED-80 | 105 appears only in a one-off cx19 config |
| two Doppler-free Φ tables per channel | 0.88–0.92× on the A40 — a loss | `shared_phi` and its tables were removed after `be7d46e8e` |
| tiled shared-memory streaming | 0.60–0.70× at every geometry | `launch_waveform_tiled` was removed after `be7d46e8e` |
| #90, #91, #92 — three trim-pathology fixes (admission gate, brownout holds, the trim handover with the gather's `/adjust_trim`) | built, disarmed again since 10-11, base rate ~0 on the healthy fleet so never judgeable; removed 2026-10-13 | `--fleet-trim-brownout-hold-s`, `--code-bias-brownout-hold`, `fleet-trim-rebase-adjust`, `reseed-spec-tau` are gone from `cli.py`; the code is at `be7d46e8e` |
| #194 | the F-engine return permuted the second NIC port's comb on cx19/cx42/cx43/cx51 (14/12/15/13 mod 16 where the node table said 12/13/14/15): their GPU-1 chains despread channels 195–390 kHz off the replicas — two at the noise floor, two at a seventh of the fleet — while `FLEET-INST` counted them present and chan_norm missed every record | `af12c4220` (10-17): table read from the taps, `node_channels` from the union of the GPU combs; the four nodes, the aggregator and the broker cycled 13:58–14:01Z | every instance's tap `freq_ids[0]` equals its `channel_ids[0]`; FLEET-INST per instance at fleet level on every band (cx19.1 E5b 1× → 803×); chan_norm applied, miss 0, on all 11. Lives on elsewhere: a per-instance dark detector (#192), cx27's GPU-1 comb when its port returns |
| #176 | the joint clock adopted a one-off 0.8-chip outlier inside the C33 transit and every 10.23-Mcps chain lost its prompts for ~1.5 s | `6e73de015` (10-13): a joint level more than 0.3 chips from the last adopted one is refused until it has held 30 s; live since 10-13 21:05Z | offline the C33 adoption is refused and the C13 walk followed; on sky 10-17 12:57Z the establishment level was refused once (−0.62 chips, held 0/30 s) and adopted 30 s later with no fleet-wide prompt loss. `gnss_broker.test_jointstep`; `grep 'REFUSED (step' broker.log` through a 1176-MHz transit after the realignment is the standing watch |
| #175 | after a transit's null crossing every E6 tracker was re-seeded ~0.45 chips off its peak by a simultaneous 'displaced' presence admission and railed there for an hour | `9f193b74e` (10-13): displaced admissions refused while a BROWNOUT is open or when three or more lie within 0.1 chips (on-peak rows exempt); live since 10-13 21:05Z | offline the 10-11 22:34:52Z admission is refused as one offset; 10-17 12:47–13:17Z: 69 refusals, every one a chain event. The clamp-escape half is #195 |
| #129 | no GLONASS positions at all: `parse_rinex_nav` was G/E/C only, so the 273 R records in every BRDC were discarded and the 1202.025-MHz L3OC signals we monitor came from a constellation we could not place | `330e11d04` (10-14): PZ-90 state-vector propagator (RK4 with the ICD's lunisolar and J2 terms), 3.4 m rms against the MGEX SP3; live at the 10-17 14:01Z broker restart | 25 satellites from BRDC 2026-290 at 19,087–19,181 km, ECEF speeds 3.15–3.59 km/s, 8 above 10° at DRAO; the railing veto and the transit sky carry the R slots. Still G/E/C-only: the supply's hourly target, `gen_fleet` `SYS_OF`, the cube's `CHAIN_SYS`; propagation past ±15 min unvalidated; the B3I list excludes C56–C58 — all #192 (e)/(f). The GLONASS chain itself is future work; C57 is the one assigned PRN absent from BRDC; #56's 1202 MHz test is now runnable |
| #154 | the shared element model latched its phase pin from its FIRST model (one shadow), so a node restarted into a dropout or before the broker's anchor pinned its global phase wrong and its records cancelled in the fleet sum (R 0.15–0.70 on 09-28 and 10-04) | the fleet reference: `elem_shared_ref.py` snapshot/post/watch, the generator's `--elem-shared-ref`, `elem_ref_cron.sh` every 5 min (re-posts, re-takes under new gains); live since 10-05, the reference re-taken 10-17 (`df37629bb`) | every restart since the 10-09 re-snapshot: R ≥ 0.95 within minutes (10-17: 0.998 four minutes after the broker restart, which followed node restarts — the no-node-restart re-base check is #196's; the six models formed under `--boresight none` slewed on within one cron interval). L2C never converges — one channel per instance — and is a #192 item |
| #170 | a model formed inside a near transit (5–8°) stayed mis-shaped for over an hour | `11ab0ee25` (10-11): formation frozen inside 8° of boresight (the broker's `_bore` post), re-seed from the reference after 600 s below the floor when the sibling pol still matches; on all six since 10-11 23:00Z | freeze half 10-11 (cx44 inside E31's pass: no model for 32 min, then sim 0.85–0.98) and 10-17 (held while G32/E10 sat in the nominal cone; six instances formed within 3 min once `--boresight none` lifted it); re-seed half 10-17: counters [0,0] through the gain change while the cron re-took. `/get_elem_cal` form_frozen and fleet_ref.reseeds. The `--boresight none` switch (`cc2b33651`, armed `55ca9a036`) stays on for the tipped week: its revert is #196 |
| #177 | `gpsL5Code` wrapped XB at 10230, not 8191: the last `advance` chips of every L5 code came from the wrong XB phase, up to 10.5 dB per PRN — the 09-22 per-satellite L5 gain was this | `e36372bee` / `a25e36345` (10-12); fleet binary d30be5f66 and every one since | canary cx44 paired in time: G25 +8.2 dB, G03 +4.6, control G28 +1.5; fleet G25 prompt C/N0 +12 dB, G26 present 93%; boost case `xb_wraps_at_8191`. 10-17: the aggregator searches on the fixed code (8 PRNs) but its per-PRN judge could not be made (the worst-loss PRNs were not up and tipped dishes make levels incomparable) — that judge and the per-satellite G re-solve are #196's |
| #180 | `generate_e1b/e1c/e6b/e6c_code` indexed the hex table before the 1..50 range check, so a swap to PRN > 50 on a live E6/E1 chain read past the array (ASan: global-buffer-overflow 8 bytes before `E1B_HEX`) | `8b3c8fa40` (10-13); on all six in 188b6e800 since 10-17 | `tests/boost/test_galileo_codes`: PRN 1/50 first-24-chip pins and the throw on 0/51; clean under ASan |
| #179 | `gnssBandPlan::half_bandwidth_hz` returned the 511.5-kHz component rate for time-multiplexed L2C, so a search covering set spanned half the main lobe | `55d222a79` (10-13): 2 × chip rate when `time_multiplexed` | `test_gnss_band_plan` pins 1.023 MHz and 7 → 11 covering channels. No live arm: the deployed aggregator searches L5 only and the node L2C chains take one channel each from the generator (closed on the test, KV 10-17) |
| the period-debounce adoption fix (`0d26b6ea9`, reverted `86cd96e61`) | **resolved 2026-09-10, see below** | — |

### The period-debounce revert, explained (2026-09-10)

`0d26b6ea9` was written against a 23.98/min storm of `period ADOPTED` on gps_l5 and reverted 31
minutes later with an empty commit body; nobody remembered why. The archived broker logs settle
it: the storm was a **restart transient** (the pre-fix log begins one minute after a broker
restart). With the fix live the rate was 0.13/min; **after the revert it was 0.45/min and never
returned to 24/min**, and on 2026-09-10 it is 0.35/min. Adoption is also gps_l5-only — the other
seven chains log zero, because only gps_l5 has a search to produce period labels. So the revert
cost ~0.2–0.3 adoptions/min on one chain and the fix was not load-bearing.
**The trap worth keeping: a fix deployed onto a restart transient will look like it worked.**

## Answered by measurement — questions that had an answer, not a fix

**#147 — the 2026-09-26 22:01Z event was a ~3 ppb step of the site frequency reference, not ours.** Every
chain and every instance at once: a carrier offset scaling with carrier frequency (E6/E5a ratio 1.09, expected 1.087), a
−37 m (−127 ns) code step, carrier residual ±2.5 Hz for more than 14 min, beam-cube 1-s coherence 0.90 → 0.03 for ~2 h,
no lock loss, no broker event, and gps_l5's code-rate clock estimate moving the same ~3 ppb. That signature is a
reference (maser/GPSDO or the F-engine clock) stepping; nothing in our chain scales with carrier frequency on its own.
Closed 2026-10-13 (KV) as a one-off with the signature recorded here so the next one is recognised in a minute and
vetoed from any coherence statistic. Evidence `fixtures/projtest/forensics/l5/ev2201*.py`.

**#56 — the hourly signal swings are near-boresight satellite transits (most of them).**
Confirmed 2026-09-10 against the cached BRDC for the archived days: bins with a GNSS satellite
inside the HWHM run a median `elem_power_max` of 34.0, bins with one at 1.24–5° run 18.6–24.5,
and bins with nothing within 5° run 9.3–9.6 — a 2.4–3.6× step, with `r(min separation, power)`
= −0.28…−0.41 and `r(n within 5°, power)` = +0.44…+0.52. The dishes are 8.59° south of zenith
(boresight az 180, el 81.41, FWHM 2.48° at L5), a transit takes tens of minutes, and it rails the
4+4b voltages — which is exactly the burst length and shape the archive shows. A residual
population with no GNSS satellite in the beam is still open under #56.

**#56 (the same measurement) — the swings are a real source, not gain or quantiser scaling.** Answered
2026-09-10 offline from the two archived `rf_rail` days (8137 and 8635 samples, 12 instances), by
the entry's own discriminator: `r(clip, power)` = **+0.88…+0.91** across hours and
`r(elem_clip, elem_power)` = **+0.96…+0.97**, so clip rises *with* power. ⚠️ The entry's "~5×" was
the wrong quantity: hourly medians move **1.6–1.8× in mean power** and **2.1–2.9× in peak element
power**, while **clip moves 22–65×** — clip is a tail statistic against a fixed threshold, so a
2–3× power rise amplifies into a 60× clip rise. Write future budgets in element power, never in
clip. The swings are bursts of tens of minutes (10-min peaks of 40–61 against a 7–20 hourly
baseline) recurring at roughly repeated UTC times. Celestial vs terrestrial is still open (#56)
because a sky source drifts only 4 min/day and three days cannot resolve it.

**#94/S2 — the prior gauge's falsifier passes, and it never needed a deliberate satellite drop.**
The pre-registration was "drop a satellite, `clk` must not move"; the fleet churns membership on
its own, so it is answerable by inspection. Live, 730 `JOINT[shadow]` samples: over **22 warm
membership changes** (both sides ≥3 sats) |Δclk| was median **0.016**, p90 0.096, max **0.119
chips**, against 707 unchanged-membership steps at median 0.008, p90 0.045, max **0.169** — the
largest jump on a change is *smaller* than the largest with membership held, and all of it is
inside the reported σ of 0.045. Zero changes moved `clk` past 0.2 chips. ⚠️ The raw statistic
looks alarming (max 150.8 chips) because it includes two transitions through n = 0: that is filter
birth, not the gauge. **The lesson: a falsifier written as an intervention may already be
answerable by observation, if the system perturbs itself.**

**A single node roll costs nothing; a fleet-wide roll costs ~30 minutes.** Measured 2026-09-10
from the archived broker log with the broker up throughout. cx43 alone was absent 01:24–01:54 and
returned: max|readback trim| stayed **0.27–0.56 chips** across both the loss and the return,
indistinguishable from baseline. The fleet-wide roll at 02:02–02:10 climbed 0.4 → 0.87 → 1.85 →
2.94, was **pinned at the ±3.000 clamp 02:18–02:24**, and returned to the 0.2–0.4 baseline only at
02:38–02:44. The standing C++ trim lives **on each node**, so one roll discards a twelfth of the
fleet's trim state while ten combiners carry the measurement, and a fleet roll discards every
satellite's at once and the DLL re-establishes all of them from zero.

**#148 — cx19's ~0 beam-cube coherence was the statistic, not the instance.** Answered 2026-09-28 from the
per-record viscap capture (`fixtures/projtest/viscap0928/out/cube148_0927T19.txt`, `coh_w1.txt`,
`coh_w2.txt`). The cube's cells are exact against the capture: worst relative difference 5.96e-8,
and |γ| = 1.0000000 on cx19 as on cx51. Each cell is corr/E, so its noise variance scales as 1/E, and the
within-window coherence summed the channels unweighted. Each cx19 instance has one channel next to a
BPSK(10) null, and that one channel's noise swamps the sum:
- gnss0 at 1166.41 MHz (null at 1166.22), E/median 1.1e-3;
- gnss1 at 1186.72 MHz (null at 1186.68), E/median 1.4e-4.

Weighted by E² (corr units), all 12 instances read 0.91–0.95 on e5a and b2a (09-27 19–20Z). Unweighted,
the statistic tracks the smallest channel's E ratio: cx44 gnss1 at 4.4e-3 read 0.37/0.33, cx42 gnss0 at
4.8e-3 read 0.70/0.30 (the entry's two 0.4–0.6 instances), and instances at ≥ 1e-2 read ~0.9. Dropping
the minimum channel only patches it (it costs cx51 0.93 → 0.82). ElemCal, the header and the replay sum
corr, not corr/E, and never saw it. L5's 0.02 on every instance was #141. ⚠️ **Any statistic that sums
cube cells across channels must weight them by E².**

## Closed with a full write-up — the three worth reading before touching these areas

### #153 — node processes aborted at random with `malloc(): unaligned tcache chunk detected` ✅ FIXED (set-once 10-01, #1714 10-09; verified 10-11)
Found 09-29 on the d53b6a254 bundle (4 aborts in 17 h, auto-relaunched in 20 s, every REST switch lost). Root-caused
09-30 from the first two cores: a double destruction of a `chordMetadata`. The writer was `cudaRFISKtilde` replacing
`bf_mask_applied`'s ring slot 0 **every frame**, which breaks `NDArrayRingBuffer::set_metadata`'s set-once contract;
the reader was `cudaCopyFromRingbuffer` on another thread copying that slot every frame through an unlocked
`get_metadata()`. A `shared_ptr` copy racing an assignment of the same `shared_ptr` frees the object twice, and the
core showed it (a `FREQ_UPCHAN_FACTOR` key-string chunk on both a fastbin and a tcache). A second, start-up segfault
was a `FramePrefetchService` use-after-free of the stream-ID vector. Fixes: `2aac0bc53` (publish the echo's metadata
once), upstream #1713 (prefetcher, merged 09-30) and #1714 (a second `set_metadata` is fatal, merged 10-06; in our
tree 10-09). The "silent" cx19 restarts were `needrestart` after unattended-upgrades (16f82b73a). Verified 10-11: no
`malloc()` line and no relaunch on any node since the set-once build; the core recipe (core_pattern + prlimit) is in
`docs/CHORD_GNSS_RUNBOOK.md`. Lesson: a backtrace in an assembler thread does not implicate the assembler — the
tcache is per thread, so the aborting thread is only the one that tripped over the damaged chunk.

### #140 — the gather dropped the broker's telemetry socket, and the reconnect cost 5 s on every chain ✅ FIXED 09-18 (in the tree; live at the next broker restart)
With #139 fixed the 2-hourly breaks were gone, but four fleet-wide breaks remained in the first
8 h (18:27, 19:25, 19:58, 21:54 UTC), each `gap ~500 / dark 0 / vouch 0` on all eight chains.
Each sat 5.0 s after a broker line `telem: gather ... unavailable (gather closed the
connection)`, and the gather's log had the other half: `dropped client fd 14 -- it could not
take a frame within 200 ms` (GnssTelemGather's `send_timeout_ms`). The stream carries every
chain on one loopback connection at ~45 MB/s, and the kernel held ~10 MB for it (4 MB send +
6 MB receive) = 0.22 s of slack; `ss -tmi` showed the receiver window-limited 5.7 % of the time
and its buffer full 11,412 times in 50 min. The old reader parsed each frame and took the ring
lock between two `recv()` calls, so any stall on that thread -- the lock is contended by every
chain's pass, and the VM runs at load 6-7 on 6 cores -- became a dropped connection, a fixed
`retry_s = 5 s` wait, and ~500 records missing on every instance of every chain. Not a
whole-process pause: the chain threads logged through all four. Fixed in `gnss_broker/telem.py`:
the socket thread does nothing but `recv()` and a deque append (bounded in bytes; over the
bound it sheds the OLDEST frames, counted as `pending_dropped`, never the connection); a store
thread parses and takes the ring lock; a connection the gather closed after it LIVED
(`fast_retry_after_s`, 10 s) is reconnected at once, the backoff kept for a gather that is
down. `stats()` carries `pending / pending_peak / pending_dropped / fast_reconnects`.
Tests D-E in `test_telem.py` run against a real socket with a 64 kB send buffer: the old client
blocks the sender while the ring lock is held, the new one drains 2 MB through it. Belt and
braces, root side: raise `net.ipv4.tcp_rmem`/`tcp_wmem` maxima so the kernel holds seconds
rather than a fifth of one; the autotune caps apply to the live connection without a restart.

### #139 — every fleet-ADR arc on every chain broke every 7200 s: a 2.7-s telemetry ring and a reload on the pass ✅ FIXED 09-18
After #136 (the synchronous BRDC merge) the arcs still broke fleet-wide twice every two hours,
to the second. A break diagnostic in `fleetadr.fold_record` (9d7f04908) read `gap 193-253
records / dark 0 / vouch 0`, every instance's last record exactly `gap` behind: telemetry the
fold never saw, not trackers gone dark. A telemetry window is one frame = 4 records = 41.9 ms,
so the broker's 64-window ring held **2.7 s** (its docstring said "~168 ms" per window) against
a fold that runs once per 2.0 s control pass. The 2-hourly dead-reckon reload held the pass
2-2.6 s longer -- `parse_rinex_nav` plus `fetch_dcb`, whose day-by-day walk pays a live HTTPS
404 to CDDIS for every day younger than the ~5-day-late product -- and the windows arriving
meanwhile were evicted unfolded. The routine breaks were the same mechanism in miniature: a
pass 0.7 s longer than usual lost one or two windows (`gap 5-9`), a few times an hour. Fixed
`8eeb1affb`: the reload's I/O runs on a thread and is swapped in between passes (no seed
touched; first load stays synchronous), and `shared_client` holds 256 windows = 10.7 s (~+380
MB). Ruled out first: seeds cleared (never), locked satellites re-born at the reload (they are
slewed), JOINT-CLK adoption (per-cycle), any whole-process or per-chain log gap over 3 s.
⚠️ The whole-process log-gap scan cannot see a per-chain stall in a free-threaded broker, and a
per-chain scan at a 3 s threshold could not see a 2.6 s one: the instrument for a stall is the
consumer that loses data, not the log.

### #138 — the fleet ADR random-walked on weak satellites: clustered half-cycle slips through the re-seat ✅ FIXED 09-18
The geometry-free TEC was 0.06 TECU at 1 s and useless by the hour. A triple-frequency closure on
the archive (`fixtures/tec_wander/`) showed the instrument itself random-walking from 1 s
(0.018 cycle/√s per band), heavier on weak satellites, common to every instance of a chain. Live
telemetry captures replayed through `fleetadr.fold_record` found it: a weak satellite's instances
slip their squared-phasor sums by exactly ±0.5 cycle, and they slip TOGETHER (E25: 134
slip-seconds in 10 min, 38 pairs, 8 triples, 3 quads, one six). The fold re-seated slippers to a
median taken with the slipped values in the sorted list, which moved that median by their ranks;
the re-seated copies then held the ranks, so each cluster left a permanent step. Strong satellites
(no slips) were flat at 0.16 TECU over 300 s throughout. Fixed `b444d2926`: repair each slip by
whole half-cycles before the median. Replayed: E25 1.41 → 0.50 TECU at 300 s, E05 1.06 → 0.77,
strong satellites bit-identical. Ruled out with numbers: multipath/elevation, carrier-trim
leakage, the median combine, per-instance Doppler. Open: weak satellites remain ~3× worse than
strong ones (sub-threshold excursions), and why slips cluster.

### #135 — the obs writer latched the F-engine sample-0 epoch once, and filed three days of data onto one ✅ FIXED 09-17
`gnss_observables.py` read `/telescope/time0_ns` at startup and stamped every row `t = frame0 +
hop/HPS` for the life of the process. An F-engine re-base restarts the hop counter without moving
that value, so a writer that outlived one (they ran for days) filed the new session onto the OLD
epoch: rows tens of hours in the past, several sessions stacked into one day file with `t`
non-monotonic in file order, and every geometry column evaluated at the wrong instant. In
`*_20260916.jsonl` 71% of rows with geometry put a tracked satellite below the horizon (median el
−46° for >35 dB-Hz); E13 read range-rate +441 m/s against a tracked Doppler of −90 Hz.

Measured with `obs_retime.py census` (the time shift at which the ephemeris Doppler matches every
strong PRN's `dop_rec_hz` at its `rec_hop`, rms 0.00–0.05 Hz, snapped to the anchors the brokers
logged): `*_20260913` holds three sessions at +136968 s, +143735 s (anchor 1789418417) and
+248166 s (1789522848); `*_20260914` is the +143735 s session past UTC midnight; `*_20260916` is
+84351 s (1789607198 — the whole 09-16→17 overnight run, stamped a day early). GAL and BDS date
every segment identically. **The broker was on the right epoch throughout** (9c56c1682 makes it
exit 3 on a moved anchor; its residuals were ±10–45 Hz all night); the writer had no such rule.

Fixed `27f91a4a1`: `--frame0-recheck-s 60`, exit 3 on a moved value, systemd restarts into the
new epoch and a new day file (exercised against a fake endpoint: exit 3 within 2 s of a +24 h
move). The writer also read the nav cache through `fetch_brdc()` — one of EIGHT extra writers of
the shared cache, the 08-27 fault — and now reads `cached_brdc()`. The archive was re-timed into
`fixtures/obs_retimed/` (`obs_retime.py apply`, never in place; `t_raw`/`retime_s` kept per row).

⚠️ **The lesson:** a value fetched once is a latch, and every latch needs a re-read that can
ACT. The broker had exactly this fault two days earlier and got the rule; the writer, a
separate process reading the same endpoint, did not. When one process learns to detect a moved
anchor, ask which OTHER processes hold the same anchor. Tracking is blind, so nothing downstream
ever complained — the data kept flowing, on the wrong day.

### #136 — the BRDC hourly merge ran synchronously in the control loop and broke every fleet-ADR arc at once ✅ FIXED 09-17
`fetch_brdc` did the station-hourly merge (up to nine files, 20 s timeout each) and the daily
mirror fetches (30 s) inside the broker's control pass. The cf06 log of 09-16 goes silent for
24.3 s (14:50:32→14:50:57) and resumes with `BRDC hourly merge: 9 station-file(s)` on every
chain; the FADR lines show every satellite's arc restarting at 14:50:56. The telemetry ring is
64 windows = 10.7 s, so a stall longer than that loses windows and `fleetadr.fold_record`
breaks every arc on the chain — correctly, an increment it cannot account for is a break. On the
09-13 run the breaks hit 16 satellites at once at :03/:18/:33/:48. Each one throws away the
carrier level of every satellite, which is the one thing a geometry-free TEC cannot recover:
this, with #137 and the ADR random walk (`fixtures/tec_wander/README.md`), is why TEC was
"useless by the hour".

Fixed `d40e35d29`: `fetch_brdc` hands the loop what is on disk and runs the fetch on one daemon
thread; synchronous only for a pinned replay, an empty cache, or `block=True` (a caller with no
ephemeris yet). Verified in production 18:59:41: the merge line with zero log-timestamp gaps.

### #137 — `gnss_tec_chord.py` keyed joint arcs on `fadr_arc` alone, which restarts at 1 ✅ FIXED 09-17
FleetAdr forgets a satellite absent 30 s and starts it again at arc 1, so the arc before and
after a dropout both read 1 and the producer joined them: two accumulators with unrelated
origins, a step of ~1e5 TECU that then dominated the satellite's statistics. Fixed `67643f642`:
the continuity key is (arc, hop0).

### #134 — the systemd unit dropped the broker's BLAS thread cap, and the 2026-08-15 fault came back ✅ FIXED 09-17
`broker_restart.sh` has exported `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1` since 2026-08-15, when numpy's OpenBLAS pool — one **busy-spinning**
worker per core, woken by the joint filter's small (~57×57) matmuls — measured **~53 cores of
pure spin** on cf06 and starved the telemetry reader at the OS level: the gather dropped its
client every 200 ms and lost 5 of 6 frames.

The first systemd unit reproduced the command line faithfully and **silently dropped the
environment**, so the fault returned in miniature on 6 cores. `perf top` on the live broker:

    55.56%  libscipy_openblas64_.so   blas_thread_server     <- the spin-wait
     3.61%  libscipy_openblas64_.so   exec_blas_async_wait
     3.44%  libscipy_openblas64_.so   inner_thread
     3.25%  libscipy_openblas64_.so   inner_thread
     0.52%  libscipy_openblas64_.so   dgemm_kernel_SKYLAKEX  <- the actual arithmetic

Two thirds of the process spinning to do half a percent of maths. Fixed by putting the four
variables in `gnss-broker.service` with the reason beside them.

| | before | after |
|---|---|---|
| broker CPU | 4.31 mean / 4.85 max | **1.23 / 1.48** (cf06 ran 1.75) |
| broker threads | 16 | **11** (cf06 ran 11) |
| stack total, of 6 cores | 5.18 / 5.66 | **2.14 / 2.39** |
| load average | 9.61 | 3.07 |

⚠️ **The lesson is not "set OPENBLAS_NUM_THREADS".** It is that **a wrapper script's `export` is
part of the program's contract, and porting the command line is not porting the program.** The
faithfully-reproduced `ExecStart` was exactly what made this invisible. When converting any
launcher to a unit, diff `/proc/<pid>/environ` between the old and new process — not the argv.

⚡ It was NOT the whole story on late frames: 0.28% → 0.238%, against cf06's 0.23% lifetime. At
6 cores the five spinners cost throughput, not the reader starvation that 53 caused.

### #132 — the gather's cpu_affinity named cores that do not exist on the gnss VM ✅ FIXED 09-16
The stack moved to a 6-vCPU host and `chord_gnss_gather.yaml` still pinned to cores 19, 24, 31,
57, 58 and 59 — cf06's 64-core map — so every start logged six
`Failed to set thread affinity/name ... error code 22` (EINVAL) and fell back to unpinned.

Two things made this more than cosmetic. kotekan's `Stage` reads `cpu_affinity` through a
**required** `get` (`lib/core/Stage.cpp:29`), so the key cannot simply be dropped — a wrong core
is an error per stage, not a no-op. And six ERROR lines at every start is the noise that hides a
real one later, which is exactly how #98 ate the hold chain for five weeks.

Fixed with a `--gather-cores` flag on `gen_chord_gnss_config.py` rather than a hand-edit: the
file says DO NOT HAND-EDIT and the generator had no core option at all. The list is consumed
exactly as the built-in one is (first four for the buffer, then `[4]` and `[5]` modulo its
length), so a single-socket host lists all its cpus. Regenerated with `0,1,2,3,4,5`; verified the
pins now take — `/telem_recv` on 0-3, `/telem_gather` on 4, `/fleet_trim` on 5 — with zero errors.

⚠️ The same latent problem sits in `build_aggregator_instance` and `build_cube_archive_instance`,
which take their cores from the same base-config list. Whichever moves next needs the same flag.

⚡ Also closed a smaller gap found on the way: the gather config recorded no regeneration
command, unlike the node configs, so its exact flags lived only in `gather_up.sh`'s help text.
It now carries its own `REGENERATE WITH:` line. Before changing anything, the documented flags
were shown to reproduce the live file byte-for-byte — otherwise the regeneration would have been
silently changing more than the affinity.

### #54 — the yardstick was the defect (2026-09-10)

For a month this read as "the GPU replica is wrong at 3.5% and gets worse with uptime". It was the
**CPU reference** that was wrong. `hoprate_stream_into` and `channels()` had both had their
CARRIER promoted to long double, each with a long comment about the one-radian ULP at CHORD's
absolute sample index — and in the same function the **code** phase was left as
`cp0 + (double)n_m * cps`. At n_m ~ 1.9e15 that product reaches 3.8e14 chips, binade [2^48, 2^49),
where a double's ULP is **0.0625 chips**.

The reason it hurts more than a phase error should: the tap spacing in that loop *is* `cps`
(~0.2 chips), so 0.0625 chips is ~30% of a tap. It does not merely shift the replica, it hands
taps to the neighbouring chip — the same `2*proto[k]` error the boundary snap a few lines below
exists to prevent, except at a rate set by the sample counter instead of by a rational
coincidence. `chip_at`, the lambda the snap compares against, was quantised the same way, so the
corrector was correcting toward a corrupted target. That is why the growth is not a power law
(×10.3, ×1206, ×45 per decade of anchor): it is smooth while the ULP is small against a tap, then
turns discrete as it approaches one.

The prime suspect from 2026-08-13 (`C_P = cp0 + n_m*cps` **in the kernel**) was right about the
expression and wrong about the side. `c38f0f138` fixed the kernel's copy and the number barely
moved, which is exactly what should have happened — the same defect was sitting in the reference
it was being measured against, and only the reference's half was still live.

Fixed by promoting all three sites to long double (`CL`, `chip_at`, and `channels()`'s per-sample
phase, the last reducing mod the code length so `code_chip` and the NH overlay both come off one
long-double evaluation). Verified on an anchor sweep across three decades:

| anchor | prompt before | prompt after | peel per-sample before | after |
|---|---|---|---|---|
| 0.007 d | 7.92e-08 | 7.92e-08 | 2.482e-07 | 2.482e-07 |
| 0.068 d | 8.16e-07 | 8.16e-07 | 5.447e-07 | 5.447e-07 |
| 0.678 d | 6.50e-04 | **1.43e-06** | 3.227e-02 | **8.619e-07** |
| 6.781 d | 3.48e-02 | **4.79e-05** | 9.551e-02 | **3.310e-05** |

Reference cross-terms went FAIL 7.94e-02 → OK 4.00e-06. Every bit-exactness gate in the suite
(split-vs-fused, 4+4b-vs-float, N×M element axis, cross-terms off→on) still reports exactly
0.000e+00, and `test_gnss_channelized_replica` passes all 17 cases including the NH overlay.

**HOW IT WAS RETIRED (the tolerance, not a tighter fix).** The residual 4.79e-05 is the
long-double *anchor* floor: the kernel reduces mod the code length once at `n0`, the reference
re-evaluates every hop, so the two differ by O(1) long-double ULPs of the anchor product. Giving
the reference the kernel's convention would drive that to the float floor **and make the term
unable to fail**, so the two are deliberately left independent and the gate now scales with the
arithmetic instead:

```cpp
const double ulp_anchor = std::ldexp(1.0, std::ilogbl(anchor_chips) - 63);  // chips
const double TOL = 1e-5 + 32.0 * ulp_anchor;
```

That is 1.10e-05 at 0.007 d rising to 9.87e-04 at 6.8 d, against measured worsts of 4.92e-07 and
1.71e-04 — margin 22× and 5.8×. **The gate still bites:** with the double-precision reference put
back it FAILS at three of the four anchors (8.79e-03 vs 1.76e-05; 4.41e-02 vs 1.32e-04; 1.41e+00
vs 9.87e-04 — by 500×, 334× and 1429×). It passes only at 0.007 d, where the double ULP is
genuinely below the float floor — which is precisely why a month of prototype-scale testing could
not see this.

**Two things this cost, worth remembering.** The defect was invisible to every boost test because
they all anchor at `start = 0`; and it survived a month of being attributed to the GPU because
nothing ever asked whether the reference was right. A reference that is never itself checked is an
assumption wearing a measurement's clothes.

⚠️ **The unrelated red gate found while checking this is now diagnosed and fixed**, and so is the
live gap it exposed (#128, closed below). `test_gnss_channelized_acquire`'s
`doppler_parabola_refine_beats_grid` had been failing (err_ref 44.14 against a 25.0 bound),
identically before and after this fix. It was **decayed tooling, not a live fault**: the test
gridded Doppler at 100 Hz against a 1000 Hz transform bin (0.1 bin) and took its "truth" from a
step-5 grid (0.005 bin), both far inside one bin. `6e30b654c` wrote the test alongside a
*parabola* refine, which degrades gracefully on a flat top; #105 (2026-08-30/31) replaced it with
the amplitude-ratio estimator, whose domain is `step == 1 bin` — and the test was never updated.
In that regime the neighbour ratio is ~1 whatever the offset, the inversion saturates at
d → 0.5, and the refine reports half a step toward the neighbour: 5.5 Hz of cell error became
44 Hz of refined error. The estimator was behaving exactly as designed — the sinc model
reproduces the observed δ = 0.4964 to 0.2%.

Retested in its proper domain (bin-spaced grid, truth 0.3 bin off a cell, analytic reference) it
recovers **δ = 0.2936 against a true 0.300 — 6.4 Hz of 1000, a 47× improvement on the raw cell**.
**#128 CLOSED.** The precondition was enforced on one path of two: `GnssCudaAcquire` declines a
non-bin-aligned grid and logs why, but with `use_cuda_acquire: false` that branch is never
entered, so the CPU acquire ran the out-of-domain refine silently. The check now sits where BOTH
paths pass — before the `#ifdef GNSS_CUDA` — and warns (rate-limited 1/512) when the grid step is
not a whole multiple of `fs/(Mp*fft_len)`. Verified against the acceptance rule it mirrors: 31.25
on a 62.5 Hz bin warns, 62.5 and 125 do not. The generator now emits the bin-aligned 62.5 Hz
**unconditionally** — it had been conditional on `--cuda-acquire`, reasoning about the GPU's
cyclic-shift speed win and missing that the refine needs bin alignment for *accuracy* on either
path. It also halves the CPU grid, so it is free.

⚠️ **Four generated aggregator configs still carry `doppler_step: 31.25`** (`chord_gnss_agg.yaml`,
`agg2`, `agg6`, `agg8`). They were NOT regenerated: `gen_chord_gnss_config.py` requires `--base`,
a live production config pull, so regenerating would fold in unrelated drift into files nothing
currently runs. The live aggregator is `chord_gnss_agg6_cuda.yaml` at 62.5 Hz. The guard is what
makes this safe — starting from one of the stale four is now loud instead of silent, which is the
property that was actually missing.

Also worth noting: the old first assertion `err_ref < err_grid` required beating a *lucky* 5.5 Hz
cell, where the expected error on a 100 Hz step is 25 Hz — badly conditioned regardless of the
estimator. Both assertions are now well-conditioned: the cell error is 0.3 bin by construction,
and an inert refine (300 Hz) or a saturating one (200 Hz) both fail the 0.1-bin bound.


### #117 — the grid hops were never missing, only unsampled (2026-09-11)

The broker folds a record on every `GRID_HOPS` (96×2048 ≈ **1.0066 s**) — the epochs at which
every chain's records carry the same hop, which is what makes two bands pair exactly. It kept
**one** snapshot, the newest. The observables writer polls at **2 s**. That is exactly Nyquist,
so each poller saw its own alternating half of the grid: **50% per chain measured over 3392 s**,
and because each chain lands on its own alternate, a band **pair** shared **40%** and a triple
**35%**. Nothing was lost in the broker; it was lost in the poll.

`SatAdr.grid` is now a list of the last `GRID_KEEP = 4` snapshots, published as
`fadr_g_hist` = `[[hop, dop_cycles, cycles, n_rec], …]`, oldest first. Four spans ~4 s, so
consecutive 2 s polls overlap by two and no hop is lost unless a poll is more than four grid hops
late. The scalars (`fadr_g_hop` and friends) are untouched and remain the newest entry, so every
existing consumer keeps working; `gnss_tec_chord.py`'s loader expands the history into its
`{prn: {g_hop: …}}` map, dating each entry back one grid cadence per step.

⚠️ **Only the current arc is published.** A break resets the accumulator, so an older arc's phase
is not continuous with the current one and must never pair against it — the filter that used to
guard the single slot now filters the list.

⚠️ **The trap that made the test hard to write:** stepping a whole grid hop per fold is a
96-record gap, which breaks the arc on every step (`max_gap_rec=3`), leaves `n_rec` at 1, and
publishes nothing. Chains fold *every* record; only every 96th is a grid hop. A first version of
the test did step by grid hops, and its most important case silently skipped.

Cost: **+203 bytes on a 1898-byte observables row (+10.7%)**; one chain-day is ~830 MB.
`test_grid_history.py` — 7 tests: retention bound, grid-multiples-only, newest-last, the arc
rule, the published shape against the scalars, and the pairing yield itself modelled across six
poll phases (newest-only < 65%, with history > 99%).


**VERIFIED ON SKY 2026-09-11.** Broker restarted 20:22, observables writers 20:48. The A/B is on
**identical rows** — every row carries both the scalar and the history, so the same 540 s window
was counted twice with no confound, over 16 PRNs and 8349 grid hops:

| | newest-only | with history |
|---|---|---|
| per-chain coverage | **50.0%** | **99.8%** |
| pair coverage (gal_e5a × gal_e6) | **8.1%** | **99.6%** |

The 50.0% reproduces the recorded per-chain baseline exactly. The pair figure came in at 8.1%
rather than the recorded 40% because these two chains happened to be polling in near-antiphase in
this window — which is the point: **the old yield was a function of poll phase and nothing else**,
so it could be anywhere between ~0% and ~50% on any given pair at any given time. With the history
it is 99.6% regardless.

⚠️ `gnss_tec_chord.py` completes its science and writes the `.npz`, then dies on
`import matplotlib` — not installed in cf06's `venv-ft`. Plot-only, pre-existing, harmless to the
measurement, but it means the tool cannot draw on cf06.


### #93 / GAP 3 — closed on the disturbed regime, with the lever checked (2026-09-12)

#93 named its own closing condition: *"the honest test is the first DISTURBED window, not calm
data: re-run the same statistic there, and if it holds, close GAP 3 as 'no aid available' rather
than 'not yet tried'."* The 2026-09-10 soak is the first dataset with the power to do it — a
genuine **24.00 h, 591k rows/chain at 6.8 rows/s**, against the 475 shadow lines the previous
verdict rested on. Pre-registered in `fixtures/expectations_20260912_gap3_disturbed.txt`;
tools `fixtures/gap3_regimes.py` and `fixtures/gap3_pooled.py`, statistic and gates unchanged
from `fadr_gap3_shadow2.py`.

**THE ANSWER: no aid available.** Pooling all eight chains (legitimate once code rate is divided
by `k = f_chip/f_c`, which makes the physics chain-independent), and reporting the **slope —
which IS the aiding gain GAP 3 would deliver** — not merely `r`:

| regime | n | aiding gain | 95% CI (bootstrap) | |
|---|---|---|---|---|
| ramps 0.02–0.2 chips/min | 438 | **−0.041** | [−0.105, +0.017] | excludes ±1 |
| ramps ≥ **0.06** chips/min (the motive) | 57 | **−0.062** | [−0.209, +0.063] | excludes ±1 |
| all pairs | 7013 | −2.480 | [−6.15, −0.391] | **underpowered** |

⚠️ **THE SENSITIVITY CHECK IS THE RESULT, NOT A FOOTNOTE.** A null is worthless without its own
lever. Here the CI excludes ±1 by ~10× in the ramp band and ~5× at the motivating level: the
test demonstrably *could* have seen a physical coupling and did not. The aid is bounded to
**≤10% of the ramp** over the ramp band.

⚠️ **P3's escape hatch does not apply.** The worry was that a null might just mean the day was
calm. It was not: **57 pooled pairs sit at or above the 0.06 chips/min that motivated GAP 3**,
inside the clean ramp band. The phenomenon is present and the coupling is still absent.

**THE ONE "SIGNIFICANT" RESULT IS THE TRAP, AND ITS CAUSE IS WORTH MORE THAN THE RESULT.**
All-pairs gives r = −0.068 at p = 0.024 — with a slope of −2.48 whose CI runs −6.1 to −0.4 and
distinguishes nothing. It is driven entirely by a tail reaching **242 chips/min**.

⚠️ **`code_resid_m` IS WRAPPED AT ±½ CODE PERIOD, AND DIFFERENCING IT ACROSS A WRAP FABRICATES
THE RATE.** Measured over the 24 h on gps_l5: min −5060 chips, max +5104, **span 10164 against a
10230-chip code period (0.9936 of one period)**. The largest in-arc "excursions" are 9357–10140
chips — i.e. exactly one wrap — and **34 of 406 carrier arcs contain one**. A first pass here
called these slips or reacquisitions; they are not, and the distinction matters because a slip
would be a real event while a wrap is an artefact of the observable's range. **Any `d/dt` of
`code_resid_m` must unwrap or band-limit first.** The banded analysis above does band-limit — a
wrap in a 600 s window implies a slope of order 10230/600 chips/s, orders outside the
0.02–0.2 chips/min band — so the verdict is unaffected; only the un-banded row is poisoned.

**gps_l5's exception is confirmed as the documented artefact, and now quantified.** #93 warned
its r = −0.417 was "manufactured by slew transfer (its seed absorbs trim content every ~600 s)".
Sweeping the fitting window shows exactly that — and the control shows it is specific to gps_l5:

| W | gps_l5 r (slope/k) | gal_e5a r (slope/k) |
|---|---|---|
| 300 s | **+0.026** (+3.10) | −0.034 (−0.35) |
| 600 s | −0.118 (−6.99) | −0.005 (−0.03) |
| 900 s | −0.112 (−2.41) | −0.028 (−0.07) |
| 1200 s | **−0.501** (−21.05) | −0.077 (−0.19) |

gps_l5's sign flips and its magnitude swings 20× with the analysis window; gal_e5a is stable and
null throughout. A real per-satellite physical coupling cannot do that. Band-limiting to ramps
collapses gps_l5's slope/k from −6.99 to −0.04 and its p from 0.027 to 0.349.

**Per-chain, W = 600 s, ramp band.** One nominally significant result — gal_e5b, r = +0.257 at
p = 0.035 — with `slope/k = +0.10`, a tenth of physics, and it is one of 16 tests (8 chains × 2
cuts), where ~0.8 hits below p = 0.05 are expected by chance. Not evidence.

**What would reopen it.** This is one day of *ordinary* operation, ramp tail included. An
extreme plant episode (an E3-style event) is not in it, and the bound above does not speak to
one. But the burden has moved: GAP 3 is closed unless such an episode is caught and shows a
coupling at physics strength.


## Faults still worth reading in full

Three closed items teach something no summary carries. Their narratives live in git history;
these are the one-line reasons to go and read them.

- **#98 — a log-only monitor between `if` and `elif` ate the hold chain for five weeks.** An
  `elif` has no name for what it chains to, and a judging window must overlap the regime that
  arms the suspect.
- **#107 — a post-publication patch of ring slot-0 metadata killed nine nodes.** A gate for a
  producer must run the producer; and "verified" over a window in which the fault cannot appear
  is a gate that cannot fail.
- **#99 — the station position was the array origin, 155 m off.** Never map a solved offset to a
  correction by reading sign conventions out of code: command a step and read the response.
