# CHORD GNSS — closed items

The durable record of what was fixed, so `CHORD_BUGLIST.md` can hold only what is open.
One line per item: **what it was, what closed it, how you would check.** Commit hashes are the
addressable half and are kept; **line numbers are deliberately stripped** — at the 2026-09-10
reconcile 6 of 12 citations in the old closed index pointed at code that had moved, and a wrong
address is worse than none. The full narratives, including every retired hypothesis, are in this
file's git history (`git log -p docs/CHORD_BUGLIST.md`).

Reconciled 2026-09-10 against HEAD `a5e7d686c`, 496 commits after the previous pass, by seven
parallel section audits that each read the tree AND the live fleet.

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

## Withdrawn — tried, measured, backed out (verify these stay dead)

| what | why | the grep that proves it is dead |
|---|---|---|
| deep-gate widening 100 → 50 | an active confound | `dll-deep-gate` is `4 9 27` / `33` / `33`, never `all` or `50` |
| GAP 2 feed on a model-primary chain | it measured its own seeded rate (the mirror) | the spec-anchored successor is armed on gal_e5a only |
| `joint-consume: slew` on a feeding chain | closes the loop | only gps_l5 takes `slew`, and it has no feed flag — **but see #124: the joint state is band-scoped, so this is a per-chain label on a shared object** |
| freezing the fast loop through establishment | made L5 worse (57–95% → 14–38%) | `--establish-hold` default 0.0, absent from the chains yaml |
| B1 split-aperture | comparator error | recorded in `gnss_fleet_chord.yaml` |
| `despread_max_chips 105` | superseded by CENTERED-80 | 105 appears only in a one-off cx19 config |
| two Doppler-free Φ tables per channel | 0.88–0.92× on the A40 — a loss | `shared_phi` appears in no config; only bench callsites |
| tiled shared-memory streaming | 0.60–0.70× at every geometry | `launch_waveform_tiled` has zero production callsites |
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
