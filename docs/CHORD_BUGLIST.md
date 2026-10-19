# CHORD GNSS — open items

What is open on the CHORD side (branch `kv/chord-gnss`). Closed items are in
[`CHORD_BUGLIST_CLOSED.md`](CHORD_BUGLIST_CLOSED.md); their narratives are in git history.

**Last reconciled: 2026-10-13 after the dead-code merge** (previous passes 2026-10-11 at `0011cf3b1`, 2026-09-12 at
`b45949144`). The 10-13 pass re-checked every open entry's anchor against the tree after `d505f3943`: all still
present except the 09-14 mitigations of #131 (gone with the launchers) and the generator's profiling switch #114 relied
on. Closed since 10-11: **#55, #62, #63, #113, #115, #121, #125, #144, #146 residual, #150, #159, #161, #172, #173,
#174**, the #1750 latent lock (PR #1763), and the 10-11 batch (#31, #34, #88, #102, #107 residual, #112, #123,
#130/#153, #133, #149, #160, #162). Opened: **#175–#189**. The same evening the ready-now set ran in two reviewed lanes and merged: **#186, #183** closed; **#176, #175** have their guards built (live since the 21:05Z broker restart); **#119, #152** are instrumented; **#94**'s gauge test is in; **#131**'s shutdown race is fixed and its death not reproduced. Overnight (10-14) the hygiene set merged, nothing live changed: the `test_skyscope` flake and the replay goldens fixed, **#129**'s GLONASS propagator landed, **#180** and **#179** fixed in the tree for the next node binary. The 10-11 and 10-13 memory notes record every check.

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

### #207 — every search-derived code phase carried range_rate x 16 ms: `detection_phase` left the replica anchor's code-Doppler advance on `cp0` and `cp_long` (2026-10-18, live, measured three ways; ROOT CAUSE FOUND + FIXED 10-18 21:30Z, DEPLOYED 22:53Z, live gate passed)
**What [live 10-18, gps_l5 + gal_e5a]:** the broker's search-derived offsets (`_det_offset`: the detection's sample-0
argument re-advanced to `ref_hop`, minus `cp_predicted` at that hop) feed the clock solve, the joint state's b_sat and
the SEARCH-MONITOR. Within one satellite's pass they move linearly with its Doppler: GPS b_sat vs the tracker Doppler,
per PRN over the day, slopes -4e-5..-1.8e-4 chip/Hz with r down to -0.87 (G03 over -2.4..+2.5 kHz), pooled over 11
PRNs and 3,293 points **-8.6e-5 chip/Hz = a +9.9 ms lever** (d(cp)/d(dop) = chip_rate/f x dt); SEARCH-MONITOR gal_e5a
per PRN -9.6e-5 (r -0.83, +11 ms); GPS b_sat per PRN -1.2e-4 (+13.7 ms). The tracker path has NO such term: today's
obs `code_resid_m` vs `range_rate_mps` with per-satellite intercepts gives -1.8/+1.2/+1.8/+0.5 ms on gal_e5a/bds_b2a/
bds_b3i/gal_e5b. So a range-rate x ~10 ms error (+-8 m at +-2 kHz) sits in every search-derived measurement and in
nothing the loop converges on; the audit's "b_sat swings to -1.2 chips at rise" (10-12) is partly this, and the joint
clock's median is biased by the fed set's mean range rate x 10 ms. **Where it is not:** the shipped harness run at
nonzero Doppler (scripts/gnss/e2e, rebuilt 10-18 on cf05 against builds/fleet-954af6047; its default `--dop 0`, the
only setting it had ever been read at, cannot see a lever, and its 27-channel stride-4 search comb aliases by 4096
samples = 13.09 chips at |dop| >= 1 kHz, so judge it at +-500..700 Hz) puts the cp0 path within the refine's +-0.07
chip of truth at the labelled snapshot start at every Doppler (new CP0 LEG ERROR line), while `cp_at_ref`, the
transported seed, carries a ~-23 ms lever (+0.081 at +500, -0.171 at -500, -0.247 at -1000 Hz) that the tracker loop
absorbs; on live detections the two differ by one hop (52.4 chips) plus a 16.0 ms lever. **Where it must be:** between
the detection's label and the model's epoch on the aggregator path: `_snap_start_hop` is the first frame's
`sample_seq / fft_len` (GnssChannelizedSearch.cpp:891), `t_i = ref_hop / hops_per_sec`, and `cp_predicted` propagates
range from `t_eph_age` (wall axis) to `t_i` (hop axis); a ~10 ms disagreement between those axes, or a one-record
(10.49 ms) offset in the search feed's `sample_seq` against the nodes' records, gives exactly this. The 0.486 ms
residue of a one-record error would land a 4,972-chip constant in the clock, which the model-primary chains would not
survive, so a round 10 ms (an axis) is likelier than a record (a label). **Next:** pin the live number with the fit
above every hour (it is 10 ms to +-2), then compare the search feed's `sample_seq` with the same node's record hops
for one frame, and `utc0_sample0` against the hop axis (the F-engine `time0_ns` convention: frame start or end).
**Fix [broker or aggregator]:** whichever axis is wrong; the harness then gets a `--dop` sweep in its gate.
**21:00Z, NARROWED TO THE SEARCH STAGE (e2e.cf05, built 10-18 from the tree against builds/fleet-954af6047):** with
PRODUCTION parameters (`--s-stride 1 --s-nchan 79 --refine-step 103 --fine-step 128 --refine-span 512 --refine-hops
391 --cuda-refine`) the cp0 leg error is a clean line in Doppler: +0.389, +0.197, +0.102, +0.087, -0.010, -0.103,
-0.214 chip at -2000..+2000 Hz = **-1.5e-4 chip/Hz = a 17 ms lever on cp0, the quantity the broker's clock solve,
joint feed and SEARCH-MONITOR use**, while cp_at_ref (the tracker's seed) stays flat (+0.03..+0.11) and the CUDA
refine equals the CPU one bit-for-bit. With the harness's DEFAULT parameters (27-channel stride-4 comb, fine refine
step) the roles swap: cp0 flat, cp_at_ref -23 ms. The two phases come from ONE refined peak through
`detection_phase`, so the lever is in how that peak's epoch is referenced versus the refine window/grid
(`refine_span`/`refine_step`/the comb), not in the broker's Python (ruled out: trkresid uses the same
`cp_predicted(v, hop/hps)` and is clean; the VM's NTP is at 40 us). Single-parameter toggles at +-2 kHz on the
default comb (which aliases by 13 chips at |dop| >= 1 kHz, so only the sub-chip rows count): comb79 alone cp0
+0.744/+0.035; refine-step 103 alone +0.151/-0.247; span 512 alone +0.145/-0.564 -- each moves it. **Next
(cf05, harness only, no sky):** print the acquire's coarse peak and the refine window's first/last hop against
truth in e2e, read `detection_phase` (lib/stages/gnss/gnssSeedTransport.*) and `refine_peak`'s reference epoch, find
the interval the Doppler advance is applied over for cp0 vs cp_at_ref, fix it in the stage, gate with a `--dop`
sweep at production parameters. Then a broker restart (the aggregator binary carries the stage: a rebuild + KV
cycle of gnss-aggregator) and re-fit b_sat vs Doppler live (must go to 0 +- 2 ms).
**21:30Z ROOT CAUSE (by reading, then three gates).** The refined peak `best_cp` is the argument of a replica
generated at the search's fixed `anchor` (= Mp*fft_len samples, one replica period = 16 L = 16 ms of code at CHORD)
that matches the data at the snapshot start S_snap. Every generator forms C(n) = arg + n*cps*(1 + s*dop/f) over the
ABSOLUTE sample index, so matching the two gives arg0 = best_cp - (S_snap - anchor)*cps_d. `detection_phase`
(lib/stages/gnss/gnssSeedTransport.cpp) removed the advance over S_snap alone: the anchor's nominal share, 16 L, is 0
mod L and hid the omission from `cp0`, but its code-Doppler share, 16 L * s*dop/f, is not -- a Doppler lever of
exactly one replica period, 16 ms, on `cp0`; and 16 L is not 0 mod 20 L, so `cp_long` was also 16 (= -4 mod 20)
periods off at every Doppler (latent: the tracker prefers the `cp_at_ref` phase, which adds the anchor's advance
back and was always right -- the roles are the reverse of what `_det_offset`'s docstring claimed). Predicted slope
16 L / f = 1.392e-4 chip/Hz; the 10-18 harness sweep measured -1.5e-4 through a 0.33-chip refine step, and the live
"cp_loc - cp_at_ref = one hop + 16 ms" was the same term. Why it was never seen: the acquire tests and the
harness's only-ever `--dop 0` all have the data AT the anchor or at zero Doppler, where both shares vanish.
**Fix [committed 10-18]:** one advance over `S_snap - anchor` in long double (as `window_advance_chips`, with the
generators' own double `cps_d`; the old `% Mp` shortcut also assumed Mp*fft_len*cps is exactly 16 L in floating
point, ~1e-3 chip at 4e7 periods), reduced mod L for `cp0` and mod n_nh*L for `cp_long`; `cp_at_ref` unchanged.
**Gates:** (1) tests/boost `detection_phase_removes_the_anchor_at_every_doppler` (data 3.2 days from the anchor;
the anchor-replica convention checked by despread; cp0 == arg0 to 1e-5 at +-4 kHz; cp_at_ref mod L == cp0
re-advanced); (2) the harness, now with a CP_LONG LEG line: OLD library at production parameters cp0 +0.087/-0.214
chip at 0/+2000 Hz and cp_long +4 periods at both; NEW library at -2000, -1000, -500, 0, +500, +1000, +2000 Hz: CP0 LEG +0.112, +0.060, +0.034, +0.088, +0.059, +0.034, +0.062 chip (flat, the refine's 0.33-chip step), CP_LONG and SEARCH LEG identical to cp0 at every point with 0 periods, CUDA refine = CPU to the bit; (3) LIVE, on the running
aggregator before the fix (scratch script polling /gps_search/get_detections): resid = cp_at_ref - cp0 re-advanced
equals 1.3913e-4 chip/Hz x dop on all 5 GPS detections to 4 decimals = a 16.00 ms lever, no noise -- after the cycle
it must read 0.0000 on every PRN. The per-PRN b_sat-vs-Doppler re-fit (pooled +9.3 ms over 18:40-21:10Z) is NOT a
clean gate: within a pass Doppler and time are degenerate and unmodelled tropo/iono (0b) is elevation-dependent, so
per-PRN slopes come out mixed-sign (G4 +16.6, G9 +15.6, G6 -15.0 ms); read it only as "the pooled slope moved by
~-1.4e-4 chip/Hz". **Consequences to re-read after the cycle:** SEARCH-MONITOR gal_e5a (its per-PRN scatter, MAD
0.15-0.33, was this lever: the +0.49 chip Galileo median (#191) was taken over 11 PRNs with mixed Dopplers and may
move); the joint clock's bias by the fed set's mean range rate x 16 ms; the PVT self-survey, which fed the search's
code phase (#99 era) and so carried Doppler-correlated pseudorange errors -> 0b's re-survey. **Deploy:** the stage
lives in the aggregator binary (build_nodpdk/kotekan/kotekan, shared with the gather and cf06's cube archiver):
builds/agg-<sha> on cf05 with the gather recipe, staged by builds/stage_agg_<sha>.sh (ldd + three check-configs +
hard-linked rollback) immediately before KV's `systemctl --user restart gnss-aggregator`; the gather keeps its
mapped inode until its own next restart, which will then run this build too. **Restart the broker right after
the aggregator** (skeptic review, 21:35Z): its per-PRN histories (cp-rate fit window, innovation and SEARCH-MONITOR
5-min medians, the joint state's b_i) would otherwise mix pre- and post-fix detections, which step by 1.39e-4
chip/Hz x dop (0.56 chip at 4 kHz) and by 4 periods in cp_long at the cycle; the 100-chip jump guard does not
refuse that, and the cp-rate fit would read it as a slope for one window. Behaviour change to know: the seeding
fallbacks that strip `cp_at_ref` (hold_freeze with no at_ref, dr_seed_phys/tracker_phase_at's argument branch) were
seeding 4 overlay periods off from `cp_long` and are right after the cycle. **Reading the live result:** the
`resid == 0` poll proves the new stage is running (both fields come from one best_cp by construction); the
independent live gate is a search-vs-tracker clock per PRN against Doppler (SEARCH-MONITOR gal_e5a per-PRN values,
or the GPS search clock minus trkresid), whose slope must move from ~-1.39e-4 chip/Hz to ~0. The pre-fix b_sat
slopes (-8.6e-5..-1.2e-4) were 0.6-0.9 of the prediction, so a residual of ~-3e-5 chip/Hz after the cycle is the
pre-existing elevation/iono degeneracy (0b), not a failed fix.
**22:53Z DEPLOYED (KV):** builds/agg-daee06222 staged into build_nodpdk (md5 72d3a36c4b2b; rollback
`kotekan.prev_4514061a6_20261018`), gnss-aggregator then gnss-broker restarted at 22:53:02Z; 0 CHAIN DIED, cross-band
bootstrap at T+15 s. **Live gate PASSED at 22:54Z:** resid = cp_at_ref - cp0 re-advanced = +0.0001 chip on all six
GPS detections over -2834..+2236 Hz (the old stage would have read -0.394..+0.311), i.e. the lever is gone from the
stage. The gather still runs the previous inode (00d51e99b6e4) until its own restart. Independent live read-back
(search-vs-tracker per PRN vs Doppler, SEARCH-MONITOR gal_e5a, the b_sat fit) follows once the joint state has
re-established. **23:11Z (T+18 min):** invariant still 0.0000 on 7 PRNs (-2911..+2236 Hz); joint state re-converged
(n=7, clk 151.83 +- 0.08 chip, 0 CHAIN DIED); SEARCH-MONITOR gal_e5a per-PRN scatter tightened to MAD 0.07-0.16 chip
(pre-fix 0.15-0.33: that scatter WAS the lever), 5-min median +0.577 chip (+16.9 m, p10/p90 +0.45/+0.69) against +0.49
pre-fix -- the datum (GPS L5 search clock) and each Galileo PRN moved by 1.39e-4 x their own Doppler, so the #191
constant now reads +0.58 chip with the lever gone. The b_sat-vs-Doppler fit over 22:56-23:11Z (196 points, each PRN
spanning ~300 Hz) is not interpretable: the post-restart settling aliases into a within-pass slope (per-PRN +242..-36
ms); re-fit over full passes (hours) before reading it. **#207 is closed on the stage and the live invariant; the
pooled-slope re-fit is the remaining read-back (0b territory).** **10-19 10:47Z re-fit over the night (01:30-10:47Z,
7,076 points, 18 PRNs, with 0b live): pooled slope +2.3e-5 chip/Hz = -2.7 ms, from -8.1e-5 (+9.3 ms) before; the
change is +1.0e-4 of the predicted +1.4e-4, the rest the within-pass degeneracy (per-PRN -11..+1 ms). CLOSED.**

### #209 — `--detectors-monitor` (the E5a search experiment) silently disables the slewed seed refresh: every locked gal_e5a satellite is re-born every ~8 min with a trim wipe, and the obs rows carry 100-1000 m excursions since 2026-10-18 18:37Z (2026-10-19, overnight check; FIXED 2672b773b, LIVE since the 12:18:53Z broker restart)
**What:** gal_e5a's `code_resid_m` rows have >20 m excursions on most PRNs since hour 18 of 10-18 (308 of 3,995 rows in
that hour, 8 the hour before, ~600/h all night; 10-18's night had 3 in 14,826), with the per-PRN medians still fine (+-1 m)
but p10/p90 of -100..+200 m (E2 -384/+192, E11 p90 +3,310). The night re-survey fit on gal_e5a reads rms 7 m with half
the rows clipped, while gal_e5b (no detectors) is clean (rms 0.75). Broker log since the 00:56Z restart: gal_e5a 147
`SEEDAUDIT STEP PRN N: -7.7 chips (= -0.004 Hz x lever) ddop ... dt ~460-500 s [dop=dr_birth cp0=dr_birth ...]` + 312
`SAWTOOTH ... BARE-WIPE` + 374 HOLD-BY-PROMPT, against 3 / 0 / 41 on gal_e5b. **Why (deadreckon.py dr_seed):**
`_slew = prn in seeds and (not ctx.detectors or prn in mp_flipped) and ...` -- the slewed refresh of a locked
dead-reckon seed is gated on "this chain has no detectors"; the monitor mode sets `ctx.detectors` (the e5a_search URL)
although its detections never seed, so every locked PRN takes the plain-hold path and is re-born on the hold cadence,
each re-birth re-basing the seed under a standing trim (the tap moves, the trim rebuilds = the sawtooth). The stage-9
reviewer flagged it on 10-18 20:45Z ("--detectors-monitor also turns off dr_seed's slewed refresh ... consistent with
E33 walking to the 3.0 trim clamp"); it sat in #192's list. **Not** the 22:53Z (#207) or 00:56Z (0b/position) restarts.
**Fix:** every seeding decision that asks "does this chain have detectors" must ask "do its detections seed" --
`ctx.detectors` and `ctx.args.detectors_monitor` together (a helper); then a broker restart. Until then gal_e5a's code
observables are unusable for the fits (the SEARCH-MONITOR itself is fine: it reads the search, not the tracker).

**12:19Z LANDED:** gnss-broker restarted 12:18:53Z on 06be1fdd8 (seeds_are_model_owned); 0 CHAIN DIED, no tracebacks. The
verdict is the SEEDAUDIT STEP / SAWTOOTH rate on gal_e5a and the obs excursions over the following half hour (overnight
~600 excursions/h; gal_e5b ~0).

### #211 — restart ORDER: a broker restart within minutes of an aggregator restart bootstraps on a cold search, and every model-primary chain is thrown onto grating lobes for half an hour (2026-10-19 12:17-12:50Z, measured, self-healing)
**What:** gnss-aggregator restarted 12:17:26Z, gnss-gather 12:18:03Z, gnss-broker 12:18:53Z (the post-merge regen cycle).
From 12:24-12:28Z every satellite on the model-primary chains without a search (gal_e5a, gal_e5b, bds_b2a, bds_b3i;
gps_l5 and gps_l2c unaffected) jumped by hundreds of metres in the tracker code residual (up to -1,545 m) and then
walked back toward 0 at exactly ONE GRATING LOBE (95 m = 0.32 us = 3.27 chips, the tracker comb's 3.125 MHz channel
spacing) per ~2 min, the trim-clamp sawtooth pulling each tap home lobe by lobe; most PRNs were back by 12:46-12:48Z,
the rest a lobe away by 12:50Z. Presence counts never dropped (served C/N0 and q are blind to a lobe). The 00:56Z
restart (broker alone, on a search warm for 2 h) shows none of this. **Chain of events (broker log):** the restarted
aggregator's search had no overlay labels or hints (9,207 LABEL CONSENSUS / nh-hint lines in its first minutes); at
12:19 gps_l5 took its seeds from those detections -- "PRN 24 overlay segment DERIVED from consensus: +5 period(s) off
the weak measurement", "period DEBOUNCED", "SEEDAUDIT STEP PRN 24: -2633.99 chips [cp0=nh_lift]", G18 +-390..-4,908
chips over the next minutes -- so the joint state re-established on 2-3 satellites with sigma +-1.0 chip (12:22:56Z,
152.107 vs 151.75 a minute later); the consuming chains adopted it and at 12:23:11-12:23:12 every PRN on gal_e5b and
bds_b2a was slewed ("dead-reckon SLEW PRN N: model-held +1.35 chips, step +0.050 (cap 0.05)"); the fleet DLL trims rose
from +0.4 to the +3.0 clamp by 12:24 (visible in the obs dll_trim_cpp), the wipes dropped the taps onto far lobes, and
the walk back began. The gather restart (trims HELD then adopted, standing +0.3..+0.4 chips) was not the trigger: the
trims only moved once the slews began. **Rule:** the aggregator is the one instance never to cycle beside the broker;
when it must be cycled, wait until its LABEL CONSENSUS / relabel lines have quieted (>= 10 min) before any broker
restart, and never restart the broker on a search younger than that. Candidates for a code-side guard: the broker
refuses dr_birth/nh_lift seeds while the search reports relabels, or the joint state's first estimates (n < 4 or sigma
> 0.5 chip) are not adoptable (the 0c coast already bounds adoption by joint_clk_max_chips; a sigma gate would have
held this one). Related: #106 (the establishment transient), #201 (the BOC side peaks: the same lobe blindness).

### #210 — glo_l3oc's code residual carries a GLONASS-only range-model term: +44 m up-like, +8 m north, +5 m east against the GPS/Galileo/BeiDou chains (2026-10-19, the 0b night gate)
**What:** the night re-survey fit (gnss_resurvey.py --hours 6-13) on 10-19 with the model live reads glo_l3oc model -
true = +44.1 U / +8.2 N / +5.2 E (rms 3.5 m, 5,382 of 6,521 rows) while gps_l5, gal_e5b, bds_b2a and bds_b3i read
+0.6..+1.2 U, +-0.4 N. On 10-18 (model without atmosphere, position 545) it read +59.3 / +12.0 / -3.6 with the
atmosphere added back, against +18 / +3.3 for GPS: the GLONASS excess (~+41 U, +8 N) was there before the position
change, so it is in the GLONASS range model, not the site. **Candidates:** the state-vector integration from tb (a
position error that grows with age; GLONASS_MAX_AGE_S), the PZ-90.11 frame or the GLONASS time scale (UTC(SU) + 3 h,
tau_c, no leap seconds) in the light-time / Sagnac path, or the L3OC code bias. A height-like error of 40 m at 2 x
10^4 km is ~2 ppm of range -- an orbit integration error of that size over hours is plausible. **Next:** per-satellite
residual vs ephemeris age on glo_l3oc; compare the integrated position with a precise orbit for one satellite.

### #208 — wall time sits in the radiometric path: the ephemeris is evaluated at `now_w` and labelled `t_eph_age = now_w - utc0_sample0` (2026-10-18, design, KV: nothing radiometric may depend on wall time; FIXED 10-18 23:45Z: every radiometric "now" is the hop axis (transport.axis_now); lands at the next broker restart)
**What [tree]:** deadreckon.py:2224-2292 runs `predict_all` at the wall clock `now_w` and stamps the rows'
capture-age epoch as `now_w - utc0_sample0`; `cp_predicted` then propagates range at `range_rate` from that wall-derived
epoch to hop-axis epochs (`ref_hop / hops_per_sec` for detections, `rec_hop / hops_per_sec` for records). The comment
there already knows the two are different clocks. `--dr-fengine-axis` (on, common:) moves "now" onto the hop axis
for the seeds but not this epoch. Every wall error (NTP offset, the VM's clock) enters as range_rate x error per
satellite; the VM's chrony reads 40 us today (RMS 0.3 ms), so no number moves now, but the rule stands: any time
used radiometrically derives from frame 0 (`time0_ns`) and a sequence number. Also wall-stamped: `det_fresh` (a
freshness gate, not radiometric) and the joint-feed warm-up (bookkeeping). **Fix [broker]:** evaluate the ephemeris
at the hop-derived now (the fe-axis newest hop / hops_per_sec, converted to GPST through `time0_ns` + leap) and set
`t_eph_age` to that hop time; keep wall time only for elapsed-time bookkeeping. A few lines in stage_dead_reckon +
the selftest that pins "no wall time in cp_predicted's inputs" (grep-based, like test_gauge_churn). Needs a broker
restart.
**10-18 23:01Z THE DEAD-RECKON EPOCH IS ON THE HOP AXIS (9f9808c7c + the merge follow-up):** `dr_eph_epoch(ctx)` in
gnss_broker/deadreckon.py returns (t_eph_age, UTC seconds) = (t_now_abs, utc0_sample0 + t_now_abs): under
`dr-fengine-axis: 1` (every chain) t_now_abs is the F-engine hop counter interpolated by wall ELAPSED seconds only, placed
on UTC through utc0_sample0, the F-engine's own time of sample 0. All seven model evaluations in the dead-reckon block
(predict_all x3, predict_from_decoders x3, transit_sky), `decoded_entries` (its epoch argument was dead and read
`_drp.now_w`; fixed in the follow-up) and the `t_eph_age` label take that epoch. **What this changes numerically: almost
nothing, and the 9f9808c7c commit message overstates it** (skeptic wf_7bc7b2f7, by a numeric check): cp_predicted
carries range from the evaluation's OWN label, so range(T_e) + rr x (t_abs - (T_e - utc0)) = range(utc0 + t_abs) to first
order for any T_e; the old wall form was exact given NTP (40 us here) and the move is second order, a x (t_abs - label)
x lag ~ 1 cm at the forecast lead, ~1 m for a 60-s-old detection in the clock solve. The one first-order change is the
seed Doppler carry (dop_geo now evaluated at the label it is carried from): drate x (lag + NTP) ~ 0.04 Hz, below
anything observable. So there is NO 0.3-chip re-base to expect at the restart; the change is design (KV's rule, no
wall-clock reading in a radiometric epoch), and it lands at the next broker restart with no visible step. Selftest:
3 checks (epoch = utc0_sample0 + t_now_abs labelled t_now_abs; a 1-day wall step does not move it; no
`fromtimestamp(ctx.drp.now_w` left in deadreckon.py). test_drift_freeze/axis_freeze/clockguard/timebase/dr_reload/
seed_bias/context pass. **STILL OPEN, the wall-time uses the skeptic listed (radiometric, NTP-true today):** (1) the
almanac stage evaluates at the wall clock (`_alm_now()` in gps_distributed_broker.py -> almanac.py t_pred -> ctx.pred:
search hints and coasted-seed Doppler); (2) seeding.py builds a detection age and the cp0 retag from
`now_w - utc0_sample0` against hop labels; (3) searchhint.py builds a hint ref_hop from `ctx.t0 - utc0_sample0`.
Each wants the hop-axis now (ctx.drp.t_now_abs) in place of the wall difference; one pass, one broker restart.
The WALL-vs-F-ENGINE tripwire's text now says what a wall step does (nothing to the models; those three paths).
**10-18 23:25Z THE REMAINDER, DONE (one pass):** `transport.axis_now(ctx)` is the one "now" every stage refers to: the
F-engine hop counter interpolated by wall ELAPSED seconds (the filtered fe_off plus the frozen cycle clock; the raw newest
hop plus the seconds since its poll while the filter has no sample; the frozen clock minus utc0_sample0 only before the
first telemetry or without --dr-fengine-axis), None until utc0_sample0 is known. Consumers: (a) dead-reckon's t_now_abs
(the block that used to compute this inline, now the only definition); (b) the ALMANAC epoch (`axis_now_utc(ctx)`, the
wall clock only before the anchor is known). The TIME-BASE DETECTOR (gnss_broker/timebase.py) compares the search's
measured Doppler with ctx.pred to catch a stale sample-0 epoch from the nodes; a wall-evaluated almanac never moved with
that epoch, AND (skeptic wf_12d15128) its freshness gate compared the wall clock with stamps built as utc0_sample0 +
ref_hop/hps, so under a past-stale epoch every detection read hours old and the detector starved either way. Both now on
the hop axis (`transport.now_utc_s` in timebase.observe and the almanac's bias solve): a stale anchor cancels in the age,
the prediction moves with the epoch and the measurement does not, which is the spread it looks for; the bias then reads
stale, so searchhint keeps the wide +-3000 Hz margins (~2 h of epoch error) and the search keeps acquiring while
geometry is withheld. (c) seeding.py's coast age and cp0 retag (hop-axis now minus ref_hop, the axis ref_hop
is on); (d) searchhint.py's nh-hint ref_hop and the UTC its predicted alignment is evaluated at. `_alm_now()` keeps the
wall clock for the BRDC store's day key at setup (a `208-allow` line). Selftest: five axis_now cases (no telemetry, raw
hop + elapsed, filtered offset + frozen clock, flag off, no anchor), the dr_eph_epoch pair, and a package-wide source grep
(model-at-wall-time and wall-minus-anchor patterns over gnss_broker/*.py + gps_distributed_broker.py, test files
excluded, the two allowed lines marked). All 27 gnss_broker test modules pass. **Lands at the next broker restart** (KV,
when convenient; the 22:53Z restart is settling). Expected visible change (the skeptic's accounting): the coasted seed's
age and retag used `ctx.drp.now_w`, which only stage_dead_reckon sets, AFTER stage_coast_drop and only every 2 s, so the
old "now" there was 0.25-2 s stale (and None before the first dead-reckon pass); the hop-axis now corrects the stored
coasted Doppler by up to ~2 s x rate (~0.8 Hz). Hints and the almanac move by the telemetry lag x rate (~0.04 Hz). Also:
the almanac and hint stages run before the cycle's telemetry ingest, so they see the previous cycle's axis advanced by the
frozen clock (ms; a snap's worth for one cycle after an offset snap); the transcript digests (broker_equiv) move with this
change (coast retag, nh-hint ref_hop) and are to be re-blessed when #192 (g) makes them deterministic; the selftest's
first grep could not match the call spellings it named (a `\b` after `)`) -- fixed and self-checked.

### #206 — viewer: L2OC's dark probes grey out satellites that L3OC holds, and the not-transmitted table is GPS-only (2026-10-18)
**What [live 13:05Z]:** glo_l2oc runs with `--probe-dark`, so its three noise probes are GLONASS-M satellites above the
horizon (R4, R5, R12 at el 62/62/9 deg) that cannot broadcast L2OC and that glo_l3oc holds (q 5.9/5.6/3.4 at 13:05Z). The viewer
merges the chains' rows per satellite, and app/panels/gps_feed.js:390 promotes any one signal's `noise_probe` to the row
("a satellite that is a noise probe on any signal is below the horizon on all of them": true until 10-18, false now),
then :420 forces the row inactive: R4/R5/R12 render as unlocked (grey row, hollow ring, not in "N locked") in the table and
the sky while their L3OC cell shows a green q. That is 3 of the 7 GLONASS rows reading unlocked, each of them locked
on L3OC, beside an L2OC column with 4 blanks (those three plus R21) for a chain that holds 3 of its 3 capable
satellites: the "half of L2OC unlocked" on the viewer. Second: livebeam_server.py `_gps_signal_capability()` builds
the not-transmitted table from the IGS GPS block registry only, so GLO_L2OC_P, GLO_L3OC_P, BDS_B2A_P and BDS_B2B_I
ship as EMPTY lists with `_prns` = 1..32; gps_feed.js `not_transmitted()` then returns true for every blank
GLONASS/BeiDou cell with PRN <= 32 (under the C/N0 metric; the q metric prints the number), so the table prints "·"
(not broadcast) where it should print "—" (no detection). In the 13:05Z snapshot the masked cell is R6's L3OC: a K2 that
does transmit L3OC, whose L3OC loop sits off-peak (q 0.03, early tap 32x the prompt) and prints "·" as if incapable;
R9 (K1B, never acquired on L2OC, #205 a) and R11 print the same when up. GAL_* and BDS_B3I ship the GPS 1..32 list
too (harmless beyond PRN 32). **Fix [viewer, script-side]:**
keep `noise_probe` per signal (`sig_by[key]`), judge `active` over the non-probe signals, render a probe cell as
"probe" in grey; build the capability per constellation (the server already has `_glonass_capable_set()` via
`prnmap.registry_capable(sigid)`; BeiDou from prnmap's BDS-3 set) and key `_prns` by constellation. **Fix shipped 10-18 (viewer code; the restart is KV's):** gps_feed.js keeps `noise_probe` per signal,
counts probe signals per row and calls a row a probe only when every signal reporting it is one; the row summary (sig,
C/N0, q) skips probe signals; gps_table.js renders a probe cell as "probe" (greyed value when the metric has one);
livebeam_server `_gps_signal_capability()` builds the lists per constellation from `prnmap.prn_blocks(sys)` and ships
`_prns` as {sys: [...]}, and `not_transmitted(key, prn, sys)` looks the PRN up in its own constellation. Checked: the
builder on the live inventory lists L2OC = the K slots only and L3OC = K + M+, B2a/B2b = BDS-3 only; test_lock_metrics.py
(dukpy, three new cases) OK. **Judge after `systemctl --user restart gnss-viewer`:** R4/R5/R12 rows filled, their L2OC
cell "probe" under every metric (the first cut passed the q/sig/hold/duty formatters' own red markup through,
KV 10-18 14:30Z); R6's L3OC cell "—", not "·"; "N locked" counts them.

### #205 — GLONASS, the first night: what the two OC chains showed, kept as watch items (2026-10-18)
**What [overnight 02:43-12:30Z, judged from broker.log, the obs rows and chain_health]:** (a) **R9 never acquires on
either band**: armed on glo_l2oc 07:56-11:21Z up to 26.3 deg (2,590 rows above 20 deg) and never present; glo_l3oc the
same (2,578 rows). Not L2OC-specific, but both bands share our slot-to-code mapping and the beam, so "satellite side" is
not shown (the 10-17 L3OC WATCH: code index != slot?). It left glo_l2oc at 0 present 08:00-08:59Z, and chain_health
(cf06) flagged glo_l2oc 08:35-09:20Z (present 0 %, rayleigh 93 %; runs 09:00-10:15 exited 1); that episode is the sky
(R9 the only capable satellite above el_min), new only because the chain is. (b) **R11 on L2OC**: 4 present rows all
night (12:03Z, ~10 deg, 25-26 dB-Hz) against 2,327 on L3OC at the same elevations; never above 14.6 deg. (c) **q on peak
above the 4.37 model** (CHORD_MULTIBAND 6.2): R23 L2OC n=6974, p50 6.19, p90 9.32, max 17.92 (03:38:55Z), above 4.37 in
95.7 %; the same R23 on L3OC p50 3.39; R6 4.70, R8 4.52, R19 3.85, R20 3.27. Mostly R23 (K1B); re-derive the spacing
from live taps before touching it. (d) **Arcs**: R23 02:44-06:57Z (one 6-min loss 03:16:54-03:22:52Z at 52-55 deg: L2OC q
6.0 -> 1.65 in 2 min, PROMPT-CN0 "no PRN above the noise", L3OC lost it only 38 s), R6 03:48-05:36 (<= 10 deg), R8
05:33-07:59 (<= 20 deg), R19 from 09:00Z, R20 11:11-11:29Z (obs rows el 0.0: no BRDC); R22 (no BRDC) 32 present rows
03:53-04:20Z at q ~1.1 while L3OC held it at 3.67. (e) Dark probes R4/R5/R12 never present (0 of 16,263 rows each),
noise median 0.80-1.24. (f) q STALL fires every 5 min on glo_l2oc 07-12Z and glo_l3oc 04:40-11:20Z: few K satellites up.
(g) **elem_ref_cron watches 8 bands, not glo_l3oc/glo_l2oc.** **Next:** (a) first: check R9's code/slot and BRDC record
against the ICD; (g) add the two bands to the watch.

### #204 — fleet-wide window starvation: five storms overnight, the 00:23 rotation stalls the broker, and the runbook's check is gone (2026-10-18)
**What:** five clusters of FADR BREAK on 8-10 chains within ~2 s, all `silent0` gap 255-263 on every PRN: 03:07:44,
03:08:20, 03:09:16, 03:26:18, 03:57:30Z; none after 03:58Z (10-17 13-24Z ran 5-85 such clusters per hour, and 00:23-02:4xZ
today 1-24 per 10 min, so better, not new). In 3 of the 5 every chain first logged "COMB-DLL: no windows yet" /
"PROMPT-CN0: no estimate (no windows)" for ~11-16 s while TELEM kept counting frames (gaps 0, ~715 windows per 30 s) and
the broker log had no gap over 0.8 s: the loss is inside the broker (the ~10.7 s window ring) or between TELEM and the
chains, cause not found. **The runbook's check (section 4) cannot be run:** the gather exports no
`kotekan_buffer_recv_dropped_frame_total{stage_name="/telem_recv"}` (kotekan omits a counter until it first counts; it
has never fired since 10-13 21:08:57Z), so a gather overflow is not what these were. **The 00:23Z logrotate stalls the
broker:** copytruncate + gzip of ~1.9 GB (broker 750 MB + aggregator 1.0 GB at the time) made broker passes overrun the
2-s interval 138 times in 00:23-00:25Z (49/52/32 a minute, worst 3.42 s, against <= 6 a minute otherwise). Tonight's
pair is projected at ~2.1 GB aggregator + ~1.55 GB broker (aggregator ~87 MB/h of per-pass /gps_search lines since the
F-engine return; not new: 10-12 was 1.10 GB). Also 11:20-11:28Z: 69 overruns with user CPU at 66-67 %, cause unknown,
and the nodes logged a burst of "trim EXPIRED" then. **Fix [script]:** rotate under `nice -n 19 ionice -c3` and
`compresscmd` at low priority (or delaycompress), or cut the aggregator's per-pass INFO lines (NHPROF + per-PRN result,
~160 lines/s) to DEBUG; re-add a drop counter the runbook can read (export 0 at start); then find what empties the
broker's window ring for 11-16 s.

### #203 — the follower chains' clocks run off when gps_l5's solve is not established, until a cross-band re-bootstrap (2026-10-18)
**What [overnight]:** gps_l5's search clock solve was "not established" in stretches (03:10-03:50, 04:30-05:00,
05:40-06:20, 09:3x, 11:00-11:30Z, and still at 12:0x-12:1x): REFUSED 42 times overnight (11 on 10-17), scatter MAD up to
4879 chips (06:12:49Z), the cross-band donor WITHHELD from the others. The WITHHELD stretches are not new (10-17: 5-124
lines/h). **New tonight:** while the donor was withheld, JOINT-CLK adoption kept pulling the followers' legacy clock back
2.4-4.5 chips every 30 s; at 04:53:03Z it was refused for the first time (delta +5.1, bound 5.0), and DRCLK on glo_l3oc,
bds_b2b, bds_b3i and gal_e5b ran 151.26 -> 143.70 (04:54:57) -> 141.81 -> 137.03 (04:56:27Z) (gal_e6 and glo_l2oc by the
same time in their own chips): ~14 chips (~1.4 us) off for ~2 min, until all four re-bootstrapped cross-band from gps_l5 at
04:56:51Z (150.99 chips). Again 06:12-06:14Z (REFUSED at +7.1, +11.1; DRCLK 141.92 ~30 s; recovered 06:14:28Z with no
BOOTSTRAP line). On 10-17 no JOINT-CLK refusal exceeded the bound. No visible tracking damage (no FADR break 04:54-04:57Z,
bds_b2b presence held). **Fix [broker]:** belongs with #200's clock work: a follower whose donor is withheld should hold
its last good clock, not walk; and why the gps_l5 solve scatters to thousands of chips while 5-7 satellites are present.

### #202 — the viewer re-reads every observables day file since September every 15 s (2026-10-18)
**What:** `stack_components.sh:120` gives the viewer `--pvt-obs-globs "$GNSS_OBS_OUT/[gb]*_20[0-9]{6}.jsonl"`, which
matches all 281 day files (37 days), not today's 10. For each file on every PVT recompute (`pvt_ttl_s` 15 s, polled all
night by an open tab: 17,235 /decode_health requests since 00:23Z) `livebeam_server.py:1222-1229` seeks to size - 4 MB and
reads to EOF: ~1.1 GB per pass. Since its 02:43Z start the viewer read rchar 2.63 TB (75 MB/s), 657 GB from storage
(18.8 MB/s), the most of any process on the VM; NFS read calls 351-962/s against 23-249 on 10-11; the viewer sits in D
state (folio_wait_bit_common) in 7 of 10 samples. It grows by 10 files a day. **Context, the load:** VM load 8-9 (sar)
since the F-engine return 10-17 ~13Z, not the L2OC restart (10-11, 8 chains: ~5); idle 18-22 %; broker ~2.5 cores, gather
1.17, aggregator 0.62; nothing starving (broker passes 0.78-0.93 s of 2.0 s); MemAvailable ~2.0 GB drifting -20 MB/h,
no swap. **Fix [script]:** glob today's (and yesterday's, across 00:00Z) files only, or cache per file by (size, mtime).
Judge: viewer rchar rate and NFS read calls/s drop by an order of magnitude; load average.

### #201 — L2OC: the fleet loop parks one chip off the model and stays there, and nothing in the chain can tell (2026-10-18, CONFIRMED from the obs rows: four episodes in 11 h, one live)
**What [10-18 09:00-13:20Z, glo_l2oc obs rows + broker.log; skeptic-checked]:** four times today a glo_l2oc track
settled one L2OC chip (146.5 m) away from where it had been, inside the satellite's own arc, and stayed there with every
row flagged present and locked, while glo_l3oc's residual for the same satellite moved < 13 m: (a) R19 at -1 chip
11:09:40-11:31:06Z: `code_resid_m` 4253 m against 4412 before (-154 m, -1.05 chip) and 4381 after; the entry was a 3-min
walk (11:06:40-11:09:40Z, trim -0.23 -> -1.28, from q 4.1-4.5 with no presence loss); ended by the D3 SAWTOOTH bare-wipe
at 11:31:11Z (readback -1.42 -> -0.35), i.e. the model re-seed put it back; 657 of 658 rows present+locked. (b) R19 at
+1 chip 11:50:35-12:57:35Z: 4524.6 m against 4381/4386 (+150 m, +1.02 chip); the entry walk (11:49:48-11:50:35Z) crossed
a 16-s presence loss during which the trim moved 0.39 chip; all 1,930 plateau rows present+locked; ended 12:57:35-41Z
by an uncommanded walk-back (-148 m, trim +0.60 -> -0.37; no broker step logged). (c) R06 13:01:29-35Z: +148 m in 6 s
(trim -0.24 -> +0.78) eight minutes after a birth seed on a 2.3-h-old BRDC (`eph_age_s` 8130, so the seed's own error
is of order a chip); which side the main peak is on is NOT shown. (d) R20 13:04-13:06Z, LIVE: after a 20-s q loss
(1.5-1.8) the readback walked -0.38 -> +0.73 in two minutes; since then q 2.0-2.3 on the lock bar and every row
present+locked (239 by 13:21Z); no residual for R20 (no BRDC). Plus R19 10:39-10:46Z: -84 m (-0.57 chip) through a real
10-12 dB fade (q 6 -> 1.4-2.0), 140 displaced rows flagged locked. R19 spent ~87 of its ~250 locked minutes today a
chip off. **Reading:** the displacement is one chip to +-0.05 both ways, so the natural lock point is the BOC side peak
at +-1.05 chip (CHORD_MULTIBAND 6.2), but that is an inference: no served statistic confirms it. The prompt C/N0 is
blind to code error (instruments.py: "never judge lock on it") and read 52.25 dB-Hz on the -1 plateau, equal to the
main peak; q was unchanged in (a) and ~45% lower in (b); `sig` unchanged; the standing `dll_disc` is -trim/2.3 on
every plateau (leak balance), so its sign says where the trim stands, not which peak. The only witnesses are the
one-chip residual step inside an arc and the L2OC-vs-L3OC disagreement. Mechanism: the fast loop's only gate is prompt
power against the probe floor (CHORD_FAST_TRIM 11.2, "NOT on q"), which a chip-off prompt passes untouched on this
comb, so the integrator servoes whatever zero it is in and the leak holds it (the railed bound is 1.25 chips,
gnssFleetDll.hpp:98, past 1.05). What kicks a settled loop off its zero is open: (a) walked from the arc's best q with
no loss; three of the four R19 transitions sit 1-13 s after an ELEM-GAIN post (120-s cadence). **Why nothing caught
it:** the designed BOC safeguard is the search-vs-track referee (seeding.py:841; cli.py `hold_max_cp_err` 0.4 chip,
"stable false equilibria ... that a hold would otherwise servo forever"), and glo_l2oc has no search stage (`search:
None`), so the referee never runs on the one BOC chain we have. The obs writer, chain_health, the viewer and the PVT
self-survey all took the rows as locked. **Fix [broker]:** give the chain a referee: (1) cross-band: the same
satellite's glo_l3oc track is in-process; when |resid_L2OC - resid_L3OC| exceeds ~0.4 chip (60 m) for a few cycles,
re-seed L2OC's trim from L3OC's (or wipe to the model, which cured (a)); (2) at birth on a BOC chain, test the seed at
cp and cp +-1 chip for a few seconds and keep the strongest fleet q; (3) hold the integrator while the chain calls the
PRN not present (removes part of the (b) walk, not (a) or (c)); (4) publish `prompt_lock=false` while the referee
disagrees, so consumers see a gap instead of a 150-m bias. **Next:** the tap check (E/P/L at +-1 chip) to name the
peak. Applies to every BOC chain added later.
### #200 — the whole broker hangs on one node: cx19 is the time anchor and the bare endpoints (2026-10-18)
**What [live, 10-18 02:25-02:41Z]:** cx19's node was stopped cleanly at 02:25:52Z, and the broker restarted at
02:27:33Z without it. `common: rest-url: http://cx19:12048` makes cx19 the source of `time0-endpoint` (the F-engine
frame-0 anchor) and the host of every bare `combiner:` name, so every model-primary chain logged "time anchor
unavailable" every cycle and seeded only its three probes. Only gps_l5, the blind search, tracked. When cx19 came back at
02:39:24Z, each model-primary chain gave up waiting and seeded on the 0.00-chip PRIME ("seeding on the 0.00-chip PRIME
after waiting 710 s"), about 151 L5 chips off the true clock (TRKRES b3i +39.6 chips), and gps_l5 had still logged no
"receiver clock BOOTSTRAP" two minutes later, so nothing could adopt a clock phase. The fix was a broker restart with
every node up (the 10-01 rule).
**The failure mode, end to end:** a broker that starts without the anchor never measures a clock, even after the anchor
returns. gps_l5's dead-reckon primes the clock at 0.00 chips "REPLACED outright by the first multi-sat solve", but that
solve never ran in the 16 min the process lived (no "receiver clock BOOTSTRAP" line, though gps_l5's search kept 5-7
satellites present throughout). The model-primary chains wait up to ~710 s for a clock to adopt, then seed on the
prime, and from then on the whole fleet tracks nothing but L5 while every node, the gather and the telemetry look
healthy. Cycling the nodes does not help: it brings the anchor back, not the clock. **Signature, one minute:** `grep
DRCLK` in the broker log reads `clk_now 0.000 chips` on every chain; "seeding on the 0.00-chip PRIME after waiting N s"
once per model-primary chain; no "receiver clock BOOTSTRAP" since the last "starting N chain(s)"; TRKRES tens of chips.
**Remedy, confirmed 10-18:** `systemctl --user restart gnss-broker.service` with every node answering. Restarted
02:43:22Z: BOOTSTRAP 151.95 chips (6 sats) at 02:43:25.7Z, 0 CHAIN DIED, and at 02:46Z gal_e5a/e5b/e6 11 present,
bds_b2a/b2b/b3i 8, gps_l2c 9, glo_l3oc 5, glo_l2oc 1 (its first satellite).
**Fix [broker + config]:** (a) the anchor from any node: `time0-endpoint` tried across the trackers' hosts, first answer
wins, and the anchors compared (they must agree to the ns); (b) no bare names: each chain's `combiner` as a full URL with
a fallback list, or chosen at run time from the instances that answer; (c) a broker that starts without the anchor must
bootstrap gps_l5's clock once it arrives, as at a clean start (today it never did), and the model-primary chains must
wait for that clock rather than seed on the prime after a fixed wait. Check: stop one node's kotekan (cx19 included),
restart the broker, and every chain seeds on a measured clock within a few minutes.

### #199 — the cf06 cube leg is not recording: the compactor dies in flush, the archiver is restarted by SIGTERM, the senders sit reset (2026-10-17)
**What:** found while judging the glo_l3oc roll-out, 22:5xZ. (a) `gnss_cube_compact.py` dies on every pass since the
17:51:46Z pass (the first after the stage-1 canary cycle; the log opened 13:07:53Z): `flush` -> `np.asarray(self.win["freq_id_lo"])`
"inhomogeneous shape (121,)", i.e. one (pointing, sender, day) file accumulated windows whose bin count differs, which
the "geometry changed mid-day" guard at line ~215 did not catch; 5 tracebacks over 37 passes, the loop then "stopped on
SIGTERM" 6 times. (b) Every pre-10-17 raw file reads "size 91094400 -> 91101600 since it was compacted ... SKIPPED"
from the first pass on (8 bytes per frame: a frame-format change the index predates, or the archiver appending). (c) The
archiver has 33 starts in `/tmp/gnss_cubearch.log`, the last ones "Got SIGINT or SIGTERM" (22:52:49Z, 22:59:50Z, ...
every ~7 min, NRestarts 0 and Result success, so an external restart, not a crash; whose?). Each restart resets the
nodes' cube senders mid-header ("Connection reset by peer, failed to send header"), and between restarts they log
"Connection refused"; at 22:54-22:59Z cf06 had 0 established connections on :11070 and at one look no listener at all.
Not the roll-out's doing: (a) and (b) predate it, and the l3oc senders behave as the others. **Judge:** `ss -tan state
established | grep -c 11070` on cf06 ~90 (11 instances x 9 chains), raw files growing continuously, a compactor pass
without a traceback. **Fix:** find the restarter first (KV?), then make `flush` refuse the mixed-geometry file with the
sender's name instead of dying, then the size check. The viewer's beam cube is dark until then.
**(c) answered 10-18: the restarter is the login session.** cf06 has `Linger=no` for kvand (`loginctl show-user kvand
-p Linger`; /var/lib/systemd/linger/ empty; the gnss VM has Linger=yes). `gnss-cube.target` is enabled through the
SHARED home's default.target.wants (since 10-17 13:07, ConditionHost=cf06), so the user manager on cf06 starts the
archiver and compactor when someone logs in and stops them (SIGTERM) when the last session ends: seen 10-18 12:09:04Z,
when a checker's ssh started the user manager, archiver and compactor in the same second, and the nodes logged "Broken
pipe" when it logged out 12:11-12:12Z. Overnight every node's cube senders logged "Connection refused" to
10.222.3.6:11070 every 30 s (11.7k-22.3k each). cf06 did not reboot (up since 10-15 03:04Z). **Fix:** `sudo loginctl
enable-linger kvand` on cf06 (KV), then judge as above; (a) and (b) remain.

### #198 — reopened: the fleet cycle still ran at ~30 s per node with the bufferSend fix in the binary (2026-10-17)
**What:** KV cycled the five nodes onto `b4ab492da` (which carries `d624d7784`, confirmed by `--version` and the
source) at ~22:40Z and saw the 30 s per node again, with recv1 (restarted 22:48:34Z) and the cube archiver (#199) down
during the cycle. The closed row's judge is therefore not met and its diagnosis incomplete: the sliced wait is in
place and `kotekanMode` stops every stage before joining any, so the senders' reconnect wait is not where the time
goes. Unmeasured: whether the 30 s is the stop (SIGTERM -> exit) or the start (`node_up.sh` waits for 12048 to
answer, which comes after ConfigTracker and the GPU set-up; nothing in `node_up.sh` sleeps). **Next cycle:** time
`ssh <node> sudo systemctl stop gnss-node` and `node_up.sh <node>` separately on one node, and read
`/tmp/gnss_node.log.1`'s tail (the previous run's shutdown) for the last stage to go.

### #196 — the tipped-dish week: what is switched off, and the checklist for the realignment (2026-10-17)
**What:** the dishes are tipped in semi-random directions until next week's realignment, so the broker runs
`--boresight none` (`cc2b33651`, armed in `config/gnss_chains_chord.yaml` `common:` by `55ca9a036`, live since 14:45Z):
the D3 transit veto, the dead-reckon clock's transit freeze and the assemblers' model-formation freeze are OFF. The
fleet element reference in use (`config/elem_shared_ref_20261017.json`, `df37629bb`) was taken at 13:30Z with the dishes
tipped and 7 of 11 instances healthy. `config/pointings.yaml` p1_scattered_20261017 has no boresight, so the nightly beam
map refuses and the array epoch is still p0.
**At the realignment, in order:** (1) delete the `boresight: none` line and restart the broker (T+60 s CHAIN DIED check);
(2) a fresh `elem_shared_ref.py snapshot` with all 11 instances once their models are warm, copy into `config/`, point the
manifest's `elem-shared-ref`, regenerate (a node restarted after a cron re-take and before the commit runs up to 30 min on
the stale file, `ACT_EVERY_S`); (3) a new array epoch and a boresight in `pointings.yaml` before any beam map or the
64-element step (#168; rows C/D now deliver); (4) read cx27's GPU-1 comb from its tap when its port is back and fix
`gpu_offsets_mod16` if it moved (#194's leftover); (5) the judges that need a pointed, quiet sky: #154's no-restart re-base
check (R ≥ 0.95 in every band within ~10 min after an F-engine re-base with no node restart — never yet run as stated),
#177's per-satellite aggregator L5 search SNR and the per-satellite G re-solve (expect DRAO's 0.64 dB spread), #184's fresh
replays, #155's transit recount, #191's timing steps, #176's standing `REFUSED (step` watch through a 1176-MHz transit.
**Check:** `grep -c 'BORESIGHT TRANSIT' broker.log` non-zero again after (1); R ≥ 0.95 at the next cron pass after (2);
the beam map builds after (3).

### #195 — a PRN whose disc sits at the clamp for minutes has no way back to its peak (2026-10-17, split from #175)
**What:** the #175 refusal stops a chain-wide displaced admission, but a PRN that is already railed stays railed: a railed
disc is leak-bounded (`gnssFleetDll::dll_tau` clamps at ±1, at most ±0.25 chips of trim) and the model holds the displaced
position, so only a fresh displaced-presence admission pulls it back — E4 took 50 min on 10-11, E16 longer. The
`PRESENCE +displaced` line is rate-limited to one per 120 s, so the log under-reports admissions.
**Fix (not built):** a per-PRN 60-s history beside D3 (the fleet row's disc, the readback trim and `spec_fit[prn].tau`) and
an actuator — release the C++ loop's standing trim or re-seed the PRN from SPEC-FIT — not a presence verdict. Design choice
first.
**Check:** the dim E6 PRNs' DLL disc after a null crossing (needs pointed dishes); a disc at the clamp for > 60 s with
SPEC-FIT tau disagreeing should end within a minute.

### #184 — four finished A/B arms stay in the broker because retiring them moves the replay digests (2026-10-13)
**What [tree]:** the removal pass retired every finished A/B except the ones whose live arm changes the `broker_equiv`
digest of the seven recorded replays: `--n2-combiners` (the path-B population), `--seed-bias-source` ema/slow with
`--seed-bias-alpha`, state_filter's median gauge, and fits' optional `fft_len` precision argument. Each keeps a dead arm
and its flag. The replays predate the L5 code fix (#177) and telemetry v8, so re-blessing their digests under the new
arms would bless a run nobody can reproduce.
**Fix:** once the F-engine is back, record fresh replays from the live chains, retire the four arms and re-bless the
digests in one commit. Check: `broker_equiv` identical on the new replays before and after the retire.

**10-13, the goldens (`8a392bd66`, `d6042214b`, `8b276f8ab`):** three of the six committed `scripts/gnss/fixtures/*.digest`
had been red since c4512de93. Bisected on cf06: `broker_onsky_l5` moved at f29211e30 then 34acbcf90 (#161: the NH hint
learns nothing while the solve is unestablished); `broker_onsky_e5a_20260826` and `b2a_20260826` at f94da37de (the lobe
DLL changed the gather taps' shape, so the recorded per-instance replies no longer parse and the two comb cycles fall to
the polled discriminator). Every move has its commit; all seven are re-blessed at the current tree, `l5_holds` has its
first golden, and `broker_equiv.py check` is green on all seven. **What the bisect exposed:** the three 08-26 replays
diverge from their recordings on the gather GETs (~1760 `TRANSCRIPT DIVERGENCE` lines each), so no golden exercises the
C++-taps arm or `PROMPT-CN0` from recorded taps — the gate covers the polled fallback, not the arm that runs live. The
fresh captures this entry waits for must be recorded with the gather GET order the replay asks for (or the replay made
order-tolerant), or the comb-taps arm stays uncovered.

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

### #190 — project ALL KNOWN satellites out of the science N^2 from the lane directions, not the N^2's own eigenvectors (2026-10-13, KV: "swap over once we get back on sky")
**What [tree]:** GnssN2Project takes its subspace from the science N^2 block's own top-k eigenvectors, gated at
lambda0 >= 3x mean auto (per-element INR ~ -12 dB over 48 live elements), frac0 >= 0.4, PR >= 6, k <= 3, EMA 0.5 s.
Below that gate the eigenvectors of many comparable faint satellites are mixtures (10-02: no quiet-sky eigenvector is a
single satellite, cos^2 <= 0.26 vs the tracked rows), so the stage removes bright objects, whatever they are (the sun
at ~+18 dB and Cas A/Cyg A at ~-7 dB per element in the beam trigger it in the lobe channels; FRBs at ~-38 dB do not),
and nothing faint. The synthetic lanes are unused on the science path.
**Design (docs to follow; memory chord-project-all-known-sats-design):** per channel, directions a_p from (1) every
tracked PRN's P row (the N x M column, a matched filter per code: separable to ~-30 dB per element at SNR ~10 per
element per 0.5 s), (2) BRDC-predicted steering vectors (positions + element cal) for ephemeris-known satellites with
no lane (P(Y)-only, unhealthy, slotless), (3) TLE-predicted vectors (Space-Track/Celestrak GNSS + SBAS groups) for
catalogued satellites without BRDC (BDS-3 C43+, GLONASS until #129, QZSS L6 on E6, the WAAS GEOs on L5). Q = orth([a_1
.. a_K]); V' = (I - QQ^H) V (I - QQ^H) with the stage's existing V - QW^H - WQ^H + QMQ^H; the data-driven solve then
runs on the RESIDUAL as the detector of unknowns and feeds the list. Each satellite costs one synthesised beam of sky
at its position (K/N of the noise); at 128 elements two dozen satellites are ~20%, accepted (KV 10-13). Depth =
min(1/SNR^2 of the direction, the clipping floor -20 dB rank 1).
**Plan:** (a) the stage reads the chains' epl rows (a process-wide board: no buffer crosses chains today, see
chord-projection-pathway) or the assembler publishes per-channel directions per PRN per frame; (b) predicted vectors
from gnssElemSteer for tiers 2-3, a TLE fetch beside the BRDC supply; (c) offline first on the viscap captures
(cf06:/var/tmp/viscap_20260927, 16 mixed + 3 AA tiles per channel per record: every block is reconstructible) measuring
depth vs the current stage per satellite; (d) canary cx51, judge = recv1 subset-vis A/B against cx19 (n2proj_ab.py)
through a quiet hour and a transit, then the fleet. Before any of it, at the F-engine's return: check the 10-03..10-12
Cas A / Cyg A transits against the stage's k/lambda0 gauges (archive) for continuum triggers in the lobes.

### #191 — close the joint receiver state out: find the per-signal timing steps, feed every chain, retire the fallbacks (2026-10-14, KV: "work on getting this finished when the sky comes back")
**What [tree, measured 10-12 00:30-12:00Z and 09-25..10-12, `fixtures/jointstate/`]:** the joint state is live on every
chain but fed by gps_l5 alone (#125), so b_sat exists for GPS only and the other seven chains' per-satellite code
residual lives in the gather's trims. The trims and the obs `code_resid_m` show no per-band receiver delay (E5a/E5b/E6
agree to 0.2 m on the same satellites; `tau_band` dropped) but per-SIGNAL offsets that are OURS: with the atmosphere
removed per row they hold in every elevation bin and every day since 10-03 (Galileo +11..+14 m vs gps_l5, B2b/B3I
-6..-9, B2a +6..+11, L2C -5..-9); our own range model run at six Septentrio IGS stations (DRAO, BREW, CHWK, HOLB, NANO, WILL) on
their pseudoranges puts Galileo E5a +1.6..+1.8 m above GPS L5 at all six, L2C -1.7..-2.2 at five (BREW +1.0), B2a
+6.8..+7.6 and B3I +4.6..+7.0 at the five that track BeiDou (fixtures/jointstate/README.md); the shipped
seed-to-despread leg places L5_Q_NH, E5A_Q_CS and B2A_P_CS identically to 0.001 chips; chains on one band share their
channels and nothing in the tracking path branches on constellation or data-vs-pilot. Three steps of ~+12 m (40 ns)
remain: Galileo E5a over GPS L5 at one frequency on identical channels, L5 over L2C, B2a over B3I. Not tiling (B2a
sits with L5), not data-vs-pilot alone, not the model, not the node bookkeeping. Also: every chain carries the same
-18 m x sin(el) term (model altitude or atmosphere; re-survey), the fallbacks never fire (JOINT-CLK adopted 1366/1376
per chain, joint rate every cycle, SatBiasFilter 0.01 chips), the P1 spectrum fit's peak/floor is 1.00 on a closed
loop and cannot gate a feed, and #93 closed code-carrier coupling by measurement.
**Plan (docs/CHORD_JOINT_TRACKING.md section 6):** NO per-constellation bias state (KV 10-14): the steps are to be
found. Offline now: (0a) the model's group-delay gaps (L2C, E6, B2b terms; the constellation mean under CAS); (0b)
troposphere + Klobuchar + a dual-frequency re-survey in the range model, gate b_sat inside +-0.2 chips and dU -> 0
on the 10-12 replay, digests re-blessed; (0c) one code unit with conversion at the feed, coast-on-refusal consumer,
selftests. At the return, first day, one node: (1) the acquisition on the GAL_E5A_Q primary vs the tracker vs the
model (search = tracker = +12 names the table or the sky; search = 0 the tracker path); GPS_L5_I beside L5_Q and
B2A_D beside B2A_P (I/Q are synchronous at the satellite: a step names data-vs-pilot); a model-seeded L5 chain beside
the search-anchored one. Gate: every chain within 1 m of GPS L5 on the same satellites. Then (2) the trkresid
observable from every chain as a SHADOW feed gated on present + fleet q + |spec_tau| + the transit veto + a set gate,
no fallback path, judged by the fed-vs-unfed trend test and 24 h of zero latch alarms; (3) `joint-consume:
clk,rate,slew` everywhere with the seed offset clk + b_i, judged by the trims going to zero; (4) retire the
SatBiasFilter, the l-a seed rate and the sibling-level adoption, re-bless. Carrier commands on the five uncommanded
chains, P3 and the node model are separate.
**Check:** each step's gate above; the standing readback trims per chain median within +-0.05 chips after step 3.
**Step 1(a) prepared 10-18:** `config/gen_chord_gnss_config.py --aggregator-extra-search GAL_E5A_Q` emits a second
GnssChannelizedSearch (`e5a_search`) on the merged L5/E5a comb (the same 79 channels, 36 Galileo PRNs; the 100 CS
alignments are scanned blind, since gal_e5a posts no nh hint); the broker's `--detectors-monitor` makes a chain poll
its detectors for the record only: `stage_dead_reckon` builds the offsets exactly as the solve does and
`search_monitor` logs SEARCH-MONITOR (search clock minus joint clock in the chain's chips, per PRN, and a 5-min
median) while `best` is emptied before any seed, hold, escape, deep gate, clock solve or joint feed sees it;
gal_e5a carries `detectors: http://localhost:12050/e5a_search` + `detectors-monitor: true` + `narrow-search: true`
(config/gnss_chains_chord.yaml; the first run without narrow-search left e5a_search with "0/36 PRNs hinted" for ten
minutes: the hints are the only thing the search scans).
**Step 1(a) RESULT, 10-18 18:37-18:50Z [live]:** the E5a search (11 Galileo PRNs, snr 20-600) measured against the
model and the GPS-fed joint clock: **search - joint = +0.49 chip median (+14.3 m) over 91 cycles, p10/p90 +0.37/+0.64**
(5-min medians +0.45/+0.46/+0.49), while gal_e5a's standing trims read +0.36 median (10-12: +0.39, obs +11.8 m).
Search = tracker: the Galileo signal arrives ~0.45 chip (40 ns) later than our model + our E5a replica place it, and
the tracker merely follows. **The tracker path is exonerated**; the offset is in the signal as our comb and code
tables see it against the model. Per PRN the search is noisy (MAD 0.15-0.33 chip; E04 +0.07, E10 +0.62, E12 +0.73,
E29 +0.69) and tracks the trims only loosely, so the chain median is the number. Caveats: (i) GPS L5 is the DATUM
(the joint clock is its search), so "+12 m on Galileo" and "-12 m on GPS L5 with Galileo right" are the same
observation; B2a at +0.3 m against the station reference argues for Galileo being the odd one, B3I (-12.6) and L2C
(-6.8) say every signal needs its own check. (ii) The station test (six IGS receivers, our model) clears the
ephemeris/clock/group-delay model to ~2 m, so what is left is the replica bank's placement of each signal's code
against the sky, a convention shared by the search and the tracker (both build from the bank) and invisible to the
e2e harness (its sky is the same bank). **Next:** a bench test with an INDEPENDENT reference: generate E5a-Q, L5-Q,
B2a-P and B3I from the ICD code definitions outside the bank (chip-0 at the code epoch, plain BPSK at the sample rate),
run them through the e2e harness and read where the bank's prompt sits per signal; 0.45 chip on E5a and 0 on L5 names
the bank, equal placements name the model's time datum (then (c): a model-seeded L5 beside the search-anchored one).
Side watch: E33 (the deep-gated PRN, setting, cn0 48-61 dB-Hz) walked +0.45 -> +1.15 -> the 3.0 clamp at
18:26-18:30Z and oscillates +0.8..+2.1 since while the search holds it at +0.84: a loose loop on a strong signal at
el 15-22 deg, the #201 class on a BPSK chain. The experiment stays live (costs: e5a_search on the aggregator GPU,
nothing on the broker's seeds).
**What the same evening excluded [10-18 18:50-19:30Z]:** (i) in-band dispersion: per-channel prompt phases
from the aligned spectra (`fleet_spectrum_aligned`, 72 channels / 11 instances) on locked PRNs give the same delay
from the lower and upper halves of the comb to +-4 ns for Galileo AND GPS, linear-fit residuals 0.01-0.03 rad, the
same spectral envelope; (ii) the DCB path: `gnss_dcb.signal_bias_s` is algebraically the ICD's -gamma x BGD
(datum - b_x = gamma d15/(gamma-1)), the live broker loads CAS day 287, and the per-PRN Galileo offsets do not track
the C1C-C5Q DSBs (E10 -0.1 ns and E29 +2.3 ns read the same +0.6 chip); (iii) the bank: `code_chip` is one
floor(chip_phase) mapping for every signal (BOC in the expanded table), so a fractional-chip per-signal offset cannot
be a code-table convention; (iv) beamformer/ElemCal/projection: the search tap is single-element and unprojected and
reads the same +0.45 as the fleet tracker; (v) the model: the station test's per-satellite Galileo residuals are
+0.5..+2.8 m under the same `predict_all(signal=, dcb=)`. In TIME the Galileo signals agree (E5a +40 ns, E5b +42 ns,
E6 +35 ns at half the chip rate), and the audit's day series grew +7.7 -> +10.3 -> +12.3 m over 09-25..10-03 before
settling, which no static convention does. Open. **Candidates left:** an extra-signal GAL_E5A_I chain beside E5A_Q on
one node (one node regen + cycle; the same +0.45 says Galileo-wide, 0 says the Q pilot's code/secondary); and the
only receiver-independent test, a short voltage capture of the L5/E5a lobe correlated outside the bank (no capture
path exists in the tree; the replay has no voltages). The ICD-generator bench test is withdrawn: (iii) answers it.
**23:11Z, after the #207 cycle:** SEARCH-MONITOR gal_e5a reads search - joint = +0.577 chip (+16.9 m) 5-min median,
p10/p90 +0.45/+0.69, per-PRN MAD 0.07-0.16 (the lever's scatter gone); the Galileo constant stands, now +0.58 chip.
**Later the same evening (before the fix):** the per-PRN scatter in the search's Galileo numbers is #207 (a Doppler lever on every
search-derived offset), not noise and not DCBs; with it removed the Galileo constant is **+0.43 chip at zero Doppler
(+12.6 m)**, and it stands in both the search and the converged tracker.
 Test: gnss_broker/test_searchmonitor.py. **To run (KV):** add the flag to the
stack's aggregator regen (the invocation in scripts/gnss/site/stack_components.sh), restart gnss-aggregator, then
gnss-broker; read `SEARCH-MONITOR gal_e5a` in broker.log. **Read:** search - joint ~ +0.4 chip (+12 m, the standing
trim) says the signal sits where the tracker found it (the code table or the sky); ~0 says the tracker path.
**Step 0c DONE 10-18 (a3bfeabfc, merged 123e2acc6):** the joint state holds ONE unit, 10.23-Mcps reference chips:
`Receiver.joint_receiver(band, code_len, chip_rate_hz)` converts the modulus, dr_joint_shadow converts its offsets,
sigma and warm start in (MINNOV/P2C read back out), every consumer converts through `joint_unit()` = (reference rate,
L); the per-feeder declaration and the mixed-unit refusals are gone (a second constellation at another chip rate joins
without lengthening the modulus or moving the clock: selftest, test_one_unit). COAST: a refused joint clock (bounds,
step guard, period check) or a state not fit to consume (thin, deaf, absent) puts clk_now on the last adopted level
carried at the rate STAMPED at that adoption, never the median and never the state's live rate, inside
`--joint-clk-coast-max-s` (300 s; 0 off) and bounded by joint_clk_max_chips; no stamp moves on a coast (the freshness
ages, the bootstrap re-arms). This is the #203 mechanism closed. Review caught three callers of the new signature
left behind inside try/except (the rrate feeds, the joint-notes drain: a TypeError per cycle logged once per 300 s);
test_gauge_churn now pins the arity. Gates: test_jointclk/gauge_churn/clockperiod/jointstep/clockguard, the migrate
test, selftest, and the onsky_l5_holds replay (no "feed skipped", the rrate feeds run) all green. The broker_equiv
gate could not judge it: the replay is NON-DETERMINISTIC (two replays of the unchanged 5a1522f3f tree disagree,
`broker_equiv.py selftest` says so; three trees gave three digests under one golden) = #192 (g), open. LIVE after the
next broker restart.
**10-19 00:4xZ (commits 23:47Z..00:50Z) STEP 0b (CHORD_JOINT_TRACKING.md 6.3) -- troposphere AND ionosphere shipped, the re-survey
measured and explained:** gnss_atmos.py (Saastamoinen zenith delays from a standard atmosphere at the site height, Chao
mapping; Klobuchar at the chain's carrier; RINEX IONOSPHERIC CORR parse + a coefficient cache); predict_all carries
tropo_m / iono_m as separate delays (0.0 off), cp_predicted adds them (every model-evaluated code phase: seeds,
offsets, trkresid, and the decoded fallback), the obs rows publish both; `--tropo saastamoinen` and `--iono klobuchar`
are the defaults. Offline gate on 10-18 (code_resid_m, receiver clock removed per 10-s epoch, five chains): the
low-elevation plunge (-8.0/-1.8/0.0 m at 0-10/10-20/20-30 deg) flattens to +0.18/+0.17/+0.15 m with Saastamoinen --
right sign, size and shape. The remaining sin(el) term is SUN-LOCKED (skeptic wf_83656700: night fits dU +12 m, day
+2..5 m, ramping at sunrise/sunset): the daytime L5 ionosphere aliases into an apparent height error as the troposphere
did, and my first reading ("no ionosphere-shaped term at L5") was wrong -- the Klobuchar "doubling" of the tilt was the
unmasking of a half-cancelled constant. With Klobuchar on the 10-17 broadcast set the night fits read +17.7/+17.4/+18.0
m (gps_l5/gal_e5a/bds_b3i) and the day fits +14.1/+14.3/+16.5 m: three quarters of the split closed, the remainder
Klobuchar's own ~50 % error (the 08-27 set closed it fully; sets differ that much). ON by default; a GIM later. Supply:
the dailies carry GPSA/GPSB on most days (late in a 119-line header); the 10-18 daily and our hourly merge did not,
because the merge kept the first station's header -- it now re-splices the IONOSPHERIC CORR lines from the first
station that has them on every merge (second skeptic, wf_55f690aa: a splice-once froze the set for the life of the
rolling store), test_brdc_golden re-blessed (three header lines, diffed old vs new); the last set seen is cached with
its age (the broker writes with a per-process temp name, the twelve obs writers read, a pinned replay never touches the
live cache -- all three from that review). **The re-survey:** model - true = (+17.4..+18.0 U, +3.2..+3.9 N, 0.0 E) m on
the night rows of three chains (mean-demeaned fixed effects; the first +7.2 m was a day/night average through a median
demeaning that also biased it ~10 % low), and the same on 10-12 with the dishes aligned (+18.3/+17.5 U, +3.5 N): a
standing position error, not the tipping. The 18 m is the GEOID: 545 m is an orthometric height, _ecef_of_llh takes an
ellipsoidal one, and EGM2008 gives N = -16.55 m at the site (the EGM96 grid on the box -16.29) -> h = 528.4 m; the model
has stood 16.5 m too high since #99. The +1.5 m residual is of the size of Klobuchar's night error (0.29 m/TECU at L5)
or of the hand-typed 545's precision, not the troposphere's (0.1-0.3 m). KV's config change: `alt: 528.4`, `lat:
49.3199845` (3.3 m south) + node-config regen + broker restart; gate = gnss_resurvey.py --hours 6-13 --klobuchar <the
day's set> on the next night's obs within +-2 m of 0 (not previewable offline: the rows' residuals were formed live).
Expected live effect of the atmosphere at the next broker restart: b_sat spread and the SEARCH-MONITOR per-PRN values
lose their elevation and local-time terms, the joint clock shifts by the fed set's mean delay (~5 m by day),
code_resid_m in the obs flattens vs elevation; the broker_equiv digests move. Tools: scripts/gnss/gnss_resid_vs_el.py
[--atmos tropo|both] [--hours], scripts/gnss/gnss_resurvey.py [--hours] [--klobuchar GPSA,GPSB] [--f-hz].
**10-19 00:56:47Z DEPLOYED (KV's go):** config e5c42b7ec (`alt: 528.4`, `lat: 49.3199845` in the chains yaml common and in
the stack's GNSS_SITE_LL; the node configs do not take the position, no regen), gnss-broker restarted 00:56:47Z (#208 +
0b + the position land together), the ten obs writers and the viewer restarted 00:58Z for the position. T+15 min: all
fifteen units active, NRestarts 0 everywhere, 0 CHAIN DIED, no tracebacks; BRDC loaded on every chain in 14 s; the
Klobuchar set came from the freshly merged hourly (the splice working: alpha 2.5146e-08 ..., newer than the 10-17
daily's); gps_l5 clock 151.91 chips, JOINT re-established (n=5, +-0.10 chip); SEARCH-MONITOR gal_e5a 5-min +0.474 chip
(+13.9 m, p10/p90 +0.37/+0.56) against +0.58..+0.60 just before (#191's constant moved ~3 m with the atmosphere and
position); obs rows carry tropo_m (3.19 m at el 44) and iono_m (12.0 m at L5, 18:12 local); the #207 invariant still 0;
viewer 200. **GATE (tomorrow):** `scripts/gnss/gnss_resurvey.py --hours 6-13 --klobuchar <the day's GPSA,GPSB from
~/.cache/kotekan_gps/iono_corr.json> /home/kvand/gnss/fixtures/obs 20261019 gps_l5 gal_e5a bds_b3i` must read dU, dN
within +-2 m of 0 (expect dU ~ +1.5 if the residual is real), and `gnss_resid_vs_el.py --atmos both --hours 6-13 ...`
flat; then 0b closes.

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
**10-17:** `joint_latch_detect.py` over the 12:47–13:17Z log: 21 scored minutes, no latch (healthy d p50 −0.01, p99 +0.89 chips, the establishment transient inside). Continue the week.

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
(live since the 10-13 21:05Z broker restart) logs the last 12 (hop, cp in seed currency, det dop Hz) at every FLUSHED. What the
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

**10-13 [tree]: the replay gate now reaches the guard.** `broker_equiv.py` takes broker flags after `--` for `digest`
and `bless` (appended after the transcript's argv, so they override it) and `bless --variant <name>` writes a second
golden that records those flags; `check` verifies every golden of a transcript (`d6042214b`).
`broker_onsky_l5_holds.jsonl.gz.cprate.digest` (`8b276f8ab`) replays the holds capture under the live knobs
(fit-hist-len 256, fit-min-snr 30, cp-rate-model-tol 0.5, fit-flush-on-reject 3). On that 210-s capture the guard is a
no-op — 0 REJECTED, 0 FLUSHED, max fit-to-clock deviation 0.14 chips/s — so the golden pins the no-op; the branch is
reached (tol 1e-6 / flush 1: 18 rejections, 54 flushes). Not covered: `--fit-gap-s 1200` and `--cp-rate-model-primary 1`
(the transcript records 3600 and the variant appends only the four knobs).
**10-17 12:48–12:50Z:** 15 `cp-fit history FLUSHED` on gps_l5 (PRNs 10 and 23, the fleet re-establishment) and 15 `history at flush` lines — exactly one per flush. The dump works; the garbage's origin is still to be read from those histories.

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

**10-14: the reboot is ready, and the pieces this entry asked for exist.**
* **The console survives the reboot.** `claude-screen.service` (`~/.config/systemd/user`; scripts in
  `~/.config/claude-screen`: `sessions`, `start.sh`, `window.sh`) starts the detached `claude` screen with one window
  per listed session, each `claude --resume <id> --remote-control <name>` from a login shell, dropping to a shell when
  claude exits so the window is never lost. Dry-run under a unit-like environment: both windows came up in
  `/home/kvand/gnss` with `claude` on the path. `--resume` keeps the session id and restores the session's permission
  mode (docs); `--resume <id> --remote-control` parses on 2.1.281. **Before the reboot:** `systemctl --user
  daemon-reload && systemctl --user enable claude-screen`.
* **The cube leg has units:** `scripts/gnss/site/systemd/gnss-cube.target`, `gnss-cubearch.service`,
  `gnss-cubecompact.service` (`ConditionHost=cf06`; installed beside the stack's, NOT enabled). The archiver runs
  `cubearch_up.sh --check` (new: the pre-flight alone — config present, archive dir writable, sender/receiver
  frame_size agree; it passes on cf06 today) as ExecStartPre and then `run_component.sh cubearch`; the compactor runs
  `run_component.sh cubecompact` with `KillMode=mixed`, and `cubecompact_loop.sh` now traps TERM so a stop finishes
  the in-flight pass instead of orphaning it (tested with a fake tool: the pass completed, the loop exited, no
  orphan; a stop during the sleep is immediate). Both components are in `stack_components.sh`. **At the return, on
  cf06:** stop the bare archiver (pid 5407, on a deleted inode) and compactor, then `systemctl --user daemon-reload &&
  systemctl --user enable --now gnss-cube.target`; the archiver then runs the current `build_nodpdk` binary.
* **Reboot checklist (KV, any time both Claude sessions are idle at their prompts).** Before: enable `claude-screen`
  (above); know that the aggregator moves from d30be5f66 to 4514061a6 and GLONASS enters the broker's pooled veto
  (#129) — both wanted. After, on the VM: `loginctl show-user kvand -p Linger` = yes; `systemctl --user list-units
  'gnss-*' claude-screen` all active; every unit's `NRestarts` 0 (`for u in $(systemctl --user list-units 'gnss-*'
  --plain --no-legend | awk '{print $1}'); do echo "$u $(systemctl --user show -p NRestarts --value $u)"; done` —
  the obs writers wait for their anchor, logging every 600 s, and must NOT loop); `grep 'CHAIN DIED'
  /var/tmp/gnss-logs/broker.log` at T+60 s = none; `screen -ls` shows `claude` and `screen -r claude` shows two
  windows, each at its session's prompt with a Remote Control URL. Then the VM half of this entry closes; what stays
  open is the target-stop propagation (journal access) and the cube target's enablement at the return.

## Open — the fix is in the NODE BINARY (queue for the next cycle)



**Top priority: #177.** The others below it came from the same 10-12 comment review and are
unconfirmed or dormant.

### #181 — the replica bank applies no shared overlay for B2I, L3OC and L2OC (2026-10-12, dormant)
**What [tree]:** the shared-overlay ladder in the `ChannelizedReplicaBank` constructor omits `BDS_B2I`, `GLO_L3OC_P/D`
and `GLO_L2OC_P`, although each has `secondary_length > 0`; `channels()` / `hoprate_stream` apply no overlay for them
and the NH search reports "no secondary code". No live chain uses them. Only `BDS_B2I` (`B2I_NH20`) and `GLO_L3OC_D`
(`L3OC_BC5`) keep a shared `gnssOverlay` registry row; the pilots' rows (`L3OC_NH10`, `L2OC_OC2`) were removed
2026-10-13, though their arrays stay in the code generators.
**Fix:** add them to the ladder (for B2I and L3OC_D the shared overlay can come from the registry). Check: a replica
test with `nh_phase >= 0`. (10-18: the CHORD chains sidestep it, as glo_l3oc tracks `GLO_L3OC_P_NH` and glo_l2oc
`GLO_L2OC_P_OC`, the overlays baked in; the ladder matters only to a search on the bare descriptors.)

### #182 — the combiner's overlay + navwipe path wipes suffix rungs at the full window's overlay phase (2026-10-12, dormant)
**What [tree]:** `GnssCoherentCombiner.cpp` (the `!_secondary.empty() && _navwipe_bit_records > 0` branch) passes the
full window's `best_ph` to `overlay_apply` for every rung, but `overlay_apply` counts the chip index from the first record
of the window it is given; a suffix rung starting d records later needs `best_ph + d` (the navwipe pin beside it is
shifted). No live config sets `secondary_overlay` or `navwipe_bit_records`, so the path does not run.
**Fix:** shift the phase by the rung's record offset, as `pin_here` does.

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
leaves it open with the harness as the standing check. **10-13 21:08:57Z:** the gather runs the fix (`build_nodpdk` = `4514061a6`, rollback
`kotekan.prev_d30be5f66_20261013`; the aggregator takes the same binary at its next restart). KV's cf06 kernel log for
2026-09-14 20:50–21:05 has no entries, so the 09-14 exit stays unexplained: open, with `telem_flap.py` as the standing
check and the next death as the evidence (the gather now runs under its unit with `Restart=on-failure`).

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

### #193 — string pass (comment pass, stage 9): help text, log text and emitted headers still narrate their history (2026-10-15, PLANNED for the Monday week-limit rollover, or Sunday if credits remain)
**What [tree]:** stage 8 (#188) left every string literal alone, because strings are code. They carry the same history
the comments did. Inventory at `6c0ac9b93` (`~/gnss/builds/stage9/inventory.py`, json beside it): **166 literals with
hard markers** (dates, `#NNN`, task refs, emoji, commit hashes, "used to") — cli.py help 49, other help 9, broker
log/print 20, tool log/print 9, other strings 48 (mostly the generator's emitted config headers), shell echo 8, viewer JS
14, C++ log 8 — plus ~236 multi-word ALL-CAPS phrases, many of them log TAGS that must stay. The stage-8 agents also listed
~250 strings (`string_cruft` in `~/gnss/builds/commentpass/findings_s8_{configs,broker,rest}.json`) that are stale
without any marker (e.g. `--prn-reconfig-interval-s` help still says one slot moves per interval; `--fit-min-snr` says
"CHORD wants ~60" against a live 30; `--rest-port` help says "NOT 12048" for a 12048 default). 417 argparse flags in all
(cli.py 108, the generator 42, 32 other scripts 267).

**Scope, in tiers:**
- **9a, help text = documentation.** Every `help=` / `description=` / `epilog=` / `usage=` string, reviewed in full for
  accuracy like the stage-8 docstrings (each flag's default and consumer checked), history out.
- **9b, operator messages.** Log/print/raise/SystemExit text, shell echo/usage, viewer UI strings: only literals that
  carry history or are wrong (the inventory + the `string_cruft` lists), plus ALL-CAPS emphasis that is not a tag.
- **9c, C++ log text** in lib/stages/gnss (8 literals): rides the next node binary; needs the cf05 build checks.
- **9d, the generator's emitted config headers** (gather, cube archiver, node headers). This changes every generated
  config's header, so after the merge KV regenerates (gen_fleet manifest + the three instance recipes; auto mode blocks
  it) or `node_up.sh` warns "no longer matches" and `gen_fleet --check` is red. Harmless while the nodes are down.

**Never change (the protected list, built deterministically before any agent runs):** a log message's leading tag
(CHAIN DIED, REFUSED (step, PRESENCE ±displaced, JOINT-CLK, COMB-DLL, SEED-OFFSET, JRRP, MODEL-UNTRUSTED, REBASE-WIPE,
BARE-WIPE, DEEP GATE, FADR BREAK, LABEL CONSENSUS, CLOCK BIAS ...); any fragment quoted in docs/*.md (41 quotes in the
buglist, closed list, runbook, joint-tracking doc), in the memory files, in the site scripts' grep/awk patterns, in the
14 tests that match log or note text (test_brdc_golden, test_clockguard, test_disp_refuse, test_epl_admit,
test_fleetadr, test_jointclk, test_jointstep, test_skyscope, test_timebase, test_trimarm, test_arraymap,
test_obs_day_roll, test_rrate_state, test_state_filter); format specifiers (`%s %d %(name)s {} {:d}`) in count and
order; key=value field names; units. argparse help is %-formatted: a literal `%` must stay `%%`.

**Proof, per edit (prove_s9.py, written first, tested against deliberate bad edits as prove_s8 was):** Python — the AST
is identical after blanking only the allowed constants (help/description/epilog/usage values; the message argument of
log/print/raise/SystemExit calls; the generator's header strings), the format-spec signature of every changed literal
is unchanged, every protected fragment in an old literal is still in the new one, pyflakes and CI's black as stage 8;
shell — bash's re-serialisation identical outside the echo/printf lines, `bash -n`; JS — acorn tokens identical except
string tokens; C++ — tokens identical with string literals masked, placeholders unchanged, CI's clang-format. The broker
replay digest hashes the POST stream only (`broker_equiv.py`), so log text cannot move a golden: all seven must stay
equivalent.

**Method:** as stage 8 — branch `kv/string-pass-s9` off `kv/chord-gnss`, worktree `~/gnss/strings9-wt`; two sequential
lanes (2 agents at a time), rewrite → adversarial review per group, one consistency pass; brief = STYLE_S8.md plus a
string section. ~13 groups: cli.py help ×2 sections; generator help + emitted headers; broker messages ×3; tools help +
messages ×3; scripts + site ×2; viewer JS + server; C++ log text. ~27 agents, about one stage-8 family of cost
(~8–10M subagent tokens), 3–4 h. On a session limit: freeze the journal results, relaunch a reduced script, tell
cut-off groups their files are partial (never `resumeFromRunId` across parallel lanes).

**Verify:** prove_s9 over every file; `--help` for every argparse script in both trees (exit 0, no traceback);
cf06 `~/gnss/builds/commentpass/{verify_s8.sh,verify_s8_tools.sh}` base vs edit (broker tests incl. the 14 above,
selftest, the seven goldens, the tool tests, pyflakes); `check_js.sh` on the gnss VM (cf06 has no node); `gen_equiv.sh`
comparing config bodies (headers differ by design under 9d); 9c: the cf05 CUDA + CPU builds, boost, `--check-config`.
Then ask KV to merge; after it, KV regenerates the configs (9d) and restarts the Python services when convenient.
**10-18 DONE, NOT MERGED.** Branch `kv/string-pass-s9` (worktree `~/gnss/strings9-wt`): five string commits on
kv/chord-gnss 96a341e7b (broker; generator + emitted headers; C++ log text; tools, scripts, site, viewer; the strings of
the #191 step 0c/1 commits, re-verified against the joint-clock coast and the search monitor) plus one docs/comments commit
(this entry, #192's list, CHORD_VIS_CAPTURE.md steps 2-4, three `_FROZEN` unit comments, the cube viewer's pol comment).
627 literals in 104 files; prove_s9 OK against kv/chord-gnss on all of them. Verified base vs edit on the rebased tree:
cf06 broker tests, selftest, pyflakes, tool tests identical and all seven goldens EQUIVALENT (8 digests, identical);
`--help` for 38 scripts; check_js 29/29; the viewer tests; generated config bodies identical (headers changed by
design); cf05 CUDA + CPU builds (WERROR), 57/57 boost, `--check-config` 9/9, the five C++ tools compile. Tools and
results: `~/gnss/builds/stage9/` (prove_s9.py, s9lib.py, protected.json, allow9.json, s9_results.json). `test_context`
fails one check at 96a341e7b itself (base and edit alike). The merge goes to whichever session holds kv/chord-gnss;
after it, regenerate the fleet and the three instance configs (every header changes) and close this entry.
**10-19 12:17Z post-merge steps done:** the six node configs regenerated from the manifest (gen_fleet.py, --check ALL 6
match: the stage-9 headers + the staged choco EOP table 10-17..10-22), the aggregator and gather configs regenerated
(bodies byte-identical, headers only) and the two instances restarted on them: gnss-aggregator 12:17:26Z, gnss-gather
12:18:03Z -- the gather's first restart since 10-13 21:08, now on the staged build_nodpdk binary (72d3a36c4b2b, the
#207 build) instead of its 10-13 inode. The EOP cron's "on-disk configs differ from the manifest beyond live data; NOT
regenerated" (since 01:17Z, three runs) was this header change; with the regen it is back to pushing and regenerating.
The cube archiver's config (cf06) was not regenerated: its header carries no recipe (it is in cubearch_up.sh) and the
cf06 leg is not touched from here. No node cycle is needed for any of this: the live nodes' EOP tables are pushed hourly
by the cron, the on-disk configs are now what a relaunch should start from, and nothing in the node binaries changed.


### #192 — comment pass, stage 10: the repair run for the ~200 small bugs the stage-8 reviewers confirmed (2026-10-15, PLANNED after stage 9)
**What [tree]:** reading every comment against its code in stage 8 (#188) surfaced 199 findings that an adversarial
reviewer judged real (184) or unsure (15): configs + generator 25, broker 74, tools/scripts/site/viewer 100. Raw lists
with file, line, mechanism and the reviewer's note: `~/gnss/builds/commentpass/findings_s8_{configs,broker,rest}.json`,
digests `s8_{broker,rest}_summary.txt` (the config family's list is in the memory note for #188). At least 20 are
repeats of the same fault, so ~175 unique. Rough split: ~30 log or text only (fold into #193), ~40 latent, 14 tests or
gates that cannot fail, ~13 config/generator choices, the rest small behaviour bugs, most in offline tools and the
viewer, a minority on the broker's live path. The `cruft_candidates` lists in the same files are a second dead-code
round's input.

**Fixed ahead of the run (10-15):** the viewer's lock rule — `gps_feed.js` `_merge` never copied `q`, `q_floor` or
`noise_probe` onto the row, so the q lock of `caaedbca1` never ran, the feed panel locked on `sig` and a noise probe
could show locked; `test_lock_metrics.py` now also runs its rows through `_merge` (3 checks failed before the fix).
And `stack_components.sh` hard-coded `_2026` in the viewer's `--pvt-obs-globs` (the PVT panel would have found no obs
file from 2027-01-01); the glob now takes any 8-digit date and matches the same 262 files. Live: the panel at the next
browser reload, the glob at the next gnss-viewer restart (stack_contract_gate flags the viewer's argv until then).

**For KV, not the run:** `bds_b2a` sets `rrate-command` with `rrate-feed-applied` at its default 1 (gal_e5a/e5b set 0;
the flag's help says 1 double-counts on the folded assembler); `config/chord_band_plan.py`'s `half_bandwidth_hz` missed
#179's change, so node L2C chains still cover ±511.5 kHz (changing it changes node channel sets: sky-judged); the GNSS
GPU0 merged cudaProcess is pinned to core 4, production's `frame_service_cpu` (19/24/31 overlap half 1); the B3I PRN
list seeds C06–08/C11–14 under the broker's floor of 19; `joint-p2c-rotate` withholds the best satellite from the
live-consumed joint solve.

**Plan for stage 10 (after stage 9, #193, so the agents read clean text):** step 0, one agent: dedupe, re-verify the "unsure"
ones, and tier the list — T1 offline-fixable with an offline test (tools, viewer, scripts, the gates that cannot fail:
the L3OC/L2OC/GLONASS code checks exit 0 when their C++ fails to compile, three tool tests cannot fail, ft_stress's
torn-read check cannot fire, `broker_equiv`'s dirty-tree check ignores modules the broker imports); T2 broker live path
(re-base hazards in `fleetadr.last_win` and `fits.unit_hop` — latent while the broker exits on a re-base; the q-stall
guard's `dr_state` None → CHAIN DIED; the 120-sample innov/minnov caps that make the "10-min" p95 ~4 min; `telem._index`
reading PRNs from slot 0 only; the TLE path posting az 0; gal_e6 shrinking the shared joint modulus to 5115) — fixed
offline like the ready-now set, goldens re-blessed with each mover named, judged on sky at the return; T3 KV decisions
(above). Then file-disjoint groups, fix + test + adversarial review per group, two lanes of one agent; the cf06/cf05
checks as stage 8. Cost about one and a half stage-8 families (~15M subagent tokens), 5–6 h.
**10-17 additions to the run's list:** (a) the viewer's rail panel should cover `live_element_ranges` only (the full-crate rows C–H rail 5–12% under the flat gains and are not ours); (b) a per-instance DARK detector: an instance whose prompt power sits at the floor on every PRN while its tap sees normal per-channel power is a config/wiring fault (#194), yet `FLEET-INST` counts it present and the viewer shows locks — compare the tap's power with the despread's per instance and name the instance; (c) L2C has nothing for the shared-reference machinery to hold (one channel per instance): exclude it or derive its reference from L5's (#154, 10-17); glo_l2oc (10-18) is the same case, one or two channels per instance, and has no pin in the shared-reference file either; (d) `gps_distributed_broker.py:1140` loop variable `sys` shadows the import (pyflakes); (e) ~~still G/E/C-only after #129: the BRDC supply's hourly coverage target (`_HOURLY_TARGET`), `gen_fleet`'s `SYS_OF`, the cube's `CHAIN_SYS`, and GLONASS propagation past ±15 min of a record is unvalidated~~ — done with the glo_l3oc roll-out 10-17 (docs/CHORD_MULTIBAND.md §6: all three tables carry R; propagation validated against the next record, 4.6 m line-of-sight median at 1 h, 18 m at 2 h, 117 m at 4 h); the review of that commit added the registry refresh + 6-h capability re-read, the 2.5-h R ephemeris cap (`GLONASS_MAX_AGE_S`), the transit-arc table and the viewer's registry-based GLONASS filter; what remains of (e): the daily file never covers today for R, so with the hourly supply down the chain goes dark after 2.5 h; (f) the B3I chain's PRN list 19–42 excludes C56–C58, assigned per SINEX and carried by BRDC. (g) 10-18: four of the six on-sky replay goldens are red without a code change: `broker_onsky_{l5_holds,l5_20260826,e5a_20260826,b2a_20260826}` replay to new digests (`0acfa2733cd5`, `621f54637e84`, `a78a963b04c7`, `cacaf230f995`) at HEAD, at 1e83eef62 and at their own blessing commit 8a392bd66 alike, while `broker_onsky_{l5,e5a}` (08-08) and `broker_fake_l5` stay EQUIVALENT. So an input the replay does not pin has moved since 10-13 (the BRDC pin's fingerprint has not): candidates are the IGS registry (refreshed 10-17 22:08; prnmap reads it for G and C whatever the flags) and the MGEX DCB product. Worse, after 00:00Z 10-18 `l5_holds` stopped being deterministic: seven runs gave seven digests, clean HEAD among them, while it gave `0acfa2733cd5` five times before midnight; so the replay also reads something dated today (the BRDC store or its background refresh are the first suspects). Pin every input beside `brdc_pin`, check two runs agree, then re-bless from a clean tree. (h) 10-18: noise probes feed the receiver-wide range-rate rows. The coarse rate feed (`gps_distributed_broker.py`, the `resid2` coherence filter) admits a probe whose noise passes coherence_s > 0 or coh_frac >= 0.3, while the KCOH feed skips probes by name; `broker_onsky_l5_holds` shows it happens (excluding all probes moves its digest). glo_l2oc's dark probes are excluded, since they are satellites glo_l3oc tracks; deciding the below-horizon ones is a behaviour change on every chain.

(g) cx19 drops packets at the distributor ring in bursts (645 `rx ring was full` warnings in its 13:48-18:32Z run,
up to 478k per poll; ~100-1000/s steady after the 18:33Z cycle, the NIC counter flat, the other five at 0-20/s):
pre-existing, not the 10-17 change; measure its lcore load and the recv1 reconnect bursts before blaming the
search or the control path.

**10-18 additions from stage 9 (#193):** the stage-9 reviewers confirmed 27 more code bugs while checking strings against
code: `~/gnss/builds/stage9/s9_bugs_real.txt` (one line each, with the reviewer's note; raw results
`s9_results.json`, `s9_delta_results.json` beside it). Five were text-only and are fixed on kv/string-pass-s9; the
rest join this run. Among them: `--frame0-nano` exits against a reachable fleet (the frame-0 check reads the node file
before the override); a stale DCB product logs a fall-back to broadcast group delays but keeps the stale table; the
gather header's REGENERATE recipe is sliced every 92 characters, mid-flag, and keeps `--out`; deadreckon's tripwire
divides by 29.3 m/chip on every chain; `gnss_beam_cube build --source elem` reports a health mask it never applies;
`node_wedge_probe.sh`'s remote `pgrep -f` matches its own shell. Live and for the #191 owner rather than this run:
`--detectors-monitor` also turns off dr_seed's slewed refresh (`_slew` tests `not ctx.detectors`), so gal_e5a's locked
satellites get the plain hold -- consistent with E33 walking to the 3.0 trim clamp 18:26-18:30Z; the monitored chain
never learns the nh hint (dr_solve_healthy is False with no offsets); its publisher reports has_search.
On (g): cf06 replayed all seven goldens EQUIVALENT at 19:30Z and 20:13Z 10-18, on e116f8b7b and on 96a341e7b, so check
whether the red/nondeterministic replays are specific to the VM (its BRDC store or registry) before re-blessing.
**How to run it (KV 10-18):** one subagent at a time -- two parallel lanes hit the session cap twice in stage 8 and once
in stage 9.

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
**10-17:** #129 is closed — GLONASS positions exist now (`gnss_ephemeris.sat_pos_clk` for R records), so the 1202 MHz half is testable against real positions from the archive. Not run yet.

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
**10-17, twice, from recv1's journal (KV):** 12:48:24Z `ConfigTracker: conflicting upstream content present for host: 10.222.1.19, port: 12048` (cx19, the canary on the new binary) → `/buffer_recv_bad_feed_mask: bufferRecv worker thread caught FatalError … shutting down` → `Main process exited, status=1` at 12:49:23, up again 12:49:33; 12:56:17Z the same for 10.222.1.27 (cx27, with the other five) → `code=dumped, status=6/ABRT` at 12:57:23, up 12:57:33 and stable since (every later cycle was the same binary). Each costs recv1 about a minute of N² plus the restart's #164 exposure; the second death was an abort with a core, not a clean exit.

**10-17 third recurrence:** 17:35:29Z, two minutes after cx51's #189 canary cycle (NRestarts 5).

**10-17 evening:** two more, at the stage-2 canary (19:49:45Z) and the fleet cycle (20:03:06Z); NRestarts 8.

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
**10-17 recount (cx27/cx44, 75 min from the 12:56Z start, crs-control as the controller):** FPGA-controller read failures 0, socket errors 0 (both were chive); `trim EXPIRED` 55/84 and `NIC dropped … rx ring was full` 394/207 — higher than 10-11's 42/134 per 110 min, but the window holds two broker restarts, four node cycles and the #194 dark instances. Recount at a quiet hour before reading anything into it.

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
