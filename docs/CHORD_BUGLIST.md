# CHORD GNSS — open items

What is open on the CHORD side (branch `kv/chord-gnss`). Closed items are in
[`CHORD_BUGLIST_CLOSED.md`](CHORD_BUGLIST_CLOSED.md); their narratives are in git history.

**Last reconciled: 2026-11-21**, against kv/chord-gnss `5e7ad2eb0` and gnss-broker `fcf2f1bd` with live reads: two
reviewers read every open entry and a third tried to refute each proposed closure (summary
`~/gnss/builds/buglist_review_20261120.md`). Closed: **#207, #209, #208, #206, #202, #200, #229, #168, #164, #163**
(fixed); **#212, #213, #199, #196, #214, #114, #124, #131, #75, #14** (premise gone); **#203, #210, #227, #226, #56,
#169, #192** closed with a residual line kept below. Kept open on the check: **#204**. Opened: **#232-#235**. Previous
passes: 2026-10-13 after the dead-code merge, 2026-10-11 at `0011cf3b1`, 2026-09-12 at `b45949144`.

**Fleet state (11-21):** six GNSS X-engines (cx19 cx27 cx43 cx44 cx51 cx52) on the R1+R2 binary `437d6a9cd` with
config/generated; cx42 and cx47 back on stock kotekan under choco since 11-21 02:05Z. Broker stack on the gnss VM from
gnss-broker `fcf2f1bd` (per-half trim targets), the gather on `gather-e4a01a357`. recv1 on chord `a22ff0da41`.

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

### #235 — tle_refresh.py is in no crontab: the DTV pass freeze's cx1.tle is 11 days old (2026-11-21, review)
The viewer keeps its own TLE caches fresh, but `/var/tmp/kvand-dtvsteer/cx1.tle` (cf06, #231's `--pass-tle`) is refreshed by
nothing, so the CHUANGXIN freeze windows drift as the TLE ages. Add a daily refresh, or point the watcher at a cache that has one.

### #233 — the viewer's metrics should be a Prometheus endpoint for the CHORD Grafana (2026-11-20, KV: "Ultimately we should be publishing all the viewer metrics to a prometheus endpoint so it can be scraped and viewed in our larger CHORD grafana setup")
The /alerts panel (#171) is the stopgap. Serve the viewer's numbers (and the ALERT/heartbeat states) as /metrics in the
Prometheus text format, so the site's Grafana scrapes and alerts on them.

### #231 — CHUANGXIN 1-03 / 1-04 burst at 369.05 MHz on every pass and trip the DTV rolling fit on freq IDs 1887-1892: freeze the fit through each pass now, null the satellites later (2026-11-09, KV: "put a note in the todos to try and null these late as well, but for now, yes, let's freeze during the transits")
**What:** two CAST store-and-forward LEO satellites (NORAD 37930, 40137; one sun-synchronous plane,
P 100.7 min, i 98.7°) emit ~1.5 MHz at 369.05 MHz (freq IDs 1886-1893; x2 shoulders over 1880-1899 on
strong passes). A pass above ~5° gives an 8-10 min burst, x4-220 in channel power. The rolling fit
(tau 1.5-8 s) follows it, and the channel trips (residual / input > 2) when it ends: every pass-end
trip from 11-06 to 11-09 was on 1887-1892, 4-9 min before the satellite set. Identified 11-09 from
the fringe motion (bin-to-bin phase steps on the redundant 6.3 m E-W and 8.4 m N-S pairs, fitted
against every LEO in CelesTrak's active set: 6-26°/bin for these two, 29-100 for the next best) and
from the pass list (all 13 bursts fall on their passes). No published frequency found.
**Freeze (status 2026-11-09):** `dtv_roll.py watch --pass-tle` (kv/dtv, not yet committed or live;
TLEs in cf06:/var/tmp/kvand-dtvsteer/cx1.tle) freezes every source on the 8 GPUs that hold 1886-1893,
from 60 s before each rise to 60 s after each set: about 15 % of the day. Bursts start >= 80 s after
rise and end >= 90 s before set; the replay covers every burst in the 11-06..11-09 extracts. The
watcher logs the TLE age at each freeze; refresh the file every week or two.
**Todo (null them):** steer a source onto each satellite from its TLE for the pass. dtv_steer.py is
built for MUOS: its 10 s update, 30 s rate and per-pass eigen start need checking against a LEO that
crosses the sky in ~13 min. The other route is to project them out of the N2 from TLE directions, as
the GuoWang lanes do (#190).
**11-21:** the pass freeze is LIVE (kv/dtv `cd9eb77d2`; cf06 gnss-dtv-watch runs `--pass-tle`). Open: refresh `cx1.tle` (11 days old, no cron, see #235) and null the bursts (#190 tiers 2-3).

### #230 — no alert for the first GuoWang satellites at 1174.404 MHz: the filed odd 50° planes would land inside the L5 / E5a band, and railing will be the first sign (2026-11-07, KV: "our first indication will be railing at 1174, so maybe we should set up some sort of watch there, maybe flag as a todo, we can likely hand that to Jim to add to the Prometheus / grafana alerts")
**What:** the ITU filing (CM20 2023) puts 216 more GuoWang satellites in 12 odd 50° planes at
1174.404 MHz (GPS L5 - 2.046 MHz), RAAN midway between the 12 flying 1202.025 MHz planes; the polar
rings may instead use 1178.496 MHz (L5 + 2.046). None fly yet. The gw_1202 chain and the beam dump do
not cover either carrier, but the F-engine does: a satellite at the filed pfd (-129.3 dB(W/m2/MHz), the
same as 1202) clips the 4-bit samples in a narrow block of channels on a close pass, as the 1202 ones do.
**Watch:** freq IDs 6003-6023 (1172.36-1176.45 MHz) for 1174.404, and 6024-6044 (1176.45-1180.54 MHz)
for 1178.496 (f = freq ID x 195.3125 kHz). Signature: the clip fraction rising together across the
block for 1-3 min, fleet-wide in the same seconds, with no GPS or Galileo L5 transit to explain it.
Quiet-sky L5 already has a broad clip hump over 1173-1180 MHz, so alert on a pass-shaped excess over
each channel's quiet-sky rate, not a fixed threshold. Source: gnss-rail.service (rail_watch.py record,
per-channel hi/lo clip fraction every second).
**Proposal (for Jim):** export a per-band clip fraction for the two blocks to Prometheus and add a
Grafana alert on that excess. When it fires, match the time against the predicted 1174 slots and add the
channels to the beam dump to decode codes and IDs.
**Status:** filed 11-07, not started. Orbits, numbers and channels: the GuoWang Full Build page
(https://claude.ai/artifact/FKaSJgsCPb3HYuxg6DVEMR; sources in builds/gw1202/future/).

### #222 — a broker cold start after a long outage predicts from an expired BRDC set and hints nothing for up to 15 min: the collapse guard cannot see a sky that was empty from the first parse (2026-10-30, live)
**What:** the stack had been down 19 h. The broker started at 16:44:45Z and every chain parsed the cached
BRDC files at 16:44:47, whose records were all more than 4 h from toe, so `brdc_predict` kept none: `ctx.pred`
empty, no `narrowed search` line, `n_prn 0` on every chain in `/get_status`, and the aggregator logged "0/N
PRNs hinted ... scanning NOTHING" on all three searches. The hourly merge landed at 16:45:12 ("BRDC hourly
merge ... Store 0 -> 566 record(s)"), but the sky re-parses only every `_SKY_REFRESH_S` (900 s), and the
early re-parse in `sky.py` fires only on a collapse below half of `peak_n` with `peak_n >= 4`. A sky that was
empty from the first parse has `peak_n` 0, so it is never a collapse. A broker restart at 16:49:49Z after
the merge gave 9-15 PRNs per chain and hints within 2 s.
**Fix (candidate):** treat an empty first sky like a collapse: with `peak_n` 0 and the store's newest record
older than the 4 h window, re-parse on the bridge gate (60 s) until a satellite appears. **Workaround:** after
an outage longer than ~4 h, restart the broker once the `BRDC hourly merge` line has appeared, or wait 15 min.

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
**13:20-13:50Z OUTCOME:** gal_e5b, bds_b2a, bds_b3i walked home by ~12:50Z. gal_e5a did not: E5/E9/E15 parked at +85 m
(one lobe) with the trim railed at exactly +3.000 and E34/E36 at -95 m with -2.6/-2.7 -- the clamp (3.0 chips) is
smaller than the lobe (3.27), so the trim alone cannot make the last step, and the wipe that does it was refused by the
displaced-presence referee: "PRESENCE -displaced: same offset -- 3 PRN(s) at off +2.9 REFUSED, a chain event rather
than N detections" (31 refusals; E5b's PRNs reached the last lobe one at a time and were admitted). The referee's
chain bar was any three satellites within 0.1 chip; three of thirteen parked on one lobe beside eight on their peaks
is not a chain event. **Fix bfe3d4802:** `displaced_chain_event` needs the same-offset group to be at least half of
the satellites the chain sees (present on their peaks plus the displaced candidates; a null crossing still
qualifies, the three-PRN bar stands without a count); test_disp_refuse gains the cases. Broker restarted ALONE
13:26:44Z on a warm search (the gps_l5 overlay-period lines had fallen to 0 per 10 min): at T+20 the five PRNs read
0 +- 1 m with trims 0.3-0.4, every model-primary chain 0 excursions, no lobe walk (the rule above honoured), 0 CHAIN
DIED. Open: a code-side guard for the cold-search restart itself (the joint state's n/sigma gate or a relabel holdoff). **10-20:** the lobe referee
(185c96248, live 01:32Z) bounds the trims (clamp under half a lobe, no trim during a slew, release on lost xcoh), but
the first restart with it showed the root: before the joint feed, the fresh broker's dead-reckon model lacks each
band's learned tau and slews trackers that are on their peaks (TRKRES ~0) by a chain-wide gap (B2B +19.25, L3OC +7.77,
B3I +12..20, E5b -2.6 chips); chains were ~100 m off for ~5 min. No slew while the joint state is unestablished and the
tracker is proven on peak is the missing gate.

### #219 — gps_l2c's code reads ~+4 m against gps_l5 on the same satellites, so L2C cannot join the tracker-fed joint state (#191 step 3) (2026-10-28, KV: "file discrete bugs for the L2C and GLONASS bias offsets")
**What:** step 3 admits a chain's code to TRK_STATE only once its signal reads within ~1 m of GPS L5 on the same
satellites (#191 §6.2). gps_l2c does not, so it is unfed and stays `joint-source: search`.
- **Same-satellite l5 - l2c** (isb_obs_ext, 10-26 06-13Z, 15 PRNs): median -3.94 m, p10/p90 -7.25/-0.28. L2C reads
  late by ~4 m and with a wide spread.
- It moved from -13.9 to +8.6 m when the DLL spacing went 0.4 -> 0.33 (10-25). The cleanup read +22 m on 10-25
  against ~+4 m on 10-26. The spacing, the coherent N-channel Dirichlet over its 4-6 contiguous channels
  (gen_chord_gnss_config.dll_spacing_chips) and the q ~2.0 with cx42/cx47 out (#216 note) all bear on where the loop
  sits on the peak.
- The mirror test (10-26) showed L2C's own large excursions (G30 a 17-chip swing).
**Next:** root-cause before any calibration constant (KV 10-24):
- with the fleet at 8 nodes (6 contiguous channels), re-measure l5 - l2c vs the DLL spacing and the per-channel tap
  geometry;
- check the CM/CL combination and the L2 group-delay model term (§6.3 0a: "still to add: the L2C ... terms");
- then the step-3 admission test.

### #220 — the GLONASS chains (glo_l3oc, glo_l2oc, glo_l2of) carry per-satellite / per-signal code offsets against the GPS/Galileo/BeiDou chains, so none can join the tracker-fed joint state (2026-10-28)
**What:** all three GLONASS chains are unfed and stay `joint-source: search`.
- **Chain medians** (m, clock included, isb_obs_ext 10-26 06-13Z): glo_l3oc 4463.48 vs gps_l5 4450.93, so L3OC sits
  ~+12.5 m.
- **Per satellite:** R10 -37 m (trim -1.08), R16 -14 m.
- **#210** found glo_l3oc's +44 U / +8 N / +5 E (10-19) to be a SATELLITE-SIDE term (the FDMA broadcast clock that
  the R records carry), with a month-scale day-to-day of 4-8 m.
- **New 10-28: glo_l2of** (FDMA L2, live since 17:00Z) adds per-frequency-channel (k) inter-frequency biases, the
  known FDMA hardware delay per carrier, on top. glo_l2oc is CDMA, and its offset has not been measured.
**Next:**
- re-run isb_obs_ext with glo_l2of rows once the obs archive has a night of them;
- the per-k structure of L2OF against L2OC/L3OC on the same satellites (M+/K carry several);
- decide whether GLONASS joins TRK_STATE with its own clock term (an inter-system bias state, which §6.2 ruled out
  for the per-signal offsets) or stays on search.
Root-cause first: no calibration table (KV 10-24).

### #210 residual — the L3OC bias table: monthly re-measure, R28 unmeasured, and the calibration-table rule (2026-11-21)
**Open:** re-measure the L3OC table monthly or at a slot change; SVN 809 (R28) is unmeasured (0). The table is per-satellite: check it against KV's 10-24 rule on calibration tables before calling it the fix.
**Closed 11-21:** `GLO_L3_BIAS_M` table (`5e9befd9c`), the ISB question moved to #220 (CHORD_BUGLIST_CLOSED.md).

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
**11-21 review:** the GC pacing is gnss-broker `23357a68` (live as gc-interval-s 21600), not `decca102e` (the fleet-ADR non-finite fix), and nothing links the 10-18 storms to GC. Not done: nice/ionice/delaycompress in scripts/gnss/site/logrotate.conf, and a drop counter on the gather's :12051/metrics.

### #203 residual — gps_l5's search clock solve drops to not-established with 5-7 satellites present (2026-11-21)
**Open:** gps_l5's search clock solve drops to "not established" with 5-7 satellites present (64 WITHHELD lines 02:41-03:04Z 11-21); why it scatters is unexplained.
**Closed 11-21:** #191 step 0c COAST `a3bfeabfc`: bds_b3i held 152.26 -> 151.88 over 02:47-03:07Z 11-21, 0 JOINT-CLK refusals (CHORD_BUGLIST_CLOSED.md).

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
**11-21:** fixes (1) and (4) built: `--xband-referee glo_l3oc` on glo_l2oc (kv/l2oc-referee `fdca22258`), NOT deployed (broker restart). 11-20 data: the L2OC-L3OC level is -37 m on every PRN; R06 sat a chip high most of the day, R10 and R20 a chip low for long stretches.
### #198 — reopened: the fleet cycle still ran at ~30 s per node with the bufferSend fix in the binary (2026-10-17)
**What:** KV cycled the five nodes onto `b4ab492da` (which carries `d624d7784`, confirmed by `--version` and the
source) at ~22:40Z and saw the 30 s per node again, with recv1 (restarted 22:48:34Z) and the cube archiver (#199) down
during the cycle. The closed row's judge is therefore not met and its diagnosis incomplete: the sliced wait is in
place and `kotekanMode` stops every stage before joining any, so the senders' reconnect wait is not where the time
goes. Unmeasured: whether the 30 s is the stop (SIGTERM -> exit) or the start (`node_up.sh` waits for 12048 to
answer, which comes after ConfigTracker and the GPU set-up; nothing in `node_up.sh` sleeps). **Next cycle:** time
`ssh <node> sudo systemctl stop gnss-node` and `node_up.sh <node>` separately on one node, and read
`/tmp/gnss_node.log.1`'s tail (the previous run's shutdown) for the last stage to go.

### #195 — a PRN whose disc sits at the clamp for minutes has no way back to its peak (2026-10-17, split from #175)
**What:** the #175 refusal stops a chain-wide displaced admission, but a PRN that is already railed stays railed: a railed
disc is leak-bounded (`gnssFleetDll::dll_tau` clamps at ±1, at most ±0.25 chips of trim) and the model holds the displaced
position, so only a fresh displaced-presence admission pulls it back — E4 took 50 min on 10-11, E16 longer. The
`PRESENCE +displaced` line is rate-limited to one per 120 s, so the log under-reports admissions.
**10-20:** built as the lobe referee (185c96248, live 01:32Z): an armed PRN with xcoh < 0.3 for 30 s and |trim| >= 0.25 lobe has both trims released until erased, then restarts from the seed; LOBE RELEASE in the broker log. Close after a week without a hand disarm. **Original fix (superseded):** a per-PRN 60-s history beside D3 (the fleet row's disc, the readback trim and `spec_fit[prn].tau`) and
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

### #216 — the fleet's node list is typed by hand in ~15 places; adding or removing a node is hours of edits (2026-10-25, KV: "fix up hardcoded node lists, make sure we're able to scale up and down quickly")
**What:** cx42 came back and cx47/cx52 were offered (10-25 ~01:55Z), which would give all 16 combs (every channel
of every lobe). The hardware matches (A40 x2, Gold 5416S, same isolcpus and hugepages, E810 on vfio, the NFS builds
visible), and the binary needs no change. The node list, though, is hand-maintained in gnss-broker:
- `config/gnss_chains_chord.yaml`: ~32 URLs per node across the chains' trackers / spectrum-endpoints /
  dll-combiners. L2C and L2OC are enumerated per GPU from the combs by hand ("re-derive the list if the comb
  offsets change").
- `config/gnss_fleet_chord.yaml` `nodes:`: search-port-base = 11040 + 2 x the node's position in the aggregator's
  `--aggregator-instance` order. Any other value reads one node's comb as another's.
- `config/chord_gnss_node.yaml`: `eop_reference_nodes: [cx47, cx52]`, the stock fallbacks for EOP and bad inputs,
  plus `stock_parity.py` and test_probe_ports.
- About a dozen site scripts, each with its own list: stack_components.sh, eop_cron.sh, eop_push.sh,
  bad_inputs_cron.py, rail_watch.py, fengine_status.sh, node_wedge_probe.sh, elemsweep.py, viscap.py,
  port_axis_gate.py, telem_e2e.py, node_up.sh. The elem shared-ref JSONs are keyed per instance.

Estimated at 1-2 h of edits and tests, then aggregator restart -> 10 min -> broker restart.
**Direction:** one fleet list (nodes, combs, GPUs, roles: GNSS / control / stock reference) in the fleet yaml.
The generator derives everything else from it: the chain URL lists (from each chain's channels and
gpu_offsets_mod16), the aggregator order and ports, and the scripts' host lists (emitted into gnss_vars, or read
at run time). Then adding or removing a node is one line, a regen and the aggregator/broker restarts. A check
in `gen_fleet --check` should refuse any host named outside the fleet list. Note the stock-side dependencies:
with no stock node left, the EOP and bad-input fallbacks and stock_parity have no reference.
**Built 2026-10-25 (kv/fleet-nodes, not deployed):** `nodes:` in config/gnss_fleet_chord.yaml is the list, its
order the slot (port 11040 + 2k); a leaving node is `role: off` and keeps its slot. config/gnss_fleet.py derives
the ports, the aggregator's feeds (gen_fleet writes the aggregator config from the manifest's `aggregator:`
block), every chain's endpoints (chains-yaml keys set to `fleet`, expanded at broker load by the generator's
covering rule), the stock references (site hosts not in the fleet) and the site scripts' host lists.
`gen_fleet --check` refuses hand-written endpoint keys and foreign hosts and compares the derived chain
instances with what each config builds; test_fleet_hosts.py refuses hand-typed host lists. On the 10-25 tree the
six node configs regenerate byte-identical, the aggregator's body is identical and every chain resolves the same
except sbas_l5, which gains cx42. Adding a node: site table entry with its combs, choco maintenance mode, one
line, regen, aggregator, node, recv1, broker >= 10 min later (CHORD_GNSS_RUNBOOK.md section 5). Left: the
obs writers move their frame-0 node to the first fleet host at their next restart, and the aggregator
config keeps its agg6 file name.
**Combs read from the frames (2026-10-25, kv/combs-live):** `gnss_fleet.py combs` reads each host's combs from
the freq_id of its n2_buffer frames (stock nodes too) and `--write` sets the table; gen_fleet refuses to write
and --check fails on a fleet host whose frames disagree; the broker's RF poll logs COMB MISMATCH per tap
while running. All eight site hosts read equal to the table on 10-25 (cx27's GPU 1, never read since 10-17,
is 8 as entered).
**Left (2026-10-26): L2C's tap spacing does not follow the fleet.** L2C sums its contiguous channels
coherently (6283-6288, one per GPU: 6283 cx47/1, 6284 cx42/1), so the correlation's mainlobe widens as
nodes leave. The chain's `dll-spacing-chips` (its `chains:` row in config/gnss_fleet_chord.yaml, read by
the node taps and the broker) is a hand-set 0.33 for five or six channels. With cx42 and cx47 out for site
tests (10-26 15:54Z) four remain (6285-6288), the first null moves 0.44 -> 0.66 CM chips, and q sits at
2.0-2.1 on every PRN at any power, under every 2.2 lock bar (lock_q, dll_quality_min, q_stall_bar); disc
~0 and 8 present. Nodes now come and go, so derive it from the fleet's count of contiguous L2C channels
(~0.5 for four, 0.33 for five or six), at each regen.

### #215 — changing the live element set costs a release cycle (2026-10-25, KV: "a weirdly large amount of processing")
**What:** at the F-engine's return, row H's Y pol went out and H-X stayed good. That needed a new epoch.
The element count is part of the epoch's identity, and the positions file and the shared ref are keyed by
the epoch key, so both re-key. The epoch table, the manifest and the keyed files live in the split repo, so
the change runs kv/chord-gnss -> cf05 extract + suite -> push -> gnss-broker pull -> regen + --check (plus
previews on both binaries): ~20-30 min for one flag on 8 inputs. A hand edit of the generated configs is
undone by the hourly eop_cron and flagged by node_up.
**Why it is odd:** the epoch table says an element going dark does not start a new epoch, and the bad-input
flagger already takes flagged inputs out of the projection stations, chan_noise_norm and the shared ref
(d5c07bde2). The element sum needed nothing either: ElemCal gates dead inputs on coherence. And the key
hashes cube_order, the pointing name and the element records, yet here its digest stayed 441d2daf: only the
epoch NAME (rows-ag112 -> ah120) moved the positions and the ref.
**Measured 2026-10-25 (KV: the release, not its existence, should not take tens of minutes):** every
mechanical step together is ~2 min: kotekan clone for the split 20 s (293 MB over NFS), filter-repo 2.4 s,
the gnss-broker suite 31 s (140 jobs), regen of six nodes 12 s, gen_fleet --check 13 s, --check-config 0.8 s
per config and binary, push/pull seconds. The rest is procedure: the epoch edit and the hand re-key of the
positions and the ref, keeping two repos and split_paths.txt consistent (append-only: the first ah120 cut
replaced two lines and stopped the extract fast-forwarding), preview passes and two sessions handing over.
The complexity is in the number of hand-run steps and invariants, not in compute.
**Directions (not decided):** (a) treat inputs that are dark or removed for hardware reasons as a health
state (bffs or a manual mask read at regen) rather than an identity change, so a new epoch means a
feed's cabling or position changing (G01/G02 here) or a repointing, never a dead input; (b) key the positions and the shared ref by dish, pol and pointing rather
than by the epoch's element set, so a mask change does not re-key them; (c) a live mask set over REST, like
the elem-proj switch, persisted by the manifest at the next routine regen; (d) one scripted release
(epoch edit -> re-key -> split release -> regen -> --check) that enforces the invariants instead of a
recipe two sessions follow by hand.
**Built 2026-10-25 (kv/epoch-health, (a), (b) and part of (d); not deployed):** (a) config/gnss_bad_inputs.json
holds the inputs out for hardware reasons (row H's Y pol, 120-127); the generator's bad_input_set returns it with
bffs's list, so the stations, chan_noise_norm_elems and the shared ref's zeros leave it out even with bffs
unreachable; the outage guard counts bffs only and updatable_config/bad_inputs stays bffs's. gnss_arraymap check
requires every masked input in the open epoch's cube_order and every `removed` element of it masked. The manifest's
stations are back to 0-6,8-127; with the same bffs block the eight node configs regenerate byte-identical apart
from the recipe line. Fixed with it: elem_shared_ref.py post and watch --act posted the reference whole, so every
re-post put the zeroed inputs back at full weight; each POST now carries the instance's generated zeros.
(b) an epoch's `wiring` id (ah120: g0102-fixed-20261024) and ArrayEpoch.ref_key() = pointing + digest of
cube_order and wiring key the positions and the ref; key() is unchanged (ah120 still 441d2daf). A positions file
with `pointing` and `positions_by_dish` is read per dish (config/elem_positions_taua_20261024_by_dish.json);
`elem_shared_ref.py relabel` re-keys a ref (--swap for a pol swap at the feed), and the 10-25 ref is relabelled
to config/elem_shared_ref_20261025_f0reset_refkey.json. Regenerated, the configs differ only in
elem_positions_epoch (and the recipe line). (d) split_paths takes config/elem_*.json by glob; test_split_paths
checks the append-only rule. A dead input is now one mask entry, a regen and a node restart: no epoch, no re-key.
**Left:** (c) the live REST mask (the restart a mask change costs), and (d) the full scripted release.

### #218 — ⚠️ IMPORTANT: GuoWang LEO satellites transmit GNSS-like signals at 1268.52 and 1202.025 MHz; identify the code before the constellation fills the sky (2026-10-28, KV: "potentially disastrous if they're all broadcasting, we don't have the modes to blank that much sky")
**Found (10-27/28; scratch e6lobe1027/, emit1027/, nearfield1028/, burstspec1028/; memory
chord-e6-lower-lobe-is-an-emitter-20261027):** the 1263-1274 MHz projection residual (#214's "E6 lower lobe", 2-3 dB)
is NOT Galileo E6-A or the lane transfer. Satellites there are removed -14..-16 dB, and the transfer caps at
12-18 dB, worth < 0.1 dB if improved. 65-75 % of what is left is this emitter.
- **Source:** China's GuoWang constellation (CelesTrak GROUP=hulianwang, "HULIANWANG DIGUI-nn"; no "guowang" group).
  - Burst tracks are far field, 0.29-0.32 deg/s at el 58-68, headings 112/67 deg (~50 deg inclination), range
    >= 24 km.
  - 5 of 7 tracks match a DIGUI satellite at 1-8 deg with matching angular velocity (~5 % by chance per burst).
  - Trains of 2-4 bursts every 12.67 +- 0.30 min = consecutive satellites in one plane (DIGUI-48 then -47).
  - ADS-B (adsb.lol) and 13k other catalogued satellites ruled out.
- **Signal** (raw channels only, 300-1500 MHz, 78 rail bursts 10-25..27, 53 archive bursts 10-26):
  - Lit at exactly two places. (1) 1258.3-1278.6 MHz, BPSK(10.23) nulls at 1268.52 (the BDS B3 carrier), centre
    1268.41 +- 0.09, flatter top than BPSK. (2) 1200.1-1204.1 MHz centred 1202.026 +- 0.026 (the GLONASS L3 carrier),
    nulls every 2.0 MHz (~2.05 Mcps).
  - Continuous and noise-like (duty >= 33-50 %), fixed frequency (drift < 0.01 MHz/s); not pulsed, swept or DME.
  - Three variants, constant within a train: 1202 lobe with filtered sidelobes (50), with full sidelobes (16), no
    1202 lobe (5).
  - Polarisation varies burst to burst (not GNSS RHCP).
  - Bursts ~5-10/h, x3-50 band power. Plus a continuous ~+35 % at 1267-1270, likely the ~15 DIGUI always above the
    horizon seen in sidelobes (not yet shown).
  - The "1200.78/1202.73 lines" (#214) were cx19's raw combs sampling the 1202 lobe in a projected stretch.
- **Why it matters:** ~180 launched; ~13,000 filed (~7,000 near 1,145 km, ~6,000 at 500-600 km).
  - Now ~15 above our horizon, ~6.6 above 15 deg, ~1.5 above 45 deg (8 % of a ~1150 km shell).
  - Fully deployed: ~800-900 above the horizon, 300-400 above 15 deg, ~70 above 45 deg, and main-beam crossings
    every ~10-20 s.
  - Spatial nulling cannot scale (113 elements per channel). A known code would allow WAVEFORM SUBTRACTION (despread,
    track, subtract per element), which removes only that signal's energy and costs no spatial modes.
  - 1215-1300 MHz is RNSS space-to-Earth, so it is likely in-allocation.
  - No public report of a GuoWang L-band/nav payload found (web search 10-28). SNIFFLES I (arXiv 2608.12999) tracked
    GuoWang at 1-26 GHz with Mopra/ATCA; its abstract names nothing below 1613 MHz (full tables unchecked).
- **TODO:**
  1. A BASEBAND capture through a predicted near-boresight DIGUI pass. KV is talking to the team planning a
     full-band, full-array 50-100 s dump (~week of 10-28). Our own capture over just 1194-1210 + 1248-1292 MHz may
     allow a much longer recording. Either way, include both lobes and time the window from fresh TLEs.
  2. A pass schedule for the chosen dates: every DIGUI pass within ~3 deg of boresight. Bracket the middle (strongest)
     burst of a 12.7-min train; refresh the TLEs the day before.
  3. An offline search pipeline, ready before the data:
     - code period from the per-channel line spacing (B3I-like = 1 ms);
     - the B3I PRNs, then the B3I-style Gold family (~8k codes), on a cf06 GPU;
     - LEO Doppler +-30 kHz at up to ~100 Hz/s.
     Validate on the MEO B3I/GLONASS satellites in the same data, so a null result means something.
  4. Then: a live b3i-chain trial and waveform subtraction; or, without a code, TLE-steered geometric lanes (tier 2:
     TLE ~0.05 deg against a synthesised beam ~0.18 deg).
  5. Tell the SNIFFLES authors if they lack 1202/1268.
**Update 10-28 (tier2_1028/RESULTS_tier2_20261028.txt):**
- **Correction:** the first TLE matches converted Unix time with skyfield `ts.utc(1970,1,1,0,0,t)`, which counts the 27
  leap seconds, so positions were 27 s (~8 deg) early. Use `ts.from_datetime`.
  - Corrected: 37 of 43 10-26 bursts sit on a DIGUI TLE track at dt 0.0 +- 0.2 s, 0.3-8.9 deg from boresight, each a
    different satellite; the 12.7-min trains are consecutive DIGUI numbers.
  - 40 of 44 passes within 6 deg give a burst. DIGUI-01/-03 (the first launch, 2024-240) are silent.
- **TLE pointing** is 0.007-0.036 deg (0.04-0.2 synthesised beams).
- **Projecting the TLE track** removes a burst by ~10 dB (16.5 ideal). It is limited by direction-dependent element
  phase, not pointing. The eigen path already takes ~7 dB, so tier-2 lanes buy <= 0.5 dB; use the TLE to gate and seed
  the eigen path instead.
- **The continuous +35 % is the sum of the ~14 DIGUI in view through dish sidelobes** (60-70 % from satellites
  45-90 deg off boresight). It is spatially spread: 14-28 extra modes gain only 0.7-1.6 dB over a null.
- **At ~850 above the horizon** that sum is ~+10 dB over system noise in 1258-1279 MHz and full rank, so no projection
  removes it. Code-based waveform subtraction (TODO 3-4) is the only scalable fix.
**2026-10-29 (docs/CHORD_TAUA_GUOWANG_20261029.md):** the bursts with no catalogued match come from uncatalogued
sets of nine at the in-plane midpoints of three catalogued 50-deg planes (RAAN 24.5 / 54.1 / 84.4 deg; orbit fitted
from recv1 burst images, TLEs in taua1029/hidden_guowang_20261029.tle). The burst spectrum is BPSK at 10.23 Mcps on
1268.52 MHz and 2.046 Mcps on 1202.025 MHz (null positions), with a flat-topped, high-sidelobe envelope that a
white-code sinc^2 does not give. Deprojection removes ~70-85 % of a burst in projected channels.

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
**10-25 on sky, after the F-engine's return: the fix holds.** SEARCH-MONITOR gal_e5a search - joint -0.045 chip (+0.47 before); trims E5a -0.15, E5b -0.09, E6 -0.05, B2a -0.19 chips, as B2b/B3I; chain medians vs gps_l5 (17:05-18:55Z against the 10-24 night) E5a +12.9 -> -1.6 m, E5b +12.9 -> -0.6, E6 +11.0 -> -0.8, B2a +7.9 -> -5.2; same-satellite differences within a constellation <= 1.2 m. Step 2 built the same day (jtrk.py, the shadow feed). Open: the BeiDou chains' common -4..-5 m vs L5 (per constellation; the station comparison), and gps_l2c moved -13.9 -> +8.6 m vs L5 after its 0.4 -> 0.33 spacing change.
**10-24 ROOT CAUSE FOUND AND FIXED (offline, F-engine down): a floating-point modulo in the broker's range model.**
Two things first. (i) The station comparison above was made with opposite signs: `code_resid_m` is tracker minus model
(a late arrival reads NEGATIVE; the pre-geoid -18 m x sin(el) term of a model 16.5 m too high reads that way), while
`drao_resid.py` is P minus model (late reads POSITIVE), so the receiver-side excess is `code_resid + r_station`, not
the difference. With the flip the pattern is not "+9 Galileo, -14 B3I, -8.5 L2C, 0 B2a": it is **the four chains whose
tracker code has a 100 ms period (E5a-Q, E5b-Q, E6-C, B2a-P) at +12.7 / +10.7 / +13.4 / +13.9 m, and every 20 / 10 / 1
ms chain (L5-Q, B3I, L3OC, B2b) at 0 +- 2 m** (10-24 02-17Z against the 10-12 station run; two Javad stations, PRDS +1.5
and SASK +2.1, agree with the six Septentrios on E5a-vs-L5, so the model is clean on 8 receivers). The night series
(`fixtures/jointstate/isb_series/`, 06-13Z, 09-10..10-24) has the level at +9..+15 m on every night through every
fleet, model and position change; the "growth 09-25..10-03" was the day window's gps_l5 datum wobble. (ii) The whole
node side was cleared by an independent ICD-tiled sky through the production bank, the fused despread, the Psi
synthesis, the 4+4b pack and the record slot (cf05, `/var/tmp/kvand-halfchip/`): all four signals placed alike to
0.3 ns. **The cause:** `deadreckon._dr_apply_reload` set `t0m = gpst_of_utc(utc0_sample0) % ctx.drp.t_code` with
`t_code` a double. 0.1 s as a double is 0.1 + 5.6e-18; over the 1.48e10 periods of the 2026-10-23 frame 0 (GPST
1476826616.000004) the exact remainder of that wrong period is **82 ns short on the 100-ms chains and 31 ns short on
the 20 / 10 / 2 / 1 ms ones** (0.02, 0.01, 0.002, 0.001 err alike). The common 31 ns lands in the receiver clock; the
**51 ns difference = +0.524 chip = +15.4 m** is the step, with the right sign (model low, residual positive), the same
time on E6, static (frame 0 moves by days: < 0.1 ns), seen by the search and the tracker alike (both read
cp_predicted), invisible to the e2e harness (small epoch) and to the station script (metres, no modulo). **Fix:**
`deadreckon.t_mod_code` reduces with the exact rational period (Fraction); `cp_predicted`'s own `% t_code` stays on
small numbers (bound 0.0015 chips at 30 days, documented). `test_dr_reload.TestT0mExact` pins exactness for all seven
periods and the defect itself (> 50 ns). 124 unit tests + selftest green; replays run clean, the e5a and the jwarm0
l5_holds digests MOVE by design (every 100-ms seed shifts; the joint clock level shifts 0.31 chip), the plain and
cprate l5_holds digests are EQUIVALENT (the clock solve absorbs a common model shift exactly). **Expected on sky at the
next broker restart:** E5a/E5b/E6/B2a standing trims +0.4 -> ~0 chip, `SEARCH-MONITOR gal_e5a` +0.47 -> ~+0.05 chip
(the station-level +1.7 m), those chains' `code_resid_m` medians down 15.4 m, every other chain unchanged, the joint
clock +0.31 chip at the bootstrap. Steps 2-4 above are unblocked with no bias state and no constant; the step-1(b)
split-chain experiment (`fixtures/jointstate/split_draft/`) is not needed.

### #171 — the ALERT files reach no one (2026-10-09)
**What [tree]:** `chain_health_cron.sh` (cf06, */15) and `elem_ref_cron.sh` (gnss, */5) write ALERT files that
nothing reads or forwards; both faults they are built for (#159's lost sky, #154's mis-pins) were found by a person
reading logs. **Fix:** tie them into CHORD's monitoring and alerting (KV: "ultimately"); until then a one-line summary
in the viewer would do.
**11-21:** stopgap built: the viewer serves /alerts and shows an Alerts panel (the ALERT files with age and first lines, plus heartbeats: the elem_ref cron, the beam-cube fold, the eop cron), kv/viewer-alerts `276be8269`, NOT deployed. From #131: the broker also fails quietly when the gather link is refused (telem.py reconnects forever, nothing shows a down gather). The real fix is #233.

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
**11-21:** the tool measured from the pre-10-24 boresight (BORE_EL 81.41) since the Tau A repoint; it now takes the boresight per epoch from config/pointings.yaml (kv/transit-bore `e38194bd5`). Open: rerun the transit census.

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
**11-21 ROOT CAUSE [live, archive 11-11..11-21; builds/fitflush_20261121/]:** in-band boresight transits. A bright 1176-MHz satellite within 5 deg of boresight leaks into the other PRNs' searches at snr >= 30 (fit-min-snr); the cp-fit history admits them with no transit gate (seeding.py), the fit lands far from the clock, three rejections flush it, and it rebuilds from more false points every 10-40 s until the transit ends. 94.5 % of gps_l5 flushes (97-98 % on 11-19/20) fall within +-2 min of a transit hold; in 518 of 566 dumps the slope equals f_chip/f_c x (Doppler of the transiting satellite - the victim's) to a median 0.08 chips/s. Per-transit counts are steady day to day (G13 ~410, G3 ~360): the 'rise' is full-sky days, not the 11-19 binary or broker. The joint clock coasts through and the code-rate clock is held, so the datum is unharmed; un-held victim PRNs take false seeds (ESCAPEs and 2-5 chip slews just after the 11-20 G13 transit). **Fix:** gate the history in the transit cone (_in_transit_cone, as the code-rate clock already is), keep un-held PRNs on their model seed there; count only out-of-cone flushes in the revert trigger.

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
**11-21:** the cube fold was dead 11-14..11-21: the corrcal reboot killed the transient loop, and after its relaunch the loop skipped the fold because the stage glob overflowed ARG_MAX at ~30k files (kv/cube-glob `ce2ce04c6`, relaunched 03:03Z 11-21, folding 34k files newest first). Open: a durable unit (KV's call).

## Open — the fix is in the NODE BINARY (queue for the next cycle)

### #234 — gw_1202's telem_pack drops PRNs beyond its 14 wire rows (2026-11-21, review)
Every node logs a WARNING that gw_1202 holds more PRNs than telem_pack's 14 wire rows; the rest are not published.
Size the rows from the chain's PRN count, or cap the chain's PRN list at the row count.

### #232 — GPS L2C is tracked on CM, the data code, only; the CL pilot has no chain (2026-11-20, KV: "can we add L2C CL to the buglist / todo? I'd like to do that, but we'll have to come back to it in a bit")
**What:** gps_l2c despreads GPS_L2C_CM, the 10230-chip, 20 ms data component on the even chips of the 1.023 Mcps
combined stream (comb_mult 2). CL, the dataless 767250-chip, 1.5 s pilot on the odd chips, has a descriptor
(`[tree]` gnssSignal.hpp GPS_L2C_CL) and a code generator (gpsL2CCode.hpp) but no chain. CM alone carries half the
L2C power, and its CNAV symbol (20 ms, one per code period) caps coherent integration at 20 ms unless the symbols
are decoded and wiped; CL is dataless, and CM+CL together carry the full L2C power.
**Why CM:** `[tree]` config/gnss_chains_chord.yaml (gps_l2c) and config/gnss_fleet_chord.yaml: CM's 20 ms period
fits the record geometry the fleet runs, CL's 1.5 s does not and needs its own bring-up. CL cannot be searched
blind; its phase follows from GPS time once CM is locked.
**History:** the airspy prototype ran a CL sibling tracker seeded from the CM solution (clsibling.py), retired in
c4c076bda (10-08) since only the airspy launcher used it; the wall-clock CL time assist went as dead code in
gnss-broker ff167396 (10-13). The CHORD fleet never tracked CL.
**Work:** a CL record path in the node (a 1.5 s code against 20 ms records); CL phase from GPS time in the broker
once CM is locked; then the CM+CL combination, with #219's open L2 terms. cnav_predictor.py's docstring ("CL
already gives L2C a dataless pilot") holds only after this.
**Status:** filed 11-20, parked by KV. Needs a node binary and sky time.

### #228 — gw_1202 lanes are held for up to elem_dir_hold_s = 1500 s, a GNSS-sized hold on LEO satellites: held GuoWang lanes fill the lane budget (2026-11-06, KV: "hold off on the lane hold fix, please file as a todo")
**What:** a slot whose row falls below elem_dir_coh_min keeps publishing its last coherent direction, steered with
the current geometry, for elem_dir_hold_s (GnssGpuRecordAssemble). The value is 1500 s on every assembler. That
suits a MEO satellite whose bright pass takes the tracker's lock; a GuoWang satellite crosses the beam in seconds
and sets minutes later.
**Evidence:** cx27/gnss0 lane dump, 10:14:13Z 11-06, 1200.000 MHz: k 39 of k_lanes_max 40, with gw_1202 lanes held
5-1058 s among them; null depth there 5.7 dB. Each lane costs 1/n of the sky. One channel only; no measurement yet
that the held lanes cost depth on a satellite in the beam.
**Proposal:** a per-chain hold (seconds for gw_1202), or drop a held lane once its satellite is far from boresight.
Buys lane budget, not depth: tracked GuoWang lanes are rebuilt every frame (11-06 dumps: coh 0.68-0.76, held 0 s).
**Status:** parked by KV 11-06; ~/gnss/builds/cube-vm-move/lanedump_read.py reads the dumps.

### #227 residual — the GuoWang code-to-satellite database is incomplete (2026-11-21)
**Open:** the code-to-satellite database is incomplete: 8 code groups unknown, 5 all-silent (GuoWang handoff 11-11).
**Closed 11-21:** `beca29fea` (GW_1202 code and nav), the gw_1202 chain live (CHORD_BUGLIST_CLOSED.md).

### #226 residual — the DTV subtraction's four open design questions (2026-11-21)
**Open:** the four questions left open before building (lag correlation of the peeled timestreams, per-arrival direction fit, the MMSE weak-source cost, the tracker time constant): answer or drop them.
**Closed 11-21:** `cudaDtvSubtract` built and live since ~11-06 (12 of 16 GPUs on 11-21: cx42/cx47 on stock) (CHORD_BUGLIST_CLOSED.md).

### #225 — GnssN2Project runs at 77-86 % of its core per GPU after the 2026-10-30 widening (43 channels, 20-30 lanes): the lane projection's n^2 k per channel-frame is the cost, and the headroom shrinks as satellites rise
**What:** the stage is single-threaded on one core per GPU (`n2-project-cpus` 15/19). With the 2026-10-30 widening to
5939-6626 (the eigen path confined to the lobes by `eigen_channels: list`) the fleet's cores read 77-86 % at 28-30
lanes per lobe channel, up from 65-73 % at 32 channels, with no backlog in host_correlation_buffer. The lane path
(gnssN2ProjectMath::project, W = VQ and the rank-k update of V, ~4 n^2 k complex MACs) dominates; the 40-lane cap is
the worst case, ~30 % above tonight's. **Levers:** project() on the upper triangle only (V is Hermitian; ~1.5-2x),
channel-parallel threads on the idle HT siblings 47/51, a lane budget ordered by strength rather than board order.
Related: the relay's copies of a GPU's own satellites on far targets (docs/CHORD_FLEET_LANES.md, Fleet).
**11-21:** now 87-95 % of a core per GPU (per thread). Threading taken by gnss-main, parked; scouting in memory chord-225-n2project-threading-parked-20261120.md: 43 live channels per GPU, the serial 129 MB frame copy caps the gain unless workers copy their own channels, OpenMP as N2Accumulate does.



### #217 — the assemblers' transit freeze is not observable (2026-10-26)
**What:** `GnssGpuRecordAssemble::shared_frozen()` and `gnss::formation_frozen()` hold the element
model through a boresight transit on the broker's `_bore` post, but log nothing and export no gauge. At G03's
transit (10-26 18:02Z, 0.72 deg) the broker posted SAT-GEOMETRY to 12/16 assemblers, and whether the freeze
engaged cannot be read. **Fix:** a `kotekan_gnss_record_assemble_shared_frozen` gauge (0/1) plus the
`_bore` separation, and one INFO line on entry and exit.

The entries below came from the 10-12 comment review and are unconfirmed or dormant.

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

### #223 — GDV map, after a few days' soak: the day/night amplitude, direct surface evaluation, gps_l5 (2026-10-30, KV: "file the GDV questions as todo for once we've soaked a few days")
Live since 10-30 17:41Z: config/gdvmap_taua_20261030.json, a per-band penalised (az, el) surface fitted on 66 h
(builds/gdv_refit_1030/out/RESULTS.md), sampled into the broker's cells and read by the 2-deg kernel lookup. Open:
- **Diurnal amplitude.** The north-low excess is x0.5-0.9 of the all-hours map from 20 to 06 h local mean solar time
  and x1.3-1.55 at 12-16 h, on every date and satellite, frequency-independent (not the ionosphere). An amplitude in
  local time buys a further 0.15-0.45 m at north-low in cross validation. Decide after a stable week: is it local
  solar or sidereal time (separable after ~2 weeks), does it follow site temperature, and is a time term in the
  range model worth it?
- **Refit on the soak.** Rerun builds/gdv_refit_1030 (leave-one-day-out) on the week from 10-31 with no change to
  the cp_predicted terms, dll spacing, chain set or pointing; rows carry gdv_m, so the pre-map residual is
  code_resid_m - gdv_m.
- **Lookup.** Sampling the surface into cells and re-smoothing with the 2-deg kernel costs up to 0.2 m at north-low
  (E5b, B2a, B2b) against evaluating the surface; a node-grid format with bilinear lookup would remove it.
- **gps_l5** is not mapped (the map file carries a gps_l5 band). Decide whether L5, the clock donor, takes it.
- Standing level differences the map does not touch: Galileo E5 vs E6 ~+1.4 m; per-satellite B2a/B2b offsets.

### #224 — Tau A beam, after a few days' soak: per-satellite levels, the east-west core, residual compression (2026-10-30, KV: "file the beam questions as todo for once we've soaked a few days")
State: builds/beam_taua_1029/out/RESULTS.md (final, unbinned, envelope, night2night). The builder now subtracts the
5-min pedestal and divides by its low envelope (5bb428fb2), keeps the closest satellite's rows in a vetoed minute
(--veto-keep-own) and places each l0 row at its own time (27a47a23e). Open:
- **Per-satellite levels draw the streaks.** Passes differ by 1.4-1.9 dB (per-satellite constants 1.5-1.9 dB rms
  on 10-28 against DRAO's 0.6 dB transmitter spread). Solve beam + per-satellite + per-pass jointly (the 09-22
  streak_joint method) on the soak's rows; compare per PRN with DRAO; Galileo IOV still reads ~2.4x DRAO in dB.
- **The east-west core.** Every boresight pass on 10-28/29 ran within 16 deg of N-S, so only the N-S width is
  measured (13-17 % wider than early October on L5/E5/B2, 29-35 % with L2C/B3I/GLONASS; pol 1 flat-topped, P 1.6-1.85).
  A 2-D Tau A core needs passes on other headings or one satellite's level tied across its passes.
- **Residual compression.** Per-input clip at the transit peak is 12.5-15 % now against 7-8 % in early October;
  L2C reaches 27-32 % one-sided in-band and B3I's quiet level is 6 LSB^2. Revisit with the F-engine gain table
  (fixtures/obs/fengine_gain_cuts_wanted_20261030.csv) once it is loaded.
- **Far sidelobes are upper bounds** until a --unbiased master (the default drops samples under the pedestal).

### #221 — the beam-cube level still varies 0.6-2 dB over a day, and a single night's skirt repeats only to ~1-1.4 dB: origin unknown (2026-10-30, KV: "figure out the origin of the residual variation in the level")
Since 5bb428fb2 a master divides each sample by the low envelope of its pedestal (10th percentile of the 5-min
pedestal over +-2 h) instead of the 5-min pedestal, which carries every satellite near the beam (0.7-1.4 dB between
quiet bins 10-20 min apart, 6-9 dB over 10-29, x3-10 in a boresight transit). What is left and not explained:
- **The envelope itself moves 0.6-2 dB over the day** (10-29 rung12, per sender/subband/element; < 0.1 dB bin to
  bin). Gain drift (analog, temperature) or sky noise (the Galaxy through the beam and sidelobes)? Test: is the drift
  common to every element and instance (gain on a shared path, or sky) or per element (front end); does it follow
  LST (sky) or local time / site temperature (gain); compare the out-of-band channels the rail recorder taps.
- **Night-to-night per-cell scatter** (envelope masters, single night, nside 128): 1.1-1.4 dB at 5-8 deg, 0.9-1.1 at
  8-15, 1.0 at 15-30, 1.2-1.3 at 30-90. Each cell is one satellite track, so per-satellite level (EIRP, attitude,
  1.5-1.9 dB rms per-PRN spread) is the leading suspect; the masters carry no per-satellite normalisation.
- **Beyond 2.5 deg the map core falls 2-5 dB less than the transit cuts**: other satellites' passes at 2.4-4.8 deg
  fill those cells at their own levels (same suspect).
Analysis and products: builds/beam_taua_1029/out/{envelope,night2night,early_oct}/ and RESULTS.md; scripts
scratch_kv/cubefix/pedestal_variability.py, pedestal_lowenv.py.

### #192 residual — refine_peak_cuda takes no nh_phase (2026-11-21)
**Open:** `refine_peak_cuda` takes no nh_phase (gnssSeedTransport.hpp; the CPU refine_peak does).
**Closed 11-21:** `fe24af48b` (43 decisions), `13036dea8` (re-bless), `0aa92696c` (fixtures retired), `273f3cb16` (gather F1/F2); item (g) gone (CHORD_BUGLIST_CLOSED.md).

### #157 — `test_buffer_peek` fails its `peeks_won > 0` check under ThreadSanitizer (flaky under slowdown; found 2026-09-30)
- **What:** under TSan's slowdown the worker thread wins every iteration, so `BOOST_CHECK(peeks_won > 0)`
  (tests/boost/test_buffer_peek.cpp:430, and the similar check near :490) fails. It failed 6/11 runs on #1714's commit and
  5/11 on develop. It passes without TSan, and TSan reports no race.
- **Why it matters:** it blocks running the buffer tests under TSan in CI. #153 showed that TSan is the tool that catches
  this class of bug.
- **Fix:** make the contention deterministic (a barrier so the peek goes first at least once), or check "not vacuous" some
  other way. Upstream test code.

### #56 residual — the 09-10/09-11 bursts were never checked against GuoWang passes (2026-11-21)
**Open:** the 09-10/09-11 burst times (18:26, 19:13Z ...) were never checked against GuoWang TLE passes; one guowang_passes run settles it.
**Closed 11-21:** superseded by #218: 37 of 43 10-26 bursts on DIGUI tracks, the N^2 stripes at 1194-1208/1250-1285 MHz on GuoWang passes (CHORD_BUGLIST_CLOSED.md).

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



### #165 — recv1's ConfigTracker exits whenever an X-engine reconnects with a different config or binary, so every node cycle restarts recv1 (2026-10-10; widened 2026-11-21)
**What [live 10-10/11]:** `ConfigInfo` carries the sender's version, commit and cmake settings; when a node
reconnects on a new binary the receiver FATALs on the mismatch, so every binary rollout costs recv1 a restart and
about a minute of N² gap — and every recv1 restart is a #164 exposure.
**Fix (chord):** compare the fields that describe the stream, not the build. Ours: none.
**10-17, twice, from recv1's journal (KV):** 12:48:24Z `ConfigTracker: conflicting upstream content present for host: 10.222.1.19, port: 12048` (cx19, the canary on the new binary) → `/buffer_recv_bad_feed_mask: bufferRecv worker thread caught FatalError … shutting down` → `Main process exited, status=1` at 12:49:23, up again 12:49:33; 12:56:17Z the same for 10.222.1.27 (cx27, with the other five) → `code=dumped, status=6/ABRT` at 12:57:23, up 12:57:33 and stable since (every later cycle was the same binary). Each costs recv1 about a minute of N² plus the restart's #164 exposure; the second death was an abort with a core, not a clean exit.

**10-17 third recurrence:** 17:35:29Z, two minutes after cx51's #189 canary cycle (NRestarts 5).

**10-17 evening:** two more, at the stage-2 canary (19:49:45Z) and the fleet cycle (20:03:06Z); NRestarts 8.
**11-21:** broader than the title said: configTracker.hpp FATALs on any change to an upstream's ConfigInfo (config or build) and on a failed fetch, so every X-engine cycle restarts recv1: 3 restarts at the 11-19 R2 cycle, 12 at the 11-20 calibration-mode revert. KV is raising it with Jim.

## Open — measured, real, and nobody's lever



### #169 residual — trim EXPIRED with no /set_trim on every node (2026-11-21)
**Open:** trim EXPIRED with no /set_trim: cx43 1587, cx27 1277 in ~25 h (gw_1202 the largest share); satellites setting, or still the #169 symptom?.
**Closed 11-21:** 0 FPGA-controller, socket, rx-ring or corrupt lines on cx43/cx27 since 11-20 01:55Z (CHORD_BUGLIST_CLOSED.md).

### #122 — the readback carries a chain-common offset on the non-primary bands
**[live]** per-PRN means: bds_b2b −0.42, bds_b3i −0.40, gal_e6 +0.29, gal_e5a +0.20, gal_e5b
+0.16, against gps_l2c −0.01 and bds_b2a +0.08. A per-chain constant the loop is not closing.
b3i and e6 are, for the first time, two fresh bands to test the standing BeiDou offset against.

### Smaller, still true
- **R09 as a noise probe** (K1B, no CDMA signals) logs ~30 `FADR BREAK glo_l2oc` lines a minute (2026-11-21 review).
- **#37** GPS block map vs sky: the registry exists; the join to the beam maps does not.
- **#40** the deep fold's rate search wrong-bins onto overlay sidebands — **demoted**: the coarse
  feed is now only the re-acquisition path. The fix is to retire the consumer (use phase
  accumulation for re-acquisition too), which needs a slip-recovery path first.
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
- **Stagger node rolls: one at a time costs nothing, all eight at once costs ~30 minutes.** The
  applied trim lives on each node, so a single roll discards a sixteenth of the fleet's trim state and
  ten combiners carry the measurement — nothing moves. Roll them together and every satellite's
  trim restarts from zero, the loop pins at the clamp for ~6 min and takes ~30 to settle.
- **A pipeline fault scales with a command; a physical cause scales with a physical variable.**
  Regress per-satellite rates against Doppler, Doppler rate and elevation before believing any
  new observable.
- **Nothing is per-node or per-instance.** A per-instance quantity is a bug (this cost the
  frame-boundary carrier phase five weeks), and a per-GPU asymmetry is a bug until proven
  otherwise (#114, measured symmetric 11-21).
- **An artifact nobody regenerates is a souvenir**, and an instrument nobody owns is dead within
  a fortnight (#56's `rail_watch`, #120's missing units).
- **A gate vouches for what its fixture runs and nothing else.** #108's fix rode the fleet for
  12 hours before anything tested it; the only prior evidence was an offline harness.
