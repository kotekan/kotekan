# CHORD GNSS — open items

What is open on the CHORD side (branch `kv/chord-gnss`). Closed items are in
[`CHORD_BUGLIST_CLOSED.md`](CHORD_BUGLIST_CLOSED.md); their narratives are in git history.

**Last reconciled: 2026-10-11 at HEAD `0011cf3b1`** (previous pass 2026-09-12 at `b45949144`). Closed in
this pass, each verified against the tree or the fleet: **#102, #107 residual, #112, #130/#153, #133, #149, #160,
#162**; moot: **#31** (roadmap, not a bug), **#88** (folded into the upstream PR plan), **#123** (the stack left
cf06). Opened: **#164–#174**. The 10-11 memory note `chord-buglist-triage-20261011` records every check.

**Fleet state: six nodes on `0011cf3b1`** since 10-11 00:55Z (cx51 back 10-10), broker on the gnss VM, recv1 taking
all three N² legs. The 10-10 re-base changed the per-board phases and the #154 cron re-took the reference. The F-engine
runs the 10-07 gain table (#167) and does not send rows C/D (#166). `[live]` marks dated before 10-10 sit across a
re-base and a binary change; treat them as `[carried]` until re-measured.

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

---

## Open — the fix is in the BROKER or a script (ships today, no node cycle)



### #171 — the ALERT files reach no one (2026-10-09)
**What [tree]:** `chain_health_cron.sh` (cf06, */15) and `elem_ref_cron.sh` (gnss, */5) write ALERT files that
nothing reads or forwards; both faults they are built for (#159's lost sky, #154's mis-pins) were found by a person
reading logs. **Fix:** tie them into CHORD's monitoring and alerting (KV: "ultimately"); until then a one-line summary
in the viewer would do.

### #158 — projection phase 2 leftovers: `b` reaches only the lobe fold, `k`/source are not exported, the dB judge never ran (2026-10-02)
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
### #151 residual — the seeds and the sky still keep two ephemerides (EPH-REBASE is now off everywhere)
The reload cadence is fixed (`b2223cc7a`, closed). 10-11: `eph-rebase` is off on every chain (#125), so the "no step is
computed" half below is parked with #101; what remains is the two-copies design note. The entry's own "check #101 first":
- **Two copies, two phases.** The seeds' copy (`dr_state["eph"]`) reloads on the broker's start phase and
  the sky's (the shared `receiver.brdc()` store, where EPH-REBASE computes its step) on its own, so a seed
  model step can land up to 15 min from the step computed for it.
- **No step is computed at all.** Zero `eph-rebase census` lines in 16.4 h (09-28 20:18Z – 09-29 12:40Z)
  with `eph-rebase: 1` armed on gal_e5a, although the census prints at every re-parse, zero included, so
  its silence cannot mean "nothing moved". Both halves of a per-chain decision live on the SHARED store:
  `almanac.py` writes the calling chain's `eph_rebase` flag onto it just before `brdc_predict`; whichever
  chain finds the store stale does the re-parse and reads that flag (gal_e6 on all three `ephemeris
  refreshed` lines of this run, unarmed); and a stored step is `pop`ped by whichever chain's almanac pass
  comes next, armed or not. The same shape as the 08-27 prediction-collapse bookkeeping (the note in
  `sky.py`). [tree + live; the interleaving is inferred from the code, not traced]
- **Fix:** one ephemeris for the seeds and the sky, the step computed once per reload and handed to every
  armed chain (keyed per consumer, never popped from the shared dict). **Check:** a census line at every
  reload, and `EPH-REBASE:` posts on gal_e5a after the merge-adjacent ones. The geometry-only transit sky
  moved to #144.

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
**Kick (parked) — none exists today** (`publish.py` POSTs only `/set_carrier_trim` and `/set_nh_prn_offset`). In order
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

### #155 — dTEC arcs die in transits of 1176-MHz emitters: the fleet ADR holds the arc while the victim band's residual is dark (2026-09-29)
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
and the cube leg's unit. `stack_down.sh` relies on that stop — until this is understood, stop the broker and gather by name.

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



### #170 — a model formed inside a near transit (5–8°) stays mis-shaped for over an hour (2026-10-09)
**What [live 10-09 19:12–20:30Z]:** L5/E5a/E5b pol-1 models that formed on the 19:12Z restart during the C21 pass
(19:40–19:48Z at 4.9°, outside the 2° freeze) stayed at shape similarity 0.07–0.33 to the fleet for over an hour while
the pol-0 models and the other bands were fine. #154 corrects only the global phase; nothing corrects a shape, and the
adaptation EMA is the only recovery. (The cable work that evening is a confound; the pass is the candidate.)
**Fix:** freeze model FORMATION (not steering) inside a wider cone, ~8°, or re-seed a model whose similarity to the
fleet reference's shape stays below 0.5 for N minutes from that reference. Check: restart one node during the next
5–8° pass and watch `elem_shared_ref status` similarity recover within minutes.
**10-11: both built (`11ab0ee25`), awaiting the next node cycle.** `elem_sum_shared_form_deg` 8.0 (no first model inside the
cone; a warm one keeps adapting), `elem_sum_shared_ref_reseed_s` 600 (a warm half below the floor that long takes the
reference's shape, once per interval) gated on the sibling pol still matching at `elem_sum_shared_ref_reseed_sib_sim` 0.8 —
both pols off at once means the reference is stale and the cron's re-take must win. `/get_elem_cal` reports `form_frozen`,
`form_deg`, `fleet_ref.reseeds`. Boost test 12 cases. Judge after the cycle: restart one node during a 5–8° pass
(`form_frozen: true` until it clears, then sim ≥ 0.8 within minutes); after the next F-engine gain change the re-seed
counters must not move while the cron re-takes.

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

### #146 residual — upstream develop still re-parses the FatalError message
Fixed on `kv/chord-gnss` (`35d7120d0`, closed). develop carries the same code. The patch and a Boost case
that fails on the old code are in `fixtures/upstream0928/`. Open the PR with #1699 as its stated
dependency: without #1699, a controlled shutdown can hang in `~gpuProcess` where the abort used to exit.

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
  `config/elem_shared_ref_20261006.json`, mode `log`.
- Tool `python/scripts/gnss/elem_shared_ref.py` (snapshot | post | status | watch). The 10-06 snapshot is
  R 0.94–0.99 in every band, and healthy instances match its shape at a median similarity of 0.88–0.99
  (a noise model scores ~0.2).
- Safety net `scripts/gnss/elem_ref_cron.sh` (log-only unless `ELEM_REF_ACT=1`); not yet in cron.
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
it. A node restarted after a re-take and before the commit runs up to 30 min on the stale file (`--act-every`).
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

### #114 — `cudaRFISKtilde` is 6× slower on gpu0 than gpu1 on the same node
**[live]** 12.89 ms vs 2.12 ms per frame on cx43 with identical per-GPU configuration, and it is
the largest non-inject kernel on the busier GPU. Per the standing rule that **nothing is
per-node or per-instance**, an asymmetry like this is a bug until proven otherwise. Investigate
before optimising anything else on the GPU: it is a bigger lever than anything left in the GPU
queue.

### #115 — gps_l2c silently drops live PRNs past the 14th wire row
**[live]** `GnssTelemPack: more live PRNs than the 14 wire rows -- DROPPING the rest (PRN 30
onwards)`, 401+ windows on cx43 today. Live satellites are tracked and then not shipped, which is
invisible to every downstream consumer. **Fix: `telem_max_prn`** — a wire-shape decision, so it
pairs with #63.

### #62 — element cal is causal in the record path and not in the comb path
**[tree]** `GnssGpuRecordAssemble` combines then updates (causal, and the comment says so), but
the `tap()` lambda used by the comb, the spectrum ring and `_chan_export` runs *after* the
update, so each record is weighted by a cal that already absorbed it. Self-weight α ≈ 0.005 at
the configured 2 s EMA, so the honest expectation is small. **Fix: hoist the update below the
last tap use, or snapshot the weights for the tap.** Judge with a before/after on the comb DLL
discriminator rather than assuming it matters.

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

## Open — the fix changes the WIRE (needs a telemetry version bump)

### #63 — the purge: the tracker's summed prompt still rides every frame
**[tree]** The record header keeps its summed prompt so nothing downstream has to change, and its
last consumer is the deep fold's re-searching estimator — now an admission gate on 3 of 8 chains.
The slots ride every wire frame on the link that is the fleet's measured constraint (#123).
**Fix, in order: move those 3 chains' admission onto the fleet-DLL taps (5 chains already are),
then drop the slots from the record and the wire.** v6 → v7, with the same mutual-rejection rule.
Pairs naturally with #115.

---

## Open — bench or offline, no deployment at all

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

## Parked — built, disarmed, and currently unjudgeable

**#90, #91, #92 — three trim-pathology fixes, built and all disarmed again since 10-11** (#92's handover had stayed
armed on gal_e5a from 08-25 until #125's re-twin). Base rate ~0 on the healthy fleet (0 brownouts, 0 latches, 1 sawtooth
of the class the handover cannot catch, in an hour), so an arm has nothing to remove and they cannot be judged in
this regime. Flags: `--fleet-trim-brownout-hold-s`, `--code-bias-brownout-hold`, `fleet-trim-rebase-adjust`,
`reseed-spec-tau`, `LatchDetector.min_absence_s`. **Re-judge only after a plant disturbance** — #106's establishment
excursion is the one that reproduces on demand.

---

## Standing traps — the ones that have cost the most

- **Verify the tree, not the summary.** Every mark here names a grep, a commit or a live number.
  `[verified]` with no named check is indistinguishable from `[carried]` a week later.
- **Delete stale measurements; do not date them and move on.** A table with a date on it still
  gets read as the answer. This pass deleted several rather than re-dating them.
- **A fix deployed onto a restart transient will look like it worked** — and one reverted on a
  restart transient will look like it was wrong. Judge a broker restart at t+10 min, and separate
  the sky (setting satellites) first.
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
