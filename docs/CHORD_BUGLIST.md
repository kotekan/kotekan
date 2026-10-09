# CHORD GNSS — open items

What is open on the CHORD side (branch `kv/chord-gnss`). Closed items are in
[`CHORD_BUGLIST_CLOSED.md`](CHORD_BUGLIST_CLOSED.md); their narratives are in git history.

**Last reconciled: 2026-09-12 at HEAD `b45949144`** — 57 commits after the 2026-09-10 pass,
which was itself 496 commits after 2026-08-22. Closed since that pass: **#54** (the CPU reference
quantised the code phase, then retired on an anchor-scaled tolerance), **#65**, **#93 / GAP 3**
(no aid available, bounded by measurement), **#111**, **#116**, **#117**, **#118**, **#128**.
Opened: **#129** (no GLONASS positions anywhere), **#130** (cx42 heap corruption).

**Fleet state: UP since 2026-09-14** (down 09-11 21:04 → 09-14 for the handover; see
[`CHORD_STACK_SHUTDOWN.md`](CHORD_STACK_SHUTDOWN.md), which now covers both directions and the
four faults that stopped that bring-up). `[live]` claims are re-checkable again.

⚠️ **But every `[live]` mark dated 09-10 or earlier now sits across a full teardown, a develop
merge, a fleet rebuild and an F-engine re-base.** Treat those as `[carried]` until re-measured —
the instrument they were measured on is not bit-for-bit the one running now. Archive work is
unaffected: `fixtures/obs/*_20260910.jsonl` is a genuine 24 h at 6.8 rows/s, and is what closed #93.

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

## The instrument as configured, 2026-09-10

Resolved through the broker's own loader (`scripts/gnss/broker_multi.load` on
`config/gnss_chains_chord.yaml`) rather than by reading the yaml by eye — the 08-22 pass recorded
that check as not run, and doing it is what found the two errors below. **Eight chains, not five.**

| chain | search | C++ code loop | joint | deep gate | carrier command | notable |
|---|---|---|---|---|---|---|
| gps_l5 | **yes** (only chain) | yes | `rate,slew` | hand list `4 9 27` | — | `model-primacy-max: 32` |
| gal_e5a | no | yes | `clk,rate` | `33` | **on**, cap 10 Hz | feeds the joint state (shadow) |
| gal_e5b | no | yes | `clk,rate` | `33` | **on**, cap 10 Hz | `rrate-phase-feed` |
| gal_e6 | no | yes | `clk,rate` | — | — | |
| bds_b2a | no | yes | `clk,rate` | — | **on**, cap **40 Hz** (default) | see #121 |
| bds_b2b | no | yes | `clk,rate` | — | — | twin of b3i |
| bds_b3i | no | yes | `clk,rate` | — | — | twin of b2b |
| gps_l2c | no | yes | `clk,rate` | — | — | `dll-spacing 0.4` (lobe fix) |

**All eight chains now hold a closed C++ code loop and consume the joint state.** The old
three-tier split ("three chains have a loop, two have nothing") is dead. Only gps_l5 has a
search, which is still the frame: every `search admits` mechanism is gps_l5-only, and the other
seven are dead-reckon.

⚠️ **The gal_e5a/gal_e5b A/B pair is confounded** — see #125. The genuinely twinned pair is now
**bds_b2b / bds_b3i**: same constellation, same six live PRNs, `joint-consume` and nothing else.

---

## Open — the fix is in the BROKER or a script (ships today, no node cycle)


### #159 — the nightly beam-cube build rewrites the live broker's ephemeris store with yesterday's sky (2026-10-08)
**What happened [live 10-08]:** at 01:34:59Z `beamcube_daily.sh` (cf06) built the 10-07 cube. `Geometry.__init__`
(`gnss_beam_cube.py:182`) calls `fetch_brdc(<day> 12:00Z)` on the default cache, `~/.cache/kotekan_gps`, which is the
NFS home the broker reads too. The broker's copy was 42 min old (last merge 00:52Z, past `_HOURLY_TTL_S` = 1800 s), so the
cube's background refresh merged against `when` = 10-07 12:00, pruned the rolling store to that window and rewrote
`hourly_MN.rnx.gz` with records at toc 10-07 08:00–14:00. The broker's dead-reckon reload at 01:36:50Z loaded it ("BRDC
loaded (101 sats)"), logged `PREDICTION COLLAPSE (0 of peak … in the eph window)` for G, E and C, and dropped every
satellite "below BRDC horizon": no real-satellite observable on any chain from 01:36:50Z, and gps_l5 (the only search
chain) re-acquired on noise 01:46–01:54Z (lock 0.23, kcoh ~0 dB-Hz). The file's new mtime then pins it for 30 min, so the
forced reload at 01:51:51Z got the same stale sky.
**Why not every night:** the cube's merge writes only when the broker's file is over 30 min old. The build normally runs
~00:25Z, and on 10-02..10-07 the obs files show no empty 5-min bin after it; tonight it ran at 01:34.
**Fix (scripts, no node cycle):** offline callers must not write the live store. Give `gnss_beam_cube.py` its own
`cache_dir` (or pin `GNSS_BRDC_DIR`), and make the supply merge read-only for a `when` more than ~1 h from the wall clock;
a past `when` must never prune the store's newest records. Other past-`when` callers: `gnss_beam_elem2obs.py:170`,
`b1c_predict_lags.py`, `diag/navbit_brdc_test.py`.
Check: `[archive]` broker.log 01:36:38–01:36:50Z; newest toc in `zcat ~/.cache/kotekan_gps/hourly_MN.rnx.gz`;
`grep -rn "fetch_brdc(" python scripts` for the callers.

### #161 — gps_l2c's clock can latch whole milliseconds off from a cross-band bootstrap, and the mod-1-ms JOINT-CLK check then holds it there (2026-10-08)
**Measured [live 10-08], knock-on of #159:** with no sky, gps_l5 locked noise and its `clock-mod-20-ms` epoch went bad.
gps_l2c's JOINT-CLK was REFUSED from 01:44:48Z (sigma over 0.5), so its local clock went 300 s stale and the dead-reckon
re-bootstrapped from gps_l5: 02:16:30Z `clock BOOTSTRAP 2564.97 chips` (+5 ms), 02:21:50Z `5122.47 chips` (+10 ms; it was
7.47 before). The seeds now carry `off +5122.512`, half the 20-ms CM period, so gps_l2c has had fleet presence 0 since
the sky came back at 02:22Z while the other seven chains recovered. JOINT-CLK reads `legacy 5122.5 joint 7.51 ... delta
+0.001 -> ADOPTED` every 30 s: the comparison is modulo 1 ms (`diff ... mod 511.5`), so the 10-ms error is invisible,
the local clock never goes stale again, and the bootstrap that could correct it never re-arms. Only a broker restart
clears it today.
**10-08 10:09Z: a broker restart did NOT clear it.** gps_l2c came up at clk +0.23 and re-bootstrapped to 5122.45 at 10:10:07Z. The wrong epoch is gps_l5's NH alignment: during the frozen sky the nh hint offset walked 16 -> 17 -> ... -> 19 -> 0 -> ... -> 10 -> 6 (01:46-02:19Z) and NH-JOINT followed it (0.02 -> 4.8 -> 7.0 -> 10.02 ms); the fresh broker relearned offset 7 from its first 6 detections (10:09:51Z) and NH-JOINT resolved 10.020 ms again. The hint narrows the search to +-2 of the learned offset (`searchhint.py`), so a wrong offset confirms itself while detections keep coming. **Cleared 10:17Z** by stopping the broker, restarting the aggregator and starting the broker: NH-JOINT resolved 0.019 ms, the hint came back at offset 16, gps_l2c bootstrapped to 7.50 chips and was at fleet presence 1.00 from 10:20Z. So the stale phase lives in the aggregator's detection table plus the hint narrowing; the broker alone cannot escape it.
**Fix direction:** do not bootstrap a clock from a donor whose own lock is not established (gps_l5 was on noise); compare
JOINT-CLK and the cross-band donor modulo the chain's own code period (20 ms for CM), not 1 ms; and refuse a bootstrap
that moves the clock by whole milliseconds without a detection that confirms it.
Check: `[archive]` broker.log gps_l2c 02:16–02:34Z (`BOOTSTRAP`, `JOINT-CLK`, `BIRTH-STEP ... off +5122`).

### #158 — projection phase 2 leftovers: `b` reaches only the lobe fold, `k`/source are not exported, the dB judge never ran (2026-10-02)
**Phase 2 is closed on its judge** (KV, 10-02): E5a × E6 arcs through transits 39% (freeze) → 74% (09-30) → 82% (10-01)
→ 97% (28/29, night of 10-01/02, freeze 2°); lock held 149/149; closure median 0.57 TECU = quiet sky. Three items the plan
(`fixtures/projtest/PROJECTION_PLAN.md` phase 2) listed were not built; none blocks phase 3.
- **The export is `b` only.** `REC_PROJ_COST` (record slot 29, telem v7) carries b = cos² of the victim's live weights in the
  projected subspace per record row. Per-record `k` and the source kind (own row / sibling board / probe stack) exist only
  in `/get_elem_cal` (the 10-s poller). Adding them is one header slot each = a WIRE bump (v8): queue for the next flag day.
- **`b` is consumed by the fleet DLL lobe fold only** (`gnssFleetDll.hpp` fold + `combdll._lobe_fold`, weight 1−b). Not by
  JFEED's `joint_sigma` (0.3 for every satellite, `gnss_broker/cli.py:897`; the #152 item "freeze it inside the transit
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

### #144 — the transit sky (`_bore` freeze, railing veto) cannot see unhealthy or non-BRDC satellites
`nearest_boresight` runs over `brdc_predict` → `gnss_ephemeris.predict_all`, which drops G/E records with
health ≠ 0 (~l.477) and stale ephemerides, so neither the shared-model transit freeze nor the railing veto
fires for E18 (1.6 deg 09-26 09:42Z), E14 (0.66 deg 09-28 04:51:54Z), or objects in no BRDC at all:
NORAD 40748 "BEIDOU-3S M2S" (Celestrak C58; 09-27 12:40Z at 0.66 deg, +10 dB in 1176/1207/B3I for up to
24 min) and GLONASS (L3OC at 1202 MHz sits in e5b/b2b). **E14 is E202** (SINEX: GALILEO 6, GAL-2 = FOC,
launched 2014-08-22 into the eccentric orbit with E18 = E201): flagged unhealthy, and a strong E5a emitter.
On its 09-28 pass it was the N² top vector inside 2° (cos² 0.56–0.61 onto its model subspace, control
0.077), `shared_frozen` stayed 0, and the live shared model turned toward it fleet-wide: similarity to the
pre-pass model fell to 0.14–0.73 on L5 and 0.45–0.83 on the other E/B bands (fleet minima), against
0.88–0.99 in an unfrozen gap with no unhealthy satellite, while L2C, which E14 does not reach, held
0.87–0.90 [archive: viscap replay, `fixtures/projtest/viscap0928/`]. That is this item's measured cost.
Next exposures (predicted): NORAD 40748 09-30 17:13Z at 1.34°, E18 10-01 04:11Z at 4.75°. Fix: build the
transit sky with the health gate off and a long age limit, supplemented from TLEs; reuse it for the
railing veto. This is also #151's geometry-only sky: `predict_all`'s docstring allows a widened
`max_age` for geometry callers, never for seeds.
Evidence `fixtures/projtest/live/passes/`, `fixtures/projtest/unk/`.

### #150 — C11–C14 are BDS-3 MEOs now, and no chain can track them (found 2026-09-28)
**Three places still encode "PRN < 19 = BDS-2", and IGS retired that on 2026-04-19..21.** The SINEX
`SATELLITE/PRN` block (`~/.cache/kotekan_gps/igs_satellite_metadata.snx`) moved BDS-3 satellites onto
the decommissioned BDS-2 PRNs: **C11 = C234, C12 = C233, C13 = C235 (ex-C49), C14 = C232**, all BDS-3 MEO
(a = 27 906 km); C01–C04 and C06–C08 went to BDS-3 GEO/IGSO (not visible here). Each of the four MEOs is
above 10° for 6–8 h a day at DRAO, C13 peaking at 82° — as much sky as C19 or C33 [archive: BRDC 09-28].
The rule lives in `signals.py` `_CHAINS` (`min_prn 19` on bds_b2a/bds_b2b, and its "C1-C18 are BDS-2
birds" comment), in the manifest lists (`BDS_B2A_P_CS`, `BDS_B2B_I`, `BDS_B3I_NH` = 19–42,
`config/gnss_fleet_chord.yaml` l.547/603/615), and in `gen_fleet.py --check-prns`, whose `MIN_PRN` table
declares the B2a/B2b exclusion "CORRECT and must never be reported as a fault" — so the gate built to
catch exactly this is silenced by the same rule. B3I has no floor in that table, so `--check-prns`
should already be reporting C11–C14 EXCLUDED on `BDS_B3I_NH` [tree; not run — it re-fetches the BRDC
cache the live broker reads].
**They are also the brightest satellites we do not track.** A near-boresight pass raises L5, E5b, B3 and
E6 but not L2 (BeiDou has nothing at 1227.6 MHz): **C12 at 0.6° on 09-25 13:58–14:20Z** (power ~7×,
plotted rail 3.5–6.5% — first misread as gain work), **C13 at 3.0° on 09-28 12:46–13:08Z** (+4.4 / +4.8 /
+7.0 / +3.5 dB in L5 / E5b / B3 / E6; plotted rail peak 0.6% in B3, ×5.3 true; the L5 search clock broke
at 12:58 and #142 carried it into e5a/b2a) [live, `rf_chan_*.jsonl` + broker log]. The broker's transit
veto and freeze do see them — `dr_pd` pools the whole BRDC — subject to #151's expiry.
**Fix:** take BDS-3 capability from the SINEX block type (`BDS-3*`), the standing rule for capability,
in all three places; add C11–C14 to all three lists (B3I stays matched to B2a/B2b for tau_band);
regenerate the node configs; node cycle (KV). **Check:** `--check-prns` lists C11–C14 EXCLUDED before the
change and is clean after; each gets a `fleet_present` row within a pass of rising.

### #151 residual — the seeds and the sky still keep two ephemerides, and EPH-REBASE (#101) never fires
The reload cadence is fixed (`b2223cc7a`, closed). What is left is the entry's own "check #101 first":
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
**Kick — none exists today** (`publish.py` POSTs only `/set_carrier_trim` and `/set_nh_prn_offset`). In order
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

### #153 — node processes abort with `malloc(): unaligned tcache chunk detected` (heap corruption; 4 restarts in 17 h on the d53b6a254 bundle) (2026-09-29)
**⚡ ROOT-CAUSED 09-30 from the first two cores, and a fix is staged, not yet deployed.** cx51 crashed twice right
after KV's 12:45Z node_up restarts; cores are in cx51:/var/crash.

- **The abort is a double destruction of a chordMetadata.**
  - The writer: fork-only `cudaRFISKtilde.cpp:390` replaces `bf_mask_applied`'s ring slot 0 **every frame**. That breaks
    `NDArrayRingBuffer::set_metadata`'s set-once contract. It came in with bad4892eb and has been in every fleet build since
    abb8f4f20 (08-31); upstream chord has since dropped the echo (78822148a).
  - The reader: `cudaCopyFromRingbuffer.cpp:112`, on another GPU 0 main thread, copies that slot every frame through
    `GenericBuffer::get_metadata()`, which took no lock.
  - A shared_ptr copy racing an assignment of the same shared_ptr means the object dies twice, so its json nodes and key
    strings are freed twice.
  - The core shows it: the aborting thread's bin-4 link has its low 32 bits zeroed (a live rb-node's red colour store),
    and a `FREQ_UPCHAN_FACTOR` key-string chunk sits on both an arena fastbin and a tcache.
- **The segfault at capture start is a FramePrefetchService race.** The prefetcher holds an unlocked reference into the
  static stream-ID vector while another port's worker `resize()`s it (reallocates and frees it). The core has port 1's
  first `push_back` landing in the freed array.
- **cx19's "silent" restarts are not crashes.** 09-29 06:52Z and 09-30 06:45Z were unattended-upgrades plus `needrestart`
  restarting gnss-node. Fix (KV's sudo): `/etc/needrestart/conf.d/gnss.conf` with
  `$nrconf{override_rc}{qr(^gnss-node)} = 0;` on every node.
- **Fixes:**
  - Upstream #1713 (prefetcher; approved).
  - Upstream #1714: first taking the buffer mutex in `get_metadata`. jbmertens asked for something finer, so it now uses
    shared_ptr atomics on the slots, commit 1cfecda36; adversarially reviewed, TSan clean, and its new frame-cycle test
    catches a reverted store.
  - Fork: set-once on `kv/rfi-bfmask-set-once`.
- **Deploy:** `build/kotekan/kotekan.next_fix153_20260930` (md5 16959baf0a18).
  - Built from branch kv/proj-phase1-fix153 = kv/proj-phase1 + #1713 + set-once + the mutex version of #1714.
  - See `fixtures/prefetch_fix/STAGED_BINARY.txt` for the swap command.
  - When the F-engine returns: wait for the broker's re-anchor, swap, then run the node_up restart loop.
  - Close #153 after a few node-days on it with no `malloc()` line, no unexplained relaunch and no new core.
- **Follow-ups (upstream code):**
  - `#ifdef DEBUG` in FramePrefetchService is always true (DEBUG is the logging macro), so its per-frame freq-ID string
    loop runs in production.
  - The readiness wait sleeps 1 ms while holding `global_stream_id_mutex`.
  - #1714's list of direct slot readers.
  - #156 and #157 below.
  - cx51 has 4 of the 7 aborts, so rule out its RAM too.

**Since the node bundle d53b6a254 (#145 + #1699 + #146) went fleet-wide on 09-28 18:33-20:39Z, kotekan
has died with glibc's heap check on three nodes and restarted silently on two:** cx51 01:43:20Z
(`Main process exited, code=dumped, status=6/ABRT`, systemd relaunched it 20 s later; the node ran
the phase-1 projection build, a dirty d53b6a254), cx43 ~05:26Z and cx42 ~11:24Z on the STOCK bundle
binary (no projection code), all three with `malloc(): unaligned tcache chunk detected` as the last
line of `/tmp/gnss_node.log` before the new `Kotekan version ... starting`; cx42 also restarted once
earlier and cx19 at 06:52:05Z with NO message and no operator (cx19's unit shows NRestarts=0, so it
was not a systemd relaunch -- unexplained; a segfault prints nothing, so those may be the same bug
with a different symptom). cx27 and cx44 ran 16 h clean. ⚠️ **No 422ea1bf8 node log survives**
(corrected 09-29): node_up.sh keeps one rotation, so every node's oldest `/tmp/gnss_node.log.1` starts
on the bundle (`gd53b6a254b`) and holds only its first ~2 h, which are clean. The only pre-bundle
baseline is the journal: one ABRT on 422ea1bf8 (cx42 09-28 05:13Z) in ~160 node-hours, against 4-5
events in ~90 on the bundle. At equal rates, 4 or more of 5 events landing in the bundle window has
p ≈ 0.06 (5 of 6: 0.025), so the rise is suggestive, not established. The message means a freed chunk's tcache metadata was overwritten: a heap overflow/underflow
or a write through a stale pointer somewhere in the process; the abort comes at a LATER malloc, in
whichever thread happens to allocate, so the last log lines (routine n2-send refusals, valve drops)
say nothing about the writer. Nothing broker-side lines up with the times (the BIRTH-STEP lines
before each one recur every few seconds all day).
- **Which binary died (from each lifetime's `Kotekan version` line in the node log; the fleet
  file on disk has been a projection build since 09-28 21:20Z, so a relaunch after that runs
  projection code with the mode off):** cx43's aborting lifetime = the STOCK bundle (`gd53b6a254b`,
  no projection code) -> the corruption predates the projection; cx42's stock lifetime (20:39Z
  start) died silently, its next lifetime (projection build, mode off) took the malloc abort;
  cx51 = projection build in shadow. An adversarial memory-safety review of the projection diff
  (09-29 12:00Z) found no out-of-bounds write, no use-after-free and no cross-thread write into
  a reallocating container; it found only torn reads of fixed-size POD vectors by the REST thread,
  and its size guards are in the 1c build.
- **From the journals (KV, 09-29 12:15Z):** cx43 05:26:01Z `status=6/ABRT`; cx42 11:23:32Z
  `status=11/SEGV` then, 45 s into the relaunched process, 11:24:37Z `status=6/ABRT` (the malloc
  line 78 log lines after its start: the corruption can strike during start-up); cx42 ALSO aborted
  09-28 05:13:00Z on 422ea1bf8, before the bundle (its message is in a log rotation that no longer
  exists), so the bundle is not clearly the origin, and only the rate may be new; cx19's 06:52Z restart has
  no failure line at all -- a deliberate `systemctl restart`, not a crash (not KV; the other
  session is the candidate). Every 09-24/25 entry is the shutdown-hang / F-engine-outage history.
- **#1699 and #146 are cleared; #145 is the only bundle change left** (09-29). Two adversarial
  reviews of #1699 found no memory-safety defect: on the normal path it adds only a stack guard
  object and one atomic load per frame, and everything else it adds runs after a stop, an exception,
  or in the destructor. #146 is reached only from FATAL_ERROR, /kill and the teardown config report.
  So the stock bundle's only C++ change that runs mid-run is #145, and its review (09-29) cleared it
  too:
  - its changes write only locals and skip one `hold()` call; nothing is resized or shared across
    threads;
  - a sanitizer harness of the deployed ElemCal and assembler code ran clean (ASan/UBSan,
    `_GLIBCXX_ASSERTIONS`, TSan; ~426M records including starts inside a freeze, reference swaps and
    slot resets; `cf06:/var/tmp/kvand-review145/`).

  **None of the bundle's three changes can write the heap.** The writer predates the bundle, or it
  lies in code not yet reviewed: DPDK, the GPU stages, bufferSend/Recv, the N²/eigen stages, the
  combiner. The projection build's code had its own review (above). The node config is mostly not a
  confound: the 422ea1bf8 lifetimes ran the same a16ba9bdd configs from KV's 09-27 ~18:2xZ cycle
  (per the 09-28 handoff; no log survives to check).
- **Latent, pre-existing, not live (found by the #145 review; cheap to harden, none explains #153 as
  deployed).** Each needs a producer or config that breaks the contract, and none does today: every
  producer's channel count matches, `n_rows_spec` is 4, and there are 96 position values.
  - GnssGpuRecordAssemble's chan_export zero-fill spans `frame_floats(n_prn, _n_elements, hdr.n_chan)`,
    with the frame's `n_chan` unchecked against the configured channels.
  - GnssN2RecordAssemble takes offsets from the input header's `n_chan` and loops to its `n_rec`, with
    no MAX_REC check.
  - The stack arrays `g3[6]`/`e3[6]` are indexed by the frame's `n_rows_spec`, never checked against 6.
  - Two constructor FATAL_ERRORs lack a `return`, so a short `elem_positions_enu` would then be
    indexed out of range.
- **Where it is detected is not where it was written.** Each assembler thread frees and allocates
  two 0x210 chunks and one 0x110 chunk per record per PRN (rebuild_split, mag), so those threads are
  the likeliest to trip over a damaged tcache entry. A backtrace in an assembler thread does not
  implicate the assembler. (Inference: with `USE_NUMA=ON` kotekan frames come from
  `numa_alloc_onnode`, which is mmap-based, so the damaged chunk is a heap object such as a vector,
  string or json, and a frame-buffer overrun would not hit a chunk header directly.)
- **Why it hides:** the unit's `Restart=` policy relaunches within 20 s, the models re-form in
  minutes, and the node's stdout goes to `/tmp/gnss_node.log` (node_up.sh rotates it to `.1` on a
  manual restart) rather than the journal, so nobody sees the message unless they grep for it. Any
  live switch set over REST is lost on the relaunch (that is how the projection canary lost its
  shadow mode overnight; now a config key).
- **Cost:** a node's chains drop for ~2 min and its shared element model re-forms cold (a restart
  inside a transit re-forms it frozen and single-element until the freeze lifts).
- **Diagnostics, cheapest first:**
  - (1) Give the nodes a core. Two things stop one today. First, the unit's SOFT core limit is 0:
    `LimitCORE=infinity` is only the hard limit, `LimitCORESoft=0`, and apport writes nothing under
    a 0 limit. Second, even with the limit raised, apport's `consistency_checks()` drops the whole
    crash, core included, when `/proc/<pid>/exe` no longer exists or is newer than the process,
    which a binary swapped under a running node is. So use a plain file pattern. cx27 runs
    systemd-coredump instead (ProcessSizeMax=64G since 09-14; it holds a 09-24 kotekan core), and
    the same sysctl overrides either. Per node, without a restart (one `ssh -t`, one sudo prompt):
    `p=$(systemctl show gnss-node -p MainPID --value); sudo sysctl -w
    kernel.core_pattern=/var/crash/core.%e.%p.%t.%s && sudo prlimit --pid $p --core=unlimited &&
    sudo cp /proc/$p/exe /var/crash/kotekan.$p && grep "core file" /proc/$p/limits`.
    - The `cp` keeps the exact binary, because the fleet file is often replaced under a running
      process.
    - `%e` is the faulting THREAD's name (Stage names its threads, with `/` written as `!`), so
      the file name alone says which thread died.
    - A relaunch starts at soft 0 again, so each arming yields one core per node until the node's
      next node_up start, which now passes `--property=LimitCORE=infinity` (d543d3abc). All six
      nodes were armed this way on 09-29 at about 19:40Z.
    - Expect ~44 GB per core: 5 GB of heap plus 38 GB of `MAP_PRIVATE` hugepage buffers, which the
      default coredump_filter 0x33 includes. Every node has ≥ 2.5 TB free on /.
    - apport restores its own pattern if its service restarts (e.g. a package upgrade), so
      re-check `/proc/sys/kernel/core_pattern` before trusting a quiet night. To disarm:
      `sudo systemctl restart apport` (cx27: `sudo sysctl --system`).
    - Afterwards, `gdb /var/crash/kotekan.<pid> /var/crash/core.*.<pid>.* -batch -ex 'thread
      apply all bt 12'` names the aborting thread's stage. That is the victim, often near the
      culprit.
  - (2) `MALLOC_CHECK_=3` in the unit environment makes glibc check on
  every free and abort at the first corrupted chunk, closer to the writer (glibc >= 2.34 also needs
  `LD_PRELOAD=libc_malloc_debug.so.0`); the tcache is per thread, so the aborting thread's
  backtrace names the stage that FREED the damaged chunk and the bin size names the object
  (0x210 = a 32-element vector<complex<double>>); (3) an ASan build of
  the CPU stages (build_nodpdk on cf06 is USE_DPDK OFF) replaying a captured tiles+ctl stream
  through GnssN2RecordAssemble -> GnssGpuRecordAssemble -> GnssTelemPack catches anything on the
  host side deterministically; (4) A/B: one node on 422ea1bf8 against the bundle fleet -- it now
  tests only the three reviews, and at the bundle's ~1 event per 20 node-hours a clean node-day still
  has p ≈ 0.3, so it takes several; the core (1) is the better next step.
- **Check:** `for n in cx19 cx27 cx42 cx43 cx44 cx51; do ssh $n 'grep -a -c "malloc()" /tmp/gnss_node.log*; systemctl show gnss-node -p NRestarts -p ExecMainStartTimestamp'; done`
  and `sudo journalctl -u gnss-node | grep -E "Main process exited|Scheduled restart"` for the
  exit code of every relaunch (status=6/ABRT = this; status=11/SEGV = the silent kind).

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

⚠️ **Separately, found while chasing this: `--fit-gap-s 3600` exceeds the unwrap's unambiguous
range.** `fit_cp_rate` unwraps nearest-wrap, so it is only unambiguous while consecutive samples
move less than half a code period: 5115 chips at the ~3.45 chips/s receiver clock (4.05 with code
Doppler) = **21–25 minutes**. The configured tolerance accepts gaps up to 60. Any re-entry in the
25–60 minute band is unwrapped onto the wrong branch, silently. Latent today; cap `fit-gap-s` at
~1200 s, or make the unwrap gap-aware and reset instead of guessing.

### #120 — cf06 has zero systemd units; nothing survives the weekly reboot
⚠️ **The DOWN half is now solved and the UP half is not** — `scripts/gnss/stack_down.sh` stops all
eight components in dependency order and archives the logs first
([`CHORD_STACK_SHUTDOWN.md`](CHORD_STACK_SHUTDOWN.md)). What is still missing is anything that
brings them *back* unattended.
⚠️ **Do not enable a boot-time unit while the array is handed over.** cf06 reboots ~03:03 on
`unattended-upgrades`' own schedule, so an autostart would bring the GNSS stack up inside someone
else's run — contending for cf06 (GPU 0 carries the aggregator and cube) and writing cube data
through their window. Write the units by all means; arm them when the array comes back.
**[live]** `systemctl list-units --all | grep -iE 'gnss|broker|gather|agg|viewer|kotekan'` returns
nothing. cf06 reboots weekly and re-fired on 2026-09-05 with 10 headless hours. The exposure has
grown from "the gather" to six manual `nohup` processes plus a cube archiver that
`gnss_fleet_chord.yaml` says must come up *before* the nodes. One boot-time unit closes it; the
constraint that deferred this ("the node restart is scarce") is gone.

### #55 — collapse the `carrier_phase_from_ref` A/B
It is **armed in production on both node legs** and #52 (the frame-boundary jump) is now fixed
fleet-wide, so the A/B has a winner. The reference count has gone 22 → 28 → **38** code sites in
9 files plus 115 doc/yaml mentions while it waited. **[tree]** Hardcode the winning mode, delete
the other branch and the mentions, in one commit.

### #125 — the A/B pair is confounded again, and three documents claim otherwise
**[tree, resolved via `broker_multi.load`]** gal_e5a and gal_e5b differ on at least seven
non-endpoint flags (`rrate-phase-feed`, `eph-rebase`, `fleet-trim-rebase-adjust`,
`joint-feed-spec`, `joint-model-primary`, `joint-shadow`, `post-sat-geometry`). Three places
assert the opposite or are stale: this buglist's old "differ on no non-endpoint flag";
`CHORD_VECTOR_TRACKER_PLAN.md` ("`rrate-command` armed on 1 of 5 chains" — it is 3 of 8); and
`gnss_fleet_chord.yaml`'s "all five chains". Also **[tree]** the bds_b2a chain block states it
"consumes NOTHING from the joint state" eleven lines below `joint-consume: clk,rate`, and calls
itself the first non-GPS clock feeder when the flag that arms that feed is gal_e5a-only.
**Fix: re-twin the pair or re-designate bds_b2b/b3i as the control pair, and correct the three
documents.** A confounded pair silently voids every future verdict taken on it — it has already
voided one.

---

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


### #160 — `gnssN_n2dual` spends 3–4 s on its first frame building per-PRN Phi tables, so both ports' DPDK workers fall 5–39 frames behind at every node start (2026-10-08)
**Measured [live 10-08, all six nodes, timed from packet sequence numbers]:** every start, both ports, catches up 5–27
frames (0.2–1.1 s) about 1 s into capture (cx19 18:13Z: 32 and 39). A second trigger mid-run: 02:22:00–01Z on the five
nodes whose logs cover it, port 1 only, 2–3 s after the broker's BRDC reload re-seeded every chain from ~3 to ~12 sats;
the 10:09Z and 10:17Z broker restarts, which re-seeded the same set, did not trigger it.
**Measured [live 10-08, cx19, three traced starts: `/buffers` at 10 Hz, `/gpu_profile` at 1 Hz]:** every stock pipeline
finishes its first frame within 0.7 s of its port starting; `gnss0_n2dual` at +2.7, +3.6 and +4.0 s, `gnss1_n2dual` at
+6.0, +5.7 and +7.2 s (port 1 starts 1.8–2.8 s after port 0), i.e. 2.7–4.4 s after its own port. `gnssN_n2dual` reads
`host_voltage_ringbuffer`, so `run_send_voltage` blocks, `host_voltage_buffer_N` is full at +1.2 s, then
`host_pl_mask_buffer`, the transpose and the 24-frame `network_input_buffer_N`, and the workers run dry. The first
version of this entry blamed GPU 0 and cited `host_bf_mask_buffer` 48/48; that buffer sits at 48/48 in steady state.
`CUDA_MODULE_LOADING=EAGER` (cx19 18:13Z) changed nothing, so it is not lazy module loading.
**Cause [code read, not yet profiled]:** `GnssCudaDespread::build_jobs` calls `Impl::ensure_phi` for every spec. A PRN
with no table, or whose Doppler moved more than `refresh_hz` (100 Hz), gets `ChannelizedReplicaBank::hoprate_filter`
(per channel, 65536 taps × two complex-double `std::exp`), an fp16 conversion, a `cudaMalloc` and two synchronous
`cudaMemcpy`, all on the `gnssN_n2dual` thread and serially across its seven `cudaGnssInject`. The streams are blocking
(`cudaStreamCreate`) and the uploads use the legacy default stream, so each upload is also a device-wide barrier. On cf06
the `exp` loop alone is 14–29 ms per 7-channel PRN.
**Fix options:** (A) generate the taps in `hoprate_filter` with a phasor recurrence re-anchored every 1024 taps: 1.3–2.5 ms
per 7-channel PRN, within 1.2e-9 of the `exp` loop (cf06 bench); (D) upload with `cudaMemcpyAsync` on the command's
stream from pinned staging; (B) build tables off the frame thread (at seed arrival, or on a worker) and leave a PRN's lane
dark for a frame rather than block; (C) arm the shared Doppler-free Phi (`set_shared_phi`), built once per chain, which
changes the replica and needs validating. A and D should bring the first frame under the ~2 s of slack; fixing this
removes most of #1750's catch-ups.
Check: `[live]` perf or offcputime through one start, for the build-versus-barrier split; after a fix, a traced start
shows `gnssN_n2dual`'s first frame within ~0.7 s of its port and no "frame(s) past" warnings.

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

### #130 — cx42 died of heap corruption 12.5 min into the merged binary
**[live]** `malloc(): unaligned tcache chunk detected` and the process is gone. cx42 started
20:10:59 on `build/kotekan/kotekan` (the develop-merge build) and aborted **20:23:33**; the other
five were still up 20 minutes in. Glibc heap corruption, so the abort site is not the bug site —
the log's last lines are only the benign `buffer_send` retries and post-restart trim expiries.
No core (`/var/crash` empty), and **no precedent for this signature in the archived node logs**.

⚠️ **`ARCH=native` was the obvious suspect and is FALSIFIED.** `build/` sets it and the binary was
compiled on cx43, so a CPU mismatch would have been the tidy explanation — but cx42, cx43, cx19
and cx51 are all Xeon Gold 5416S with **byte-identical `/proc/cpuinfo` flag sets**. The binary is
valid on cx42.

**[live, 2026-09-11 20:34] IT DID NOT RECUR.** Restarted on the same binary, cx42 ran 24 min —
nearly twice the 12.5 it managed before — with zero aborts, and was still clean when the fleet was
stopped at 21:04. So this stands as a **single unexplained event**, not a reproducible fault, and
the discriminators below are for whoever sees the second one.

**What is and is not known.** The same merged source ran ~72 min on all six as `build-merge`
(built without `ARCH=native`, `WITH_TESTS=OFF`) with no abort, and `build/` lost one node in
12.5 min. That is one crash, not a rate: it does not distinguish "the merge carries a latent
corruption that fires occasionally" from "the build/ flags expose it" from "it was always there".
**Do not conclude from a single event.** The cheap discriminators, in order: re-run cx42 on the
same binary and see whether it recurs; if it does, run one node under the `build-merge` binary in
parallel; only then suspect the flags.

**[carried]** The `buffer_send_n2_subset`/`n2_full` connection refusals to `10.222.0.51:11025/11027`
are ⚠️ **NOT** related — those lines are byte-identical in the config before and after the
regeneration, nothing has listened on those ports for some time, and all six nodes log them.


### #112 — `set_bf_mask` free-runs, and it is what makes #108's fuse 12 h instead of years
**[live]** 170–188k frames/s on all six nodes, with no pacing anywhere in
`lib/stages/bufferBadInputs.cpp` **[tree]**. That is ~3 cores and ~2.3 GB/h of copies per node
producing nothing, and it is the reason the 2^33 `frameID` counter is reached in ~12 h rather
than in years. #108's wrap is fixed and verified, so this is now cost and fuse-length, not a
crash — but it is a large amount of heat for a mask that changes rarely.
**Fix: pace the emitter (it needs to publish on change plus a slow heartbeat, not per frame).**

### #113 — `eigencalc: Only 0 of 32 elements are unmasked`, 148× in 12.5 h
**[live]** on cx43 today, with `valve_bf_mask_*` reporting 8.16e9 dropped frames. The bf mask is
masking everything and the valve is discarding the result. Unchanged since 09-02 and previously
buried as #107 collateral. Related to #112 (same emitter) but a separate question: **is the mask
correct and the consumer wrong, or is the mask wrong?**

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

### #102 — per-element steering is disarmed behind a loader that cannot read the measured table
It shipped 2026-08-29 and was disarmed 08-31 pending a measured dish table. That table now exists
(`fixtures/chord_dish_layout_measured_20260907.json`, 32 entries with `measured_ENU_m`, from the
visibility capture). The blocker is one function: the generator derives ENU by regex on the
element *name* against a nominal grid, and the measured file carries the truth in a field.
**Fix: teach the loader to read `measured_ENU_m`, then re-arm with e5b as the sign control.**
⚠️ Re-scale the entry: the measured array is 43.6 m × ~10 m ≈ **1.5 chips** of differential
delay, not the 6 chips the 200 m projection assumed.

### #34 — the producer still publishes a forecast `ref_hop` as `pow_hop`
**[live]** 16 `fe-axis: newest pow_hop … FUTURE` + 8 `FUTURE instance(s)` per hour, the same rate
as when it was parked. The *consequence* is now bounded at both ingestion and filter
(`7a4bc0089`, `babd111c7`), so it is harmless — but the producer half is untouched. ⚠️ The old
"⚡ THE LEAD: exactly 100 records = 25 frames — find what holds 100 records" line is **falsified**
and has been deleted: the measured band was 1.01–1.13 s and `dr-forecast-lead-s: 1.0` sits inside
it. Do not go hunting a ring lap.

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


### #149 — N² to recv1: our nodes cannot satisfy upstream `chord`'s receiver (SHELVED by KV 2026-09-27)
recv1 (template + binary updated 09-19, `origin/chord` 2c588db08) requires (a) the frame-descriptor
handshake on every port — fixed on our side in a16ba9bdd (`use_frame_desc`, `reconnect_time`, full leg
only) — (b) a bad-feed-mask stream covering every frequency with data (`hdf5N2Write::_bad_feed_mask_finish`
FATALs at the first file close, ~3.5 min: "File N has data at frequency 1536, which no bad feed mask stream
covers"), and (c) a DishInputs subset layout. (b) and (c) need machinery only upstream `chord` has: 340
commits since our 09-10 merge-base, 10 conflicting files incl. N2Accumulate/bufferRecv. KV: no N² until he
talks with Jim & Andre (no tug-of-war between develop and chord). recv1's unit crash-loops every ~3.5 min
until stopped (`sudo systemctl stop kotekan` on recv1). Also: its subset writer failed on a STOCK frequency
(1539) with only 2 of 4 stock mask streams in the file — possibly recv1's own problem.

### #107 residual — the last publish-then-mutate sites
The root of the nine node deaths is fixed and running. **[tree]** Eight more `lib/cuda` stages
were fixed upstream on `kv/chord-rfimask-ds-race` (`6651a2440`, through review) and that commit is
**not an ancestor of `kv/chord-gnss`**. One of the unfixed sites, `cudaPLMaskExpander`, sets
metadata and then mutates through the returned pointer, and is instantiated ×2 in the live node
config. **Held deliberately: this is part of a set of upstream fixes under discussion — take the
conclusion when it lands rather than cherry-picking one commit.** Whether the remaining sites can
actually fire needs #107's own arithmetic (does anything observe the frame between publish and
mutate), which has not been done for these two.

---

## Open — measured, real, and nobody's lever


### #147 — 2026-09-26 22:01Z: a fleet-wide ~3 ppb reference-frequency step
All 8 chains and every instance: a carrier offset that scales with carrier frequency (E6/E5a 1.09 vs 1.087
expected), a −37 m (−127 ns) code step, carrier residual ±2.5 Hz for >14 min and beam-cube 1-s coherence
0.90 → 0.03 for ~2 h, with no lock loss and no broker event; gps_l5's code-rate clock estimate moved
−0.001 → +0.002..+0.003 ppm (also ~3 ppb). Maser/GPSDO or F-engine reference? Not ours to fix, but it
must be vetoed from any coherence statistic. Evidence `fixtures/projtest/forensics/l5/ev2201*.py`.

### #133 — the RF-band selector leaked the internal group key and mislabelled its frequency ✅ FIXED 09-17
**Corrected filing.** I first wrote this up as "the viewer collapses the broker's five `rf_band`
values into two", which was wrong and is retracted. The two-way split is **deliberate and
correct**: the viewer has exactly three multi-constellation COLUMN GROUPS — High (≥1.40 GHz,
nothing deployed), Mid (L2C 1227.60, B3I 1268.52, E6 1278.75) and Low (the E5/B2 complex,
1176.45–1207.14) — and `_band_of()` validates the broker's finer per-signal taxonomy against
them rather than trusting it, because taking `"E5b"` raw once dropped two tracking signals out
of the table entirely (2026-08-09). The Stream-health panel shows the broker's five-band view
and is also correct; they are two different taxonomies, both wanted.

The real defect was narrower. `BAND_LABEL = {L1: "High", L2: "Mid", L5: "Low"}` already existed
in `app/panels/gps_table.js` and `app/panels/decode_health.js`, so every table header already
read High/Mid/Low — but the **RF-band selector** was built server-side from the raw group key
plus *the first chain's carrier*, so it rendered `L2 · 1268.52 MHz`: the GPS-centric key, against
B3I's frequency, for a group whose namesake L2C is at 1227.60.

Fixed by labelling the selector from the shared `BAND_LABEL` and quoting the span the group
actually occupies — `Mid · 1227.60–1278.75 MHz`, `Low · 1176.45–1207.14 MHz`. High is absent
because nothing occupies it yet, which is the honest rendering.

⚠️ `BAND_LABEL`/`BAND_ORDER` now exist in three files (the server plus those two panels). Rename
in all three or not at all — noted in each.

✅ **The keys are renamed too** (09-17): `L1`/`L2`/`L5` → `high`/`mid`/`low` across
`livebeam_server.py` and four client panels. The sweep turned up one site that would have failed
**silently**: `gps_amp_history.js` mapped the group key to the C/N0 baseline key
(`{L1:"l1", L2:"l2c", L5:"l5"}`) behind a `|| "l1"` fallback, so every signal would quietly have
taken the L1 baseline rather than its own. Three other taxonomies were deliberately left alone —
the front-end names (`l1`/`l2c`/`l5`), the broker's per-signal `rf_band`, and the PVT
`CARRIER_NAME` — and each now says so where it is defined.

⚠️ `check_js.sh` could not be run: no `node` on cx43, cf06 or the VM. The change is confined to
comments and three const declarations, and the four panels were confirmed to serve with the new
keys present, but a browser reload is the only real proof the panels still render.

### #123 — cf06's single 1 GbE is the fleet's binding constraint
**[live]** 713 Mbit/s ingress, of which telemetry is 363 after the v6 shrink. It has already
produced two named faults — 22% late frames (since paced down to 0.2–0.5%) and SYN-ACK loss that
stalls every fresh connect from a node by 1/2/5 s. It has never had an item, and it outranks most
of what is above it. This is what #73's retirement left behind.

### #121 — bds_b2a commands 1.8–4.0 Hz of carrier trim on the default 40 Hz cap
**[live]** `JRR-CMD[C] 19:+1.76 20:-0.30 22:-0.27 29:+0.04 31:+4.04 35:-2.41 Hz` against
0.01–0.13 Hz on both Galileo chains in the same minute. b2a takes the **default 40 Hz** cap
where e5a/e5b set 10 — and the yaml itself calls 10 "the physics bound on a true residual" — and
it has no fine phase rate feed, so its command rides the coarse fold-fed rate that is recorded as
100–1000× above physical orbit error. Not visibly harmful (b2a PVT σ 1.7/1.5/3.1 m, standing
trims −0.08…+0.29), so: **armed on an unvetted feed.** Next: the commanded-step pair that
calibrated the sign on e5b (±2 Hz both directions, gal_e6 as the untrimmed control).

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

### #31 — the data channels are built, armed, and producing nothing
**[live]** `nav_bits by source: {'none': N}; known bits: {}`, 2505 times an hour. The decoder,
the flag and the plumbing all exist. Either wire a source or say plainly that CHORD does not
decode nav bits and delete the surface.

### #88 — finish the j2 restructure (half of it moved)
**[tree]** The anti-drift half advanced — the captured base was re-taken, is date-stamped, and the
yaml documents the re-capture procedure (though that capture is itself 10 days old with no
refresh event, which is the mechanism reproducing itself). The other half is untouched: the
generator still assembles in Python the blocks the template also renders, so there are two
definitions, and the equivalence gate already proves deleting the Python side is safe.

### Smaller, still true
- **#11** boresight seeding mask: half-built for a different consumer; nothing masks seeding.
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

**#91, #92, #90 — three trim-pathology fixes, all built and disarmed, all measuring a base rate
of ~0 on the healthy fleet.** **[live]** in one hour: 0 brownouts, 0 latches, 1 sawtooth (and
that one of the SLEW-TRANSFER class the handover cannot catch, on gps_l5). The loop errors these
were written against are now 6–60× smaller. An arm has nothing to remove, so these cannot be
judged in this regime. The flags are `--fleet-trim-brownout-hold-s`, `--code-bias-brownout-hold`,
`fleet-trim-rebase-adjust`, `reseed-spec-tau` and `LatchDetector.min_absence_s`.
**Re-judge only after a plant disturbance** — and note that #106's establishment excursion is the
one disturbance that reliably reproduces on demand.

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
