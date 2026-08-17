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

Last reconciled: **2026-08-17 ~00:45 UTC**, against session tasks #6–#83.
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

## 🔴 Active — the next levers, ranked

### A1. Widen the #79 search bar, one notch at a time — gps_l5 [ready now]
Task #79 shipped `--dll-deep-gate-from-search` and armed it at snr **100**, admitting 2 new
satellites (armed 3 → 8). The search sees PRNs down to snr 30. Widen 100 → ~50 → 30, judging
each step on disc/q duty with the still-unadmitted satellites as in-poll controls. The flag's
own history warns that admitting everything at once trades the latch for an oscillation and
makes the A/B unreadable — hence notches. [verified 08-17: armed and stable, 8-min watch]

### A2. The dead-reckon chains: rate-limited pull-in, and one dead instrument
⚠️ **Rewritten 00:55 after investigating C20 — my first version of this entry was wrong.**

Two separate faults, both measured on bds_b2a:

**(i) #85 — the corrector cannot outrun the error.** C20 is strong (34.2 dB-Hz, KCOH sig 3270
when the tap lands) but its model error GROWS ~1.4 chips/min while every logged slew is railed
at the 0.05-chip cap (~1.5 chips/min of authority). So the commanded phase sweeps *through* the
peak — a transient 30× spike, then away. This is per-satellite, not the clock: every other PRN
on that chain sits at model-held 0.02–1.28 chips; C20 alone runs 1.3 → 4.7 → 7.0 (16.8 earlier).
**And the instrument already measures the cause**: `deep_rate_full_hz` reads −8 to −10 Hz for
C20 at q 10–17, mixed-sign across the chain (so per-sat, not a chain offset), and nothing
consumes it here. That points at #33 P3 / #71, re-judged against far better evidence than the
first attempt had.

**(ii) #84 — `spec_tau` is FROZEN on all five chains.** 42–43 identical samples per PRN, to
three decimals, over 25 minutes, every line reading "7 ch/1 inst" with peak/floor 0.8–1.3
(sometimes *below* 1). It is served as `spec_tau_chips` too. So the admission statistic I
proposed in the first draft of this entry **does not currently measure anything**, and #50's
premise ("the signal already exists, wire it in") is false until that is fixed.

Remaining options for admission, then: (a) fix #84 and reconsider; (b) let the gps_l5 search
admit its *band siblings* where the PRN maps across; (c) extend #49's hand list, which does not
scale. But note (i): on these chains admission may not even be the binding constraint — pull-in
rate is. [verified 08-17]

### A5. Re-arm the fast loop on the dead-reckon chains — now unblocked
The four gal/bds chains have **no fast loop of any kind**: no C++ fleet DLL (23.84 Hz), no
5 Hz Python fast thread. They get the ~12 s broker cycle only, and only for PRNs that pass
presence. `eec1d2f12` disarmed the C++ loop there last night because arming it *removed* their
pull-in path, and set one condition for re-arming: a per-PRN handover, so the Python arm keeps
acquisition authority for not-yet-armed PRNs and stands down as the fast loop takes each.
**That handover is now implemented** (authority follows the LAST POSTED armed set, so never
both arms and never neither). Arm ONE chain, keep the others as controls, and judge on
duty — not on a single satellite. ⚠️ Note #85 first: on these chains the binding constraint may
be that the seed is being swept by model error, in which case a faster loop chasing a moving
seed is not obviously the win. Arm it as an experiment with a control, not as a fix.

### A3. #83 Phase 2(b) — enable seed phase transport, kill the clock lever
`--seed-phase-transport` is committed but OFF. Enabling it removes the sample-0 back-reference
whose lever (~5300 chips/Hz at a week of run age) converts a ~3.5 Hz fleet clock wobble into
20k-chip seed steps — the residual churn class after #41. ⚠️ **Blocked on the audit's §4.6
fix first**: both broker trim-application sites write only `code_phase_chips`, never
`code_phase_at_ref_chips`, so enabling transport on the DR chains would silently disable their
only code loop. Fix the write, then enable. [carried, audit 08-16]

### A4. #76 — ✅ FIXED (eb30892b3, deployed 08-17 02:06): the trim readback
`/fleet_trim/get_dll` now serves the integrator per PRN (`trim_chips`/steps/railed/skipped/win
+ `armed`; a MISSING key means "no standing trim", distinct from a trim passing through 0.0),
and the broker GETs it back each cycle right after `/set_policy` into `_ft_readback` + a
rate-limited `FLEET-TRIM READBACK` log line. Flag-gated (`--fleet-trim-readback`, yaml-armed
on gps_l5 + gal_e5a) because pre-#76 fixtures have no get_dll entries in their get streams —
all four fixtures EQUIVALENT with the flag off. Read-only: consumption is #83 2(d).
**First minutes of data**: gal_e5a's armed PRNs carry LARGE fast-growing trims
(21: −2.28 chips within ~90 s of integrator birth) — the re-armed loop is absorbing real
standing code error on Galileo; watch for #78's clamp at 3.0. [fixed 08-17]

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
