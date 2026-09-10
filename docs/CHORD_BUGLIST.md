# CHORD GNSS — open items

What is open on the CHORD side (branch `kv/chord-gnss`). Closed items are in
[`CHORD_BUGLIST_CLOSED.md`](CHORD_BUGLIST_CLOSED.md); their narratives are in git history.

**Last reconciled: 2026-09-10 at HEAD `a5e7d686c`** — 496 commits after the 2026-08-22 pass, by
seven parallel section audits that read the tree AND the live fleet. The previous pass could not
reach the fleet and said so; this one could, which is why so much of it dissolved.

**How to read the marks.** `[tree]` = checked against the working tree, with the check named.
`[live]` = measured on the running fleet on 2026-09-10, with the number. `[carried]` = believed,
not re-checked — treat as a claim. Every entry names its check or says it has none.

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

### #119 — `--fit-flush-on-reject`'s own revert trigger is tripped, and unread
Pre-registered as "revert if flushes happen on healthy sats outside events". **[live]** 69
`cp-fit history FLUSHED` in 57 minutes on a healthy fleet, e.g. `PRN 24 … last -23.75 chips/s vs
clock -0.01`. Either the guard is correctly catching #97's continuous source defect — in which
case it is an instrument and the trigger should be rewritten — or it is churning healthy
satellites down the birth path every ~8 minutes. **Next: pair the FLUSHED PRN/timestamps against
`SOURCE PERIOD DISAGREES` and `SEEDAUDIT STEP` for the same PRN over a post-transient hour.** One
log pass; no restart.

### #118 — `dop_rate_rejected` is never cleared, and is now the loudest line in the log
`fleetdll.py` logs it and clears only `cp_rate_rejected`; the comment beside it admits the bug.
**[live]** 1552 `dop-rate: N fit(s) REJECTED` lines in 57 minutes — ~91% of gps_l5's 2 s cycles —
replaying every PRN ever rejected. This is the exact misreading trap #100 records being burned
by (identical stale values repeating for minutes, misread as a live clock fault).
**Fix: clear the dict after logging, exactly as `cp_rate_rejected` already is.** One line.

### #65 — three stack scripts truncate their log on start
`> "$LOG"` in `agg_up.sh`, `gather_up.sh` and `broker_restart.sh` **[tree]** — it was two scripts
when this was filed, it is three now. It has destroyed burst evidence twice. `>` → `>>` or
rotate. One line each.

### #116 — the observables writers never roll the UTC day
`gnss_observables.py` computes the output path once at start-up and opens it once; there is no
reopen. **[tree]** The `--out` help promises "the file ROLLS at UTC midnight" and then describes
exactly the failure it has: "a date baked in at launch keeps one day's name for as long as the
process lives, and consumers that open today's file then find nothing." **[live]** every
2026-09-10 row is in `*_20260909.jsonl`; no `_20260910` file exists.
⚠️ **Do not fix under a running soak** — the frozen name gives one seam-free file per chain, and
the analysis has to know which file the data is in. Fix at the next writer restart. Three lines.

### #117 — cross-chain grid pairing yields 40%, and the loss is sampling, not physics
TEC needs two bands at the *same* grid hop. **[live, measured over 3392 s]** each chain publishes
1686 of 3370 grid hops = **50%** — the obs writer's 2 s poll against the 1.0066 s grid is exactly
Nyquist — and because each chain lands on its own alternate, a band **pair** shares 40% and a
**triple** 35%. **Fix: publish the last ~4 grid snapshots per row instead of only the newest, so
both chains' rows overlap regardless of poll phase.** Ceiling ~100%.

### #120 — cf06 has zero systemd units; nothing survives the weekly reboot
**[live]** `systemctl list-units --all | grep -iE 'gnss|broker|gather|agg|viewer|kotekan'` returns
nothing. cf06 reboots weekly and re-fired on 2026-09-05 with 10 headless hours. The exposure has
grown from "the gather" to six manual `nohup` processes plus a cube archiver that
`gnss_fleet_chord.yaml` says must come up *before* the nodes. One boot-time unit closes it; the
constraint that deferred this ("the node restart is scarce") is gone.

### #111 — the Earthdata token fails silently and DCB has no fallback
`gnss_dcb.py` returns `None` when there is no token, with no WARN and no metric. **[live]** the
token is healthy — issued 2026-09-09 19:05, **expires 2026-11-08, 59 days out** — and DCB is
flowing daily since 08-23, so this is latent, not active. (An earlier estimate of a 09-19 expiry
was wrong.) **Fix: WARN-once distinguishing "no token configured" from "401 rejected", plus a DCB
product-age metric so staleness is visible before it is an outage.**

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

## Open — the fix is in the NODE BINARY (queue for the next cycle)

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

### #54 — the GPU and CPU replicas differ per sample, and one suspect just got worse
**[tree]** Chased 2026-08-13: GPU-vs-CPU prompt error grows monotonically with the absolute
anchor (7.92e-8 → 8.16e-7 → 9.55e-4 → 3.58e-2 over three decades), and the per-sample peel
residual was 9.55e-2 at 6.8 days of uptime. The `cp_ref` fix was explicitly recorded as *not* the
answer. Three suspects remain: the long-double CPU reference is assumed correct and never
independently verified; `m_head_for` computes its running offset in double at n₀ ≈ 1.9e15; and
the Φ tables — **which are now fp16 in production**, so that suspect got worse, not better.
**[live]** The on-sky per-record carrier floor of 0.045 cycles — common across instances,
independent of C/N0, white in time — is the same magnitude and is the first live cross-check this
has ever had. **Next: re-run the despread test with the anchor swept, once with fp16 and once
without, and promote `m_head_for` to long double.** A bench run.

### #56 — the ~5× hourly signal swings, and an instrument nobody owns
**[tree/live]** `rail_watch` has been dead since 2026-08-24, has no systemd unit and no cron
entry, and nobody owns it — while this entry says "nothing blocks this but the reading". Two
complete days are on disk, which is enough to apply the discriminator offline (clip rises with
total power ⇒ source or interferer; power flat while clip rises ⇒ gain or quantiser scaling).
**Next: run it on the archived days, then either give `rail_watch` an owner or delete it.**

### #94 — the shared-parameter estimators: 1 of 6 sites done, and the armed one was never judged
S2 (the prior gauge) is built and **armed receiver-wide for 14 days with its pre-registered
falsifier never run** — while its own in-poll control reports **39% refusals** (936 of 2371).
The falsifier is two minutes: drop one satellite and read `clk` either side. S1 (the
circular-median clock) was supposed to be demoted to a cross-check by S2 and was not; S3–S6 (the
membership-invariance property and its tests) are not started.

### #106 — the establishment excursion, and the one instrument that was never built
The withdrawn freeze is correctly fenced (default 0.0, absent from the yaml). The cause is still
unexplained and **[live]** still reproduces: after today's 13:36 broker restart the gps_l5
readback walked 0.470 → 1.780 → 2.272 → 2.502 → 2.789 → **3.000 (the clamp) at t+5.5 min**, and
recovered by t+7 min. What is missing is a number: the **pre-clamp** common-mode trim demand.
Nothing exposes it — the clamp is applied inside the loop and only a rail counter escapes.
**Fix: expose the pre-clamp demand per PRN in the FleetDll stat line, then read one restart.**
Minutes, not a soak. ⚠️ Two staleness notes: the "~8 min" figure holds for a *broker* restart, but
a full *node* cycle now takes 25–30 min to settle; and the lobe-coherent combine made the
discriminator ~1.5× steeper without the loop constants being retuned, which is the most likely
reason the excursion still reaches the clamp.

### #97 residual — the source defect fires at its original rate
**[live]** 44 `SOURCE PERIOD DISAGREES` in 57 minutes (~46/h, against the pre-fix 37–49/h), at
snr 531 with a 0.0-chip within-period residual — so the search's period label is wrong on strong,
clean detections. The command side is fixed (the label is now a fleet consensus, so no single
satellite moves its own seed) and the node-side consensus cut the ±1 class, but not ±3/±6.
**Next: the per-detection `(nh, cp_long, snr)` stream against injection.** A node-side dig.

### #93 / GAP 3 — the aiding target has largely evaporated
**[live, computed from 475 shadow lines this run]** de-meaned per poll per chain, the correlation
between the carrier rate and the trim slope is ~0 on seven of eight chains (r −0.05…+0.06), and
the trim ramps themselves are now **0.001–0.01 chips/min against the 0.06 chips/min that
motivated GAP 3** — 6–60× smaller. The one exception, gps_l5 at r = −0.417, is manufactured by
slew transfer (its seed absorbs trim content every ~600 s) and must not be read as divergence.
The carrier observable GAP 3 was blocked on now exists and is clean (single-difference Allan
deviation ~1 mHz at 256 s). **So the honest test is the first DISTURBED window, not calm data:
re-run the same statistic there, and if it holds, close GAP 3 as "no aid available" rather than
"not yet tried".** ⚠️ 598 s windows sampled at 60 s give ~6 independent samples per satellite per
hour, not the raw pair count.

---

## Open — blocked upstream

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
  restart transient will look like it was wrong. Judge a broker restart at t+10 min and a node
  cycle at t+30, and separate the sky (setting satellites) first.
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
