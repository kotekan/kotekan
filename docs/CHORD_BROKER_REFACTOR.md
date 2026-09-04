# One broker, many signals — the unification plan (task #27)

Status: **M0-M5 LANDED 2026-08-08.** One process now runs every signal, sharing the time
anchor, the BRDC store and both halves of the receiver clock. Remaining: M6 (one publisher
port / one viewer) and M7 (flag audit); M8 (readability) is optional.

    M0 ee70428e1  the equivalence gate
    M1 6b3936421  1382 stateless lines -> gnss_broker/
    M2 8ceace0d2  --signal, parsed from gnssSignal.hpp
    3ccf71463     re-plan: the closure is already the Chain object
    M3 cf0d18c4b  the shared Receiver (+ a defect in M0's own clock)
    M4/M5 4534a6de0  scripts/gnss/broker_multi.py -- one process, many signals

**DEPLOYED ON SKY 2026-08-08 18:38 UTC.** One process (PID 2739598 on cf06) runs GPS L5 and
Galileo E5a together, replacing broker_up.sh + broker_up_extra.sh. Evidence of sharing, live:

    [gal_e5a] time anchor: frame0 = 1786209501.000002861   <- ONE fetch, both chains
    [gps_l5]  time anchor: frame0 = 1786209501.000002861
    [driver]  receiver: anchor=... (gal_e5a), brdc=1 store(s)
    [gal_e5a] dead-reckon: clock ADOPTED 54.61 chips from in-process chain 'gps_l5'
              (same band 1176.45MHz, NO FILE TRANSPORT)

Gated before deployment against BOTH on-sky transcripts, byte-identical to the
single-chain path: GPS L5 86e8ad5b (56 cycles, 934 posts), Gal E5a 776c70ff (41/468).

RUN IT:  scripts/gnss/broker_multi.py config/gnss_chains_chord.yaml
GATE IT: scripts/gnss/broker_equiv.py check scripts/gnss/fixtures/broker_fake_l5.jsonl
         python/scripts/gnss/gnss_broker/selftest.py
         (multi-chain smoke test: scripts/gnss/fixtures/README.md)
 Authority for the refactor; update as milestones land.
Subject: `python/scripts/gnss/gps_distributed_broker.py`, 6411 lines.

---

## 1. Why

### 1.1 What the file is today

    lines    1-1262   module level: 27 functions + FleetPublisher
    lines 1263-2463   main(): argparse, 175 `add_argument` calls, 1200 lines
    lines 2464-3158   main(): setup, ~130 mutable closure locals
    lines 3159-6410   main(): the cycle loop, 3252 lines, ~16 phases
    line       6411   `if __name__ == "__main__"`

`main()` is 5149 lines and every piece of per-chain state is a closure local. There is no
seam anywhere: a phase cannot be tested, a second signal cannot be added, and the only way
to run two signals is to run two processes.

### 1.2 One process per signal is not a packaging choice, it is a physics error

The broker is parameterised by exactly one signal — `args.carrier_hz`, `args.chip_rate_hz`,
`args.code_length`, `args.long_code_segments`, `args.constellation` are scalars. So CHORD
runs `broker_up.sh` (GPS L5) and `broker_up_extra.sh e5a` (Galileo E5a) side by side.

**Those two chains are on the same carrier.** Both launch scripts pass
`--carrier-hz 1176.45e6 --chip-rate-hz 10.23e6 --code-length 10230`. GPS L5 and Galileo E5a
are the same 24 MHz of sky, through the same feed, the same cable, the same F-engine, the
same GPS-disciplined clock. They are not two receivers. They are one receiver being
estimated twice, badly:

* **The receiver clock is hand-carried.** E5a runs `--detectors` empty by design
  (CHORD_MULTIBAND.md §5), so it can never solve its own clock — `offs` is always empty and
  the EMA never runs. Until 2026-08-08 it held whatever `--dr-clock-chips` a human pasted
  out of the GPS broker's log, wrong after every F-engine restart and silent about it.
  `--dr-clock-adopt` now reads it from a file the sibling writes at `flush_s` cadence, gated
  on a measured slew across two reads. That is a good fix for cross-process transport, and
  it is entirely unnecessary between two chains in one process.
* **The time anchor is fetched N times.** `utc0_sample0` comes from `telescope/time0_ns`
  and is latched per process (line 3722). Two processes latch it independently and can
  disagree across an F-engine restart. It is one number for the whole instrument.
* **The ephemeris is decoded N times.** Each broker runs its own decoder registry
  (`_dec_reg`, line 3060) and its own BRDC cross-check. A CNAV decode on L5 produces the
  same ephemeris E5a needs, and today it is thrown away at the process boundary.
* **The cross-band Doppler assist goes through the filesystem.** `_xb_dir` /
  `_xb_read_dir` (lines 2703-2709) exist only because the sibling is another process.
* **The viewer needs one instance per broker.** `--publish-port` is 12060 for GPS and
  12061 for E5a, and the viewer follows a single `--kotekan-rest-port`.

Every one of these is a consequence of the process split, and every one of them cost
diagnosis time in the last two days.

### 1.3 The design already contains a second chain

`--cl-tracker` (line 6094) seeds an L2C **CL** sibling chain from inside the L2C **CM**
broker: its own combiner, its own overlay length, its own segment index `k`, its own
seed POST. It is a hand-rolled, single-purpose, partially-featured second chain living
inside a loop written for one. The unified model is not speculative — it is what this code
has already been growing toward, one special case at a time.

### 1.4 The scope note in `receiver_state.py` is about the prototype, not CHORD

> `dongle` names what physically shares an LO — one airspy per band, so every chain on a
> band is measuring ONE number. It is the correct fusion scope... Do not fuse across
> dongles: the measured per-band offsets (-151 / -15 / +31 Hz) are frac-N synthesis
> constants, not a common reference error.

That is correct and it survives. It is a statement about **airspy dongles**, each with its
own frac-N synthesiser. CHORD has no dongles: one F-engine digitises the whole band from
one GPS-disciplined reference, so **the CHORD receiver is a single dongle** and every chain
on it — L5, E5a, and eventually L2C, E6, B1C — shares one clock scope. The unified broker
keeps `dongle` as the fusion key (it is the right abstraction, and the airspy benches still
use it); on CHORD that key simply has one value.

**The one thing that does not transfer is group delay across a retune.** E5b at 1207 MHz
has a different cable/PFB group delay from L5 at 1176.45, so a *code* bias measured on one
is not the other's. The unified clock therefore carries the carrier-side term at receiver
scope and the code-side term **per band**, which is exactly what `--state-dongle l5`
asserts today by hand.

---

## 2. The object model

The instrument the broker actually talks to:

    Receiver (one telescope)
      site position, F-engine time anchor, receiver clock
      |
      +-- Sky (one)                shared across every constellation
      |     BRDC/almanac store, ephemeris (decoded + broadcast), visibility,
      |     satellite position/velocity  <-- also what node-local projection/nulling wants
      |
      +-- Band (per carrier)       code-side clock scope: cable + PFB group delay
      |     |
      |     +-- Chain (per signal: constellation x band x code)
      |           SignalDef (static)   + fleet endpoints + loops + per-PRN state
      |           GPS L5-Q  Gal E5a-Q  BDS B2a-P  GPS L2C-CM  GPS L2C-CL  ...
      |             |
      |             +-- Track (per satellite x signal)
      |
      +-- Publisher (one)          one REST port, chain-keyed rows

The chains fan out over the fleet (6 nodes x 2 GPUs today) and feed **search**, **track**
and **peel**. Projection and nulling are node-local; the broker's only obligation to them
is satellite coordinates, which the Sky object already computes and should publish as a
first-class product rather than as a side effect of seeding.

### 2.1 State census — where each of the ~130 closure locals belongs

**RECEIVER (shared, one instance):**

| group | names |
|---|---|
| time anchor | `utc0_sample0`, `_anchor_chk`, `_anchor_seen` |
| sky / almanac | `almanac_sats`, `brdc_alm`, `alm_sys`, `alm_min_prn`, `_alm_clock_offset`, `_alm_file_pos`, `_alm_now`, `_capable`, `navbrdc` |
| ephemeris store | `_dec_reg`, `_decfb`, `_decfb_log_t` |
| clock, carrier side | `clock_bias_ema`, `clock_bias_cal`, `bias_available`, `_bias_meas_t`, `bias_stale`, `_fus_cache`, `_fus_seen`, `n_sib` |
| clock, code side (per band) | `code_bias_ema`, `code_bias_cal`, `dr_state["clk"]`, `dr_state["drift"]`, `dr_state["clk_t"]` |
| state transport | `state_w`, `_state_dir`, `_xb_dir`, `_xb_read_dir`, `_xb_resid` |

**CHAIN (per signal):** everything else — endpoints (`detectors`, `trackers`, `combiner`,
`dll_combiners`, `n2_combiners`, `cl_combiner`, the six `*_combiner` decoder feeds); derived
signal constants (`CODE_LEN`, `LC_SEG`, `LC_EPOCH`, `CL_SEG_S`, `Q_ALIAS_HZ`,
`MAX_GAP_HOPS`, `HIST_LEN`); per-PRN track state (`seeds`, `status`, `cp_hist`, `dop_hist`,
`ph_hist`, `nh_*`, `gating`, `low_hits`, `hold_miss`, `wd_*`, `dr_bad`, `dr_untrusted`,
`dr_state["seeded"]`, `dr_state["pin"]`, `dr_state["pd"]`, `dr_state["pd2"]`); the loops
(`dll_*`, `car_*`, `_trim_force`); the CL machinery (`cl_k`, `cl_pred0`, `cl_segsearch`,
`_clseg_spiral`, `cl_toff`, `_kscan*`); nav (`cnav`, `cnav2`, `fnav`, `inav`, `bcnav1/2/3`,
`navbits`, `navhealth`, `dhw`).

⚠️ **`dr_state` straddles the boundary** and that is precisely the seam `--dr-clock-adopt`
papers over: `clk`/`drift` are the receiver's, `seeded`/`pin`/`pd`/`pd2` are the chain's.
Splitting this dict is the single highest-value structural change in the whole refactor.

### 2.2 The cycle, as 16 phases

Read off the current loop body (line numbers are today's):

| # | phase | lines | scope |
|---|---|---|---|
| 1 | poll detections → `best` | 3160-3229 | chain |
| 2 | sibling-state fuse, bias staleness | 3230-3277 | receiver |
| 3 | almanac/BRDC predict → `pred` | 3278-3530 | receiver (per-carrier evaluation) |
| 4 | cross-band assist + narrow-search command | 3531-3700 | receiver → chain |
| 5 | per-detection seeding (cp currency, alias fold, CL assist, fits) | 3701-4346 | chain |
| 6 | poll combiner/tracker status; nav bits | 4347-4693 | chain |
| 7 | noise probes | 4694-4718 | chain |
| 8 | track watchdog | 4719-4810 | chain |
| 9 | seed maintenance / expiry | 4811-4881 | chain |
| 10 | dead-reckon seeder | 4882-5314 | chain (reads receiver clock) |
| 11 | fleet DLL | 5315-5525 | chain |
| 12 | carrier loop | 5526-5841 | chain |
| 13 | clock/bias solve + publish | 5842-5947 | chain → receiver |
| 14 | build seed payload (nav-bit signs) | 5948-6066 | chain |
| 15 | POST seeds | 6067-6093 | chain |
| 16 | CL sibling chain; logging; state write; sleep | 6094-6410 | chain (→ its own chain) |

The shape is clean: **receiver phases are 2, 3 and half of 13; everything else is
per-chain.** The driver runs the receiver phases once, snapshots them, runs every chain
against that snapshot, and folds each chain's clock observations back in a fixed order.

---

## 3. The gate: equivalence before anything moves

A 5000-line function whose every comment block is a scar from a real outage cannot be
refactored on inspection. **M0 builds the gate, and nothing else starts until it is green.**

### 3.1 Bit-exactness is a legitimate gate here

Python does no float contraction and no reassociation — a pure code move reproduces
arithmetic exactly. (Contrast `gnss-float-contraction-is-not-yours`: that trap is CUDA's,
where `fmaf` fusion changes with kernel shape. It does not apply to this file.) So the gate
is byte-identical POST payloads, not "close enough".

The known nondeterminism sources, all handled:
* `time.time()` — 49 call sites, routed through one `_now()` and pinned by the transcript.
  `_now()` is **frozen per cycle**: this is M0's one deliberate behaviour change. Measured
  on cf06's live GPS broker (600 log lines): the intra-cycle spread of clock reads was
  median 0.035 s, p90 1.71 s, **max 1.79 s against a 2 s interval** — most of a cycle, and
  entirely a function of which endpoints were slow that pass. Every gate it feeds has a
  threshold of 10 s or more, so freezing changes no decision; it removes an arbitrary
  dependence on fleet latency and makes replay possible at all.
* `random` — one use, `random.Random(seed)` in `fleet_coherent`'s null floor: already
  seeded, already deterministic.
* dict iteration — insertion-ordered since 3.7, and every payload site already sorts.
* network — replaced by the transcript.

### 3.2 What the harness is

`--transcript-write FILE` records every `_get`/`_post` (url, request, response, `_now()`)
as JSONL. `--transcript-read FILE` replays: `_get` returns the recorded response, `_now()`
returns the recorded time, `_post` is captured rather than sent.

`scripts/gnss/broker_equiv.py` runs a transcript through the broker and hashes the ordered
POST stream. **Refactor is correct iff the hash is unchanged.**

Transcripts to capture (all offline-capable, none needs the F-engine):
1. **GPS L5 on sky** — the real thing, from a live `broker_up.sh` when data returns.
2. **E5a model-primary** — `--detectors` empty, exercises the dead-reckon seeder and clock
   adoption. Capturable now against the current (dark) fleet.
3. **`e2e_broker.py`** — synthetic detections with known truth, `--passes 5` so `cp_hist`
   and `fit_cp_rate` actually run. The only transcript that exercises phase 5's rate fit.
4. **The CM/CL sibling chain** (`--cl-tracker`, `--cl-autoseg`) — used by
   `config/run_live.sh`. ⚠️ NOT by `replay_bench_leg.sh`, which runs GPS L1 C/A + BeiDou
   B1C and no CL; and the airspy replay benches cannot run on CHORD hardware at all
   (`/home/lwlab/...`, `build_cuda/`, `/tmp/gpsin*` — none present). Checked 2026-08-08.

⚠️ **A transcript captured against a dead fleet proves less than it appears to.** Before
trusting one, confirm its POST stream is non-trivial: distinct PRNs, changing
`code_phase_chips`, a non-empty `active` list. A frozen chain replays perfectly and tests
nothing (`one-observation-is-not-a-verdict`).

---

## 4. Milestones

Each is independently verifiable and independently committable. The equivalence hash must
be green at the end of every one.

### M0 — the gate
* `_now()` indirection; `_get`/`_post` behind a `Transport` object.
* `--transcript-write` / `--transcript-read`; `scripts/gnss/broker_equiv.py`.
* Capture the four transcripts; assert each is non-trivial; store golden hashes.
* **Verify:** replay reproduces its own golden hash twice, and a deliberate one-character
  perturbation (e.g. `dll_gain` scaled by 1+1e-12) *breaks* it. A gate that cannot fail is
  not a gate.

### M1 — extract the pure helpers
No state, no risk: `_get/_post/_log/_log_rl/expand_token/resolve_prefix/parse_endpoints` →
`broker/transport.py`; `fit_cp_rate/fit_dop_rate/rate_residuals/code_clock_bias_sample/
cp_rate_from_code_bias` → `broker/fits.py`; `fleet_dll/fleet_coherent/_coherent_sum` →
`broker/fleet.py`; `brdc_predict/visible_prns/_dh_dpos` + the eight `*_brdc_xcheck`
functions → `broker/sky.py`; `FleetPublisher` → `broker/publish.py`.
~1250 lines leave the file. **Verify:** hash green; `test_code_bias.py` still imports.

### M2 — `SignalDef` and the signal registry
Collapse the ~18 per-signal scalars into one frozen descriptor, plus a registry of named
signals (`gps_l5q`, `gal_e5aq`, `bds_b2ap`, `gps_l2c_cm`, `gps_l2c_cl`, `gal_e1c`,
`bds_b1c`). Every derived constant (`CODE_LEN`, `LC_SEG`, `LC_EPOCH`, `CL_SEG_S`,
`Q_ALIAS_HZ`, `MAX_GAP_HOPS`) becomes a property of the descriptor, computed in one place.

This is where `broker_up_extra.sh`'s twelve hand-typed constants disappear. They are
load-bearing and silent when wrong: the file's own comment says getting
`--long-code-segments` wrong "does not error — the seed lands in an effectively random one
of the 100 periods". A named signal cannot be typo'd into a different code length.

**Verify:** hash green with the legacy flags; and `--signal gps_l5q` alone reproduces the
same hash as `broker_up.sh`'s twelve explicit flags.

### ⚠️ M3 onward were re-planned after M2. THE CLOSURE IS ALREADY THE CHAIN OBJECT.

The original M3 was "move the ~110 per-chain closure locals into a `Chain` class", on the
assumption that unification needed an explicit state object. It does not, and the
assumption was expensive in the wrong direction.

`main()` is a closure whose locals are *exactly* one chain's state — that is what a closure
is. Running N chains in one process therefore does not require converting 110 names to
attributes; it requires calling the closure N times, in N threads, and reaching in to share
the ~20 names that are genuinely receiver-scope. **20 targeted redirections instead of 110
blind ones, for the same result.**

The 110-name conversion and the phase split are still worth doing — a 3252-line loop body
is not maintainable — but they are *readability* work that can proceed incrementally
afterwards, behind the same gate, and they are no longer on the critical path to the thing
task #27 actually asks for. Demoted to M8.

This also explains why `--cl-tracker` looks the way it does (§1.3): someone already needed
a second chain and, with no driver to add one to, hand-rolled a partial one inside the
loop. The driver is the missing piece, not the state object.

### M3 — extract the SHARED state (the ~20 names of §2.1)
`Receiver`: time anchor, Sky (BRDC/almanac/visibility), `ReceiverClock` (carrier-side at
receiver scope, code-side per band), `EphemerisStore`. The closure's reads and writes of
`utc0_sample0`, `brdc_alm`, `clock_bias_ema`, `code_bias_ema` and `dr_state["clk"]/["drift"]`
redirect to it. Includes the `dr_state` split — the highest-value single change in the
refactor, and the seam `--dr-clock-adopt` currently papers over.
**Verify:** hash green. One chain owning its own receiver state must be byte-identical to
one chain contributing to and consuming from a shared one.

### M4 — `build_args(argv)` + `run_chain(args, rx)`
Split `main()` at the seam that already exists: 1200 lines of argparse, then the chain. No
change to either half's contents.
**Verify:** hash green; `main()` is `run_chain(build_args(argv), Receiver(...))`.

### M5 — the driver
* Config: a chain list (repeatable `--chain KEY:endpoints`, or YAML), plus receiver-scope
  settings (site, `hops_per_sec`, time0 endpoint).
* One `Receiver`, N `run_chain` threads. Threads, not sequential: ~25 HTTP requests per
  chain per cycle against a 2 s interval and 5 s timeouts — sequential across four chains
  blows the budget.
* Shared state is read-through-snapshot during a cycle; contributions fold back in fixed
  chain order, so replay stays deterministic.
* Absorb `--cl-tracker` as an ordinary chain.
* Per-chain transcript streams (M0's `_Transcript` is single-owner by design).
**Verify:** two chains in one process reproduce the two single-chain digests;
`--dr-clock-adopt` becomes a no-op between co-hosted chains and E5a's seeds are unchanged
with the file transport removed.

### M6 — one publisher, one viewer  (NEXT, not yet done)
Today each chain still binds its own `--publish-port` (12060 GPS, 12061 E5a) — which works
unchanged inside one process, so this is an improvement rather than a blocker.

⚠️ **Do not simply merge the ports.** Established while scoping this: the viewer discovers
its signal table from **kotekan's `/config`** (`livebeam_server.discover_signals`, walking
`GnssChannelizedSearch` / `cudaGnssTrack` entries), and consumes the broker's publisher as
if it were *one combiner* — `FleetPublisher`'s docstring says so explicitly ("rows carry the
field names `GnssCoherentCombiner::get_status` uses, so the viewer's `signal_metrics()`
consumes them unchanged"). An unfiltered merged `/get_status` would hand it two
constellations' PRNs in one table.

So: one publisher, rows tagged `chain`/`signal`/`band`, and `/get_status?chain=<key>`
filtered — the filtered form keeps today's viewer working byte-for-byte while an updated
panel takes the lot. Then retire the second viewer instance.

### M7 — the flag audit
Only now, and separately. Census the 175 flags against every live caller
(`broker_up.sh`, `broker_up_extra.sh`, `config/replay_*.sh`, `run_live.sh`,
`run_trim_bench.sh`, `e2e_broker.py`, `peel_bench_broker.py`). Retire what nothing sets and
what the file already documents as falsified. **Keep every scar comment** — the deleted
mechanisms are documented precisely so nobody rebuilds them (`--trim-precomp`, ALIAS ESCAPE
v1/v2, the CL k-scan's falsified segment mode). Delete code, keep history.

---

### M8 — readability: the state object and the phase split
Demoted from the critical path (see the note above M3), still wanted. The ~110 per-chain
closure locals become attributes of a `Chain`; the 16 phases of 2.2 become methods, each
taking `(self, rx, t)`; the 3252-line loop body becomes a driver over named steps with no
module over ~700 lines. Behind the same gate, incrementally, one phase at a time.

## 5. Target layout

    python/scripts/gnss/broker/
        __init__.py
        transport.py     _get/_post/_log/_log_rl, endpoint expansion, Transport, transcripts
        fits.py          cp/dop rate fits, residuals, code-clock bias
        fleet.py         fleet_dll, fleet_coherent
        signals.py       SignalDef + the named-signal registry
        sky.py           Sky: almanac/BRDC, visibility, predict, capability, xchecks
        ephem.py         EphemerisStore + decoder registry
        clock.py         ReceiverClock (carrier @ receiver, code @ band), fuse/publish/adopt
        chain.py         Chain: state + the 16 phase methods
        phases/          detect.py predict.py seed.py deadreckon.py loops.py payload.py
        publish.py       FleetPublisher, multi-chain
        config.py        CLI + YAML -> receiver config + [ChainConfig]
        driver.py        the cycle
    python/scripts/gnss/gps_distributed_broker.py
        thin shim: the 175 legacy flags -> one chain. Kept through M8 as the compatibility
        surface AND as the equivalence gate. `test_code_bias.py` imports it as a module,
        so the helper names must remain importable from it.

`scripts/gnss/broker_up.sh` ends as one invocation with a chain list, and
`broker_up_extra.sh` is deleted.

---

## 6. Risks, stated

* **The refactor is the risk.** Every comment block in this file is an outage someone paid
  for. The mitigation is M0 and the discipline of not "improving" anything while moving it.
  Behaviour changes go in separate commits, after, each with its own measurement.
* **A transcript is a recording, not a proof.** It covers the code paths it happened to
  exercise. Four transcripts were chosen to span: on-sky detections, model-primary, the
  rate fit, and the CM/CL sibling. Paths none of them reach (`--cl-autoseg` engaging,
  `--noise-probes`, the watchdog's lifecycle rescue) are covered by inspection and by
  targeted synthetic transcripts where cheap. **Say which is which in the commit.**
* **Threading in M6** is the one genuine behaviour change. Chains share nothing mutable
  during the fan-out and the fold is ordered, but the publisher is touched from threads
  (it already is — it serves HTTP from a daemon thread behind a lock).
* **No burn-ins.** Every timescale here is seconds: 2 s cycles, EMAs converging in a handful
  of them. If something "needs time to settle" after a milestone, that is a bug, not
  convergence (`no-burn-in-waits`).

## 7. Not in scope

* Any change to what the broker *computes* — loops, gates, estimators, thresholds.
* The C++ send/recv migration for full-array data rates (noted 2026-08-08, future work).
* Path B migration (#26), per-element/per-freq accumulators (#24, #25).
