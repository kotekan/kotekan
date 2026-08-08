# Broker equivalence fixtures (task #27 M0)

Recorded transcripts for `scripts/gnss/broker_equiv.py`. A transcript is one broker run's
entire conversation with the world — clock reads, GETs, POSTs — so it can be replayed
against refactored code and the POST stream compared byte-for-byte.

**These are committed, not regenerated.** A fresh capture would carry different wall-clock
timestamps, and the cycle clock feeds freshness stamps and EMA ages, so its digest would
legitimately differ. The point of the fixture is that the *input* is frozen.

| fixture | source | covers |
|---|---|---|
| `broker_fake_l5.jsonl` | `broker_fakefleet.py` | detections → seeding → cp currency → status → DLL → seed POST, 60 cycles / 59 posts, GPS L5 geometry |
| `broker_onsky_l5.jsonl.gz.digest` | live GPS L5, 2026-08-08 18:05 UTC | the real thing: 56 cycles, 1213 gets, **934 posts across 17 endpoints** including `set_doppler_hints` / `set_nh_hint`, the almanac predictor, the dead-reckon seeder and the code-bias pool |
| `broker_onsky_e5a.jsonl.gz.digest` | live Galileo E5a, 2026-08-08 18:2x UTC | model-primary: 41 cycles, **468 posts**, no detections at all — the dead-reckon seeder and clock adoption, which nothing else reaches |

⚠️ **The on-sky transcript itself is NOT in the repo** — a minute of real fleet is ~1.3 MB
per cycle (the combiner status responses dominate): 73 MB raw, 21 MB gzipped. Only the
**digest** is versioned; the blob lives at

    /home/kvand/gnss/fixtures/broker_onsky_l5.jsonl.gz     (NFS, same path on every node)

⚠️⚠️ **THE GOLDEN DIGEST ALWAYS LIVES HERE, next to this README — never beside the blob.**
It used to be written to `<transcript>.digest`, which for the on-sky fixtures put it on NFS,
*outside git*, with a hand-made mirror in this directory. The mirror drifted silently.

⚠️⚠️⚠️ **THE E5a DIGEST MOVES ON ITS OWN, AND THAT IS NOT A BUG IN YOUR CHANGE.** A
transcript freezes the *fleet* conversation, but the broadcast ephemeris is fetched by
`gnss_ephemeris.fetch_brdc` through its own urllib — **not** through `transport.py` — into
`~/.cache/kotekan_gps`, and it refreshes several times a day. So the digest is a function of
*(code, transcript, today's ephemeris)*.

Measured 2026-08-09: golden blessed at 20:36, the daily BRDC rewritten at 20:44, and every
replay after that gave a different digest with **byte-identical code** — one seed's `cp0`
moved 0.1 chips because a satellite's freshest `toe` changed. (This also retracts the earlier
claim in this file that the `776c70ff` golden came from a dirty-tree bless. It did not.
Nothing was ever dirty.)

**A model-primary chain is maximally exposed** — every seed comes from the model. GPS L5 is
search-anchored and its digest survived the same ephemeris update untouched. If E5a moves and
L5 does not, suspect the sky before your diff.

`bless` records a BRDC fingerprint and `check` calls out an ephemeris change explicitly.
Pinning the nav files to make it genuinely hermetic is task #29.

Since 2026-08-08 `bless` resolves the golden to this directory unconditionally and stamps
the commit it ran at, shouting if the tree was dirty:

    EQUIVALENT  458afb3e…  [blessed-at 01e777480]

A digest is a claim about code. If you cannot name the commit it describes, it is decoration
— and **bless from a clean tree, then commit the `.digest` in the same commit as the change
that moved it.**

`broker_equiv.py` and the broker both read `.gz` transparently, so it is used exactly like
any other fixture:

    broker_equiv.py check /home/kvand/gnss/fixtures/broker_onsky_l5.jsonl.gz

If that blob is ever lost, re-capture (below) and re-bless — the old digest is then dead,
which is correct: it described a recording that no longer exists.

Re-capture (only when the fixture must deliberately change — say a new phase needs
coverage). Note the ports: the fake fleet is deterministic in content, so the only thing a
re-capture changes is timing.

    fuser -k 12777/tcp; sleep 1
    ./broker_fakefleet.py --port 12777 &
    ./broker_equiv.py record fixtures/broker_fake_l5.jsonl -- \
        --detectors http://localhost:12777/gps_search \
        --trackers  http://localhost:12777/track \
        --combiner  http://localhost:12777/combiner \
        --rest-url  http://localhost:12777 --time0-endpoint telescope/time0_ns \
        --carrier-hz 1176.45e6 --chip-rate-hz 10.23e6 --code-length 10230 \
        --hops-per-sec 195312.5 --nh-overlay-len 20 \
        --interval 0.2 --acquire-snr 12 --dll-gain 0.25 --publish-port 0
    # ^ let it run ~12 s, then Ctrl-C
    ./broker_equiv.py selftest fixtures/broker_fake_l5.jsonl   # must print GATE GOOD
    ./broker_equiv.py bless    fixtures/broker_fake_l5.jsonl

## Still wanted

The synthetic fixture covers the seeding spine and the DLL. It does **not** reach the
almanac/BRDC predictor, the dead-reckon seeder, the carrier loop (`--carrier-gain` is 0 on
CHORD), the nav-bit path, or the CM/CL sibling chain. Those need real captures, and all
three are cheap the moment their chain is running:

* ~~**GPS L5 on sky**~~ — **DONE 2026-08-08.** Captured with
  `broker_restart.sh --transcript-write /tmp/l5_onsky.jsonl` while PRN 10 was at deep 242
  and ten PRNs were active. It closes two of the three coverage gaps:

      sensitivity OK   moved by: carrier_hz, hops_per_sec, dll_gain, code_bias_alpha, bias_alpha
      coverage NOTE    this fixture does NOT reach: carrier_gain

  **`carrier_gain` is not a fixture gap** — `broker_up.sh` runs `--carrier-gain 0.0`, so the
  carrier loop is deliberately OFF in production and no on-sky capture can reach it without
  turning it on. Covering it needs a deliberate `--carrier-gain 0.5` run, which is a
  behaviour change to the instrument, not a recording choice.
* ~~**E5a model-primary**~~ — **DONE 2026-08-08**, and it is the one that exercises the
  dead-reckon seeder and clock adoption with no detections at all:

      sensitivity OK   moved by: carrier_hz, hops_per_sec
      coverage NOTE    this fixture does NOT reach: dll_gain, carrier_gain, code_bias_alpha, bias_alpha

  Reaching fewer knobs than the L5 capture is correct, not a defect — a chain with no
  detections runs no DLL and no code-bias pool. It is the *only* fixture that covers the
  model-primary spine, which is exactly where #28's cold-start defect lived.
* **The CM/CL sibling chain** — `--cl-tracker`, which is used by `config/run_live.sh`
  (NOT by `replay_bench_leg.sh`; that runs GPS L1 C/A + BeiDou B1C and no CL at all).

⚠️ **The airspy replay benches cannot run on CHORD hardware.** `config/replay_bench_leg.sh`,
`replay_l1gps_leg.sh` and `replay_l1bds_leg.sh` came in with the prototype merge and still
point at `/home/lwlab/airspy_gps/kotekan`, `build_cuda/kotekan/kotekan` and raw captures
under `/tmp/gpsin*` — none of which exist here. Same trap as `airspy_docs/buglist.md`:
prototype artifacts sitting in this tree that look like CHORD ones. Checked 2026-08-08.

### What the e2e harness gives, and what it does not

`scripts/gnss/e2e_broker.py` DOES run here (cx19, real GPU) and puts the real broker in the
loop against a known injected truth. It is a **correctness** harness, not an equivalence
one, and the difference is not a nicety:

    ./e2e_broker.py --prn 3 --passes 5 --settle-s 30 --port 12778 \
        --broker-arg=--transcript-write=/tmp/e2e.jsonl

Its transcript replays deterministically (33 cycles, 32 posts) and is **inert** — `selftest`
says so, and a 1% perturbation of `carrier_hz`, `chip_rate_hz`, `code_length` and
`hops_per_sec` leaves the digest bit-identical. The reason is visible in the seeds it
captures: `e2e --emit-detection` emits `code_phase_chips: 0` (the real phase rides in
`code_phase_at_ref_chips`), so the broker's POSTs are a passthrough of the served detection
and depend on nothing it computes. Excellent for measuring chips of seed error against
truth; useless as a refactor gate.

⚠️ Before blessing any capture, run `selftest`. A transcript recorded against a dark or
frozen chain replays perfectly and proves nothing — that trap cost two full scans on
2026-08-08.

## Multi-chain smoke test (`two_chain_fake.yaml`)

Two chains — GPS L5 with a search, Galileo E5a with none — in one process against the fake
fleet. This is what proves the M5 driver actually shares, and it needs no F-engine:

    fuser -k 12777/tcp; sleep 1
    ./broker_fakefleet.py --port 12777 &
    ./broker_multi.py fixtures/two_chain_fake.yaml     # ~20 s, then Ctrl-C

Look for all four, not just "it ran":

    time anchor ... = <same value>        on BOTH chains  (fetched once, shared)
    receiver: anchor=<v> (<chain>)        ONE latch, named owner
    almanac: BRDC (32 G sats) / (33 E sats) with `brdc=1 store(s)` — one parse, two
                                          constellations, which is the thing two processes
                                          throw away
    dead-reckon: clock ADOPTED <n> chips from in-process chain 'gps_l5'  (no file transport)

⚠️ The fake fleet's detections are not physically consistent, so the adopted clock wanders
between cycles. That is the fixture, not the mechanism — do not read convergence into it.
