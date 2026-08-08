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

* **GPS L5 on sky** — add `--transcript-write /tmp/l5.jsonl` to `broker_up.sh`, let it run
  a minute, kill it. Needs the F-engine.
* **E5a model-primary** — same, on `broker_up_extra.sh e5a`. This is the one that exercises
  the dead-reckon seeder and clock adoption, and it needs no detections.
* **L2C replay bench** — `config/replay_bench_leg.sh`, the only chain that runs
  `--cl-tracker` and the CM/CL segment machinery. Offline, available now.

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
