# Broker equivalence fixtures

Recorded transcripts for `scripts/gnss/broker_equiv.py`. A transcript is one broker run's
entire conversation with the world — clock reads, GETs, POSTs — so it can be replayed
against refactored code and the POST stream compared byte-for-byte.

**These are committed, not regenerated.** A fresh capture would carry different wall-clock
timestamps, and the cycle clock feeds freshness stamps and EMA ages, so its digest would
legitimately differ. The point of the fixture is that the *input* is frozen.

| fixture | source | covers |
|---|---|---|
| `broker_fake_l5.jsonl` | `broker_fakefleet.py` | detections → seeding → cp currency → status → DLL → seed POST, 60 cycles / 59 posts, GPS L5 geometry. The seeding spine and the DLL only: no almanac/BRDC predictor, dead-reckon seeder or nav-bit path. Moved by `carrier_hz`, `hops_per_sec`, `dll_gain` |
| `broker_onsky_l5.jsonl.gz.digest` | live GPS L5, 2026-08-08 18:05 UTC | the real thing: 56 cycles, 1213 gets, **934 posts across 17 endpoints** including `set_doppler_hints` / `set_nh_hint`, the almanac predictor, the dead-reckon seeder and the code-bias pool. Moved by `carrier_hz`, `hops_per_sec`, `code_bias_alpha`, `bias_alpha` |
| `broker_onsky_e5a.jsonl.gz.digest` | live Galileo E5a, 2026-08-08 18:2x UTC | model-primary: 41 cycles, **468 posts**, no detections at all — the dead-reckon seeder. Moved by `carrier_hz`, `hops_per_sec` (no detections, so no DLL and no code-bias pool) |
| `broker_onsky_l5_holds.jsonl.gz.digest` | live GPS L5, 2026-08-16 23:15 UTC | 30 cycles, 434 posts, with `--telem-dll`. Moved by `carrier_hz`, `hops_per_sec`, `dll_gain`, `code_bias_alpha`, `bias_alpha`. Also has the `cprate` variant golden (below) |
| `broker_onsky_e5a_20260826.jsonl.gz.digest` | live Galileo E5a, 2026-08-26 00:33 UTC | model-primary, 113 cycles, 1474 posts (below) |
| `broker_onsky_l5_20260826.jsonl.gz.digest` | live GPS L5, 2026-08-26 01:1x UTC | search-fed, 164 cycles, 2444 posts (below) |
| `broker_onsky_b2a_20260826.jsonl.gz.digest` | live BeiDou B2a, 2026-08-26 01:2x UTC | model-primary, 172 cycles, 2210 posts (below) |

**The on-sky transcripts themselves are not in the repo**: a minute of real fleet is
~1.3 MB per cycle (the combiner status responses dominate). Only the **digests** are
versioned; the blobs live on NFS, same path on every node, e.g.

    /home/kvand/gnss/fixtures/broker_onsky_l5.jsonl.gz

**The golden digest always lives here, next to this README, never beside the blob**, so a
stale golden shows up in `git status`. A variant golden, `<transcript>.<name>.digest`, pins
the same recording replayed with flags appended to its own and records those flags beside
the digest; `check` verifies every golden of a transcript, each under its own flags:

    broker_equiv.py bless --variant cprate <transcript> -- --cp-rate-model-tol 0.5 ...

`broker_onsky_l5_holds.jsonl.gz.cprate.digest` is one: the holds capture under
`--fit-hist-len 256 --fit-min-snr 30 --cp-rate-model-tol 0.5 --fit-flush-on-reject 3`, which
runs the cp-rate guard that no recording runs on its own.

**Inputs outside the transcript.** A transcript freezes the *fleet* conversation, and
`broker_equiv.py` pins or cuts what the broker reads by other routes:

* The broadcast ephemeris is fetched by `gnss_brdc_supply` through its own urllib, not
  through `transport.py`. Replays read a pinned snapshot: `GNSS_BRDC_DIR` points at the
  transcript's `<transcript>.brdc/` sidecar when it has one, else at the shared
  `/home/kvand/gnss/fixtures/brdc_pin`. `bless` records the pin's fingerprint and `check`
  calls out a fingerprint change explicitly. A model-primary chain is the most exposed,
  since every seed comes from the model.
* The telemetry gather stream (`--telem-gather`) is not transcribed. A recording that named
  a gather replays against `DARK_GATHER` (127.0.0.1:1, a closed port): the client exists and
  never connects, so a replay on a host with a live gather cannot fold its frames in.
* The IGS satellite registry decides which satellites a chain with a capability gate seeds
  (`--signal-capability`, which a BDS-3-only signal such as `bds_b2a` implies). A
  `<transcript>.igs.snx` sidecar pins it (`GNSS_IGS_SNX`, never re-fetched); without one a
  replay reads `~/.cache/kotekan_gps/igs_satellite_metadata.snx`, which the live broker
  re-fetches weekly. `broker_onsky_b2a_20260826` depends on it.
* The CAS DCB product (`--dcb-bias`) is read from `~/.cache/kotekan_gps` and is not pinned.
  The three 2026-08-26 fixtures carry the flag, but their replays never reach the fetch.

`bless` stamps the commit it ran at, says when the tree was dirty and records the BRDC pin's
fingerprint, and `check` prints them back:

    EQUIVALENT  broker_onsky_l5.jsonl.gz.digest  026dd887…  [blessed-at 6e9341aa8 brdc e0484c330b03]

A digest is a claim about code. If you cannot name the commit it describes, it is decoration,
so **bless from a clean tree, then commit the `.digest` in the same commit as the change
that moved it.**

`broker_equiv.py` and the broker both read `.gz` transparently, so it is used exactly like
any other fixture:

    broker_equiv.py check /home/kvand/gnss/fixtures/broker_onsky_l5.jsonl.gz

If that blob is ever lost, re-capture (below) and re-bless; the old digest is then dead,
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

Before blessing any capture, run `selftest`. A transcript recorded against a dark or
frozen chain replays perfectly and proves nothing.

## Multi-chain smoke test (`two_chain_fake.yaml`)

Three chains — GPS L5 with a search, Galileo E5a and BeiDou B2a with none — in one process
against the fake fleet. This is what shows that `broker_multi`'s chains share what they
should, and it needs no F-engine:

    fuser -k 12777/tcp; sleep 1
    ./broker_fakefleet.py --port 12777 &
    ./broker_multi.py fixtures/two_chain_fake.yaml     # ~20 s, then Ctrl-C

Look for all four, not just "it ran":

    time anchor ... = <same value>        on all three chains  (fetched once, shared)
    receiver: anchor=<v> (<chain>)        one latch, named owner
    almanac: BRDC (32 G sats) / (33 E sats) / (<n> C sats) with `brdc=1 store(s)` — one
                                          parse, three constellations, which is the thing
                                          separate processes throw away
    dead-reckon: clock ADOPTED <n> chips from in-process chain '<chain>' (same band 1176.45MHz)
                                          on gal_e5a and bds_b2a

The fake fleet's detections are not physically consistent, so the adopted clock wanders
between cycles. That is the fixture, not the mechanism — do not read convergence into it.

## broker_onsky_e5a_20260826 (2026-08-26 00:33 UTC)

* chain `gal_e5a`, model-primary, the most exposed case (every seed comes from the model),
  with `rrate-phase-span-s 16`, `rrate-command true` and `joint-consume clk,rate`. The
  recording also carries `fleet-trim-rebase-adjust 1`, which replay drops
  (`broker_equiv.py` RETIRED_FLAGS).
* 113 cycles, 1474 posts, 4628 gets. Moved by `carrier_hz`, `dll_gain`; it does not reach
  `code_bias_alpha` or `bias_alpha`.
* Carries its own ephemeris in `broker_onsky_e5a_20260826.jsonl.gz.brdc/` (DOY 237+238
  + hourly). `broker_equiv._pin_for` prefers a `<transcript>.brdc/` sidecar over the shared
  pin, because the shared pin covers only the days of the fixtures without one (DOY 219-228),
  and adding a day to it would move the fingerprint of every one of them at once.

Capture recipe, one chain per run (`_TR` is a module singleton, so a second `open_write`
clobbers the first): archive the broker log, add `transcript-write: <path>` to one chain in
the yaml, restart, wait ~3 min, remove the flag, restart, gzip, snapshot the nav files into a
`.brdc/` sidecar (and, for a chain with a capability gate, the IGS registry into a
`.igs.snx` one), then `bless` from a clean tree and `selftest` before trusting it.

## broker_onsky_l5_20260826 + broker_onsky_b2a_20260826 (2026-08-26 01:1x-01:2x UTC)

Both carry their own `.brdc/` sidecar (DOY 237+238 daily + hourly, snapshotted at capture
time).

* **`broker_onsky_l5_20260826`**: the search-fed chain, 164 cycles, 2444 posts. Moved by
  `carrier_hz`, `hops_per_sec`, `dll_gain`, `bias_alpha`.
* **`broker_onsky_b2a_20260826`**: the BeiDou chain, 172 cycles, 2210 posts. Moved by
  `carrier_hz`, `dll_gain`. `--signal bds_b2a` arms the capability gate, so its digest also
  depends on the IGS registry, which a `.igs.snx` sidecar pins.

Coverage across the fixtures: `code_bias_alpha` is reached only by `broker_onsky_l5` and
`broker_onsky_l5_holds`, and `dll_gain` by every fixture except `broker_onsky_l5` and
`broker_onsky_e5a`, so keep every fixture in a broker_equiv check.
