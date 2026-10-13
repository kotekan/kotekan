# Broker equivalence fixtures (task #27 M0)

Recorded transcripts for `scripts/gnss/broker_equiv.py`. A transcript is one broker run's
entire conversation with the world — clock reads, GETs, POSTs — so it can be replayed
against refactored code and the POST stream compared byte-for-byte.

**These are committed, not regenerated.** A fresh capture would carry different wall-clock
timestamps, and the cycle clock feeds freshness stamps and EMA ages, so its digest would
legitimately differ. The point of the fixture is that the *input* is frozen.

| fixture | source | covers |
|---|---|---|
| `broker_fake_l5.jsonl` | `broker_fakefleet.py` | detections → seeding → cp currency → status → DLL → seed POST, 60 cycles / 59 posts, GPS L5 geometry. The seeding spine and the DLL only: no almanac/BRDC predictor, dead-reckon seeder or nav-bit path |
| `broker_onsky_l5.jsonl.gz.digest` | live GPS L5, 2026-08-08 18:05 UTC | the real thing: 56 cycles, 1213 gets, **934 posts across 17 endpoints** including `set_doppler_hints` / `set_nh_hint`, the almanac predictor, the dead-reckon seeder and the code-bias pool. Moved by `carrier_hz`, `hops_per_sec`, `dll_gain`, `code_bias_alpha`, `bias_alpha` |
| `broker_onsky_e5a.jsonl.gz.digest` | live Galileo E5a, 2026-08-08 18:2x UTC | model-primary: 41 cycles, **468 posts**, no detections at all — the dead-reckon seeder. Moved by `carrier_hz`, `hops_per_sec` (no detections, so no DLL and no code-bias pool) |
| `broker_onsky_e5a_20260826.jsonl.gz.digest` | live Galileo E5a, 2026-08-26 00:33 UTC | the current-regime model-primary chain, 113 cycles, 1474 posts (below) |
| `broker_onsky_l5_20260826.jsonl.gz.digest` | live GPS L5, 2026-08-26 01:1x UTC | the current-regime search-fed chain, 164 cycles, 2444 posts (below) |
| `broker_onsky_b2a_20260826.jsonl.gz.digest` | live BeiDou B2a, 2026-08-26 01:2x UTC | the first BeiDou capture, 172 cycles, 2210 posts (below) |

⚠️ **The on-sky transcripts themselves are NOT in the repo** — a minute of real fleet is
~1.3 MB per cycle (the combiner status responses dominate): 73 MB raw, 21 MB gzipped. Only
the **digests** are versioned; the blobs live beside each other, e.g.

    /home/kvand/gnss/fixtures/broker_onsky_l5.jsonl.gz     (NFS, same path on every node)

⚠️⚠️ **THE GOLDEN DIGEST ALWAYS LIVES HERE, next to this README — never beside the blob.**
It used to be written to `<transcript>.digest`, which for the on-sky fixtures put it on NFS,
*outside git*, with a hand-made mirror in this directory. The mirror drifted silently.

⚠️⚠️⚠️ **THE EPHEMERIS IS A REPLAY INPUT TOO.** A transcript freezes the *fleet*
conversation, but the broadcast ephemeris is fetched by `gnss_ephemeris.fetch_brdc` through
its own urllib — **not** through `transport.py`. So replays read a pinned snapshot:
`broker_equiv.py` points `GNSS_BRDC_DIR` at the transcript's `<transcript>.brdc/` sidecar
when it has one, else at the shared `/home/kvand/gnss/fixtures/brdc_pin`, and a digest is a
function of *(code, transcript)*. `bless` records the pin's fingerprint and `check` calls out
a fingerprint change explicitly. **A model-primary chain is maximally exposed** to it — every
seed comes from the model.

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

⚠️ Before blessing any capture, run `selftest`. A transcript recorded against a dark or
frozen chain replays perfectly and proves nothing — that trap cost two full scans on
2026-08-08.

## Multi-chain smoke test (`two_chain_fake.yaml`)

Three chains — GPS L5 with a search, Galileo E5a and BeiDou B2a with none — in one process
against the fake fleet. This is what proves the M5 driver actually shares, and it needs no
F-engine:

    fuser -k 12777/tcp; sleep 1
    ./broker_fakefleet.py --port 12777 &
    ./broker_multi.py fixtures/two_chain_fake.yaml     # ~20 s, then Ctrl-C

Look for all four, not just "it ran":

    time anchor ... = <same value>        on all three chains  (fetched once, shared)
    receiver: anchor=<v> (<chain>)        ONE latch, named owner
    almanac: BRDC (32 G sats) / (33 E sats) / (<n> C sats) with `brdc=1 store(s)` — one
                                          parse, three constellations, which is the thing
                                          separate processes throw away
    dead-reckon: clock ADOPTED <n> chips from in-process chain '<chain>'  on gal_e5a and
                                          bds_b2a (same band, no file transport)

⚠️ The fake fleet's detections are not physically consistent, so the adopted clock wanders
between cycles. That is the fixture, not the mechanism — do not read convergence into it.

## broker_onsky_e5a_20260826 — the current-regime capture (2026-08-26 00:33 UTC)

The first fixture recorded against the post-08-25 configuration, and the reason the
refactor's gate is no longer blind: the other three on-sky fixtures were blessed at
`f2524ab04`/`26211b9a8` and predate the axis fix, the GAP-1 span observable, the carrier
command, and #92's handover.

* chain `gal_e5a` — model-primary, the maximally-exposed case (every seed comes from the
  model), with `rrate-phase-span-s 16`, `rrate-command true`, `joint-consume clk,rate`
  and `fleet-trim-rebase-adjust 1` all live.
* 113 cycles, 1474 posts, 4628 gets. `selftest`: **GATE GOOD** — determinism holds and the
  digest moves under a 1e-12 perturbation of `carrier_hz` and `dll_gain`.
  ⚠️ It does NOT reach `code_bias_alpha`, `bias_alpha`; a refactor touching
  those is not covered here. That is what the next captures (a search-fed gps_l5 and a
  BeiDou chain) are for.
* **Carries its own ephemeris** in `broker_onsky_e5a_20260826.jsonl.gz.brdc/` (DOY 237+238
  + hourly). `broker_equiv._pin_for` prefers a `<transcript>.brdc/` sidecar over the shared
  pin, because the shared pin covers only DOY 219-228 and adding to it would move the
  fingerprint of all four existing fixtures at once.

CAPTURE RECIPE (one chain per run — `_TR` is a module singleton, so a second `open_write`
clobbers the first): archive the broker log, add `transcript-write: <path>` to ONE chain in
the yaml, restart, wait ~3 min, remove the flag, restart, gzip, then `bless` FROM A CLEAN
TREE and `selftest` before trusting it.

## broker_onsky_l5_20260826 + broker_onsky_b2a_20260826 (2026-08-26 01:1x-01:2x UTC)

The remaining two current-regime captures. Both blessed at `4176af3de` from a clean tree, both
carry their own `.brdc/` sidecar (DOY 237+238 daily + hourly, snapshotted at capture time).

* **`broker_onsky_l5_20260826`** -- the search-fed chain: 164 cycles, 2444 posts.
  `selftest` GATE GOOD; moved by `carrier_hz`, `hops_per_sec`, `dll_gain`, `bias_alpha`.
* **`broker_onsky_b2a_20260826`** -- the first BeiDou capture ever: 172 cycles, 2210
  posts. `selftest` GATE GOOD; moved by `carrier_hz`, `dll_gain`.

⚠️ COVERAGE LEDGER across the six fixtures: **`code_bias_alpha` is reached ONLY by
the old `broker_onsky_l5` (2026-08-08, pre-axis regime)** -- the new l5 capture moves
`bias_alpha` but not `code_bias_alpha`, so keep the old fixture in every broker_equiv check;
it is the sole cover for that knob.
