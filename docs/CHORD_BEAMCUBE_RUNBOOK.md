# Beam-cube runbook — the daily refresh, and bringing cf06 back after a reboot

**What this is for.** Every piece of the beam-cube chain on cf06 is a `nohup setsid` process
started by a `*_up.sh` script. None of it is a systemd unit, none of it is in cron, and
`unattended-upgrades` reboots cf06 on its own schedule — 2026-08-06, 08-12, 08-19, 09-05, all
at 03:03 UTC. **After a reboot, nothing in this chain is running and nothing will tell you so.**
The viewer keeps serving the last export it had, which is exactly what a healthy viewer looks
like. This file is the list of what to restart, in order, and how to tell it worked.

The physics of the builder, the units, and the subband axis are in
[`CHORD_CUBE_ARCHIVE.md`](CHORD_CUBE_ARCHIVE.md) §8; this file is only the operations.

---

## 1. The chain

```
  89 senders on 6 nodes                    ── nodes must be up (node_up.sh)
        │  bufferSend
        ▼
  cubearch    kotekan chord_gnss_cubearch.yaml   raw/gnss_cube_NNNNNNN.raw
        │                                        cubearch_up.sh      ⚠ BEFORE the nodes
        ▼
  compactor   cubecompact_loop.sh    every 300 s: raw → l0/<pointing>/<sender>/<day>.h5
        │                            after 00:20 UTC: yesterday → rung12/ and rung60/
        │                            ★ and now: build + export yesterday (§3)
        ▼                            cubecompact_up.sh
  masters     fixtures/beamcube/cube_<day>_nside64.npz
        │
        ▼
  web export  fixtures/beamcube/web/cube_<day>_<chain>.bin + index.json
        │
        ▼
  viewer      python3 -m http.server 8877     beamview_up.sh     http://cf06:8877/
```

Everything above the masters is continuous. Everything from the masters down happens **once a
day**, a few minutes after the rungs are cut.

## 2. Bringing it back after a reboot

Run these on cf06, in this order. Each refuses to run on the wrong host, so an accidental
`bash` on cx43 is safe.

```bash
ssh cf06 '/home/kvand/gnss/kotekan/scripts/gnss/cubearch_up.sh'      # ⚠ FIRST — see below
ssh cf06 '/home/kvand/gnss/kotekan/scripts/gnss/cubecompact_up.sh'
ssh cf06 '/home/kvand/gnss/kotekan/scripts/gnss/beamview_up.sh'
```

⚠️ **The archiver goes up before the nodes start pushing.** A `bufferSend` frame arriving with
no archiver is fine, but the reverse order has bitten us: buglist #110, a cube frame with no
metadata segfaulted six nodes. Archiver first, then nodes.

⚠️ `cubecompact_up.sh` needs `h5py`, so it runs `/home/kvand/gnss/venv/bin/python` and **not**
`venv-ft` — the script checks and refuses, but if you invoke the loop by hand, use `venv`.

The rest of cf06 is out of scope here and has its own owners: `gather_up.sh`, `agg_up.sh`,
`obs_up.sh`, and the broker (`broker_restart.sh`, KV's — archive `/tmp/gnss_broker.log` first).

**Checks that actually bind** (all three, in under a minute):

```bash
ssh cf06 'pgrep -af "chord_gnss_cubearch|cubecompact_loop|http.server 8877" | wc -l'   # want 3
ls -l /mnt/cs00/data/kvand/gnss_cube/raw | tail -2        # a new .raw within the last ~15 s
tail -3 /tmp/gnss_cubecompact.log                          # a "== <ts> compact" inside 5 min
curl -s http://cf06:8877/index.json | head -c 120          # the days the page can draw
```

A stale `index.json` is the failure this file exists to catch: the page renders perfectly with
week-old data. **Judge the viewer by the newest day in `index.json`, never by whether it loads.**

## 3. The daily refresh

`cubecompact_loop.sh` cuts yesterday's rungs shortly after 00:20 UTC and then builds the master
and re-exports the whole web directory (`beamcube_daily.sh`, §4). Nothing to do by hand.

**To confirm it happened** — one line, and the only daily check worth making:

```bash
curl -s http://cf06:8877/index.json | python3 -c 'import json,sys; print([d["day"] for d in json.load(sys.stdin)["days"]])'
```

Yesterday should be in that list by ~00:35 UTC. If it is not, `grep beamcube
/tmp/gnss_cubecompact.log` says why, and §5 is the hand-run.

## 4. Doing it by hand

Needed when the loop was down at 00:20 (a reboot, say), or to add **today** before its rungs
exist. Two steps, ~4 min and ~2 min:

```bash
cd /home/kvand/gnss/kotekan
B=/home/kvand/gnss/fixtures/beamcube
/home/kvand/gnss/venv/bin/python python/scripts/gnss/gnss_beam_cube.py build \
    --source l0 --archive /mnt/cs00/data/kvand/gnss_cube/rung12/p0_dec40p73 \
    --days YYYYMMDD --nside 64 --outdir $B
/home/kvand/gnss/venv/bin/python python/scripts/gnss/gnss_beam_cube.py export \
    $B/cube_*_nside64.npz --outdir $B/web --nside 32
```

or just `scripts/gnss/beamcube_daily.sh YYYYMMDD`, which is those two with the traps handled
(and is what the compactor calls). It skips the build if the master already exists —
`BEAMCUBE_FORCE=1` to rebuild — takes an `flock` so it cannot collide with the nightly pass,
and refuses a day whose rungs span two pointings rather than summing them.

⚠️ **THE MASTERS DIRECTORY IS THE PUBLISH SET.** `beamcube_daily.sh` exports
`$B/cube_*_nside64.npz`, all of it, every time — that is what makes the "forgot a day" trap
impossible. The corollary is that anything in there goes on the page: park superseded or
experimental masters in `fixtures/beamcube/attic/`, which is where the 2-chain 09-02 test cube
now lives (it published itself the first time this script ran).

⚠️⚠️ **`export` rewrites `index.json` from ITS ARGUMENTS ONLY.** Pass every master you want the
page to offer, not just the new one, or the other days vanish from the viewer. The glob above is
the safe habit. No viewer restart is needed after an export — a browser reload is enough.

⚠️ **`rung12` exists only for days that have ENDED.** The compactor cuts it after 00:20 UTC for
the day before. Today exists at L0 only, and L0 is ~10× slower to build from.

⚠️⚠️ **DO NOT READ TODAY'S L0 IN PLACE.** The compactor appends to each day file every 300 s
holding HDF5's exclusive lock, so a reader gets `BlockingIOError` partway through a build — and
worse in the other direction, a long h5py read holds a shared lock that can make the
COMPACTOR's next pass fail. Snapshot with `cp` (which takes no HDF5 lock), verify, build from
the copy: 89 files / ~160 GB runs ~12 min at P=6 into cf06 `/var/tmp/`, which has the room.

`--nside 128` masters are for the static plots (`gnss_beam_static.py`); the viewer never reads
them, so a daily run does not need one.

## 5. What each artifact costs

| | per day | note |
|---|---|---|
| master, nside 64 | ~0.6 GB | what the export reads |
| master, nside 128 | ~1.1 GB | static plots only, on demand |
| web export | ~0.2 GB | nside 32, one .bin per chain |
| build time | ~4 min | from rung12; ~40 min from L0 |
| export time | ~20 s/day listed | re-reads every master each time, so this grows |

The export re-reading every master is the part that scales badly — at ~30 days it is ten
minutes. When that starts to hurt, the fix is an incremental `index.json` merge, not a shorter
day list.

## 6. Things that have gone wrong here before

- **A day that looks thin is usually the sky, not the pipeline.** 2026-09-09 has 869 of 1440
  minutes, one fleet-wide gap 03:44–13:15 UTC, identical on all 89 senders. The builder prints
  `railing veto: N/M minute(s)` — M is how many minutes had any data at all, and it is the
  fastest read on whether a day is short.
- **A smoothed count is not coverage.** See `gnss_beam_static.py --min-occ`; it painted the
  vetoed 5° hole full of data that was never measured there.
- **Two consumers of `/get_beam_cube` split the data** — it is reset-on-read. One archiver only.
- **Restarting the compactor needs the in-flight pass to finish first.** `pkill` takes down
  the bash loop and orphans any running `gnss_cube_compact.py`; a second one on the same 89 L0
  files is a corrupting write. `cubecompact_up.sh` now waits for the child (up to 15 min) and
  refuses rather than racing it.
- **The viewer's `peak normalise` divides by the peak of what is displayed**, and the railing
  veto removed the main lobe, so that peak sits on the lobe edge. Prefer the SVD normalisation
  in `gnss_beam_static.py` when the number matters.
