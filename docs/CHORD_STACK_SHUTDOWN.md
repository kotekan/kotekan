# Stopping and starting the GNSS stack — the teardown, the bring-up, and what it costs to get either wrong

> ⚠️ **SUPERSEDED for the live stack, 2026-09-17.** The broker, gather, aggregator, obs writers
> and live viewer moved to **`gnss.site.chord-observatory.ca`** and run as systemd units.
> **Start here instead: [`CHORD_GNSS_RUNBOOK.md`](CHORD_GNSS_RUNBOOK.md).**
> What remains on cf06 is the cube leg only — archiver, compactor, static viewer :8877.

**What this is for.** Handing the array to someone else, or taking GNSS down for any other
reason. The bring-up counterpart is [`CHORD_BEAMCUBE_RUNBOOK.md`](CHORD_BEAMCUBE_RUNBOOK.md) §2
for the cube chain and the `scripts/gnss/*_up.sh` family for everything else; there was no
counterpart going the other way until 2026-09-11, so every teardown before that was assembled by
hand from `ps` output at the moment it was needed.

**One command** (cf06's cube leg only, these days):

```sh
ssh cf06 /home/kvand/gnss/kotekan/scripts/gnss/stack_down.sh
```

It archives the live logs, then stops all eight components in dependency order and verifies each.
`--list` shows what it would stop and changes nothing; `--no-archive` skips the archiving.
**The nodes are not in it** — stopping those is a separate, deliberate act (see §4).

---

## 1. What the stack actually is

Eight components, fifteen processes, all `nohup setsid` — no systemd units, nothing in cron.

| component | processes | what it is |
|---|---|---|
| obs-writers | 8 | `gnss_observables.py`, one per chain, polling the broker |
| viewer-livebeam | 1 | `livebeam_server.py` — HTTP 8080, WebSocket 8539 |
| viewer-static | 1 | `python3 -m http.server 8877` — serves the exported cube |
| cube-compactor | 1 | `cubecompact_loop.sh` |
| broker | 1 | `broker_multi.py` — REST on 12060 |
| aggregator | 1 | `kotekan … chord_gnss_agg*.yaml` — carries the search; REST 12050 |
| gather | 1 | `kotekan … chord_gnss_gather.yaml` — REST 12051 |
| cube-archiver | 1 | `kotekan … chord_gnss_cubearch.yaml` |

⚠️ **Two of these are easy to miss by hand, and both were missed the first time.** The static
viewer is a bare `python3 -m http.server 8877`: it matches no pattern containing "gnss",
"kotekan" or "broker", and the only way to find it is by its port. The cube compactor is a bash
loop, so killing what looks like the python job leaves the loop to start another.

## 2. Order: consumers before producers

Down is the reverse of up, and the table above is already in that order. It matters for two
reasons, neither cosmetic:

- Stopping the gather or the broker first leaves everything above it logging connection failures
  **into the very logs being archived**, and the obs writers writing rows against a broker that
  is going away.
- The bring-up has a hard ordering constraint in the other direction — `gnss_fleet_chord.yaml`
  requires the **cube archiver up before the nodes** — so the archiver is last down and first up.

Everything exits cleanly on `SIGTERM`; the script allows 10 s before escalating, and on
2026-09-11 nothing needed escalation. **Do not reach for `-9`**: the archiver, the compactor and
the obs writers are all mid-write on files, and `KILL` truncates whatever was in flight.

## 3. Archive the logs FIRST — this is not optional

⚠️ **`/tmp` is cleared on boot.** `/usr/lib/tmpfiles.d/tmp.conf` carries `D /tmp 1777 root root
30d`, and cf06 reboots on `unattended-upgrades`' own schedule — 2026-08-06, 08-12, 08-19, 09-05,
all at 03:03 UTC. An unarchived stack log is therefore lost **at the next boot**, not at the next
restart, and a teardown is usually followed by exactly that.

The `*_up.sh` scripts rotate their log on start (buglist #65), which protects the *previous*
run's log inside `/tmp` — and `/tmp` is what gets cleared. Rotation is not archiving.

`stack_down.sh` archives every `/tmp/gnss_*.log` touched in the last 6 h to
`/home/kvand/gnss/logs/<name>_<UTC>.log.gz`, in parallel at `gzip -1`, and skips the rest as
history. The 2026-09-11 teardown: **~8.5 GB of live logs → 1.4 GB archived in about 90 s**, with
a stale 25 GB `gnss_gather_v6.log` correctly skipped. `gzip` will say `file size changed while
zipping` for the logs still being written — that is expected, it archives what was there, and
the exit status is still 0.

## 4. The nodes

**Not in the script, on purpose.** Stopping the six node instances is a separate decision with a
different blast radius, and bringing them back costs a settle the cf06 stack does not:

- A **single** node roll costs essentially nothing.
- A **fleet-wide** cycle costs ~25–30 min, because the standing trim lives on each node and is
  wiped by the restart. You will see `cudaGnssInject: PRN N trim EXPIRED with no /set_trim` from
  every chain for the first few minutes; that is the transient, not a fault. On 2026-09-11 the
  rate went to **zero within 8 minutes** of a full cycle.

## 5. Confirming it is actually down

```sh
ssh cf06 'ss -ltnp | grep -E "8080|8539|8877|1205[0-9]|1206[0-9]"'   # expect nothing
ssh cf06 'pgrep -fc "[k]otekan|[b]roker_multi|[g]nss_observables"'    # expect 0
```

The script prints both at the end. ⚠️ **A quiet viewer is not a stopped viewer** — the static
viewer keeps serving the last export it had, which is indistinguishable from a healthy one; check
the port, not the page.

## 6. Bringing it back

**One command, then the nodes:**

```sh
ssh cf06 /home/kvand/gnss/kotekan/scripts/gnss/stack_up.sh        # --list shows the plan
for n in cx19 cx27 cx42 cx43 cx44 cx51; do scripts/gnss/node_up.sh $n restart; done   # sudo per node
scripts/gnss/eop_push.sh                                            # from a host that reaches the nodes
```

`stack_up.sh` is the reverse of the table in §1 with the environment each component needs baked
in -- above all **`GNSS_PY=/home/kvand/gnss/venv-ft/bin/python` for the broker** -- and it
verifies every component by its ports before starting the next. It refuses unless this is cf06,
`/mnt/cs00/data` is mounted, and `chive:54321/get-frame0-time` answers.

### The four things that stopped the 2026-09-14 bring-up, in the order they bit

Each is now a refusal or a script default; they are listed so the symptom is recognisable.

1. **The baked EOP table had expired** (Friday's table, −17 h by Monday). Symptom: every node
   runs ~60 s then exits on `Requesting EOP later than in table`. `node_up.sh` now refuses below
   12 h of headroom and prints the `gen_fleet.py` regen + `--check`; commit the result.
2. **choco's per-node maintenance mode was off.** Symptom: nodes die within ~60 s of coming up
   with `ERROR: /kill endpoint called` as the last log line, one per minute (a sweep). Nothing in
   this tree sends `/kill`. Maintenance mode must be ON on all six before any node start.
3. **chive still published the epoch from before an F-engine re-base.** Symptom: every DPDK
   worker logs `THE WIRE'S SEQ AXIS IS <big>s FROM THE WALL CLOCK`, `port_axis_gate.py` says both
   ports agree, the broker sees nothing. Order is chive refresh → nodes (frame0 is read once per
   process). `node_up.sh` now refuses if chive does not answer; it cannot know whether the
   answer is *current* -- compare `start_ctime` with when the F-engine actually restarted.
4. **The broker came up under the GIL** (`broker_restart.sh` defaulted to the 3.12 venv).
   Symptom: gather logs `dropped client fd N (127.0.0.1) -- could not take a frame within 200 ms`
   every ~15 s; broker logs `chain X: ALL 12 instances stale` for every chain; `FLEET-TRIM`
   shows `0 PRN(s) armed` on every chain but L5; nodes log `trim EXPIRED with no /set_trim`;
   L5 works (its fast loop is in-broker). The default is now `venv-ft`.

### Surviving an F-engine restart

The nodes cannot: `fpga_monitor` raises a FatalError when the controller's config *or timing*
changes, on purpose -- the latched `frame0` is no longer the wire's. As of 2026-09-14 the
transient unit carries `Restart=on-failure` (20 s, no start limit), so a node that dies this
way -- or crashes -- comes back on its own, re-reads chive on the way up, and converges on the
new epoch without an operator. A clean exit (`/kill`, `systemctl stop`) stays down. The node
log is appended across restarts so the fatal survives; `node_up.sh` rotates it to `.1` on each
deliberate start. The gather re-anchors by itself (`epoch_resets` in `/fleet_trim/get_stats`);
the aggregator, archiver and broker do not read chive and do not need restarting for this. The
EOP live table does NOT survive a node restart -- only the baked one does, hence the 12 h gate.

⚠️ **The obs writers and the broker must both be running the same generation of the code.** They
are separate processes with separate lifetimes: on 2026-09-11 the broker was restarted with a new
published field (`fadr_g_hist`, buglist #117) while the writers kept running Thursday's code, so
the field was published and silently not recorded for 26 minutes. If a change spans the two,
restart both.
