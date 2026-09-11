# Stopping the GNSS stack — the teardown, and what it costs to get it wrong

**What this is for.** Handing the array to someone else, or taking GNSS down for any other
reason. The bring-up counterpart is [`CHORD_BEAMCUBE_RUNBOOK.md`](CHORD_BEAMCUBE_RUNBOOK.md) §2
for the cube chain and the `scripts/gnss/*_up.sh` family for everything else; there was no
counterpart going the other way until 2026-09-11, so every teardown before that was assembled by
hand from `ps` output at the moment it was needed.

**One command:**

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

Reverse the table: cube-archiver, gather, aggregator, broker, then the viewers, the compactor and
the obs writers. `CHORD_BEAMCUBE_RUNBOOK.md` §2 covers the cube half in detail. The scripts are
`cubecompact_up.sh`, `gather_up.sh`, `agg_up.sh`, `broker_restart.sh`, `viewer_up.sh`,
`obs_up.sh` — each rotates its own log on start, so nothing needs clearing first.

⚠️ **The obs writers and the broker must both be running the same generation of the code.** They
are separate processes with separate lifetimes: on 2026-09-11 the broker was restarted with a new
published field (`fadr_g_hist`, buglist #117) while the writers kept running Thursday's code, so
the field was published and silently not recorded for 26 minutes. If a change spans the two,
restart both.
