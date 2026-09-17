# CHORD GNSS runbook — the whole stack, on `gnss`

**The GNSS stack runs on `gnss.site.chord-observatory.ca` under systemd. It is not on cf06 any
more.** If you are about to run something on cf06 that isn't the cube, stop and read §6.

Everything below was exercised on 2026-09-17, and the timings are measured, not estimated.

---

## 1. The three commands

```sh
ssh gnss systemctl --user start  gnss-stack.target     # up   (~20 s)
ssh gnss systemctl --user stop   gnss-stack.target     # down (~0 s)
ssh gnss systemctl --user status 'gnss-*'              # what is running
```

That is the whole of routine operation. **The nodes are not in it** — see §5.

⚠️ **`--user`, always.** These are systemd *user* units (there is no root on this VM yet).
Without `--user` systemd looks for system units, finds nothing, and says so confusingly.

## 2. What runs where

| | where | unit | cost |
|---|---|---|---|
| broker | **gnss** | `gnss-broker` | 1.43 cores |
| gather | **gnss** | `gnss-gather` | 0.84 |
| aggregator (GPU) | **gnss** | `gnss-aggregator` | 0.76 + 34% of an L40S |
| obs writers ×8 | **gnss** | `gnss-obs@<chain>` | 0.15 total |
| live viewer :8080 | **gnss** | `gnss-viewer` | 0.02 |
| — | | | **3.19 of 6 cores (53%)** |
| cube archiver | cf06 | `cubearch_up.sh` | writes 32.5 GB/h to NFS |
| cube compactor | cf06 | `cubecompact_loop.sh` | periodic heavy analysis |
| static cube viewer :8877 | cf06 | `beamview_up.sh` | serves the exported cube |

**The eight chains** are `gps_l5`, `gps_l2c`, `gal_e5a`, `gal_e5b`, `gal_e6`, `bds_b2a`,
`bds_b2b`, `bds_b3i`.

## 3. Is it healthy? Four numbers, one minute

```sh
ssh gnss 'curl -s http://127.0.0.1:12060/status | python3 -m json.tool | head -20'   # broker
ssh gnss 'curl -s http://127.0.0.1:12051/fleet_trim/get_stats | head -20'            # gather
ssh gnss 'curl -s http://127.0.0.1:12050/gps_search/get_detections | head -5'        # search
ssh gnss 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'  # GPU
```

Healthy, as measured today:

| | good | bad |
|---|---|---|
| gather frames | **~2120/s**, `bad_frames 0` | falling, or bad_frames climbing |
| late frames | **0.15-0.25%** | >1%, or rising steadily |
| chains | all 8, **10-15 armed / 5-10 present** each | a chain at 0 armed |
| GPU | **30-40%, ~1.5 GB** | 0% with data flowing = the search is not running |
| load average | **3-4.5** of 6 cores | >8 — something is spinning (see #134) |

⚠️ **Armed is not present.** `n_prn` is how many the broker is commanding; `present` is how many
are actually being seen. `present` well below `armed` for minutes is the interesting fault.

## 4. Auto-recovery — what fixes itself, and what does not

**Fixes itself.** Every unit has `Restart=on-failure`. Measured today by `kill -9`:

| | back in |
|---|---|
| obs writer | < 15 s |
| viewer | ~33 s (it waits for the broker's chain list to settle — that delay is deliberate, §7) |
| broker, gather, aggregator | seconds, then the fleet re-arms (below) |

**A full stop/start costs almost nothing** as long as the nodes stay up: measured today, armed
and present were back at 99/64 immediately, detections within seconds, and the trim loop
re-closed inside a minute. The nodes hold their own tracking; the VM services only re-attach.

⚠️ **A gather restart still wipes every standing trim.** The readback goes large (~1-2.5 chips)
for a minute and settles. That is the transient, not a fault.

**Does NOT fix itself:**

* **A node restart is needed** whenever `telem-host` or `search-host` changes in
  `config/gnss_fleet_chord.yaml` — nodes read those once at startup (§5).
* **An F-engine restart** needs chive refreshed *first*, then the nodes. Symptom: every DPDK
  worker logs `frames are being stamped <big>s from the wall clock`, the fleet looks alive and
  acquires nothing. Compare chive's `start_ctime` against when the F-engine actually restarted —
  the nodes will faithfully latch a stale epoch.
* **Reboot:** ⚠️ the mechanism is in place (`Linger=yes`, `gnss-stack.target` in
  `default.target.wants`) but **has not been tested with an actual reboot** — that needs root.
  Check it the first time the VM reboots.

## 5. The nodes

Not in the target, on purpose — they are six machines, they need sudo, and cycling them costs a
fleet-wide re-arm.

```sh
for n in cx19 cx27 cx42 cx43 cx44 cx51; do scripts/gnss/node_up.sh $n restart; done
scripts/gnss/eop_push.sh        # from a host that reaches the nodes
```

**Bring the gather and the aggregator up BEFORE restarting nodes into them.** Sixty senders
retrying a dead listener is ~2 log lines a second across the fleet.

The four things that have stopped a node bring-up, in the order they bit:
EOP table expired (`node_up.sh` refuses below 12 h headroom) · choco maintenance mode off (a
per-minute `/kill` sweep) · chive serving a pre-re-base epoch · the broker started under the GIL.
Details in [`CHORD_STACK_SHUTDOWN.md`](CHORD_STACK_SHUTDOWN.md) §6.

## 6. ⚠️ cf06: what is still there, and what will hurt you

**Still on cf06, legitimately:** the cube archiver, the cube compactor, the static cube viewer
on :8877. All of it is the beam-cube product — offline analysis, not the live stack. See
[`CHORD_BEAMCUBE_RUNBOOK.md`](CHORD_BEAMCUBE_RUNBOOK.md). Both its L40S are free, and that is
where GPU benchmarking should live (`CUDA_VISIBLE_DEVICES=1`); the VM's single GPU is in
production use.

**⚠️ DO NOT RUN `stack_up.sh` ON cf06.** It predates the move and would start a *second* broker,
gather and aggregator alongside the VM's. Two brokers command the same fleet. It now refuses;
if you find a copy that does not, do not use it.

The per-component `*_up.sh` scripts (`broker_restart.sh`, `gather_up.sh`, `agg_up.sh`,
`obs_up.sh`, `viewer_up.sh`) are **superseded for the live stack** by the units, but they are not
stale: since 2026-09-17 they launch through the *same* definition the units do (§10), so running
one by hand starts exactly what systemd would. They keep the host guards, the config preflight,
the log rotation and the post-start health checks that systemd does not do.

## 7. Two deliberate delays, so they are not mistaken for faults

* **The viewer waits up to 300 s to start** (`wait_for_broker_chains.sh`), until the broker's
  chain count stops changing for 15 s. It discovers the chain list **once** and caches it, so
  starting it against a half-built broker leaves it showing 1 or 2 chains of 8. That happened.
* **The broker must run free-threaded 3.14t** (`GNSS_PY=/home/kvand/gnss/venv-ft/bin/python`)
  **and with `OPENBLAS_NUM_THREADS=1`** and its three siblings. Both are in the unit. Under the
  GIL every non-L5 chain dies; without the BLAS cap the process spends two thirds of its cycles
  in OpenBLAS's busy-wait (#134). Neither failure is loud.

## 8. Logs

`/var/tmp/gnss-logs/` on the VM — local disk, not the NFS home (259 MB/h of appends where
`O_APPEND` is not atomic), and not `/tmp` (cleared on boot). ~0.7 GB/day.

```sh
ssh gnss 'tail -f /var/tmp/gnss-logs/broker.log'
ssh gnss 'systemctl --user status gnss-broker.service -n 40'
```

Archive before any deliberate teardown you might want to explain later:
`gzip -1 -c /var/tmp/gnss-logs/broker.log > /home/kvand/gnss/logs/broker_$(date -u +%Y%m%d_%H%M%S).log.gz`

⚠️ There is no logrotate yet (it needs root). Watch the directory until there is.

## 9. Profiling, when it is slow

* **py-spy does not work here.** 0.4.2 cannot read free-threaded 3.14 (`failed to get
  gil_thread_id`). Do not spend time on it.
* **perf needs root** (`kernel.perf_event_paranoid = 4`) and is the fastest real answer:
  `sudo perf record -F 999 -g -p <pid> -- sleep 30 && sudo perf report --stdio --no-children
  --sort dso,symbol`. It found #134 in one screen.
* **`kill -USR1 <broker pid>` dumps every thread's stack** to the log. **ONCE.** As a sampler it
  has killed the broker.
* For a real profile, replay under `broker_equiv` — side-effect-free, no root, no ptrace.

## 10. One definition, two ways to run it

**`scripts/gnss/stack_components.sh` is the single source of truth** for what each component is
— its argv, its environment, and its working directory. Nothing else defines those.

```
stack_components.sh            what to run, and with what environment
       |
       +-- run_component.sh    execs it in the FOREGROUND
               |        |
               |        +----- systemd  ExecStart=run_component.sh <name>     (the live path)
               +-------------- *_up.sh  daemonises it, and adds the things systemd does not:
                                        host guards, config preflight, log rotation,
                                        post-start health checks
```

⚠️ **Why the units do not simply call `*_up.sh`**, which is the obvious idea: those scripts
*daemonise* (`nohup setsid … &`, `disown`) and most `pkill` a previous instance first. Under
systemd that is three faults at once — `Type=simple` expects `ExecStart` to *be* the process,
`setsid` detaches the child from the unit's cgroup so supervision is lost entirely, and the
internal `pkill` races systemd's own restart. So the direction is inverted: both paths go
through a foreground runner.

### The gate that makes #134 impossible to repeat

```sh
ssh gnss 'cd /home/kvand/gnss/kotekan && ./scripts/gnss/stack_contract_gate.sh'
```

It compares each **running** process against the definition — **argv *and* environment** — and
exits non-zero on the first mismatch. The environment half is the point: #134 was a faithfully
reproduced command line with a dropped `export`, which an argv-only comparison would pass.

**Run it after any change to a unit, a launcher, or the definition.** Expect
`checked 12 component(s), 0 mismatch(es)`.

### Editing

```sh
# change what a component IS  -> scripts/gnss/stack_components.sh
# change how it is SUPERVISED -> scripts/gnss/systemd/gnss-<x>.service
ssh gnss 'sh /home/kvand/gnss/kotekan/scripts/gnss/systemd/install_user_units.sh'
ssh gnss 'systemctl --user restart gnss-<unit>.service'
ssh gnss 'cd /home/kvand/gnss/kotekan && ./scripts/gnss/stack_contract_gate.sh'
```

⚠️ One difference is deliberate and visible in the definition rather than hidden: `agg_up.sh`
runs the aggregator with `GNSS_SEARCH_PROFILE=1` and the systemd unit does not. cf06 ran with
that profiling on for its whole life (113,997 `[consumer]` lines in its last log); the live VM
instance runs without it. Set `GNSS_SEARCH_PROFILE=1` to restore it.
