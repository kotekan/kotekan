# Moving the GNSS infrastructure to `gnss.site.chord-observatory.ca`

**What this is.** Sizing and a migration plan for putting the GNSS stack on the provided VM
under systemd, long-term. Every number below is measured on the live stack, not estimated; how
to re-measure is at the end.

**The short version.** ✅ **DONE.** Broker, gather, obs writers and viewer moved 2026-09-16; the
aggregator followed on 2026-09-17. The whole GNSS stack now runs on one 6-core VM at **3.19 mean
/ 3.39 max cores, 53-57%**, under systemd, coming back by itself after a reboot. cf06 keeps only
the cube archiver, the compactor and the static viewer — and both its L40S are now free.

⚠️ The sizing very nearly went the other way: the first peak sample read 5.18-5.66 cores and I
recommended asking for more vCPU. That was measuring a bug (#134, a dropped BLAS thread cap),
not a shortage. See §4 — the episode is the most useful thing in this document.

---

## 1. Measured: what the stack actually costs

Sampled on cf06 with the fleet live, 30 s windows. CPU is instantaneous, not the lifetime
average (which runs ~15% lower for the broker).

| process | CPU | RSS | threads | fds | disk write | socket rx |
|---|---|---|---|---|---|---|
| **broker** (`broker_multi.py`, venv-ft) | **1.75 cores** | 582 MB | 11 | 63 | 259 MB/h | 453 Mbit/s ← from the gather, over loopback |
| **gather** (kotekan) | **1.24 cores** | 339 MB | 14 | 201 | 7 MB/h | 369 Mbit/s ← from the fleet |
| **obs writers** ×8 | 0.06 cores total | 425 MB total | 1 each | — | 420 MB/h total | — |
| livebeam viewer | 0.02 cores | 78 MB | 65 | — | 1 MB/h | — |
| static viewer (`http.server`) | ~0 | 46 MB | 1 | — | — | — |
| cube archiver (kotekan) | 0.27 cores | 335 MB | 10 | 103 | **32.5 GB/h** → NFS | — |
| aggregator (kotekan + **GPU**) | 0.91 cores | 747 MB | 44 | 119 | 103 MB/h | 125 Mbit/s |

**Totals for the infrastructure move** (broker + gather + obs writers):
**3.05 cores, 1.35 GB RSS, ~0.7 GB/day of logs, 369 Mbit/s in.**

## 2. The VM as provisioned (2026-09-16 — re-provisioned, everything below re-measured)

It is a KVM guest **on cf02, so the same silicon as cf06**, with `host` CPU passthrough.

| | value | verdict |
|---|---|---|
| vCPU | **6**, `Intel Xeon Gold 5416S` | ✅ identical part to cf06 |
| CPU features | `fma avx avx2 bmi2 avx512f avx512dq` | ✅ the `SIGILL` blocker is gone |
| RAM | 7 GB, **no swap** | ✅ 2.4 GB needed even with everything |
| Disk | 249 G, 210 G free | ✅ with rotation |
| Network | **≥5.1 Gbit/s** measured (641 MB/s off `cs00`, `iflag=direct`) on a 10G uplink | ✅✅ |
| **GPU** | **NVIDIA L40S, 46 GB, passthrough** | ✅ verified with a real kernel, see below |
| CUDA | toolkit 13.4, driver 615.71.09 (cf06: 13.3 / 610.57.04) | ✅ newer driver, forward-compatible |
| NUMA | 1 node, GPU affinity `0-5` | ✅ core-placement rules cannot bite |
| `/mnt/cs00/data`, `/home/kvand` | mounted | ✅ |
| Reachability | cx19, cx43, cf06 all connect inbound to `:11060` | ✅ |

### The binary runs, and the GPU is real

```
$ ssh gnss .../build_nodpdk/kotekan/kotekan --help ; echo $?
0                                  # was 132 (SIGILL) before the reprovision
$ ldd .../kotekan | grep "not found"
                                   # nothing missing
```

A passthrough GPU that answers `nvidia-smi` does not always give a usable compute context, so
that was tested rather than assumed — `cudaMalloc` + a kernel launch + readback returned the
right answer, `devices=1`, exit 0. All of `/dev/nvidia{0,ctl,-uvm,-uvm-tools}` are present.

### Per-core speed is cf06's, because it is cf06's CPU

Same interpreter off the same NFS venv, best of three:

| kernel | cf06 | gnss VM |
|---|---|---|
| Python dict/tuple churn | 0.0475 s | 0.0468 s |
| `json.loads` | 0.190 s | 0.185 s |
| `np.fft.rfft` | 0.119 s | 0.119 s |
| float64 matmul, default threads | 0.0104 s | 0.0139 s |
| float64 matmul, **1 thread** | 0.0052 s | **0.0028 s** |

⚠️ **The multi-threaded matmul row is not a per-core comparison and should not be read as one.**
cf06 spreads it over 64 cores and the VM over 6.

The single-thread row is the interesting one: **the VM did the same work 1.9× faster on
nominally identical silicon**, which should not happen. It is the clock. cf06 runs the
`powersave` governor under `intel_pstate` and was sampled at **819–1767 MHz**; the guest has no
`cpufreq` driver and reports a flat nominal 2000 MHz, which is a KVM fiction — but the work it
actually completed is not.

Two consequences, and the second is the useful one:

- Do not quote "per-core parity" as a measured fact. What is measured is that the VM's effective
  per-core throughput is **at least** cf06's.
- **Every core figure in this document was therefore measured on a down-clocked host, so the
  budget is conservative.** 4.23 cores on cf06 is an upper bound on what the VM will need, not
  an estimate to add margin to.

⚡ Worth someone's attention independently of this migration: cf06 serves the GPU aggregator
while sitting on `powersave` at well under half its rated clock.

## 3. Measured load, and why today's numbers are the floor not the basis

⚠️ **The F-engine went down for site work partway through this work.** The stack stays up and
connected — 89 sender connections to the gather, 12 to the aggregator — but carries no data:
`rx 0.0 Mbit/s`, `present: 0` and `n_prn: 0` on every chain, the archiver writing 0 GB/h. So
there are two sets of numbers and only one of them is a sizing basis.

| process | **with data** (the basis) | idle, F-engine down |
|---|---|---|
| broker | **1.75 cores** | 0.82 |
| gather | **1.24 cores** | 0.23 |
| aggregator | **0.91 cores** + GPU | 0.95 |
| cube archiver | **0.27 cores**, 32.5 GB/h | 0.00, 0 GB/h |
| obs writers ×8 | **0.06 cores** | 0.03 |
| **total** | **4.23 cores** | 2.03 |

Two things worth keeping from the idle column. The fixed overhead is ~2 cores, so about half the
load is data-driven and scales with the fleet. And **the aggregator costs the same either way** —
its search spins regardless (`blocked 3.6s (100%), 120 frames discarded`), so its 0.91 is a floor
that will not fall when the sky is quiet.

## 4. Sizing — measured twice, and the first answer was wrong

Estimated from cf06: 3.05 cores for broker + gather + obs. **First measurement on the VM**
(3 min at 5 s, F-engine live, 95 PRNs armed): **5.18 mean / 5.66 max of 6**, load average 9.61,
with the broker alone at 4.31 against the 1.75 estimated. On that basis I said the aggregator
could not move and recommended more vCPU.

**That was measuring #134.** `broker_restart.sh` has exported `OPENBLAS_NUM_THREADS=1` and its
three siblings since 2026-08-15; the systemd unit reproduced the command line faithfully and
silently dropped the environment. `perf top` settled it in one screen: **55.6% of all cycles in
`blas_thread_server`**, OpenBLAS's busy-wait, against **0.52% in `dgemm_kernel_SKYLAKEX`**, the
actual arithmetic. Two thirds of the process spinning to do half a percent of maths.

| | estimate (cf06) | first measurement | after #134 | final, with the aggregator |
|---|---|---|---|---|
| broker | 1.75 | **4.31** | 1.23 | 1.43 |
| gather | 1.24 | 0.74 | 0.78 | 0.84 |
| aggregator | 0.91 | — (on cf06) | — | **0.76** |
| obs ×8 + viewer | 0.08 | 0.12 | 0.12 | 0.17 |
| **total of 6** | — | **5.18 / 5.66** | 2.14 / 2.39 | **3.19 / 3.39 (53-57%)** |
| load average | — | 9.61 | 3.07 | 4.43 |

⚠️ **The lesson is not about OpenBLAS.** It is that **a wrapper script's `export` is part of the
program's contract, and porting the command line is not porting the program** — the faithfully
reproduced `ExecStart` is exactly what hid this. When converting a launcher to a unit, diff
`/proc/<pid>/environ` between the old process and the new one, not the argv.

⚡ The aggregator is *cheaper* here than on cf06 (0.76 against 0.91), which is consistent with
cf06 running `powersave` at 819-1767 MHz while this guest boosts properly.

⚡ And the cf06 estimates were not simply "low": the gather came in **under** estimate and the
broker far over. A single 30 s window on a differently-loaded host is not a budget.

### What the profilers could and could not do

* **py-spy cannot attach to this broker at all** — 0.4.2 fails with `failed to get gil_thread_id`
  against free-threaded 3.14 (a 3.12 control profiles cleanly). Do not spend time on it.
* **perf works but needs root here**: `kernel.perf_event_paranoid = 4`. `perf record -F 999 -g -p
  <pid> -- sleep 30` then `perf report --stdio --no-children --sort dso,symbol`.
* **`kill -USR1 <broker>` dumps every thread's stack** to the log — but ONCE. Driven as a
  sampler it killed the broker on 2026-08-23. It showed 10 threads with Python frames against 16
  in the process, and that gap was the clue: the missing six were native BLAS workers.
* For a real profile, `broker_multi.py`'s own header points at a **replay** under `broker_equiv`,
  which is side-effect-free and needs neither root nor a working ptrace.

RAM and disk were never close: 2.4 GB of 7, and ~0.7 GB/day of logs on 210 GB free.

## 5. The portable build: now optional, still worth doing

cf02 and cf06 are the same part, so `build_nodpdk`'s `ARCH=native` binary runs as-is — which is
why `--help` now exits 0. That removes the blocker but not the fragility: a `native` binary
breaks on any future host that is not this exact CPU, and we have now been bitten by that twice
in two days. Building `-DARCH=x86-64-v2` into its own tree remains cheap insurance and costs the
gather nothing (none of the three configs declares a `cuda*` stage, so `-DUSE_CUDA=OFF` is also
free for the gather — but **not** for the aggregator, which needs CUDA).

## 6. What moved, and what stayed

**On the VM** (13 units, all active, enabled, lingering):

| unit | cost |
|---|---|
| `gnss-broker` | 1.43 cores |
| `gnss-gather` | 0.84 |
| `gnss-aggregator` | 0.76 + the GPU (34%, 1.5 GB of an L40S) |
| `gnss-obs@` ×8 | 0.15 total |
| `gnss-viewer` | 0.02 |

Broker and gather are one machine by construction — the gather's trim-in port is bound to
`127.0.0.1:11061` and the pair exchanges 453 Mbit/s over loopback. The aggregator joining them
put its detector endpoint back on loopback too.

**Still on cf06:** the cube archiver (32.5 GB/h to NFS, and the bring-up ordering rule that it
precedes the nodes), the cube compactor (periodic heavy analysis), and the static viewer. Its
**two L40S are now both idle**, which matters: that is where GPU benchmarking lives
(`CUDA_VISIBLE_DEVICES=1`), and the VM's single GPU is now in production use. Keep the foothold.

## 7. systemd: what the units must encode

Templates are in [`scripts/gnss/systemd/`](../scripts/gnss/systemd/), and are no longer blocked
on anything — the re-provisioned VM runs the binary as-is. They are not installed yet because
that is step 1 of §8, not because they cannot be. `gnss-aggregator.service` exists but is
deliberately left out of `gnss-stack.target` until step 4. Each setting below is a fault we have
already paid for:

- **`Environment=GNSS_PY=/home/kvand/gnss/venv-ft/bin/python` on the broker.** Starting it under
  the GIL 3.12 venv instead of free-threaded 3.14t killed every non-L5 chain for two hours: the
  gather drops its consumer every 15 s, every chain logs `ALL 12 instances stale`, nothing arms.
  L5 keeps working (its fast loop is in-broker), which makes it look survivable. It is not.
- **`LimitNOFILE=65536` on the gather.** It runs at 201 fds and has hit `EMFILE` twice.
  ⚠️ While measuring this I found the **aggregator's soft limit is 1024** with 119 fds open —
  worth raising on cf06 too, independently of this migration.
- **`Restart=on-failure` with backoff** replaces the broker's hand-rolled supervisor loop
  (`broker_restart.sh:140-179`), which already does restart-on-nonzero with fast-crash backoff.
  ⚠️ But a **gather restart wipes every standing trim**, costing a fleet-wide ~8 minute re-arm,
  so its `RestartSec` should be generous and its failures should be visible, not silent.
- **`RequiresMountsFor=/home/kvand /mnt/cs00/data`.** The code, the venvs and the configs are all
  on NFS. A unit that starts before the mounts will fail in a way that looks like a code fault.
- **Logs off `/tmp`.** `/tmp` is cleared on boot and the VM will take unattended-upgrades
  reboots like cf06 does at 03:03. Journald or `/var/log/gnss/` with logrotate; budget
  ~0.7 GB/day.
- **`MemoryAccounting=yes` and a `MemoryMax`.** No swap on this box. A leak should hit a cgroup
  limit and restart one unit, not OOM-kill something else.
- **A `gnss-stack.target`** so the whole thing comes up in order after a reboot — which is the
  real prize here. Today nothing is a systemd unit and every bring-up is by hand.

## 8. Order of work — steps 1 and 2 are DONE (2026-09-16)

Done, during the F-engine site-work window:

1. ✅ **Repointed the telemetry leg.** `telem-host` 10.222.3.6 → 10.222.0.56 in the fleet
   manifest; `search-host` and `cube-host` deliberately unchanged, so the aggregator and the
   archiver stay on cf06. The broker's aggregator endpoint stopped being able to be localhost
   and is now `http://cf06:12050/gps_search`. `gen_fleet --check` green before and after; the
   regen also picked up a fresh EOP table (51 h of headroom, against a 12 h gate).
2. ✅ **Archived cf06's live logs** to `/home/kvand/gnss/logs/` — 331 MB of broker log to 91 MB,
   11 files. `/tmp` is cleared on boot, so this is not optional.
3. ✅ **Stopped broker, gather, viewer and the 8 obs writers on cf06**, consumers first, by PID.
   All went on `SIGTERM`; none needed `-9`. The supervisor was killed before the broker it
   supervises, or it would have respawned it. The aggregator, the cube archiver, the static
   viewer and the compactor were left running and verified still alive.
4. ✅ **Started them on `gnss` as systemd USER units** — 11 units plus the target, all active,
   ports listening, `broker → gather` on loopback and `broker → cf06:12050` both established,
   verified on two polls 20 s apart. A full `restart gnss-stack.target` was exercised and brings
   everything back, which is the behaviour the whole migration was for.

Still to do:

5. ⚠️ **KV restarts the six nodes** — they read `telem-host` once at startup, so until they
   cycle they are still pushing at a cf06 gather that is gone. `node_up.sh <node> restart`.
   Nothing is lost meanwhile: the F-engine is down.
6. **Take the peak-under-load sample** once the F-engine is back (§9), over an hour rather than
   30 s, against the 3.05-core budget.
7. **Then move the aggregator** — the step that takes the box to 70% and costs the spare bench
   GPU. `gnss-aggregator.service` exists and is deliberately not in `gnss-stack.target`.
8. Optionally move the cube archiver, and build the portable tree (§5) as insurance.

### Why user units, and what changes when sudo arrives

There is no passwordless sudo on the VM yet, so system units, `/var/log/gnss` and logrotate were
not available. `loginctl enable-linger` **did** work without root, and that is the part that
matters: user units with linger start at boot and survive logout, so the operational behaviour is
the same. Logs go to `/var/tmp/gnss-logs` (local disk — **not** the NFS home, where 259 MB/h of
appends would meet the fact that `O_APPEND` is not atomic there, and not `/tmp`, which is cleared
on boot).

`scripts/gnss/systemd/install_user_units.sh` derives the user variants from the canonical units
rather than keeping a second copy — three `sed` edits, listed in its header. Converting to system
units later is: install the originals into `/etc/systemd/system`, create `/var/log/gnss`, drop in
`gnss.logrotate`. Nothing else changes.

## 9. Re-measuring

Everything in §1 and §2 came from read-only sampling; nothing was installed or restarted.

```sh
# CPU per process, 30 s window (ps pcpu is a lifetime average -- do not use it)
ssh cf06 'for p in $(pgrep -f "[b]roker_multi|[k]otekan"); do ...  /proc/$p/stat fields 14,15 ... done'
# socket rx per process
ssh cf06 'ss -tinp | grep -A1 "pid=$P," | grep -oE "bytes_received:[0-9]+"'
# disk, the honest counter
ssh cf06 'grep write_bytes /proc/$P/io'
# VM network, without binding anything
ssh gnss 'dd if=<a big file on /mnt/cs00> of=/dev/null bs=4M count=100 iflag=direct'
# is the aggregator really on the GPU
ssh cf06 'nvidia-smi --query-compute-apps=pid,used_memory --format=csv'
# ⚠️ is there DATA? a sample taken with the F-engine down measures the floor, not the load
ssh cf06 'curl -s http://127.0.0.1:12060/status'      # present/n_prn zero => not a sizing basis
```
