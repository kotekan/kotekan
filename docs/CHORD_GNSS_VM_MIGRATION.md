# Moving the GNSS infrastructure to `gnss.site.chord-observatory.ca`

**What this is.** Sizing and a migration plan for putting the GNSS stack on the provided VM
under systemd, long-term. Every number below is measured on the live stack, not estimated; how
to re-measure is at the end.

**The short version.** As re-provisioned — 6 cores of `host` Xeon Gold 5416S on cf02, an L40S
passed through, a 10G uplink — **the whole stack fits, aggregator included**, at ~70% of 6
cores and a third of the RAM. Nothing more needs requesting. The remaining questions are about
sequencing, not capacity: one peak-under-load measurement is still missing, and moving the
aggregator costs the spare GPU we bench on.

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

## 4. Sizing: 6 cores fits, with one measurement still missing

**Everything, including the aggregator, fits in 6 cores at ~70%.** RAM is not close to binding:
2.4 GB of RSS against 7 GB, and the aggregator's alarming 24.9 GB `VmSize` is virtual —
`VmHWM` 823 MB, `VmLck` 86 MB, `VmPin` 0, and the VM runs `vm.overcommit_memory=0`, so the
reservation is free. **Do not ask for more RAM or disk.**

⚠️ **What I cannot tell you yet is the peak.** The with-data column is a single 30 s window taken
before the F-engine went down; I have no peak-under-load sample, and the one thing that would
change the answer is a burst. Two reasons to care on a 6-core box specifically:

- **69 threads** (aggregator 44, gather 14, broker 11) on 6 cores instead of 64. Thread count is
  not itself a problem — they are mostly idle — but scheduling latency becomes a new variable.
- **The gather has a 200 ms frame deadline** (`dropped client fd N -- could not take a frame
  within 200 ms`). On a 64-core host that deadline has enormous slack. At 70% CPU it does not.

So: take a peak sample when the F-engine returns, before committing the aggregator (§8).

## 5. The portable build: now optional, still worth doing

cf02 and cf06 are the same part, so `build_nodpdk`'s `ARCH=native` binary runs as-is — which is
why `--help` now exits 0. That removes the blocker but not the fragility: a `native` binary
breaks on any future host that is not this exact CPU, and we have now been bitten by that twice
in two days. Building `-DARCH=x86-64-v2` into its own tree remains cheap insurance and costs the
gather nothing (none of the three configs declares a `cuda*` stage, so `-DUSE_CUDA=OFF` is also
free for the gather — but **not** for the aggregator, which needs CUDA).

## 6. What moves, what does not

**Moves now — the infrastructure:**

- **broker** + **gather** — and they must stay **together**. The gather's trim-in port is bound
  to `127.0.0.1:11061`, and the broker↔gather link carries 453 Mbit/s; splitting them puts that
  on the wire and breaks the localhost binding.
- **the 8 obs writers** — they poll the broker and write to `/home/kvand/gnss/fixtures/obs/`,
  the same NFS path on the VM. No path changes.

Together that is **3.05 cores of 6 (51%)** and ~1.35 GB. Comfortable.

**Can move, second — the aggregator.** The GPU objection is gone: an L40S is passed through and
verified to run a real kernel, and the aggregator only ever used one GPU (57% of cf06's GPU 0,
2664 MiB of 46 GB). Adding it takes the VM to **4.23 cores of 6 (70%)**. Two things to weigh
first, neither of them a blocker:

- ⚠️ **cf06 has two L40S and the VM has one.** GPU 1 on cf06 is idle and is what we bench on
  (`CUDA_VISIBLE_DEVICES=1` — the #54 work lived there). Move the aggregator and any GPU
  benchmarking either contends with production or has nowhere to go. That is an argument for
  keeping a foothold on cf06, not for keeping the aggregator there.
- The peak-under-load sample in §4 should exist before this step, not after.

**Stays on cf06 regardless:**

- **the cube compactor** — periodic heavy analysis, which is what this VM is explicitly not for.
- **the beam viewers** (`livebeam_server` 8080/8539, `http.server` 8877) — they serve exported
  cube products, so they belong with the cube analysis. They cost 0.02 cores either way.

**Judgement call — the cube archiver.** 0.27 cores, and its 32.5 GB/h goes to NFS, which the VM
does at 9 MB/s without noticing. The argument for moving it is bring-up ordering: it must be up
*before the nodes*, and systemd guarantees that where a hand-run script does not. **My call:
leave it on cf06 for the first cut**, and move it with the aggregator once the VM has a few
weeks behind it.

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

## 8. Order of work

Nothing needs requesting; this is all sequencing now.

1. **Install the units** and start broker + gather + obs writers on the VM **with the fleet
   still pointing at cf06**. Both stacks run; the VM's sees no fleet traffic yet. Confirms the
   units, the mounts, the venv and the log paths with zero blast radius.
2. **Cut over the infrastructure**: repoint the nodes' telemetry at `gnss:11060`, stop the cf06
   broker and gather. Expect the usual re-arm transient — trims are wiped by the gather change.
   Watch `late_frames` and `forced_closes` in `/fleet_trim/get_stats`; the 10G uplink should make
   them *better* than cf06's, not worse.
3. **Take the peak-under-load sample** once the F-engine is back (§9), over an hour rather than
   30 s, and compare against the 3.05-core budget. This is the number §4 is missing.
4. **Then, and only then, move the aggregator** — it is the step that takes the box to 70% and
   the one that costs the spare bench GPU. Rebuild is not needed (same silicon), but it needs
   CUDA, so it cannot use a `-DUSE_CUDA=OFF` tree.
5. **Reboot the VM deliberately** and confirm the whole stack comes back with no operator. That
   is the actual prize: today nothing is a systemd unit and every bring-up is by hand.
6. Optionally move the cube archiver, and build the portable tree (§5) as insurance.

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
