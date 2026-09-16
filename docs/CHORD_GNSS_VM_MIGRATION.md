# Moving the GNSS infrastructure to `gnss.site.chord-observatory.ca`

**What this is.** Sizing and a migration plan for putting the broker, the gather and the obs
writers on the provided VM under systemd, long-term. Every number below is measured on the live
stack, not estimated; how to re-measure is at the end.

**The short version.** RAM, disk and network are comfortable — the network is *much* better than
cf06's. CPU is not: the stack needs ~3.1 cores and the VM has 2. And there is a hard blocker
that has nothing to do with capacity: **the VM's CPU model is so old that the kotekan binary
dies on `SIGILL`**. Ask for the CPU model to be changed at the same time as the cores.

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

## 2. The VM as provided

| | value | verdict |
|---|---|---|
| vCPU | **2** (QEMU Virtual CPU 2.5+) | ❌ need ~4 |
| CPU features | `sse4_2`, `popcnt`, `aes` — **no avx, avx2, avx512, fma, bmi2** | ❌ see §3 |
| RAM | 7 GB, **no swap** | ✅ (1.35 GB needed) |
| Disk | 249 G, 210 G free, local LVM | ✅ with rotation |
| Network | **≥4.3 Gbit/s** measured (536 MB/s read off `cs00` NFS, `iflag=direct`) | ✅✅ |
| `/mnt/cs00/data` | mounted, NFS4 | ✅ |
| `/home/kvand` | mounted, NFS3 from `nfs-home` | ✅ code and venvs are already visible |
| OS | Ubuntu 24.04.5, systemd 255 | ✅ same as cf06 |
| Reachability | cx19, cx43 and cf06 all connect inbound to `gnss:11060`; VM reaches all of them | ✅ no firewall in the way |

### The network is the headline improvement

cf06 is on 1 GbE and it has cost us real diagnosis time: 22% late frames traced to TCP loss,
and an ingress path that drops SYN-ACKs so every fresh connect from cf06 stalls 1/2/5 s. The
gather alone pulls 369 Mbit/s. **Moving the gather here retires that class of fault** — the VM
has roughly four times the usable bandwidth.

## 3. ⚠️ The blocker: the CPU model, not the core count

```
$ ssh gnss /home/kvand/gnss/kotekan/build_nodpdk/kotekan/kotekan --help
Illegal instruction (core dumped)          # exit 132
```

cf06 builds with `ARCH=native` on a Xeon Gold 5416S, which means AVX-512. The VM's QEMU CPU
model predates AVX entirely. Two independent fixes, and **both are worth doing**:

1. **Ask for a better CPU model** — `host-passthrough`, or any modern named model. This costs
   nothing, is the smaller request of the two, and it also closes the 12× gap in §4.
2. **Build a portable binary**: `cmake -DARCH=x86-64-v2 -DUSE_CUDA=OFF` (`ARCH` defaults to
   `native` in `cmake/Toolchain.cmake:207`; `USE_CUDA` is an AUTO tristate). Build it **on
   cf06**, not on the VM — same OS, and the VM has 2 cores. This is needed regardless: a
   `native` binary breaks on every future host move, and we have now been bitten by it twice.

None of the three cf06 kotekan configs (`gather`, `cubearch`, `agg6_cuda`) declares a `cuda*`
stage, so `USE_CUDA=OFF` costs the gather nothing.

## 4. Per-core speed: the same, except where it isn't

Same interpreter off the same NFS venv, best of three, exercising what the broker actually does:

| kernel | cf06 | gnss VM | ratio |
|---|---|---|---|
| Python dict/tuple churn | 0.0475 s | 0.0469 s | **1.0×** |
| `json.loads` | 0.190 s | 0.185 s | **1.0×** |
| `np.fft.rfft` | 0.119 s | 0.118 s | **1.0×** |
| float64 matmul (level-3 BLAS) | 0.0104 s | 0.1222 s | **11.8× slower** |

⚠️ **My first run said the VM was 2× faster at scalar work. That was cf06 load, not the VM** —
cf06's own number swung 0.0928 → 0.0475 between back-to-back runs while the VM was stable to
3%. Take minimums, and do not compare a loaded host to an idle one.

The 11.8× is OpenBLAS falling off AVX-512 onto SSE. It does not touch the broker:
`broker_multi.py` and `python/scripts/gnss/gnss_broker/*.py` contain **zero** uses of
`np.dot`/`matmul`/`linalg`/`einsum`. So the 3.05-core figure transfers about 1:1.

⚠️ **The gather is the untested half.** It is C++, and rebuilding it without AVX will make its
vectorised loops slower by an unknown factor. Budget for it, then measure (§8).

## 5. Recommended shape

| | ask for | why |
|---|---|---|
| vCPU | **4 minimum, 6 comfortable** | 3.05 measured + the unknown gather penalty + headroom for acquisition bursts. 4 leaves ~25% margin, which is thin if the gather slows down. |
| CPU model | **host-passthrough or a modern named model** | removes the `SIGILL` and the 11.8× BLAS gap |
| RAM | **8 GB is already right** — keep it | 1.35 GB in use; the rest is page cache for two NFS mounts. Do not shrink it to 4 GB: there is no swap, so an OOM kill is a fleet outage. |
| Disk | as provided | ~0.7 GB/day of logs with rotation |

**Do not ask for more RAM or disk.** They are not the constraint.

## 6. What moves, what does not

**Moves — the infrastructure:**

- **broker** + **gather** — and they must stay **together**. The gather's trim-in port is bound
  to `127.0.0.1:11061`, and the broker↔gather link carries 453 Mbit/s; splitting them puts that
  on the wire and breaks the localhost binding.
- **the 8 obs writers** — they poll the broker, and they write to
  `/home/kvand/gnss/fixtures/obs/`, which is the same NFS path on the VM. No path changes.

**Stays on cf06:**

- **the aggregator** — yes, as you thought, and it is confirmed rather than assumed: it holds
  **2664 MiB of GPU memory** with 25 open `/dev/nvidia*` fds, and its config carries
  `use_cuda_acquire: true`. A CPU acquire path does exist (it is the reference the #54 work
  fixed), but it is the expensive stage and the VM has no vector units — a bad trade twice over.
- **the cube compactor** — periodic heavy analysis, exactly what you said should live elsewhere.

**Either way, decide deliberately:**

- **the cube archiver** — 0.27 cores, and its 32.5 GB/h goes to NFS, which the VM does at 9 MB/s
  without noticing. The argument for moving it is the bring-up ordering: it must be up *before
  the nodes*, and systemd would guarantee that where a hand-run script does not. The argument
  against is that it is the recording leg for an analysis product. **My call: leave it on cf06
  for the first cut**, move it once the VM has proven itself over a few weeks.
- **the beam viewers** (`livebeam_server` 8080/8539, `http.server` 8877) — you said elsewhere,
  and they cost nothing either way (0.02 cores). They serve exported cube products, so they
  belong wherever the cube analysis lands.

## 7. systemd: what the units must encode

Templates are in [`scripts/gnss/systemd/`](../scripts/gnss/systemd/). They are **not installed**
— they need the resized VM and a portable binary first. Each of these is a fault we have already
paid for:

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

1. Request **4–6 vCPU and a modern CPU model**. Nothing else can be tested until the second one
   lands — the binary will not execute.
2. Build the portable kotekan **on cf06**: `-DARCH=x86-64-v2 -DUSE_CUDA=OFF`, into its own tree
   (not `build_nodpdk`, which the live cf06 gather and aggregator run out of).
3. **Measure the gather on the VM** against the 1.24-core baseline, from a replayed telemetry
   capture rather than the live fleet (`fleetdll --trim-in` against a `:11061` capture). This is
   the one number in this document that is extrapolated rather than measured.
4. Install the units, start broker + gather + obs writers on the VM with the fleet still
   pointing at cf06, and compare side by side before cutting over.
5. Cut over: repoint the nodes' telemetry at `gnss:11060`, stop the cf06 broker and gather.
   Expect the usual re-arm transient — trims are wiped by the gather change.
6. Reboot the VM deliberately, and confirm the whole stack comes back with no operator.

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
```
