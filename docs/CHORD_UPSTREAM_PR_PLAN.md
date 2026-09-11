# Staged upstream PR plan — `kv/chord-gnss` → `chord`

**Status 2026-09-08.** Stage 1 has **landed**: PR
[#1640](https://github.com/kotekan/kotekan/pull/1640) was approved and squash-merged into
`chord` as `a6ce5ac49`. Two fixes split out of the later stages are open and approved-or-clean:
[#1638](https://github.com/kotekan/kotekan/pull/1638) (caller-side publish-then-mutate, 7
`lib/cuda` stages) and [#1642](https://github.com/kotekan/kotekan/pull/1642)
(`cudaCopyFromRingbuffer` reads its own snapshot). **Stage 2 is in progress** — see the split
recorded under its heading below.

PR [#1618](https://github.com/kotekan/kotekan/pull/1618) remains open as a draft, base `chord`,
head `kv/chord-gnss`, **no description**, carrying the whole branch: **661 files, +220,684 /
−143, 1,741 commits, one author.** No maintainer has reviewed it, which is the expected outcome
of asking anyone to read 661 files.

This document is the plan to make it landable. It is versioned here rather than in the PR
body because it has to stay in step with the branch as stages land.

---

## 1. The number that matters is 44, not 661

The diff is **almost purely additive**: only 25 files carry any deletion at all. Added files
are cheap for a reviewer — they can be read in isolation and cannot break anything that exists
today. **Modified files are where all the risk and all the review effort live**, and there are
only 44 of them.

So the staging is ordered by *what a modification can break*, not by subsystem tidiness:

| Tier | What it touches | Files | Who has to care |
|---|---|---|---|
| A | Upstream code, zero GNSS content | ~10 | any CHORD user |
| B | Shared buffer/metadata data path | 4 | correlator + PL-mask owners |
| C | GPU scheduling semantics | 4 | anyone with a multi-stream pipeline |
| D | DPDK capture path | 4 | production N² receive |
| E | New GNSS code (added only) | ~600 | GNSS only |

A reviewer who accepts Tier A in one sitting has lost nothing if Tier E is still under
discussion. That is the whole point of the split.

---

## 2. Stage 0 — hygiene (DONE 2026-09-02)

Removed from the PR because it is not source:

* `scripts/gnss/seedchk` — a committed **218 KB ELF binary** beside its own `.cpp`.
* `scripts/gnss/{phibits,phishare,phisharegpu,wavebench}` — tracked **symlinks** to
  `.cf06`/`.cx43`/`.cx19` builds. Upstream would have received dangling links to hosts that do
  not exist. `git ls-files -s | awk '$1=="120000"'` now returns **zero** repo-wide, which is
  the check that says this class is finished rather than reduced.
* Six generated matplotlib PNGs (~1.5 MB), zero references, reproducible from
  `gps_cn0_map.py`.

Dead generator scratch:

* **`config/generated/live_*_gpu.yaml` (9 files, 163 KB)** — output of `gen_band_config.py
  --check`. A basename grep says they are referenced; they are not. Every hit is the
  *deployed twin* at `config/live_*.yaml`, and `gen_3band_config.py:67` (`HERE = "config/"`)
  is the line that settles it. The two genuine references were provenance comments, repointed
  to the byte-identical deployed copy.

And one real defect, which was ours and not upstream's to cause:

* **`config/crs_full_packet_capture.yaml` reverted.** An early sweeping commit repointed this
  *upstream* tool's metadata pool at `GnssChanMetadata` — a type that **does not exist on
  `chord`** — so the config would fail at startup for its own users. Nothing in our tree even
  references the file. Restored byte-for-byte.

> ⚠️ **The general rule this exposed.** A MODIFIED upstream file is a different risk class from
> an ADDED one. Each of the 44 needs a reason that survives being read by the file's owner.

---

## 3. The stages

### Stage 1 — upstream fixes with no GNSS in them *(~400 lines, land first)*

Every one of these helps a CHORD user who will never run a GNSS chain. None mentions a GNSS
symbol. This stage exists so the first review is a pleasant one.

* `kotekan/kotekan.cpp` — `--check-config` (static validation: unknown stage types, duplicate
  names, dangling metadata pools) and `--dry-run` (build the pipeline and tear it down). Both
  exit non-zero on failure. **Carries `lib/core/restServer.cpp`**, whose affinity guard is what
  stops `--dry-run` segfaulting on a never-started server.
* `kotekan/CMakeLists.txt` — blosc1-vs-blosc2 link fix. Identical declared asdf-cxx 8.0.0
  installs on cx19 and cf06 resolve different symbols, undetectable from the `.pc` file.
* `lib/utils/LinearAlgebra.hpp` — `to_blaze_herm()` takes `std::real()` of the diagonal and
  warns instead of letting blaze throw. **Consumers are `EigenN2Iter` and `EigenVisIter` —
  pure CHORD science code, zero GNSS.**
* `lib/stages/valve.cpp` — adds the `passed_frames` counter (the denominator) and throttles the
  "output buffer full" WARN.
* `lib/stages/bufferRecv.cpp` — `SO_REUSEADDR` unconditionally, so a `drop_frames` receiver can
  restart inside TIME_WAIT.
* `lib/stages/rawFileRead.*`, `rawFileWrite.*` — two opt-in config keys, defaults unchanged.
* `docs/bfmask_deadlock_upstream_note.md` — already written for Jim and Andre.

### Stage 2 — shared buffer / metadata data path *(CLOSED 2026-09-11, superseded upstream)*

**Nothing here is left to upstream. Every piece landed upstream on someone else's PR, and the
last survivor turned out to be defending a hazard that no longer exists.**

| piece | what closed it |
|---|---|
| `NDArrayRingBuffer::set_metadata` build-then-publish | **#1643** (`b38555c11`, on `develop`) — eschnett sets ring metadata ONCE, on the first frame. Removing the per-frame republish beats synchronising it. |
| `cudaCopyFromRingbuffer` descriptor gate | **#1647** (`1af1099d4`, jbmertens, 2026-09-10) — reads the locked snapshot `out_meta` instead of the live slot-0 object, and replaces the `instance_num == 0` branch with a `FATAL_ERROR`. Better than what we had. |
| `check_read_progress()` restoration | dropped 2026-09-09; no callers, and at the one site tried it reduces to a tautology. |
| `buffer.cpp` — `get_metadata()` under the lock | **WITHDRAWN 2026-09-11, and dropped from our tree.** See below. |

⚠️ **WHY THE `get_metadata` LOCK WAS WITHDRAWN — the premise expired under it.** That lock
(`7c025c5d0`, 08-31) was the companion to OUR `set_metadata` rewrite in the same commit: our
ring metadata was rebuilt and republished EVERY FRAME, so an unlocked reader really was racing
a live per-frame write. #1643 deleted the writer. After it, ring metadata is written once, on
frame 0, by instance 0, before any reader is gated in — readers block in
`wait_and_claim_readable`, and that ordering is established by the buffer's own mutex and
condvar, so the single write happens-before every read. For non-ring buffers the frame
handshake already orders it. Even our own off-handshake reader, `peek_newest_full_frame`, takes
`buffer_lock` itself and never goes through `get_metadata()`.
**So the race is not instantiable, and by our own rule we do not raise hazards we cannot
instantiate.** What would have remained is a consistency argument — every other public accessor
of `metadata[]` (`set_metadata`, `pass_metadata`, `copy_metadata`,
`allocate_new_metadata_object`) takes the lock and this one does not — which is not worth
upstream goodwill. The lock is now removed from our tree too, so `get_metadata` is
byte-identical to `develop` and no future merge has to re-resolve it.

**The lesson, since it cost a stage:** a fix carried across a merge keeps its code but not
necessarily its justification. When upstream changes the mechanism you were defending against,
re-derive whether the defence is still reachable before shipping it.

**N2Accumulate desync survival** was never really Stage 2; it is our own feature (268 lines
from `develop`). It still carries the START-ordering bug and the incompatible default 4, and it
goes later as its own PR. See §4.


### Stage 3 — GPU scheduling *(the highest-risk shared change; goes alone)*

* `lib/cuda/cudaDeviceInterface.{hpp,cpp}` — **removes the public device-wide
  `gpu_command_mutex`** in favour of a 64-slot per-stream array. Ten pipelines per GPU were
  serialising command queuing across blocking driver calls, wedging the whole GPU.
* `lib/cuda/cudaProcess.{hpp,cpp}` — per-stream locking, and a real multi-stream end-of-frame
  join that **closes upstream's own `// TODO, this should wait on the last event from every
  stream!`**. This changes completion semantics for any multi-stream pipeline.
* `lib/cuda/cudaCommand.cpp` — new `cuda_stream_base` key; **default 0 preserves existing
  behaviour byte-for-byte.**

### Stage 4 — DPDK capture resilience *(largest operational blast radius)*

* `lib/dpdk/crs16BoardCaptureWorker.hpp` — per-stream FPGA-seq monotonicity (the old check
  compared different boards to each other), **a `FATAL_ERROR` demoted to a counted WARN**,
  three per-packet log branches throttled (unthrottled logging filled a 3.5 TB root fs), and an
  opt-in window resync.
* `lib/dpdk/dpdkCore.{cpp,hpp}` — worker-health metrics and a FATAL when workers are missing.
  > ⚠️ **`exit_on_worker_failure` defaults to `true`** — the one modified-shared-code default
  > that changes behaviour in the non-conservative direction. A node that silently lost a
  > worker now stops instead of limping. Defensible, documented, escape hatch present — but it
  > must be a maintainer's conscious decision, not a discovery.
* `lib/dpdk/FramePrefetchService.hpp` — wall-clock-vs-seq axis watchdog.

### Stage 5 — GNSS foundation, no framework *(~8k lines, trivially reviewable)*

The 32 signal code generators, the pure value-type headers, and `lib/stages/pfbPrototype.*`.
These have **zero project includes** — C++ stdlib only — and `tests/boost/` already compiles
them directly rather than linking `kotekan_stages`, which is an existing working proof that
the tier builds with no FFTW, no CUDA and no framework. Ship the boost tests in this stage.

### Stage 6 — `GnssChanMetadata` + build wiring *(small, unlocks everything after)*

`lib/metadata/GnssChanMetadata.*`, the `metadataFactory.cpp` pool branch, and the CMakeLists
edits. 28 of the 104 GNSS sources depend on this header; without the factory branch every GNSS
config throws at startup.

### Stage 7 — CPU/FFTW GNSS chain

The channelized despread/replica/acquire/search/tracker set, plus the `fftwEngine` and
`airspyInput` modifications they require (`ensure_frame_desc`, the optional PFB, the bounded
`/adcstat` wait, the stream watchdog).

### Stage 8 — CUDA GNSS path + `external/n2k_dual`

The four `.cu` kernels, `cudaCorrelatorDual`, and the `n2k_dual` clone.
**`external/n2k` is byte-untouched and must stay so** — verified: `git diff -- external/n2k`
is empty.

### Stage 9 — broker, viewer, tooling, configs, docs

The Python broker package, the js_viewer panels, `scripts/gnss/`, `config/`, `docs/`. Largest
by line count, lowest by risk — none of it compiles into kotekan.

---

## 4. Decisions to settle before the later stages

1. **`config/generated/` is 50,345 lines — 23% of the entire diff.** Six near-identical
   ~5,700-line node configs, fully reproducible from `gen_chord_gnss_config.py` +
   `config/gnss/*.j2`, and already gated by `gen_fleet.py --check`. Shipping machine output
   upstream is a real question. *Recommendation:* ship the generator, the j2 templates and the
   manifest; consider one example node rather than six.
2. **The acceptance gate cannot run upstream.** Six of `gate.sh`'s seven fixtures are 38–100 MB
   transcripts living **outside git** at `/home/kvand/gnss/fixtures/` (NFS), with only their
   `.digest` committed — a deliberate, documented choice. Only `broker_fake_l5.jsonl` is
   in-repo. A reviewer cannot reproduce "7/7 EQUIVALENT". Either say so plainly in the PR or
   make one on-sky arm fetchable.
3. **Seven citations to `docs/gnss_gpu_migration.md`**, in shipped `CMakeLists.txt` comments and
   GNSS headers. The memo was deliberately moved out of the repo in `3f0ffc3b1` and now lives
   at `airspy_docs/`. Either bring it back (12 KB) or mark the citations external.
4. **`python/scripts/gnss/gps_distributed_broker.py` (+2,818)** appears superseded by the
   `gnss_broker/` package. Confirm before shipping both.
5. **`lib/cuda/benches/chordShapeBench.cu` (+253)** is referenced by no build file or script.
6. **⚠️ THE BIG ONE — is the gx10 airspy prototype retired, and does it still pull this
   branch?** This single question gates roughly **600 KB** of `config/`: `gnss_node.yaml`
   (119 KB), `WIRING.md`, the four `gen_*_config.py` airspy generators, every top-level
   `live_*.yaml`, `run_live.sh` (58 KB), `run_band.sh`, `run_3band.sh`, `bandctl.sh`,
   `night_batch.sh`, and the `peel_*` / `replay_*` / `capture_*` benches.

   *Evidence it is still live:* the prototype checkout at `/home/lwlab/airspy_gps/kotekan`
   **shares this git history** — `config/binomial_trial.sh` even carries a scratch path from
   that host — so deleting these here removes the prototype's own launchers on its next pull.
   The family was maintained as recently as 2026-08-07.
   *Evidence it is retired:* the prototype is documented as a different repo and instrument,
   and `airspy_docs/buglist.md`'s newest entry is 2026-07-28.

   ⚠️ **And one trap that makes this more than a tidiness question.** `config/run_live.sh` is
   the ONLY non-definition user of about ten broker flags (`--xband-combiner`,
   `--coast-to-horizon`, `--adc-stage`, `--xband-seed`, `--xband-lo-dongle`, …). The `_FROZEN`
   sweep in `gnss_broker/cli.py` freezes a flag on the premise *"not set in the production
   config, in any launch script, or in any gate or fixture in this repo"* — so **deleting
   `run_live.sh` silently makes those flags eligible for freezing.** A config deletion would
   quietly change the tuning contract. Whatever the answer, `run_live.sh` needs the frozen-flag
   audit re-run, not just a `git rm`.

   Held pending that answer: five prototype-host shell scripts that hardcode
   `/home/lwlab/airspy_gps/kotekan` and cannot execute on CHORD (`binomial_trial.sh`,
   `replay_bench_leg.sh`, `replay_l1gps_leg.sh`, `replay_l1bds_leg.sh`, `run_trim_bench.sh`) —
   the repo's own `scripts/gnss/fixtures/README.md` already names three of them as a known trap.

7. **`config/base/live_config_20260730.json`** is superseded as the fleet base, but
   `gather_up.sh:42` names it as the input that `generated/chord_gnss_gather.yaml` — a LIVE
   config — was built from. Confirm that gather config is reproducible without it before
   deleting.
8. **19 files hardcode `/home/kvand`** paths (mostly bench/test scripts). Cosmetic, but it is
   the kind of thing a reviewer notices in file one.

---

## 5. What this plan is not

It is **not** a claim that the branch is ready. It is a claim about the order in which it
becomes reviewable. Stage 1 could open this week; Stages 2–4 need conversations with the
owners of the code they touch, and those conversations are the actual long pole — not the
GNSS code, which is additive and nobody else's problem.
