# Staged upstream PR plan — `kv/chord-gnss` → `chord`

**Status 2026-09-02.** PR [#1618](https://github.com/kotekan/kotekan/pull/1618) is open as a
draft, base `chord`, head `kv/chord-gnss`, **no description**. It currently carries the whole
branch: **661 files, +220,684 / −143, 1,741 commits, one author.** No maintainer has reviewed
it, which is the expected outcome of asking anyone to read 661 files.

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

### Stage 2 — shared buffer / metadata data path *(4 files, needs the owners)*

* `lib/core/buffer.cpp` — `get_metadata()` now returns a copy of the `shared_ptr` under the
  buffer lock.
* `lib/cuda/NDArrayRingBuffer.hpp` — `set_metadata()` builds a **fresh** object and publishes
  it, instead of allocating slot 0 and mutating in place under a live reader. This is the
  torn-read fix (`9216 = 8×384×3`) behind four autopsied node deaths.
  > ⚠️ **This also restores `check_read_progress()`, which upstream deleted in `f82baaed6`.**
  > A deliberate re-revert must be discussed with that commit's author, not slipped in.
* `lib/cuda/cudaCopyFromRingbuffer.{cpp,hpp}` — descriptor publication becomes a validated
  defer-and-retry rather than an unconditional call.
* `lib/stages/N2Accumulate.cpp` — desync autopsy diagnostics; the existing FATALs are unchanged.

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
6. **Airspy prototype heritage.** `config/airspy_autocorr.yaml` carries a CORS entry for
   `http://gx10.vdl:8080`, the retired prototype host. Decide whether the airspy path is still
   supported or is dead weight to drop.
7. **19 files hardcode `/home/kvand`** paths (mostly bench/test scripts). Cosmetic, but it is
   the kind of thing a reviewer notices in file one.

---

## 5. What this plan is not

It is **not** a claim that the branch is ready. It is a claim about the order in which it
becomes reviewable. Stage 1 could open this week; Stages 2–4 need conversations with the
owners of the code they touch, and those conversations are the actual long pole — not the
GNSS code, which is additive and nobody else's problem.
