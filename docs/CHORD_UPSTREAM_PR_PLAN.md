# Staged upstream PR plan — `kv/chord-gnss` → `develop`

**Status 2026-10-08.** Upstream PRs now target `develop`; `chord` is rebuilt by Jim as
`develop` plus selected open PRs ("Sync chord: develop <sha> + PRs ..."), so a PR reaches
`chord` at his next sync without bypassing review.

| PR | Stage | State |
|---|---|---|
| [#1640](https://github.com/kotekan/kotekan/pull/1640) | 1, upstream fixes | merged (to `chord`, since folded into `develop`) |
| [#1642](https://github.com/kotekan/kotekan/pull/1642), #1643, #1647, [#1714](https://github.com/kotekan/kotekan/pull/1714) | 2, buffer/metadata | closed: done upstream |
| [#1699](https://github.com/kotekan/kotekan/pull/1699), [#1713](https://github.com/kotekan/kotekan/pull/1713) | off-plan fixes | merged |
| [#1675](https://github.com/kotekan/kotekan/pull/1675) | 3, GPU scheduling | open; Jim approved, waiting on Andre and Erik |
| [#1750](https://github.com/kotekan/kotekan/pull/1750) | 4, DPDK capture | open; running on all six CHORD GNSS nodes |
| — | 5–9, GNSS | not started; cleanup round 2 done (below) |

Against `develop`'s merge base (`3bfbba126`) the branch is **682 files, +201,512 / −308**:
631 added, 51 modified. The draft [#1618](https://github.com/kotekan/kotekan/pull/1618)
(the whole branch, no description) is superseded by this series.

This document is the plan to make it landable. It is versioned here rather than in the PR
body because it has to stay in step with the branch as stages land.

---

## 1. The number that matters is 51, not 682

The diff is **almost purely additive**: only 25 files carry any deletion at all. Added files
are cheap for a reviewer — they can be read in isolation and cannot break anything that exists
today. **Modified files are where all the risk and all the review effort live**, and there are
only 51 of them (44 when this plan was written; stages 3 and 4 account for most of the rest).

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

### Cleanup round 2 (2026-10-08)

* **Generated configs untracked.** `config/generated/` (21 files, ~65k lines) and the six
  `config/gnss/gnss_vars_<node>.j2` are generator output; they stay on disk for the services
  and are gitignored. `config/gnss/example_chord_gnss_cx51_multi.yaml` is one node's output.
  Every removed file is reproducible: node configs via `gen_fleet.py`, the gather and cube
  archive from the recipe in their headers, the aggregator from the recipe in `agg_up.sh`
  (two generator flags were added so the hand-patched live aggregator config is reproducible).
* **The airspy prototype removed.** 49 files under `config/` (its configs, generators and
  launchers) that nothing in the CHORD system reads; they were already stale. Tag
  `airspy-prototype-final` keeps them, including the only GLONASS chain configs and the
  closed-carrier-loop tuning in `run_live.sh`.
* **Stale pointers.** Citations of the out-of-repo `gnss_gpu_migration.md` memo, the unbuilt
  `lib/cuda/benches/chordShapeBench.cu`, and `docs/CHORD_HANDOFF.md` (a July orientation note).

**Left for the review / cruft pass** — code the airspy removal stranded: the stages
`GnssChannelizedTracker`, `GnssVoltagePeel`, `GnssQuantize44`, `GnssBeamCube`,
`GnssSubbandSplit`, `GnssChannelGather` and the command `cudaGnssTrack` (no live config uses
them, but live stages include their headers, so removal needs builds); our +294 lines in
`airspyInput` and +99 in `fftwEngine`; ~15 Python/shell tools reached only from the removed
launchers; and 44 broker flags whose only setter was an airspy launcher, with the features
behind them. ⚠️ Removing those launchers makes the 44 flags eligible for the `_FROZEN` sweep;
retire each flag with its feature instead, and never freeze `dr-constellation` or
`nh-overlay-len` (the `--signal` implied-value table overwrites them after `_FROZEN`).

---

## 3. The stages

### Stage 1 — upstream fixes with no GNSS in them — merged as #1640

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

**Not yet proposed** (small, no GNSS; each to be checked against `develop` before a PR):
`LinearAlgebra::to_blaze_herm` real diagonal (listed above but not in #1640), `bufferRecv`
`allow_short_frames`, `bufferSend` null-metadata fix and opt-in `SO_MAX_PACING_RATE`,
`cpuMonitor` reaping its tracking thread before the stages it reads, and `rawFileWrite`
`create_base_dir` / `continue_numbering` (the cube archiver needs them).

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


### Stage 3 — GPU scheduling — [#1675](https://github.com/kotekan/kotekan/pull/1675), open

Per-stream command-queuing locks instead of the device-wide `gpu_command_mutex`;
`cuda_stream_base` as a pipeline index (streams `3*base+0/1/2`, default 0 = existing
behaviour); a frame signalled only after every stream it used finishes (closes upstream's
`TODO` in `queue_commands`); and one owner per named GPU memory region unless both stages
declare the share. The rules are in AGENTS.md's "GPU stages" section.

### Stage 4 — DPDK capture — [#1750](https://github.com/kotekan/kotekan/pull/1750), open

`dpdkCore` catches `FatalError` on the lcores (a throw there was `std::terminate`);
`crs16BoardCaptureWorker` stops kotekan on a packet behind its active frames or 128+ frames
ahead (returning -1 only ended the worker, leaving its port's shared frames unfinished),
advances the frames toward a packet ahead of them after a downstream stall, and records each
packet's receipt bit in the frame it was copied into. The earlier five-piece plan (throttled
logs, per-stream seq check, worker-health metrics, axis watchdogs, opt-in resync) was dropped
in favour of this: with the daemon restarting on FATAL, none of it was needed.

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

The channelized despread/replica/acquire/search set. ⚠️ With the airspy prototype gone, our
`fftwEngine` and `airspyInput` modifications serve no CHORD configuration; the cruft pass
decides whether they revert to upstream rather than ship here.

### Stage 8 — CUDA GNSS path + `external/n2k_dual`

The four `.cu` kernels, `cudaCorrelatorDual`, and the `n2k_dual` clone.
**`external/n2k` is byte-untouched and must stay so** — verified: `git diff -- external/n2k`
is empty.

### Stage 9 — broker, viewer, tooling, configs, docs

The Python broker package, the js_viewer panels, `scripts/gnss/`, `config/`, `docs/`. Largest
by line count, lowest by risk — none of it compiles into kotekan.

---

## 4. Decisions

1. **Generated configs** — RESOLVED 2026-10-08: untracked; ship the generator, templates,
   manifest and one example (see cleanup round 2).
2. **The acceptance gate cannot run upstream.** Six of `gate.sh`'s seven fixtures are 38–100 MB
   transcripts outside git (`/home/kvand/gnss/fixtures/`, NFS), with only their `.digest`
   committed; several tests also read fixtures from there. OPEN: say so plainly in the PR, or
   make one on-sky arm fetchable.
3. **Citations of `docs/gnss_gpu_migration.md`** — RESOLVED 2026-10-08: dropped.
4. **`gps_distributed_broker.py`** — RESOLVED: not superseded. It is the broker's main module
   (`scripts/gnss/broker_multi.py` imports it); `gnss_broker/` holds parts split out of it.
5. **`lib/cuda/benches/chordShapeBench.cu`** — RESOLVED 2026-10-08: removed.
6. **The airspy prototype** — RESOLVED 2026-10-08: removed (tag `airspy-prototype-final`); no
   sign the prototype has pulled this branch since 2026-08-07. Stranded code is listed under
   cleanup round 2.
7. **`config/base/live_config_20260730.json`** — RESOLVED: keep. It is the base for the gather,
   aggregator and cube-archive recipes.
8. **Hardcoded `/home/kvand` paths** — OPEN, and larger than first counted: 78 files. Most are
   our deployment scripts, systemd units and runbooks, which raises the real question for
   stage 9 — which of `scripts/gnss/` and `docs/` belongs upstream at all. ~37 are code or test
   defaults pointing at out-of-repo fixtures and data, tied to item 2.

---

## 6. Cleanup round 3 — the plan (2026-10-08)

Six read-only reviewers at `9f555d4be`; condensed notes with file:line detail in
`docs/CHORD_CLEANUP_ROUND3_NOTES.md`. About 37k of the ~201k added lines go with high
confidence and no live behaviour change. Verify every batch with: CUDA `-Werror` build +
boost tests on cf05, `kotekan --check-config` on the nine live configs, `gen_fleet --check`
byte-identical, broker `gnss_broker/test_*.py` + `selftest.py` under venv-ft (niced), and for
broker changes `broker_equiv.py check` on cf05 (never `gate.sh` beside the live stack).
⚠️ Run heavy scans on cf05, not the gnss VM (six reviewers pushed its load to ~33).

**A. Fix first — blocks every upstream PR's CI.**
1. CI-flag build: `lib/testing/gpuSimulateRFISK.cpp:565` sign-compare; unused `cps`
   (GnssChannelizedSearch.cpp) and `ctl_energy` (cudaGnssInject.cpp).
2. CPU-only build (FFTW on, CUDA off): move `cudaGnssChordTrack.cpp`, `cudaGnssInject.cpp` into
   the `USE_CUDA` block; fix `_grid_bin_warns` / `gpu_ok` under `!GNSS_CUDA` in the search.
3. Pytests CI runs: delete `tests/test_gnss_channelized_correlator.py` and
   `test_gnss_record_collector.py` (stages gone); fix `test_gps_navdecode.py`'s sys.path.

**B. Delete, high confidence**, in batches (notes have the lists):
1. Stranded airspy C++: the six stages, `cudaGnssTrack`, `GnssTrackState`, `GpsReplicaCorrelator`
   (+ its two pytests), dead kernel launchers; revert `airspyInput`, `fftwEngine`,
   `airspyFrameDesc`, `config/airspy_autocorr.yaml` to `3bfbba126` (after broker B5f). ~6.5k.
2. Dead code inside live stages (reviewer 5's high-confidence list, incl. the
   `cudaGnssChordTrack` command, combiner overlay/navwipe/nh-assist/bit_export, bench-only kernel
   variants, ms-split acquire, `gnssBroker`, `bufferDedup`). ~4.9k.
3. `python/scripts`: ~80 files (airspy-era tools, closed investigations, `diag/`, `fullband`,
   copy-of-logic tests). ~11k.
4. `scripts/gnss`: 51 files (closed investigations, dead-endpoint probes, non-compiling
   `n2skyab.cpp`, the dead shared-phi feature). Repoint provenance citations to commits first.
   ~8.9k.
5. `docs`: 17 finished plans/journals/snapshots, after moving the useful bits into the runbook
   and repointing ~160 code citations (with the comment rewrite). ~7.4k.

**C. KV decisions pending.**
1. Retire the nav-decode set (26 files, 7.9k; off in production but imported at startup)?
2. Broker features set only by airspy launchers (xband, CL sibling, nh-assist, state-consume,
   coast-to-horizon, watchdog, carrier-loop knobs): retire each with its flag (B1 guard first:
   `SIGNAL_IMPLIED` names can never be frozen), each its own commit with `broker_equiv`.
3. Code generators: drop the four L1-band ones (above CHORD's band)? GLONASS (in band, no chain)?
4. `scripts/gnss/site/` boundary for our deployment tooling (needs a planned restart: units,
   crontabs, cf06 `cubecompact_loop`).
5. `n2k_dual`: propose the two-input extension to n2k upstream, or ship the clone in stage 8?
6. `dop-continuous`: retire with the others?
7. Medium-confidence C++ (carrier-phase A/B arm #55, unused assembler REST levers, debug env
   vars, FDMA offset).
8. Take develop's copies at the next merge (NDArrayRingBuffer, `chord_pathfinder_recv.j2`,
   cpuMonitor, TransposeBasebandArray, two julia `.out`); revert LinearAlgebra; re-derive the
   premise of the cudaCopyFromRingbuffer gate and the correlators' build-then-publish (kept
   deliberately on 10-01).

**D. Operations found along the way.** Logs are not rotated (`/var/tmp/gnss-logs` 44 GB in 22
days, ~2 GB/day, 161 GB free; `systemd/gnss.logrotate` targets a missing dir and is not
installed). `gate.sh` cannot pass (on-sky digests moved, `holds` nondeterministic): re-bless or
retire. `test_deep_gate` 7/8 and `test_rrate_state` 1/29 fail; 41 `python/scripts/gnss` tests
have no runner. `live_element_gate.py` broken since 10-02. `gnss_transit_arcs.py` uses the
pre-#99 station (155 m off). Viewer: layout key v7, reset clears v6.

**E. Per stage, as each PR is cut.** Comment rewrite to AGENTS.md style (ratio 0.49 vs upstream
0.28; 245 emoji, 264 dates, 230 task refs; ~1.5-2 weeks total); sphinx stubs per stage + one
`docs/sphinx/user/gnss.rst`; the test set per stage; `GNSS_FIXTURES` env default with skip.

**F. A separate airspy PR** (KV, 2026-10-08: the airspy/fftw changes are worth upstreaming).
Source: tag `airspy-prototype-final` — `airspyInput.{cpp,hpp}` (+~294: bounded `/adcstat` wait,
PFB mode, sample_seq, stream watchdog, `ensure_frame_desc`), `fftwEngine.{cpp,hpp}` (+~99),
`airspyFrameDesc.hpp`, `config/airspy_autocorr.yaml`. Base `develop`, its own worktree, so it
does not wait on or collide with this cleanup (B1 reverts those files here; when the PR merges,
the next develop merge brings them back cleanly). Must: drop the dependency on
`GnssChanMetadata` (stage 6) or land after it; fix `fftwEngine.cpp:181-188`, which dereferences
`get_gnss_chan_metadata()` for any pool (nullptr unless GNSS); strip host and dongle specifics
(gx10, serials); split the `/adcstat` hang fix out as its own small commit; tests per AGENTS.md.
Done as draft PR #1751 (branch kv/airspy-pfb, worktree ~/gnss/airspy-wt).

**Progress, 2026-10-08 (second session).** Branch `kv/cleanup-r3` (worktree ~/gnss/cleanup3-wt),
on `31896a862`, local. Each batch verified on cf05: CI's three `-Werror` builds (CUDA+OpenCL, CPU
full, CPU bare), boost, CI's pytests, `--check-config` on the nine live configs (it rejects an
unknown stage type), `gen_fleet` output byte-identical base vs branch, broker unit tests +
selftest + `broker_multi --list` base vs branch (`test_skyscope` is a race: base fails it 1/5 too).
* A done (`0fbe5071b`, `04f640012`). Also found: CI's **lint** job was red on our tree; fixed by one
  formatter pass (`72662e4d8`, black 19.10b0 / clang-format-18 / cmake-format / yamlfix) and a
  j2lint guard in `gnss_chain.j2` (`79912e69c`).
* B1 done except the airspy-layer revert (waits on C2's adcstat anchor; #1751 is its upstream
  form): `a2a2770a8`, `0cd8d3c03`. B2: only `gnssBroker` + `bufferDedup` (`bf5590704`); the rest
  is not mechanical, see below. B3 `5862dfd5b` (70 files), B4 `48c30fe54` (46), B5 `0e1457783`
  (14 docs; runbook took the bring-up failures and the epoch check; citations name 31896a862).
* Held back, with the reason: path A (`cudaGnssChordTrack` + generator branch; the generator
  records KV keeping it for single-signal debugging -> C9); combiner overlay/navwipe/bit_export
  and nh-assist (C1/C2); bench-only kernel variants and the GPU gate tools phibits/phishare(gpu)
  (stage 8, need GPU A/B); assembler chan/phi and dcyc dumps (C7); `navbit_reuse` (C1);
  `l5_band_decode`, `mid_band_decode`, `gps_l2c_subband_validate`, `gps_hoprate_validate` (cited as
  references by the generators and a boost test: stage 5/7); `gnss_tec` (feeds gnss_tec_movie);
  `kcoh_phase_series`/`kcoh_rate_probe` (C7); beam-map pipeline (6 files) and broker benches
  (medium/low); `bfmask_deadlock_upstream_note` (C11).
* New decisions: **C9** remove path A? **C10** beam-map pipeline? **C11** file the bfmask note as
  an upstream issue, then delete it? Facts for C6: `dop_continuous` is neither frozen nor in the
  live argv, so production runs without it.

**KV's decisions (2026-10-08) and what followed**, same branch:
* C1 KEEP nav decode (simplify later; keep the function). So `navbit_reuse`, the combiner's
  overlay/navwipe/bit_export and the broker's nav-bit plumbing stay.
* C2 + C6 DONE: 14 broker commits (guard on SIGNAL_IMPLIED; xband, CM/CL sibling, nh-assist,
  state-consume, coast-to-horizon, watchdog, almanac-epoch, the /adcstat anchor, warm-start files,
  tle-name-filter, --dop-continuous retired; carrier-loop knobs and seven airspy knobs frozen).
  All 7 fixture digests identical to base. Then the combiner's nh-assist pass, and the airspy
  layer back to develop's version (#1751 is its upstream form).
* C3 KEEP all code generators (the F-engine carries 0-1.6 GHz; L1 will be forwarded).
* C4 YES, at a planned restart (units, 3 crontabs, cf06 loop) -- not started.
* C5 clone `n2k_dual` for now; it ships in stage 8.
* C7 DONE except: `/set_elem_gain`, `/set_elem_sum_adapt`, `/set_reference_element` are manual
  operator levers (element-cal work), not dead -- asked KV; the kernels' carrier_phase_from_ref
  branch waits for the GPU bench pass (fixed at 1).
* C8 YES, at the next develop merge.
* C9 DONE: generator builds path B only (--no-path-a, --n2-dual, --combine-gpus,
  --local-trim-gain gone); cudaGnssChordTrack and its in-tracker trim loop removed; the N x M
  despread stays as the reference n2dualxval checks path B against.
* C10 DONE: beam-map pipeline removed; CHORD_BEAM_MAPS.md kept, marked historical.
* C11 DONE: the bfmask note is deleted; develop's #1655 (one mask stream per GPU half) answers it.
* The generator no longer emits the dead keys (`gnss{0,1}_cmb_buf`, `carrier_phase_from_ref`,
  `carrier_phase_mode`, the n2combine's `phase_dump_prns`/`_path`), so the next regeneration
  changes the node configs by exactly those keys and the provenance header.
* KV kept `/set_elem_gain`, `/set_elem_sum_adapt`, `/set_reference_element` ("we may need those
  again shortly").

---

## 5. What this plan is not

It is **not** a claim that the branch is ready. It is a claim about the order in which it
becomes reviewable. Stage 1 could open this week; Stages 2–4 need conversations with the
owners of the code they touch, and those conversations are the actual long pole — not the
GNSS code, which is additive and nobody else's problem.
