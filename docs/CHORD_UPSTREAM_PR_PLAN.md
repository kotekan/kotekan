# Staged upstream PR plan — `kv/chord-gnss` → `develop`

**Status 2026-10-09.** Upstream PRs now target `develop`; `chord` is rebuilt by Jim as
`develop` plus selected open PRs ("Sync chord: develop <sha> + PRs ..."), so a PR reaches
`chord` at his next sync without bypassing review.

| PR | Stage | State |
|---|---|---|
| [#1640](https://github.com/kotekan/kotekan/pull/1640) | 1, upstream fixes | merged (to `chord`, since folded into `develop`) |
| [#1642](https://github.com/kotekan/kotekan/pull/1642), #1643, #1647, [#1714](https://github.com/kotekan/kotekan/pull/1714) | 2, buffer/metadata | closed: done upstream |
| [#1699](https://github.com/kotekan/kotekan/pull/1699), [#1713](https://github.com/kotekan/kotekan/pull/1713) | off-plan fixes | merged |
| [#1675](https://github.com/kotekan/kotekan/pull/1675) | 3, GPU scheduling | open; Jim approved, waiting on Andre and Erik |
| [#1750](https://github.com/kotekan/kotekan/pull/1750) | 4, DPDK capture | open; running on all six CHORD GNSS nodes |
| [#1751](https://github.com/kotekan/kotekan/pull/1751), [#1752](https://github.com/kotekan/kotekan/pull/1752) | off-plan: airspyInput, fftwEngine PFB | drafts; waiting on a dongle test |
| — | 5–8, GNSS | not started; cleanup rounds 2 and 3 done (§2, §6) |

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

## 2. Stage 0 and cleanup round 2 — DONE (2026-09-02, 2026-10-08)

Stage 0 took out what is not source (a committed ELF binary, tracked symlinks to host builds,
generated PNGs, dead generator scratch) and restored the upstream
`config/crs_full_packet_capture.yaml`. Round 2 untracked the generated configs (ship the
generator, templates, manifest and one example), removed the airspy prototype (tag
`airspy-prototype-final`) and dropped stale pointers. The record is
`be7d46e8e:docs/CHORD_UPSTREAM_PR_PLAN.md` §2.

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

Every piece landed upstream on someone else's PR (#1643, #1647) or was withdrawn; `get_metadata`
is byte-identical to `develop` again. The record is `be7d46e8e:docs/CHORD_UPSTREAM_PR_PLAN.md`
Stage 2.

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

### Stage 5 — GNSS foundation

The 16 signal code generators, the pure value-type headers, `lib/stages/pfbPrototype.*`, and
`GnssChanMetadata` with its `metadataFactory.cpp` pool branch and the CMakeLists wiring. The
generators and value types include only the C++ standard library, and `tests/boost/` already
builds them without linking `kotekan_stages`; ship those boost tests here. Everything after
needs the metadata: 18 GNSS sources use it, and without the factory branch every GNSS config
throws at startup.

### Stage 6 — GNSS host-side core

`gnssChannelizedReplica` (`ChannelizedReplicaBank`, the code replicas every GPU correlation
runs against), `gnssChannelizedDespread` (the despread result type, `overlay_wipe` and a CPU
despread) and `gnssChannelizedAcquire` (the acquisition definitions and an FFTW engine), with
their boost tests. This is not a CPU pipeline: the prototype's CPU stages are gone (tag
`airspy-prototype-final`), and the live GPU path is built on these classes. The CPU despread
and the FFTW engine do not run in production (the engine is `GnssChannelizedSearch`'s
fallback when the GPU declines a grid), but they are the reference the boost tests check the
GPU path's math against, so upstream's CPU CI can test it. Whether the search keeps its CPU
fallback is a question for this PR.

### Stage 7 — CUDA GNSS path, the GNSS stages, `external/n2k_dual`

The four `.cu` kernels, `cudaCorrelatorDual`, the `n2k_dual` clone, and the stages the CHORD
configs run: the node chain, record assembly, combiner, projection, telemetry, fleet trim and
the aggregator's search. The GPU cleanup pass comes first (the kernels' carrier-phase branch,
bench-only kernel variants, `enable_split_timing`; it needs an A/B on cf06's L40S).
**`external/n2k` is byte-untouched and must stay so** — verified: `git diff -- external/n2k`
is empty.

### Stage 8 — broker, viewer, tooling, configs, docs

The Python broker package, the js_viewer panels, `scripts/gnss/`, `config/`, `docs/`. Largest
by line count, lowest by risk — none of it compiles into kotekan.

---

## 4. Decisions

1. **Generated configs** — RESOLVED 2026-10-08: untracked; ship the generator, templates,
   manifest and one example (cleanup round 2, §2).
2. **The acceptance gate cannot run upstream.** Five of the six `broker_equiv` fixtures are
   38–100 MB transcripts outside git (`/home/kvand/gnss/fixtures/`, NFS), with only their `.digest`
   committed; several tests also read fixtures from there. OPEN: say so plainly in the PR, or
   make one on-sky arm fetchable.
3. **Citations of `docs/gnss_gpu_migration.md`** — RESOLVED 2026-10-08: dropped.
4. **`gps_distributed_broker.py`** — RESOLVED: not superseded. It is the broker's main module
   (`scripts/gnss/broker_multi.py` imports it); `gnss_broker/` holds parts split out of it.
5. **`lib/cuda/benches/chordShapeBench.cu`** — RESOLVED 2026-10-08: removed.
6. **The airspy prototype** — RESOLVED 2026-10-08: removed (tag `airspy-prototype-final`); no
   sign the prototype has pulled this branch since 2026-08-07. The code it stranded went in
   cleanup round 3 (§6, B1).
7. **`config/base/live_config_20260730.json`** — RESOLVED: keep. It is the base for the gather,
   aggregator and cube-archive recipes.
8. **Hardcoded `/home/kvand` paths** — OPEN, and larger than first counted: 78 files. Most are
   our deployment scripts, systemd units and runbooks, which raises the real question for
   stage 8 — which of `scripts/gnss/` and `docs/` belongs upstream at all. ~37 are code or test
   defaults pointing at out-of-repo fixtures and data, tied to item 2.

---

## 6. Cleanup round 3 — the plan (2026-10-08)

Six read-only reviewers at `9f555d4be`; condensed notes with file:line detail in
`be7d46e8e:docs/CHORD_CLEANUP_ROUND3_NOTES.md`. About 37k of the ~201k added lines go with high
confidence and no live behaviour change. Verify every batch with: CUDA `-Werror` build +
boost tests on cf05, `kotekan --check-config` on the nine live configs, `gen_fleet --check`
byte-identical, broker `gnss_broker/test_*.py` + `selftest.py` under venv-ft (niced), and for
broker changes `broker_equiv.py check` on cf05 (never beside the live stack).
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
5. `n2k_dual`: propose the two-input extension to n2k upstream, or ship the clone in stage 7?
6. `dop-continuous`: retire with the others?
7. Medium-confidence C++ (carrier-phase A/B arm #55, unused assembler REST levers, debug env
   vars, FDMA offset).
8. Take develop's copies at the next merge (NDArrayRingBuffer, `chord_pathfinder_recv.j2`,
   cpuMonitor, TransposeBasebandArray, two julia `.out`); revert LinearAlgebra; re-derive the
   premise of the cudaCopyFromRingbuffer gate and the correlators' build-then-publish (kept
   deliberately on 10-01).

**D. Operations found along the way.** 41 `python/scripts/gnss` tests have no runner.
`gnss_transit_arcs.py` uses the pre-#99 station (155 m off). Viewer: layout key v7, reset
clears v6.

**E. Per stage, as each PR is cut.** Comment rewrite to AGENTS.md style (ratio 0.49 vs upstream
0.28; 245 emoji, 264 dates, 230 task refs; ~1.5-2 weeks total); sphinx stubs per stage + one
`docs/sphinx/user/gnss.rst`; the test set per stage; `GNSS_FIXTURES` env default with skip.

**F. A separate airspy PR** (KV, 2026-10-08: the airspy/fftw changes are worth upstreaming).
Source: tag `airspy-prototype-final` — `airspyInput.{cpp,hpp}` (+~294: bounded `/adcstat` wait,
PFB mode, sample_seq, stream watchdog, `ensure_frame_desc`), `fftwEngine.{cpp,hpp}` (+~99),
`airspyFrameDesc.hpp`, `config/airspy_autocorr.yaml`. Base `develop`, its own worktree, so it
does not wait on or collide with this cleanup (B1 reverts those files here; when the PR merges,
the next develop merge brings them back cleanly). Must: drop the dependency on
`GnssChanMetadata` (stage 5) or land after it; fix `fftwEngine.cpp:181-188`, which dereferences
`get_gnss_chan_metadata()` for any pool (nullptr unless GNSS); strip host and dongle specifics
(gx10, serials); split the `/adcstat` hang fix out as its own small commit; tests per AGENTS.md.
Done as draft PR #1751 (branch kv/airspy-pfb, worktree ~/gnss/airspy-wt).

**KV's decisions (2026-10-08) and what followed** (all of round 3 is on `kv/chord-gnss`):
* C1 KEEP nav decode (simplify later; keep the function). So `navbit_reuse`, the combiner's
  overlay/navwipe/bit_export and the broker's nav-bit plumbing stay.
* C2 + C6 DONE: 14 broker commits (guard on SIGNAL_IMPLIED; xband, CM/CL sibling, nh-assist,
  state-consume, coast-to-horizon, watchdog, almanac-epoch, the /adcstat anchor, warm-start files,
  tle-name-filter, --dop-continuous retired; carrier-loop knobs and seven airspy knobs frozen).
  All 7 fixture digests identical to base. Then the combiner's nh-assist pass, and the airspy
  layer back to develop's version (#1751 is its upstream form).
* C3 KEEP all code generators (the F-engine carries 0-1.6 GHz; L1 will be forwarded).
* C4 PREPARED on `kv/site-move`: 39 scripts + `systemd/` moved to `scripts/gnss/site/`. Switch-over,
  away from :17 and 00:15-00:45 UTC: ff the live checkout; `scripts/gnss/site/systemd/
  install_user_units.sh` (daemon-reload only, nothing restarts); repoint the gnss crontab (3 lines)
  and cf06's (1); `ssh cf06 .../scripts/gnss/site/cubecompact_up.sh`; then
  `scripts/gnss/site/stack_contract_gate.sh` (rail differs until gnss-rail restarts).
* C5 clone `n2k_dual` for now; it ships in stage 7.
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
