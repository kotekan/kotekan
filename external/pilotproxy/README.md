# Vendored PilotProxy F-statistic detector core

This directory vendors the CUDA detector core of **PilotProxy**, the ATSC
DTV pilot-tone F-statistic detector, for use by the `cudaPilotProxyDetector`
kotekan stage.

- Upstream repository: https://github.com/WVURAIL/pilot-proxy
- Upstream directory: `cuda/`
- Vendored from upstream commit: see `VENDOR.json`, which pins the full
  commit hash and the per-file digests (kernel core 2.3.x line, including
  the `FXFFT256_REF_NO_MAIN` guard); `tools/check_vendored_pilotproxy.py`
  verifies them.
- Files (verbatim copies of the upstream files):
  - `config.h` — compile-time kernel configuration (detector window length
    K selectable via `FSTAT_DETECTOR_WINDOW_SAMPLES`, supported values 64
    and 128 with a default of 128; 3 weight terms, packed complex int4,
    DP4A, uint64 power accumulation).
  - `f_statistic.h` / `f_statistic.cu` — public C API and CUDA
    implementation. All entry points are `extern "C"`; the deployed mask
    entry is `FStat_Compute_FusedFineMask_U64` (kernel core 2.3.0) and the
    coarse fallback is `FStat_Compute_NumDen_Mask_RationalHalf`.
  - `f_statistic_reference.h` / `f_statistic_reference.cpp` — bit-accurate
    CPU reference for the packed int4 dot products, row sums, and uint64
    power sums (used by `testPilotProxyDetector`).
  - `fxfft256_ref.c` — frozen fxfft256 v1 fixed-point FFT reference
    (bit-exact model of the kernel's fine-reduction transform). Compile with
    `-DFXFFT256_REF_NO_MAIN` (or include it from a TU that defines the
    macro) to get only the `fxfft256()` transform without the standalone
    file harness `main()`.

Local modifications: none. Do not edit these files here; changes belong
upstream in pilot-proxy (the detector contract is frozen and validated by
the upstream test suite, including golden vectors for the FFT and
bit-equality tests for the fused mask epilogue). To update, copy the files
from a new upstream commit and refresh `VENDOR.json`. `tools/lint.sh`
prunes `external/`, so the copies are never reformatted and stay
byte-identical to upstream (re-vendoring remains a pure copy + hash
comparison). pilot-proxy is MIT-licensed: `LICENSE` in this directory is
the upstream license text, and the repository-level LICENSE lists the core
under "Included Libraries".

Kotekan-side files in this directory (not upstream copies): this README,
`VENDOR.json`, `COMMIT` (the one-line upstream pin in the same format as
`external/ksgpu` and `external/n2k`; the check script requires it to agree
with `VENDOR.json`), `CMakeLists.txt` (build glue; the core builds as the
static library `pilotproxy`, linked into consumers via `libexternal`), and
`LICENSE`.

Notes for kotekan integration:

- `f_statistic.cu` is self-contained (only `config.h` / CUDA runtime
  includes) and compiles with the repository default
  `CMAKE_CUDA_ARCHITECTURES` (sm_86, A40); the upstream default build knobs
  (`FSTAT_USE_DP4A=1`, `FSTAT_BLOCK_THREADS=64`,
  `FSTAT_USE_SHARED_WEIGHT_LANES=1`, `FSTAT_GRID_MAX_BLOCKS=4096`) are the
  `config.h` defaults. The CHORD deployment additionally sets
  `FSTAT_DETECTOR_WINDOW_SAMPLES=64` in this directory's `CMakeLists.txt`:
  one 8192-sample GPU frame then carries the frozen 128 windows per stream
  on CHORD's 195.3125 kHz channels. Stage code reads the compiled value at
  runtime through `fstat_get_config`.
- The upstream API executes on the legacy default CUDA stream and takes no
  stream argument. `cudaPilotProxyDetector` brackets the calls with CUDA
  events to order them against kotekan's per-pipeline streams; a
  stream-taking API variant is a proposed upstream follow-up.
- The local `config.h` is included as `"config.h"` from these files only;
  quote-include resolution keeps it from shadowing other headers, and
  nothing else in kotekan includes a header of that name from this
  directory.
