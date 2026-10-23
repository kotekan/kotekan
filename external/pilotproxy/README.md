# PilotProxy detector core

The CUDA detector core is copied from `cuda/` in
[WVURAIL/pilotproxy](https://github.com/WVURAIL/pilotproxy). `VENDOR.json` names the upstream revision and
records each file's SHA-256 digest.

The seven upstream files are:

- `config.h`: window length and kernel settings.
- `f_statistic.h` and `f_statistic.cu`: the C API and CUDA implementation.
- `f_statistic_reference.h` and `f_statistic_reference.cpp`: CPU references for
  packed dot products, row sums, fine power spectra and the frame powers formed
  from them.
- `fxfft256_ref.c` and `fxfft_master_twiddle.h`: the fixed-point FFT reference.
  Define `FXFFT256_REF_NO_MAIN` to build the transform without its file harness.
  `f_statistic_reference.cpp` compiles it with internal linkage, so a test may
  include it again.

Keep these files unchanged. Make detector changes in PilotProxy, copy the new
revision here, and update the pin and the file hashes. The PilotProxy CI script
checks the local files against these hashes.

The local CMake file builds the `pilotproxy` static library and links it through
`libexternal`. `pilotproxy_core.cu` compiles the core twice, with
`FSTAT_DETECTOR_WINDOW_SAMPLES` set to 64 (CHORD, 8192-sample blocks) and 128
(CHIME, 16384-sample blocks). Each build lives in its own namespace and CMake renames
its C functions, so both link into one binary. `pilotproxy_core.hpp` returns the
functions of one build; the stage picks it with its `detector_window_samples` setting.
Other kernel settings keep their upstream defaults.

`cudaPilotProxyDetector` binds each handle to its own command stream with
`FStat_SetStream` when it first sees data, so the library's kernels are
stream-ordered with the rest of the pipeline. The vendored files include their
`config.h` with quotes, so it does not replace other headers with that name.
Kotekan code includes only `pilotproxy_core.hpp`, not the vendored headers.

This README, the CMake file, `pilotproxy_core.cu`, `pilotproxy_core.hpp` and
the manifest are maintained in Kotekan. `tools/lint.sh` excludes `external/` from formatting. PilotProxy uses the
MIT license; the upstream license text is in `LICENSE`, and the repository's
main license lists it under Included Libraries.
