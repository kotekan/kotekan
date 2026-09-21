# PilotProxy detector core

The CUDA detector core is copied from `cuda/` in
[WVURAIL/pilot-proxy](https://github.com/WVURAIL/pilot-proxy).
`VENDOR.json` names the upstream revision and records each file's SHA-256 digest.

The seven upstream files are:

- `config.h`: window length and kernel settings.
- `f_statistic.h` and `f_statistic.cu`: the C API and CUDA implementation.
- `f_statistic_reference.h` and `f_statistic_reference.cpp`: CPU references for
  packed dot products, row sums and coarse powers.
- `fxfft256_ref.c` and `fxfft_master_twiddle.h`: the fixed-point FFT reference.
  Define `FXFFT256_REF_NO_MAIN` to build the transform without its file harness.

Keep these files unchanged. Make detector changes in pilot-proxy, copy the new
revision here, and update the pin and the file hashes. Check with:

```sh
python3 tools/check_vendored_pilotproxy.py --fetch
```

The local CMake file builds the `pilotproxy` static library and links it through
`libexternal`. It sets `FSTAT_DETECTOR_WINDOW_SAMPLES=64` for CHORD, giving 128
windows per stream in an 8192-sample block. The stage reads the compiled settings
through `FStat_GetSpecs`. Other kernel settings keep their upstream defaults.

`cudaPilotProxyDetector` binds each handle to its own command stream with
`FStat_SetStream` when it first sees data, so the library's kernels are
stream-ordered with the rest of the pipeline. Its local `config.h`
is included with quotes by the vendored files, so it does not replace other
headers with that name.

This README, the CMake file and the manifest are maintained in
Kotekan. `tools/lint.sh` excludes `external/` from formatting. PilotProxy uses the
MIT license; the upstream license text is in `LICENSE`, and the repository's
main license lists it under Included Libraries.
