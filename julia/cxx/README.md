# Simple C++ kernel implementations

This directory contains simple (straightforward) C++ implementations
of some of the Julia-generated CUDA kernels. This serves to illustrate
the algorithms that these kernels use in a much simpler form. The
kernels here can also produce output that is more-or-less similar to
the output that should be produced by the Julia-generated CUDA
kernels, usually up to floating-point round-off or quantization
differences.

TODO -- there are testing Kotekan stages, eg
  lib/testing/gpuSimulateCudaBasebandBeamformer.cpp
that should either be adapted to include this code, OR be auto-generated.
