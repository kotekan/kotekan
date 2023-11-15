#!/bin/bash

set -euxo pipefail

scriptdir=$(dirname "$0")
cd "$scriptdir/.."

# Delete previous output (so that we don't accidentally re-use it)
rm -f ../lib/cuda/cudaFRBBeamformer.cpp
rm -f ../lib/cuda/kernels/FRBBeamformer.jl
rm -f ../lib/cuda/kernels/FRBBeamformer.ptx
rm -f ../lib/cuda/kernels/FRBBeamformer.yaml

# Generate kernel
julia --project=@. --optimize kernels/frb.jl 2>&1 | tee output-A40/frb.out

# Format generated C++ code
clang-format -i output-A40/frb.cxx

# Format generated Julia code
julia --project=@. --eval 'using JuliaFormatter; JuliaFormatter.format_file("output-A40")'

# Copy kernel into Kotekan
cp output-A40/frb.cxx ../lib/cuda/cudaFRBBeamformer.cpp
cp output-A40/frb.jl ../lib/cuda/kernels/FRBBeamformer.jl
cp output-A40/frb.ptx ../lib/cuda/kernels/FRBBeamformer.ptx
cp output-A40/frb.yaml ../lib/cuda/kernels/FRBBeamformer.yaml
