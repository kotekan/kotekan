#!/bin/bash

set -euxo pipefail

scriptdir=$(dirname "$0")
cd "$scriptdir/.."

setups='pathfinder chord'

# Delete previous output (so that we don't accidentally re-use it)
for setup in $setups; do
    rm -f ../lib/cuda/cudaBasebandBeamformer_$setup.cpp
    rm -f ../lib/cuda/kernels/BasebandBeamformer_$setup.jl
    rm -f ../lib/cuda/kernels/BasebandBeamformer_$setup.ptx
    rm -f ../lib/cuda/kernels/BasebandBeamformer_$setup.yaml
done

# Generate kernel
for setup in $setups; do
    julia --project=@. --optimize kernels/bb-$setup.jl 2>&1 | tee output-A40/bb-$setup.out
done

# Format generated C++ code
for setup in $setups; do
    clang-format -i output-A40/bb-$setup.cxx
done

# Format generated Julia code
julia --project=@. --eval 'using JuliaFormatter; JuliaFormatter.format_file("output-A40")'

# Copy kernel into Kotekan
for setup in $setups; do
    cp output-A40/bb-$setup.cxx ../lib/cuda/cudaBasebandBeamformer_$setup.cpp
    cp output-A40/bb-$setup.jl ../lib/cuda/kernels/BasebandBeamformer_$setup.jl
    cp output-A40/bb-$setup.ptx ../lib/cuda/kernels/BasebandBeamformer_$setup.ptx
    cp output-A40/bb-$setup.yaml ../lib/cuda/kernels/BasebandBeamformer_$setup.yaml
done
