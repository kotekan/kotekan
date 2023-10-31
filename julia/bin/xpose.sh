#!/bin/bash

set -euxo pipefail

scriptdir=$(dirname "$0")
cd "$scriptdir/.."

setups='pathfinder chord'

# Delete previous output (so that we don't accidentally re-use it)
for setup in $setups; do
    rm -f ../lib/cuda/cudaTransposeKernel_$setup.cpp
    rm -f ../lib/cuda/kernels/TransposeKernel_$setup.jl
    rm -f ../lib/cuda/kernels/TransposeKernel_$setup.ptx
    rm -f ../lib/cuda/kernels/TransposeKernel_$setup.yaml
done

# Generate kernel
for setup in $setups; do
    julia --project=@. --optimize kernels/xpose-$setup.jl 2>&1 | tee output-A40/xpose-$setup.out
done

# Format generated C++ code
for setup in $setups; do
    clang-format -i output-A40/xpose-$setup.cxx
done

# Format generated Julia code
julia --project=@. --eval 'using JuliaFormatter; JuliaFormatter.format_file("output-A40")'

# Copy kernel into Kotekan
for setup in $setups; do
    cp output-A40/xpose-$setup.cxx ../lib/cuda/cudaTransposeKernel_$setup.cpp
    cp output-A40/xpose-$setup.jl ../lib/cuda/kernels/TransposeKernel_$setup.jl
    cp output-A40/xpose-$setup.ptx ../lib/cuda/kernels/TransposeKernel_$setup.ptx
    cp output-A40/xpose-$setup.yaml ../lib/cuda/kernels/TransposeKernel_$setup.yaml
done
