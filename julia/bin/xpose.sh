#!/bin/bash

# This script generates the Julia-based Cuda kernels for the transpose kernel.
# Run it from the kotekan base directory like ./julia/bin/xpose.sh

set -euxo pipefail

scriptdir=$(dirname "$0")
cd "$scriptdir/.."

setups='chord hirax pathfinder'

mkdir -p output-A40

# Delete previous output (so that we don't accidentally re-use it)
for setup in $setups; do
    rm -f output-A40/xpose_$setup.cxx
    rm -f output-A40/xpose_$setup.jl
    rm -f output-A40/xpose_$setup.ptx
    rm -f output-A40/xpose_$setup.sass
    rm -f output-A40/xpose_$setup.yaml
    rm -f ../lib/cuda/cudaTransposeKernel_$setup.cpp
    rm -f ../lib/cuda/kernels/TransposeKernel_$setup.jl
    rm -f ../lib/cuda/kernels/TransposeKernel_$setup.ptx
    rm -f ../lib/cuda/kernels/TransposeKernel_$setup.yaml
done

# Generate kernel
for setup in $setups; do
    julia --project=@. --optimize kernels/xpose_$setup.jl 2>&1 | tee output-A40/xpose_$setup.out &
done
wait

# Check whether kernels were generated
for setup in $setups; do
    test -f output-A40/xpose_$setup.cxx
    test -f output-A40/xpose_$setup.jl
    test -f output-A40/xpose_$setup.ptx
    test -f output-A40/xpose_$setup.sass
    test -f output-A40/xpose_$setup.yaml
done

# Format generated C++ code
for setup in $setups; do
    clang-format -i output-A40/xpose_$setup.cxx &
done

# Format generated Julia code
julia --project=@. --eval 'using JuliaFormatter; JuliaFormatter.format_file("output-A40")' &
wait

# Copy kernels into Kotekan
for setup in $setups; do
    cp output-A40/xpose_$setup.cxx ../lib/cuda/cudaTransposeKernel_$setup.cpp
    cp output-A40/xpose_$setup.jl ../lib/cuda/kernels/TransposeKernel_$setup.jl
    cp output-A40/xpose_$setup.ptx ../lib/cuda/kernels/TransposeKernel_$setup.ptx
    cp output-A40/xpose_$setup.yaml ../lib/cuda/kernels/TransposeKernel_$setup.yaml
done
