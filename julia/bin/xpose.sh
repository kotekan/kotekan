#!/bin/bash

# This script generates the Julia-based Cuda kernels for the transpose kernel.
# Run it from the kotekan base directory like ./julia/bin/xpose.sh

set -euxo pipefail

scriptdir=$(dirname "$0")
cd "$scriptdir/.."

setups='chord hirax pathfinder' # smallfinder

mkdir -p output

# Delete previous output (so that we don't accidentally re-use it)
for setup in ${setups}; do
    rm -f output/xpose_${setup}.cxx
    rm -f output/xpose_${setup}.jl
    rm -f output/xpose_${setup}.ptx
    rm -f output/xpose_${setup}.sass
    rm -f output/xpose_${setup}.yaml
    rm -f ../lib/cuda/generated/cudaTransposeKernel_${setup}.cpp
    rm -f ../lib/cuda/generated/TransposeKernel_${setup}.jl
    rm -f ../lib/cuda/generated/TransposeKernel_${setup}.ptx
    rm -f ../lib/cuda/generated/TransposeKernel_${setup}.yaml
done

# Generate kernel
for setup in ${setups}; do
    julia --project=@. --optimize kernels/xpose_${setup}.jl 2>&1 | tee output/xpose_${setup}.out &
done
wait

# Check whether kernels were generated
for setup in ${setups}; do
    test -f output/xpose_${setup}.cxx
    test -f output/xpose_${setup}.jl
    test -f output/xpose_${setup}.ptx
    test -f output/xpose_${setup}.sass
    test -f output/xpose_${setup}.yaml
done

# Format generated C++ code
for setup in ${setups}; do
    clang-format-18 -i output/xpose_${setup}.cxx &
done

# Format generated Julia code
for setup in ${setups}; do
    julia --project=@. --eval 'using JuliaFormatter; JuliaFormatter.format_file("'output/xpose_${setup}.jl'")' &
done
wait

# Copy kernels into Kotekan
for setup in ${setups}; do
    cp output/xpose_${setup}.cxx ../lib/cuda/generated/cudaTransposeKernel_${setup}.cpp
    cp output/xpose_${setup}.jl ../lib/cuda/generated/TransposeKernel_${setup}.jl
    cp output/xpose_${setup}.ptx ../lib/cuda/generated/TransposeKernel_${setup}.ptx
    cp output/xpose_${setup}.yaml ../lib/cuda/generated/TransposeKernel_${setup}.yaml
done
