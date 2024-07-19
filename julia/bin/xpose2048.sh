#!/bin/bash

# This script generates the Julia-based CUDA kernels for the 2048-byte transpose kernel.
# Run it from the kotekan base directory like ./julia/bin/xpose2048.sh

set -euxo pipefail

scriptdir=$(dirname "$0")
cd "$scriptdir/.."

card='A40'
# card='GeForce_RTX_4090'
# card='L40S'
setups='chime'

mkdir -p output-${card}

# Delete previous output (so that we don't accidentally re-use it)
for setup in $setups; do
    rm -f output-${card}/xpose2048_$setup.cxx
    rm -f output-${card}/xpose2048_$setup.jl
    rm -f output-${card}/xpose2048_$setup.ptx
    rm -f output-${card}/xpose2048_$setup.sass
    rm -f output-${card}/xpose2048_$setup.yaml
    rm -f ../lib/cuda/cudaTranspose2048_$setup.cpp
    rm -f ../lib/cuda/kernels/Transpose2048_$setup.jl
    rm -f ../lib/cuda/kernels/Transpose2048_$setup.ptx
    rm -f ../lib/cuda/kernels/Transpose2048_$setup.yaml
done

# Generate kernel
for setup in $setups; do
    julia --project=@. --optimize kernels/xpose2048_$setup.jl 2>&1 | tee output-${card}/xpose2048_$setup.out &
done
wait

# Check whether kernels were generated
for setup in $setups; do
    test -f output-${card}/xpose2048_$setup.cxx
    test -f output-${card}/xpose2048_$setup.jl
    test -f output-${card}/xpose2048_$setup.ptx
    test -f output-${card}/xpose2048_$setup.sass
    test -f output-${card}/xpose2048_$setup.yaml
done

# Format generated C++ code
for setup in $setups; do
    clang-format -i output-${card}/xpose2048_$setup.cxx &
done

# Format generated Julia code
julia --project=@. --eval 'using JuliaFormatter; JuliaFormatter.format_file("'output-${card}'")' &
wait

# Copy kernels into Kotekan
for setup in $setups; do
    cp output-${card}/xpose2048_$setup.cxx ../lib/cuda/cudaTranspose2048_$setup.cpp
    cp output-${card}/xpose2048_$setup.jl ../lib/cuda/kernels/Transpose2048_$setup.jl
    cp output-${card}/xpose2048_$setup.ptx ../lib/cuda/kernels/Transpose2048_$setup.ptx
    cp output-${card}/xpose2048_$setup.yaml ../lib/cuda/kernels/Transpose2048_$setup.yaml
done
