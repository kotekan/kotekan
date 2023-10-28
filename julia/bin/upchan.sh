#!/bin/bash

set -euxo pipefail

scriptdir=$(dirname "$0")
cd "$scriptdir/.."

Ufactors='16 32 64 128'

# Delete previous output (so that we don't accidentally re-use it)
for U in $Ufactors; do
    rm -f ../lib/cuda/cudaUpchannelizer_U$U.cpp
    rm -f ../lib/cuda/kernels/Upchannelizer_U$U.jl
    rm -f ../lib/cuda/kernels/Upchannelizer_U$U.ptx
    rm -f ../lib/cuda/kernels/Upchannelizer_U$U.yaml
done

# Generate kernel
for U in $Ufactors; do
    julia --project=@. --optimize kernels/upchan-U$U.jl 2>&1 | tee output-A40/upchan-U$U.out
done

# Format generated C++ code
for U in $Ufactors; do
    clang-format -i output-A40/upchan-U$U.cxx
done

# Format generated Julia code
julia --project=@. --eval 'using JuliaFormatter; JuliaFormatter.format_file("output-A40")'

# Copy kernel into Kotekan
for U in $Ufactors; do
    cp output-A40/upchan-U$U.cxx ../lib/cuda/cudaUpchannelizer_U$U.cpp
    cp output-A40/upchan-U$U.jl ../lib/cuda/kernels/Upchannelizer_U$U.jl
    cp output-A40/upchan-U$U.ptx ../lib/cuda/kernels/Upchannelizer_U$U.ptx
    cp output-A40/upchan-U$U.yaml ../lib/cuda/kernels/Upchannelizer_U$U.yaml
done
