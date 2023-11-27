#!/bin/bash

set -euxo pipefail

scriptdir=$(dirname "$0")
cd "$scriptdir/.."

Ufactors='16 32 64 128'

# Delete previous output (so that we don't accidentally re-use it)
for U in $Ufactors; do
    rm -f output-A40/upchan_U$U.cxx
    rm -f output-A40/upchan_U$U.jl
    rm -f output-A40/upchan_U$U.ptx
    rm -f output-A40/upchan_U$U.yaml
    rm -f ../lib/cuda/cudaUpchannelizer_U$U.cpp
    rm -f ../lib/cuda/kernels/Upchannelizer_U$U.jl
    rm -f ../lib/cuda/kernels/Upchannelizer_U$U.ptx
    rm -f ../lib/cuda/kernels/Upchannelizer_U$U.yaml
done

# Generate kernels
for U in $Ufactors; do
    julia --project=@. --optimize kernels/upchan_U$U.jl 2>&1 | tee output-A40/upchan_U$U.out &
done
wait

# Check whether kernels were generated
for U in $Ufactors; do
    test -f output-A40/upchan_U$U.cxx
    test -f output-A40/upchan_U$U.jl
    test -f output-A40/upchan_U$U.ptx
    test -f output-A40/upchan_U$U.yaml
done

# Format generated C++ code
for U in $Ufactors; do
    clang-format -i output-A40/upchan_U$U.cxx &
done

# Format generated Julia code
julia --project=@. --eval 'using JuliaFormatter; JuliaFormatter.format_file("output-A40")' &
wait

# Copy kernel into Kotekan
for U in $Ufactors; do
    cp output-A40/upchan_U$U.cxx ../lib/cuda/cudaUpchannelizer_U$U.cpp
    cp output-A40/upchan_U$U.jl ../lib/cuda/kernels/Upchannelizer_U$U.jl
    cp output-A40/upchan_U$U.ptx ../lib/cuda/kernels/Upchannelizer_U$U.ptx
    cp output-A40/upchan_U$U.yaml ../lib/cuda/kernels/Upchannelizer_U$U.yaml
done
