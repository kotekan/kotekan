#!/bin/bash

set -euxo pipefail

scriptdir=$(dirname "$0")
cd "$scriptdir/.."

setups='
    chord_U8 chord_U16 chord_U32 chord_U64 chord_U128
    pathfinder_U2 pathfinder_U4 pathfinder_U8 pathfinder_U16 pathfinder_U32 pathfinder_U64
'

# Delete previous output (so that we don't accidentally re-use it)
for setup in ${setups}; do
    rm -f output-A40/upchan_${setup}.cxx
    rm -f output-A40/upchan_${setup}.jl
    rm -f output-A40/upchan_${setup}.ptx
    rm -f output-A40/upchan_${setup}.yaml
    rm -f ../lib/cuda/cudaUpchannelizer_${setup}.cpp
    rm -f ../lib/cuda/kernels/Upchannelizer_${setup}.jl
    rm -f ../lib/cuda/kernels/Upchannelizer_${setup}.ptx
    rm -f ../lib/cuda/kernels/Upchannelizer_${setup}.yaml
done

# Generate kernels
for setup in ${setups}; do
    julia --project=@. --optimize kernels/upchan_${setup}.jl 2>&1 | tee output-A40/upchan_${setup}.out &
done
wait

# Check whether kernels were generated
for setup in ${setups}; do
    test -f output-A40/upchan_${setup}.cxx
    test -f output-A40/upchan_${setup}.jl
    test -f output-A40/upchan_${setup}.ptx
    test -f output-A40/upchan_${setup}.yaml
done

# Format generated C++ code
for setup in ${setups}; do
    clang-format -i output-A40/upchan_${setup}.cxx &
done

# Format generated Julia code
julia --project=@. --eval 'using JuliaFormatter; JuliaFormatter.format_file("output-A40")' &
wait

# Copy kernel into Kotekan
for setup in ${setups}; do
    cp output-A40/upchan_${setup}.cxx ../lib/cuda/cudaUpchannelizer_${setup}.cpp
    cp output-A40/upchan_${setup}.jl ../lib/cuda/kernels/Upchannelizer_${setup}.jl
    cp output-A40/upchan_${setup}.ptx ../lib/cuda/kernels/Upchannelizer_${setup}.ptx
    cp output-A40/upchan_${setup}.yaml ../lib/cuda/kernels/Upchannelizer_${setup}.yaml
done
