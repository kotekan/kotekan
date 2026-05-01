#!/bin/bash

# This script generates the Julia-based CUDA kernels for the upchannelization kernel.
# Run it from the kotekan base directory like ./julia/bin/upchan.sh

set -euxo pipefail

scriptdir=$(dirname "$0")
cd "$scriptdir/.."

setups='
    chime_U2 chime_U4 chime_U8 chime_U16 chime_U32 chime_U64 chime_U128
    chord_U2 chord_U4 chord_U8 chord_U16 chord_U32 chord_U64 chord_U128
    hirax_U8 hirax_U16 hirax_U32 hirax_U64 hirax_U128
    pathfinder_U2 pathfinder_U4 pathfinder_U8 pathfinder_U16 pathfinder_U32 pathfinder_U64
'
# smallfinder_U2 smallfinder_U4 smallfinder_U8 smallfinder_U16 smallfinder_U32 smallfinder_U64

mkdir -p output

# Delete previous output (so that we don't accidentally re-use it)
for setup in ${setups}; do
    rm -f output/upchan_${setup}.cxx
    rm -f output/upchan_${setup}.jl
    rm -f output/upchan_${setup}.ptx
    rm -f output/upchan_${setup}.sass
    rm -f output/upchan_${setup}.yaml
    rm -f ../lib/cuda/generated/cudaUpchannelizer_${setup}.cpp
    rm -f ../lib/cuda/generated/Upchannelizer_${setup}.jl
    rm -f ../lib/cuda/generated/Upchannelizer_${setup}.ptx
    rm -f ../lib/cuda/generated/Upchannelizer_${setup}.yaml
done

# Generate kernels
for setup in ${setups}; do
    julia --project=@. --optimize kernels/upchan_${setup}.jl 2>&1 | tee output/upchan_${setup}.out &
done
wait

# Check whether kernels were generated
for setup in ${setups}; do
    test -f output/upchan_${setup}.cxx
    test -f output/upchan_${setup}.jl
    test -f output/upchan_${setup}.ptx
    test -f output/upchan_${setup}.sass
    test -f output/upchan_${setup}.yaml
done

# Format generated C++ code
for setup in ${setups}; do
    clang-format-18 -i output/upchan_${setup}.cxx &
done

# Format generated Julia code
for setup in ${setups}; do
    julia --project=@. --eval 'using JuliaFormatter; JuliaFormatter.format_file("'output/upchan_${setup}.jl'")' &
done
wait

# Copy kernels into Kotekan
for setup in ${setups}; do
    cp output/upchan_${setup}.cxx ../lib/cuda/generated/cudaUpchannelizer_${setup}.cpp
    cp output/upchan_${setup}.jl ../lib/cuda/generated/Upchannelizer_${setup}.jl
    cp output/upchan_${setup}.ptx ../lib/cuda/generated/Upchannelizer_${setup}.ptx
    cp output/upchan_${setup}.yaml ../lib/cuda/generated/Upchannelizer_${setup}.yaml
done
