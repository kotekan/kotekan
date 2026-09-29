#!/bin/bash

# This script generates the Julia-based CUDA kernels for the upchannelization kernel.
# Run it from the kotekan base directory like ./julia/bin/upchan.sh

set -euxo pipefail

scriptdir=$(dirname "$0")
cd "$scriptdir/.."

setups='
    charts_U32_K4
    chime_U2_K4 chime_U4_K4 chime_U8_K4 chime_U16_K4 chime_U32_K4 chime_U64_K4 chime_U128_K4
    chime_U2_K8 chime_U4_K8 chime_U8_K8 chime_U16_K8 chime_U32_K8 chime_U64_K8 chime_U128_K8
    chord_U2_K4 chord_U4_K4 chord_U8_K4 chord_U16_K4 chord_U32_K4 chord_U64_K4 chord_U128_K4
    hirax_U8_K4 hirax_U16_K4 hirax_U32_K4 hirax_U64_K4 hirax_U128_K4
    pathfinder_U2_K4 pathfinder_U4_K4 pathfinder_U8_K4 pathfinder_U16_K4 pathfinder_U32_K4 pathfinder_U64_K4 pathfinder_U128_K4
'
# smallfinder_U2_K4 smallfinder_U4_K4 smallfinder_U8_K4 smallfinder_U16_K4 smallfinder_U32_K4 smallfinder_U64_K4

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
