#!/bin/bash

# This script generates the Julia-based CUDA kernels for the baseband beamforming kernel.
# Run it from the kotekan base directory like ./julia/bin/bb.sh

set -euxo pipefail

scriptdir=$(dirname "$0")
cd "$scriptdir/.."

setups='charts chime chord hirax pathfinder' # smallfinder

mkdir -p output

# Delete previous output (so that we don't accidentally re-use it)
for setup in ${setups}; do
    rm -f output/bb_${setup}.cxx
    rm -f output/bb_${setup}.jl
    rm -f output/bb_${setup}.ptx
    rm -f output/bb_${setup}.sass
    rm -f output/bb_${setup}.yaml
    rm -f ../lib/cuda/generated/cudaBasebandBeamformer_${setup}.cpp
    rm -f ../lib/cuda/generated/BasebandBeamformer_${setup}.jl
    rm -f ../lib/cuda/generated/BasebandBeamformer_${setup}.ptx
    rm -f ../lib/cuda/generated/BasebandBeamformer_${setup}.yaml
done

# Generate kernel
for setup in ${setups}; do
    julia --project=@. --optimize kernels/bb_${setup}.jl 2>&1 | tee output/bb_${setup}.out &
done
wait

# Check whether kernels were generated
for setup in ${setups}; do
    test -f output/bb_${setup}.cxx
    test -f output/bb_${setup}.jl
    test -f output/bb_${setup}.ptx
    test -f output/bb_${setup}.sass
    test -f output/bb_${setup}.yaml
done

# Format generated C++ code
for setup in ${setups}; do
    clang-format-18 -i output/bb_${setup}.cxx &
done

# Format generated Julia code
for setup in ${setups}; do
    julia --project=@. --eval 'using JuliaFormatter; JuliaFormatter.format_file("'output/bb_${setup}.jl'")' &
done
wait

# Copy kernels into Kotekan
for setup in ${setups}; do
    cp output/bb_${setup}.cxx ../lib/cuda/generated/cudaBasebandBeamformer_${setup}.cpp
    cp output/bb_${setup}.jl ../lib/cuda/generated/BasebandBeamformer_${setup}.jl
    cp output/bb_${setup}.ptx ../lib/cuda/generated/BasebandBeamformer_${setup}.ptx
    cp output/bb_${setup}.yaml ../lib/cuda/generated/BasebandBeamformer_${setup}.yaml
done
