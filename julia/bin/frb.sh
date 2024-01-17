#!/bin/bash

set -euxo pipefail

scriptdir=$(dirname "$0")
cd "$scriptdir/.."

# setups='chord hirax pathfinder'
setups='chord pathfinder'

# Delete previous output (so that we don't accidentally re-use it)
for setup in $setups; do
    rm -f output-A40/frb_$setup.cxx
    rm -f output-A40/frb_$setup.jl
    rm -f output-A40/frb_$setup.ptx
    rm -f output-A40/frb_$setup.yaml
    rm -f ../lib/cuda/cudaFRBBeamformer_$setup.cpp
    rm -f ../lib/cuda/kernels/FRBBeamformer_$setup.jl
    rm -f ../lib/cuda/kernels/FRBBeamformer_$setup.ptx
    rm -f ../lib/cuda/kernels/FRBBeamformer_$setup.yaml
done

# Generate kernel
for setup in $setups; do
    julia --project=@. --optimize kernels/frb_$setup.jl 2>&1 | tee output-A40/frb_$setup.out &
done
wait

# Check whether kernels were generated
for setup in $setups; do
    test -f output-A40/frb_$setup.cxx
    test -f output-A40/frb_$setup.jl
    test -f output-A40/frb_$setup.ptx
    test -f output-A40/frb_$setup.yaml
done

# Format generated C++ code
for setup in $setups; do
    clang-format -i output-A40/frb_$setup.cxx &
done

# Format generated Julia code
julia --project=@. --eval 'using JuliaFormatter; JuliaFormatter.format_file("output-A40")' &
wait

# Copy kernel into Kotekan
for setup in $setups; do
    cp output-A40/frb_$setup.cxx ../lib/cuda/cudaFRBBeamformer_$setup.cpp
    cp output-A40/frb_$setup.jl ../lib/cuda/kernels/FRBBeamformer_$setup.jl
    cp output-A40/frb_$setup.ptx ../lib/cuda/kernels/FRBBeamformer_$setup.ptx
    cp output-A40/frb_$setup.yaml ../lib/cuda/kernels/FRBBeamformer_$setup.yaml
done
