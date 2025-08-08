#!/bin/bash

# This script generates the Julia-based CUDA kernels for the CHIME FRB beamforming kernel.
# Run it from the kotekan base directory like ./julia/bin/chimefrb.sh

set -euxo pipefail

scriptdir=$(dirname "$0")
cd "$scriptdir/.."

setups='
    chime_U16
'

mkdir -p output-A40

# Delete previous output (so that we don't accidentally re-use it)
for setup in $setups; do
    rm -f output-A40/chimefrb_$setup.cxx
    rm -f output-A40/chimefrb_$setup.jl
    rm -f output-A40/chimefrb_$setup.ptx
    rm -f output-A40/chimefrb_$setup.sass
    rm -f output-A40/chimefrb_$setup.yaml
    rm -f ../lib/cuda/cudaCHIMEFRBBeamformer_$setup.cpp
    rm -f ../lib/cuda/kernels/CHIMEFRBBeamformer_$setup.jl
    rm -f ../lib/cuda/kernels/CHIMEFRBBeamformer_$setup.ptx
    rm -f ../lib/cuda/kernels/CHIMEFRBBeamformer_$setup.yaml
done

# Generate kernel
for setup in $setups; do
    julia --project=@. --optimize kernels/chimefrb_$setup.jl 2>&1 | tee output-A40/chimefrb_$setup.out &
done
wait

# Check whether kernels were generated
for setup in $setups; do
    test -f output-A40/chimefrb_$setup.cxx
    test -f output-A40/chimefrb_$setup.jl
    test -f output-A40/chimefrb_$setup.ptx
    test -f output-A40/chimefrb_$setup.sass
    test -f output-A40/chimefrb_$setup.yaml
done

# Format generated C++ code
for setup in $setups; do
    clang-format -i output-A40/chimefrb_$setup.cxx &
done

# Format generated Julia code
julia --project=@. --eval 'using JuliaFormatter; JuliaFormatter.format_file("output-A40")' &
wait

# Copy kernels into Kotekan
for setup in $setups; do
    cp output-A40/chimefrb_$setup.cxx ../lib/cuda/cudaCHIMEFRBBeamformer_$setup.cpp
    cp output-A40/chimefrb_$setup.jl ../lib/cuda/kernels/CHIMEFRBBeamformer_$setup.jl
    cp output-A40/chimefrb_$setup.ptx ../lib/cuda/kernels/CHIMEFRBBeamformer_$setup.ptx
    cp output-A40/chimefrb_$setup.yaml ../lib/cuda/kernels/CHIMEFRBBeamformer_$setup.yaml
done
