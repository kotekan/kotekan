#!/bin/bash

# This script generates the Julia-based Cuda kernels for the FRB beamforming kernel.
# Run it from the kotekan base directory like ./julia/bin/frb.sh

set -euxo pipefail

scriptdir=$(dirname "$0")
cd "$scriptdir/.."

setups='
    chord_U1 chord_U2 chord_U4 chord_U8 chord_U16 chord_U32 chord_U64 chord_U128
    hirax_U1 hirax_U2 hirax_U4 hirax_U8 hirax_U16 hirax_U32 hirax_U64 hirax_U128
    pathfinder_U1 pathfinder_U2 pathfinder_U4 pathfinder_U8 pathfinder_U16 pathfinder_U32 pathfinder_U64
'

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
# for setup in $setups; do
#     julia --project=@. --optimize kernels/frb_$setup.jl 2>&1 | tee output-A40/frb_$setup.out &
# done
# wait
# for setup in $setups; do
#     julia --project=@. --optimize kernels/frb_$setup.jl 2>&1 | tee output-A40/frb_$setup.out
# done
for setup in $setups; do echo $setup; done |
    xargs -I'$setup' -n 1 -P 8 bash -c 'julia --project=@. --optimize kernels/frb_$setup.jl 2>&1 | tee output-A40/frb_$setup.out'

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

# Copy kernels into Kotekan
for setup in $setups; do
    cp output-A40/frb_$setup.cxx ../lib/cuda/cudaFRBBeamformer_$setup.cpp
    cp output-A40/frb_$setup.jl ../lib/cuda/kernels/FRBBeamformer_$setup.jl
    cp output-A40/frb_$setup.ptx ../lib/cuda/kernels/FRBBeamformer_$setup.ptx
    cp output-A40/frb_$setup.yaml ../lib/cuda/kernels/FRBBeamformer_$setup.yaml
done
