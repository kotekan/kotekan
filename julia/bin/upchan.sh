#!/bin/bash

set -euxo pipefail

scriptdir=$(dirname "$0")
cd "$scriptdir/.."

# setups='pathfinder chord'
setups='chord'
Ufactors='8 16 32 64 128'

# Delete previous output (so that we don't accidentally re-use it)
for setup in ${setups}; do
    for U in ${Ufactors}; do
	rm -f output-A40/upchan_${setup}_U${U}.cxx
	rm -f output-A40/upchan_${setup}_U${U}.jl
	rm -f output-A40/upchan_${setup}_U${U}.ptx
	rm -f output-A40/upchan_${setup}_U${U}.yaml
	rm -f ../lib/cuda/cudaUpchannelizer_${setup}_U${U}.cpp
	rm -f ../lib/cuda/kernels/Upchannelizer_${setup}_U${U}.jl
	rm -f ../lib/cuda/kernels/Upchannelizer_${setup}_U${U}.ptx
	rm -f ../lib/cuda/kernels/Upchannelizer_${setup}_U${U}.yaml
    done
done

# Generate kernels
for setup in ${setups}; do
    for U in ${Ufactors}; do
	julia --project=@. --optimize kernels/upchan_${setup}_U${U}.jl 2>&1 | tee output-A40/upchan_${setup}_U${U}.out &
    done
done
wait

# Check whether kernels were generated
for setup in ${setups}; do
    for U in ${Ufactors}; do
	test -f output-A40/upchan_${setup}_U${U}.cxx
	test -f output-A40/upchan_${setup}_U${U}.jl
	test -f output-A40/upchan_${setup}_U${U}.ptx
	test -f output-A40/upchan_${setup}_U${U}.yaml
    done
done

# Format generated C++ code
for setup in ${setups}; do
    for U in ${Ufactors}; do
	clang-format -i output-A40/upchan_${setup}_U${U}.cxx &
    done
done

# Format generated Julia code
julia --project=@. --eval 'using JuliaFormatter; JuliaFormatter.format_file("output-A40")' &
wait

# Copy kernel into Kotekan
for setup in ${setups}; do
    for U in ${Ufactors}; do
	cp output-A40/upchan_${setup}_U${U}.cxx ../lib/cuda/cudaUpchannelizer_${setup}_U${U}.cpp
	cp output-A40/upchan_${setup}_U${U}.jl ../lib/cuda/kernels/Upchannelizer_${setup}_U${U}.jl
	cp output-A40/upchan_${setup}_U${U}.ptx ../lib/cuda/kernels/Upchannelizer_${setup}_U${U}.ptx
	cp output-A40/upchan_${setup}_U${U}.yaml ../lib/cuda/kernels/Upchannelizer_${setup}_U${U}.yaml
    done
done
