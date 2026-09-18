#!/bin/bash

# This script runs the in-file self-test of the upchannelization kernel.
# It injects a single tone into the kernel input and compares the kernel
# output against an analytic prediction (see `main` in `kernels/upchan.jl`).
# Run it from the kotekan base directory like ./julia/bin/upchan_selftest.sh
#
# The self-test uses its own small setup (`kernels/setup_selftest.jl`) instead
# of one of the production setups. The production setups have F > F_per_U[U],
# i.e. more coarse frequencies than fit into the kernel's output buffer `Ē`,
# so that the self-test could only ever exercise a subset of them.

set -euo pipefail

scriptdir=$(dirname "$0")
cd "$scriptdir/.."

setups='
    selftest_U2 selftest_U4 selftest_U8 selftest_U16 selftest_U32 selftest_U64 selftest_U128
'

# Setups that are known to fail, with the kernel bug they expose. Both are
# special cases of the FFT decomposition (`n == 0` for U=4, the extra `Γ⁴`
# stage for U=128); all other upchannelization factors pass.
#   - selftest_U4:   a constant input produces a time dependent output, with
#                    power in three of the four fine frequencies
#   - selftest_U128: the tone lands in fine frequency `u + U/2` instead of `u`
known_failures='selftest_U4 selftest_U128'

mkdir -p output

for setup in ${setups}; do
    julia --project=@. --optimize kernels/upchan_${setup}.jl >output/upchan_${setup}.out 2>&1 &
done
wait || true

status=0
for setup in ${setups}; do
    if grep -q '^Found 0 errors' "output/upchan_${setup}.out"; then
        passed=yes
    else
        passed=no
    fi
    if [[ " ${known_failures} " == *" ${setup} "* ]]; then
        if [ ${passed} = yes ]; then
            echo "${setup}: PASS (expected to fail -- remove it from known_failures)"
            status=1
        else
            echo "${setup}: FAIL (known kernel bug)"
        fi
    else
        if [ ${passed} = yes ]; then
            echo "${setup}: PASS"
        else
            echo "${setup}: FAIL"
            sed -n '/^Checking results/,$p' "output/upchan_${setup}.out" | head -20
            status=1
        fi
    fi
done

exit ${status}
