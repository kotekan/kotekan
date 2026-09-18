#!/bin/bash

# This script runs the in-file self-test of the upchannelization kernel for all
# supported upchannelization factors. Each run injects two inputs -- a tone and
# a constant -- and compares the kernel output against an analytic prediction
# (see `main` in `kernels/upchan.jl`).
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
# stage for U=128); all other upchannelization factors pass. Both test cases
# fail for both, which is worth recording: the tone and the constant see the
# same bug from two independent directions.
#   - selftest_U4:   a constant input produces a time dependent output, with
#                    power in three of the four fine frequencies instead of an
#                    even split across the middle two
#   - selftest_U128: the tone lands in fine frequency `u + U/2` instead of `u`;
#                    a constant lands in `u = 0` and `u = 63` instead of the
#                    middle two `u = 63` and `u = 64` (`64 + 64 ≡ 0 mod 128`)
known_failures='selftest_U4 selftest_U128'

mkdir -p output

for setup in ${setups}; do
    julia --project=@. --optimize kernels/upchan_${setup}.jl >output/upchan_${setup}.out 2>&1 &
done
wait

# Each driver runs both test cases (`:tone` and `:constant`) and prints one
# "Found N errors" line per case.
expected_results=2

status=0
for setup in ${setups}; do
    # Distinguish "ran and reported" from "crashed before reporting". Without
    # this a crash looks exactly like a mismatch, and for a known failure the
    # script would report the expected FAIL and exit 0.
    results=$(grep -cE '^Found [0-9]+ errors' "output/upchan_${setup}.out" || true)
    if [ "${results}" -ne ${expected_results} ]; then
        echo "${setup}: ERROR (${results} of ${expected_results} test cases reported a result)"
        tail -20 "output/upchan_${setup}.out"
        status=1
        continue
    fi
    if [ "$(grep -c '^Found 0 errors' "output/upchan_${setup}.out" || true)" -eq ${expected_results} ]; then
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
