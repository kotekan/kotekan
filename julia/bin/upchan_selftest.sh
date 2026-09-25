#!/bin/bash

# This script runs the in-file self-test of the upchannelization kernel for all
# supported upchannelization factors. Each run injects three inputs -- a tone,
# a constant, and a set of impulses -- and compares the kernel output against an
# analytic prediction (see `main` in `kernels/upchan.jl`). The tone is then
# repeated with the time window wrapped around the end of the ring buffer.
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

mkdir -p output

for setup in ${setups}; do
    julia --project=@. --optimize kernels/upchan_${setup}.jl >output/upchan_${setup}.out 2>&1 &
done
wait

# Each driver runs all test cases (`:tone`, `:constant`, `:impulse`, and the
# tone again with a wrapped time window) and prints one "Found N errors" line
# per case.
expected_results=4

status=0
for setup in ${setups}; do
    # Distinguish "ran and reported" from "crashed before reporting". Without
    # this a crash looks exactly like a mismatch.
    results=$(grep -cE '^Found [0-9]+ errors' "output/upchan_${setup}.out" || true)
    if [ "${results}" -ne ${expected_results} ]; then
        echo "${setup}: ERROR (${results} of ${expected_results} test cases reported a result)"
        tail -20 "output/upchan_${setup}.out"
        status=1
        continue
    fi
    if [ "$(grep -c '^Found 0 errors' "output/upchan_${setup}.out" || true)" -eq ${expected_results} ]; then
        echo "${setup}: PASS"
    else
        echo "${setup}: FAIL"
        sed -n '/^Checking results/,$p' "output/upchan_${setup}.out" | head -20
        status=1
    fi
done

exit ${status}
