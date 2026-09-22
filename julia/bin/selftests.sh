#!/bin/bash

# This script runs the in-file self-test of every Julia kernel generator and
# reports one line per generator. It exits nonzero if any of them fails.
# Run it from the kotekan base directory like ./julia/bin/selftests.sh
#
# It needs a GPU: every self-test runs the real kernel and compares the result
# against a CPU reference. CI cannot do this -- every job builds with
# `-DUSE_JULIA=OFF` and has no GPU -- so this script is the only backstop these
# generators have, which is why it insists on a nonzero exit status.
#
# The generators write their kernels to `output/`, and running one rewrites the
# checked-in files under `julia/output/` without the clang-format and
# JuliaFormatter passes that `julia/bin/*.sh` apply afterwards. So the whole
# project is copied into a scratch directory and the self-tests run there; the
# working tree is left untouched. `upchan_selftest.sh` is excluded because it
# has its own `selftest` setup, writes only to gitignored paths, and sweeps all
# upchannelization factors; run it separately.

set -euo pipefail

scriptdir=$(dirname "$0")
cd "$scriptdir/.."
juliadir=$(pwd)

# generator:driver
tests='
    xpose:xpose_chord
    xpose:xpose_hirax
    xpose:xpose_pathfinder
    xpose2048:xpose2048_chime
    chimefrb:chimefrb_chime_U16_K4
    chimefrb:chimefrb_chime_U16_K8
    frb:frb_selftest
    bb:bb_selftest
'

workdir=$(mktemp -d)
trap 'rm -rf "${workdir}"' EXIT
cp -a kernels src Project.toml Manifest.toml "${workdir}/"
mkdir -p "${workdir}/output"

echo "Running self-tests in ${workdir} (the working tree is not modified)"
cd "${workdir}"
for test in ${tests}; do
    driver=${test#*:}
    julia --project=@. --optimize "kernels/${driver}.jl" >"output/${driver}.out" 2>&1 &
done
wait

status=0
for test in ${tests}; do
    generator=${test%%:*}
    driver=${test#*:}
    out="${workdir}/output/${driver}.out"
    if grep -q '^Self-test passed\.$' "${out}"; then
        echo "${generator} (${driver}): PASS"
    elif grep -q 'SELF-TEST FAILED' "${out}"; then
        echo "${generator} (${driver}): FAIL"
        sed -n '/Checking results/,$p' "${out}" | head -20
        status=1
    else
        # No verdict at all: the run died before it got to the comparison.
        echo "${generator} (${driver}): ERROR (no self-test result -- the run did not finish)"
        tail -20 "${out}"
        status=1
    fi
done

# Keep the logs; the scratch copy of the sources is not worth keeping.
mkdir -p "${juliadir}/output/selftests"
cp "${workdir}"/output/*.out "${juliadir}/output/selftests/"
echo "Logs: julia/output/selftests/"

exit ${status}
