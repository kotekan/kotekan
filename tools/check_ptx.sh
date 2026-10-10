#!/bin/bash
# Assemble every generated PTX kernel with ptxas, for the architecture its
# wrapper in lib/cuda/generated requests. The build never compiles these
# files: each wrapper compiles its PTX at run time, so this is the only
# compile check for kernels the GPU tests do not run. Needs no GPU.
# Usage: tools/check_ptx.sh [ptxas]   (run from the repository root)

ptxas=${1:-ptxas}
rc=0
for cpp in lib/cuda/generated/cuda*.cpp; do
    name=$(basename "$cpp" .cpp)
    ptx=$(grep -o 'build_ptx("[^"]*"' "$cpp" | cut -d'"' -f2)
    arch=$(grep -o -- '--gpu-name=sm_[0-9]*' "$cpp" | cut -d= -f2)
    if out=$("$ptxas" -arch="$arch" -o /dev/null "$ptx" 2>&1); then
        echo "ok   $arch $ptx"
    else
        echo "$out" | grep -v Advisory
        echo "FAIL $arch $ptx"
        rc=1
    fi
done
exit $rc
