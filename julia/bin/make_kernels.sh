#!/bin/bash

# This script generates theall-based CUDA kernels.
# Run it from the kotekan base directory like ./julia/bin/make_kernels.sh

set -euxo pipefail

./julia/bin/bb.sh
./julia/bin/upchan.sh
./julia/bin/frb.sh
