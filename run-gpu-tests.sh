#! /bin/bash

# This script should run on Blue

# Clean
rm -rf cmake-gpu-tests

# Configure
cmake -S . -B cmake-gpu-tests -G Ninja -DUSE_CUDA=ON -DUSE_HDF5=ON -DUSE_Julia=ON -DUSE_OMP=ON -DWITH_TESTS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo

# Build
cmake --build cmake-gpu-tests --target kotekan/kotekan

tests=(
    config/tests/onehot_cuda_baseband_beamformer_phase.yaml
    config/tests/onehot_cuda_baseband_beamformer_voltage.yaml
    config/tests/onehot_cuda_frb_beamformer_voltage.yaml
    config/tests/onehot_cuda_upchan.yaml

    config/tests/verify_cuda_baseband_beamformer.yaml
    config/tests/verify_cuda_n2_astron.yaml
    config/tests/verify_cuda_n2k.yaml

    config/tests/julia_hello_world.yaml
)

echo "Running Kotekan self-tests"

for test in ${tests[@]}; do
    echo "Running test $test..."
    ./cmake-gpu-tests/kotekan/kotekan --bind-address 0:23000 --config ${test}
    case $? in
        (2) echo "[SUCCESS]"
            ;;
        (3) echo "[FAILURE]"
            exit 1
            ;;
        (*) echo "[UNKNOWN RESULTS]"
            exit 2
            ;;
    esac
done

echo "Done."
