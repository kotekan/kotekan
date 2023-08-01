#! /bin/bash

# Build e.g. with:
# Clean:
#     rm -rf cmake-build
# Configure:
#     cmake -S . -B cmake-build -G Ninja -DUSE_CUDA=ON -DUSE_Julia=ON -DUSE_OMP=ON -DWITH_TESTS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
# Build:
#     cmake --build cmake-build --target kotekan/kotekan

tests=(
    config/tests/onehot_cuda_baseband_beamformer_phase.yaml
    config/tests/onehot_cuda_baseband_beamformer_voltage.yaml
    config/tests/onehot_cuda_frb_beamformer_voltage.yaml
    config/tests/onehot_cuda_upchan.yaml
)

echo "Running Kotekan self-tests"

for test in ${tests[@]}; do
    echo "Running test $test..."
    ./cmake-build/kotekan/kotekan --bind-address 0:23000 --config ${test}
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
