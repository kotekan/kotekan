Compile kotekan for FRB on new HSA -- use precompiled hasco for now:

** On site new HSA:
cmake -DRTE_SDK=/opt/dpdk-stable-16.11.4/ -DRTE_TARGET=x86_64-native-linuxapp-gcc -DUSE_DPDK=ON -DUSE_HSA=ON -DCMAKE_BUILD_TYPE=Debug -DUSE_HDF5=yes -DHIGHFIVE_PATH=/root/andre/HighFive/ -DUSE_PRECOMPILED_OPENCL=ON  ..

**On site old HSA:
cmake -DUSE_PRECOMPILED_OPENCL=ON -DRTE_SDK=/opt/dpdk-stable-16.11.3/ -DRTE_TARGET=x86_64-native-linuxapp-gcc -DUSE_DPDK=ON -DUSE_HSA=ON -DCMAKE_BUILD_TYPE=Debug ..

** In LWlab new HSA (no dpdk):
cmake -DUSE_PRECOMPILED_OPENCL=ON -DRTE_TARGET=x86_64-native-linuxapp-gcc -DUSE_HSA=ON -DCMAKE_BUILD_TYPE=Debug ..

--------------------------

Run FRB with CPU verification, one GPU only
sudo ./kotekan -c ../../kotekan/kotekan_frb_verify-cpu-oneGPU.yaml

Run FRB with DPDK+packetizer, 4GPUs
sudo ./kotekan -c ../../kotekan_frb_test_dpdk-reorder-l0-packet.yaml
