To run kotekan FRB mode, using 4 GPUs:
sudo ./kotekan -c ../../kotekan/kotekan_frb.yaml

To run kotekan FRB mode with CPU verification, using only one GPU:
sudo ./kotekan -c ../../kotekan/kotekan_frb_verify-cpu.yaml

Edited these files:
kotekan/frbMode.cpp
kotekan/frbMode.hpp
kotekan/kotekan_frb.yaml
kotekan/kotekan_frb_verify-cpu.yaml
lib/hsa/kernels/transpose.cl
lib/hsa/kernels/upchannelize.cl
lib/hsa/CMakeLists.txt
lib/hsa/hsaBeamformKernel.cpp
lib/hsa/hsaBeamformTranspose.cpp
lib/hsa/hsaBeamformTranspose.hpp
lib/hsa/hsaBeamformUpchan.cpp
lib/hsa/hsaBeamformUpchan.hpp
lib/hsa/hsaCommandFactory.cpp
lib/hsa/hsaBeamformOutput.cpp
lib/hsa/hsaBeamformOutput.hpp
lib/testing/gpuBeamformSimulate.cpp
lib/testing/gpuBeamformSimulate.hpp
lib/testing/testDataGen.cpp
lib/processFactory.cpp
CMakeLists.txt
kotekan/CMakeLists.txt
kotekan/kotekan.cpp
lib/testing/testDataCheck.hpp

Removed:
lib/hsa/kernels/Upchannelize.cl

**Notes on potential conflict:
- Changed testDataCheck from int32 to float in lib/processFactory.cpp
- Needed to add rocm path in CMakeLists.txt
- Needed to add -ldl for DPDK_LIBS in kotekan/CMakeLists.txt
