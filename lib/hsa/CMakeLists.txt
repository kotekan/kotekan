project(libkotekan_hsa)

add_library(
    kotekan_hsa
    hsaEventContainer.cpp
    hsaBase.c
    hsaCommand.cpp
    hsaDeviceInterface.cpp
    hsaProcess.cpp
    hsaSubframeCommand.cpp
    # Copy-in & general-purpose:
    hsaInputData.cpp
    hsaHostToDeviceCopy.cpp
    hsaBarrier.cpp
    hsaAsyncCopyGain.cpp
    # RFI Kernels & copy-out:
    hsaRfiVdif.cpp
    hsaRfiTimeSum.cpp
    hsaRfiInputSum.cpp
    hsaRfiBadInput.cpp
    hsaRfiZeroData.cpp
    hsaRfiBadInputOutput.cpp
    hsaRfiMaskOutput.cpp
    hsaRfiOutput.cpp
    hsaRfiVarOutput.cpp
    hsaRfiUpdateBadInputs.cpp
    # CHIME N2 Kernels & copy-out:
    hsaOutputDataZero.cpp
    hsaPresumZero.cpp
    hsaPresumKernel.cpp
    hsaCorrelatorKernel.cpp
    hsaOutputData.cpp
    # CHIME/FRB Kernels & copy-out:
    hsaBeamformReorder.cpp
    hsaBeamformKernel.cpp
    hsaBeamformTranspose.cpp
    hsaBeamformUpchan.cpp
    hsaBeamformUpchanHFB.cpp
    hsaBeamformOutput.cpp
    hsaBeamformHFBOutput.cpp
    hsaBeamformHFBSum.cpp
    # CHIME/Pulsar Kernels & copy-out:
    hsaTrackingBeamform.cpp
    hsaTrackingBeamformOutput.cpp
    hsaTrackingUpdatePhase.cpp)

target_link_libraries(kotekan_hsa PRIVATE libexternal kotekan_libs ${HSA_RUNTIME_LIBRARY})
target_include_directories(
    kotekan_hsa
    PUBLIC . /opt/rocm/hsa/include/
    INTERFACE hsa-runtime64)

add_subdirectory(kernels)
add_dependencies(kotekan_hsa makeHSACOKernels)
