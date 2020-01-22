project( libkotekan_hsa )

include_directories (${KOTEKAN_SOURCE_DIR}/include)
include_directories (${KOTEKAN_SOURCE_DIR}/lib)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/gpu)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/hsa)

include_directories ( /opt/rocm/hsa/include/ )

set ( KOTEKAN_HSA_SOURCES
      hsaEventContainer.cpp
      hsaBase.c
      hsaCommand.cpp
      hsaDeviceInterface.cpp
      hsaProcess.cpp
      hsaSubframeCommand.cpp

# Copy-in & general-purpose:
      hsaInputData.cpp
      hsaInputLostSamples.cpp
      hsaInputCompressLostSamples.cpp
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
      hsaBeamformPulsar.cpp
      hsaBeamformPulsarOutput.cpp
      hsaPulsarUpdatePhase.cpp
    )

add_library ( kotekan_hsa
              ${KOTEKAN_HSA_SOURCES})

target_link_libraries( kotekan_hsa kotekan_utils )

add_subdirectory(kernels)
add_dependencies( kotekan_hsa makeHSACOKernels )




