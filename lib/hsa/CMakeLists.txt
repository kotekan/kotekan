cmake_minimum_required( VERSION 2.8 )
project( libkotekan_hsa )

include_directories (${KOTEKAN_SOURCE_DIR}/include)
include_directories (${KOTEKAN_SOURCE_DIR}/lib)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/hsa)

include_directories ( /opt/rocm/hsa/include/ )

add_library ( kotekan_hsa
              signalContainer.cpp
              hsaBase.c
              hsaCommand.cpp
              hsaCommandFactory.cpp
              hsaDeviceInterface.cpp
              hsaProcess.cpp

# Copy-in & general-purpose:
              hsaInputData.cpp
              hsaInputLostSamples.cpp
              hsaBarrier.cpp

# RFI Kernels & copy-out:
              hsaRfiVdif.cpp
              hsaRfiTimeSum.cpp
              hsaRfiInputSum.cpp
              hsaRfiBadInput.cpp
              hsaRfiZeroData.cpp
              hsaRfiBadInputOutput.cpp
              hsaRfiMaskOutput.cpp
              hsaRfiOutput.cpp

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
              hsaBeamformOutput.cpp
              hsaBeamformOutputSolo.cpp

# CHIME/Pulsar Kernels & copy-out:
              hsaBeamformPulsar.cpp
              hsaBeamformPulsarOutput.cpp
              hsaBeamformPulsarOneFeed.cpp
              hsaPulsarUpdatePhase.cpp
              )

target_link_libraries( kotekan_hsa kotekan_utils )

add_subdirectory(kernels)
add_dependencies( kotekan_hsa makeHSACOKernels )




