cmake_minimum_required( VERSION 2.8 )
project( libkotekan_hsa )

include_directories (${KOTEKAN_SOURCE_DIR}/include)
include_directories (${KOTEKAN_SOURCE_DIR}/lib)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/hsa)

include_directories ( /opt/rocm/hsa/include/ )

add_library ( kotekan_hsa
              signalContainer.cpp
              hsaBase.c
              hsaBarrier.cpp
              hsaCorrelatorKernel.cpp
              hsaInputData.cpp
              hsaOutputData.cpp
              hsaOutputDataZero.cpp
              hsaPreseedKernel.cpp
              hsaPresumZero.cpp
              hsaBeamformKernel.cpp
              hsaBeamformTranspose.cpp
              hsaBeamformUpchan.cpp
              hsaBeamformOutput.cpp
              hsaCommand.cpp
              hsaCommandFactory.cpp
              hsaDeviceInterface.cpp
              hsaProcess.cpp
              hsaRfiVdif.cpp
              hsaRfi.cpp
              hsaRfiOutput.cpp
              )

add_subdirectory(kernels)
add_dependencies( kotekan_hsa makeHSACOKernels )
