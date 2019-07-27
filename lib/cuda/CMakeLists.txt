cmake_minimum_required( VERSION 2.8 )
project( kotekan_cuda )

include_directories(${CUDA_INCLUDE_DIR})

include_directories (${KOTEKAN_SOURCE_DIR}/include)

include_directories (${KOTEKAN_SOURCE_DIR}/lib)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/gpu)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/cuda)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/cuda/kernels)

set ( KOTEKAN_CUDA_SOURCES
      cudaCommand.cpp
      cudaDeviceInterface.cpp
      cudaEventContainer.cpp
      cudaProcess.cpp

# Copy-in & general-purpose:
      cudaInputData.cpp
      cudaOutputData.cpp

# Simple test
      cudaSaxpyKernel.cu
    )

add_library ( kotekan_cuda ${KOTEKAN_CUDA_SOURCES} )

#set_target_properties ( kotekan_cuda PROPERTIES LINKER_LANGUAGE CUDA )
add_dependencies( kotekan_cuda kotekan_gpu )
