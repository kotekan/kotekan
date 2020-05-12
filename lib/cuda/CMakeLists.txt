project(kotekan_cuda)

include_directories(${CUDA_INCLUDE_DIR})

include_directories(${KOTEKAN_SOURCE_DIR}/lib)
include_directories(${KOTEKAN_SOURCE_DIR}/lib/gpu)
include_directories(${KOTEKAN_SOURCE_DIR}/lib/cuda)
include_directories(${KOTEKAN_SOURCE_DIR}/lib/cuda/kernels)

add_library(
    kotekan_cuda
    cudaCommand.cpp
    cudaDeviceInterface.cpp
    cudaEventContainer.cpp
    cudaProcess.cpp
    # Copy-in & general-purpose:
    cudaInputData.cpp
    cudaOutputData.cpp
    cudaOutputDataZero.cpp
    # Simple test
    cudaCorrelatorKernel.cu)

target_link_libraries(kotekan_cuda libexternal)

target_compile_options(kotekan_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode
                                            arch=compute_75,code=sm_75 -lineinfo>)

add_dependencies(kotekan_cuda kotekan_gpu)
