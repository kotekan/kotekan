project(kotekan_cuda)

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

target_link_libraries(
    kotekan_cuda
    PRIVATE libexternal -lstdc++
    INTERFACE ${CUDA_LIBRARIES})
target_include_directories(kotekan_cuda PUBLIC . ${CUDA_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})

target_compile_options(kotekan_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode
                                            arch=compute_75,code=sm_75 -lineinfo>)

add_dependencies(kotekan_cuda kotekan_gpu)
