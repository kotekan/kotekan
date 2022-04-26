project(kotekan_cuda)

add_library(
    kotekan_cuda
    # Core
    cudaCommand.cpp
    cudaDeviceInterface.cpp
    cudaEventContainer.cpp
    cudaProcess.cpp
    # Copy & general-purpose:
    cudaInputData.cpp
    cudaOutputData.cpp
    cudaOutputDataZero.cpp
    # Kernels
    cudaCorrelatorAstron.cpp
    cudaShuffleAstron.cu
    kernels/FRBBeamformer/average.cu
    kernels/FRBBeamformer/average.fatbin.c)

target_link_libraries(
    kotekan_cuda
    PRIVATE libexternal kotekan_libs -lstdc++
    INTERFACE ${CUDA_LIBRARIES} cuda nvrtc)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --forward-unknown-to-host-compiler")

target_include_directories(kotekan_cuda PUBLIC . ${CUDA_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})

target_compile_options(kotekan_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode
                                            arch=compute_86,code=sm_86 -lineinfo>)

add_dependencies(kotekan_cuda kotekan_gpu)
