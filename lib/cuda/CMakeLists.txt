project(kotekan_cuda)

if(${CUDA_ARCHITECTURE})
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
else()
    # NVIDIA Ampere
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

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
    cudaSyncStream.cpp
    cudaSyncInput.cpp
    cudaSyncOutput.cpp
    # Kernels
    cudaCorrelatorAstron.cpp
    cudaShuffleAstron.cu
    cudaBasebandBeamformer_chord.cpp
    cudaBasebandBeamformer_pathfinder.cpp
    cudaFRBBeamformer_chord.cpp
    cudaFRBBeamReformer.cpp
    quantize_kernel_wrapper.cu
    cudaQuantize.cpp
    cudaRechunk.cpp
    cudaTransposeKernel_chord.cpp
    cudaTransposeKernel_pathfinder.cpp
    cudaUpchannelizer_U16.cpp
    cudaUpchannelizer_U32.cpp
    cudaUpchannelizer_U64.cpp
    cudaUpchannelizer_U128.cpp
    chordMVPSetup.cpp)

# C++17
set_property(TARGET kotekan_cuda PROPERTY CUDA_STANDARD 17)

find_package(GPUTILS)
if(${GPUTILS_FOUND})
    message("gputils support enabled")
    find_package(N2K)
    if(${N2K_FOUND})
        message("n2k support enabled")
        target_include_directories(kotekan_cuda SYSTEM PRIVATE ${N2K_INCLUDE_DIR})
        target_link_libraries(kotekan_cuda INTERFACE ${N2K_LIBRARY})
        target_sources(kotekan_cuda PRIVATE cudaCorrelator.cpp)
    else()
        message("n2k not found, support disabled in kotekan_cuda")
    endif()
    target_include_directories(kotekan_cuda SYSTEM PRIVATE ${GPUTILS_INCLUDE_DIR})
    target_link_libraries(kotekan_cuda INTERFACE ${GPUTILS_LIBRARY})

else()
    message("gputils not found, support disabled in kotekan_cuda")
endif()

target_link_libraries(
    kotekan_cuda
    PRIVATE libexternal kotekan_libs -lstdc++
    INTERFACE ${CUDA_LIBRARIES} cublas cuda nvrtc nvptxcompiler_static)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --forward-unknown-to-host-compiler")

target_include_directories(kotekan_cuda SYSTEM PUBLIC ${CUDA_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
target_include_directories(kotekan_cuda PUBLIC .)

target_compile_options(kotekan_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode
                                            arch=compute_86,code=sm_86 -lineinfo>)

add_dependencies(kotekan_cuda kotekan_gpu)
