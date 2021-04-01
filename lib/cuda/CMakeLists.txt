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
    PRIVATE libexternal kotekan_libs -lstdc++
    INTERFACE ${CUDA_LIBRARIES})

#set(CUDA_PROPAGATE_HOST_FLAGS OFF)
#set_property(TARGET kotekan_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")

set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --forward-unknown-to-host-compiler")

target_include_directories(kotekan_cuda PUBLIC . ${CUDA_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})

target_compile_options(kotekan_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode
                                            arch=compute_86,code=sm_86 -lineinfo>)

add_dependencies(kotekan_cuda kotekan_gpu)



#project( kotekan_cuda )
#
#include_directories(${CUDA_INCLUDE_DIR})
#
#include_directories (${KOTEKAN_SOURCE_DIR}/include)
#
#include_directories (${KOTEKAN_SOURCE_DIR}/lib)
#include_directories (${KOTEKAN_SOURCE_DIR}/lib/gpu)
#include_directories (${KOTEKAN_SOURCE_DIR}/lib/cuda)
#include_directories (${KOTEKAN_SOURCE_DIR}/lib/cuda/kernels)
#
#set ( KOTEKAN_CUDA_SOURCES
#      cudaCommand.cpp
#      cudaDeviceInterface.cpp
#      cudaEventContainer.cpp
#      cudaProcess.cpp
#
## Copy-in & general-purpose:
#      cudaInputData.cpp
#      cudaOutputData.cpp
#      cudaOutputDataZero.cpp
#
## Simple test
#        cudaCorrelatorKernel.cu
#        #cudaCorrelatorRomein.cpp
#    )
#
#add_library ( kotekan_cuda ${KOTEKAN_CUDA_SOURCES} )
#
#target_compile_options(kotekan_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_75,code=sm_75 -lineinfo -src-in-ptx>)
#
#add_custom_target(cuda_kernel_copy
#        COMMAND ${CMAKE_COMMAND} -E copy_directory
#        "${KOTEKAN_SOURCE_DIR}/lib/cuda/kernels"
#        "kernels/")
#
#add_dependencies( kotekan_cuda cuda_kernel_copy kotekan_gpu )