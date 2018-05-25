cmake_minimum_required( VERSION 2.8 )
project( kotekan_opencl )

include_directories(${OPENCL_INCLUDE_DIRS})

include_directories (${KOTEKAN_SOURCE_DIR}/include)

include_directories (${KOTEKAN_SOURCE_DIR}/lib)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/opencl)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/opencl/kernels)
include_directories (${KOTEKAN_SOURCE_DIR}/kotekan/opencl/kernels)

add_library ( kotekan_opencl
              clDeviceInterface.cpp
              clEventContainer.cpp
              clCommand.cpp
              clCommandFactory.cpp
              clProcess.cpp
#              timer.cpp

# Copy-in & general-purpose:
              clInputData.cpp

# CHIME N2 Kernels & copy-out:
              clOutputDataZero.cpp
              clPresumZero.cpp
              clPresumKernel.cpp
              clPreseedKernel.cpp
              clCorrelatorKernel.cpp
              clOutputData.cpp

# CHIME/Pulsar Kernels & copy-out:
              clBeamformPhaseData.cpp
              clBeamformKernel.cpp
              clOutputBeamformResult.cpp

#RFI Kernels
#              rfi_kernel.cpp
#              output_rfi.cpp

              )

add_custom_target(kernel_copy
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${KOTEKAN_SOURCE_DIR}/lib/opencl/kernels"
        "kernels/")

#target_link_libraries( kotekan_opencl kotekan_libs )

add_dependencies( kotekan_opencl kernel_copy )
