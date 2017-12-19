cmake_minimum_required( VERSION 2.8 )
project( libkotekan_opencl )

include_directories(${OPENCL_INCLUDE_DIRS})

include_directories (${KOTEKAN_SOURCE_DIR}/include)

include_directories (${KOTEKAN_SOURCE_DIR}/lib)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/opencl)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/opencl/kernels)
include_directories (${KOTEKAN_SOURCE_DIR}/kotekan/opencl/kernels)

add_library ( kotekan_opencl
              device_interface.cpp
              callbackdata.cpp
              beamform_kernel.cpp
              beamform_phase_data.cpp
              correlator_kernel.cpp
              input_data_stage.cpp
              offset_kernel.cpp
              output_beamform_result.cpp
              output_data_result.cpp
              preseed_kernel.cpp
              gpu_command.cpp
              gpu_command_factory.cpp
              clProcess.cpp
              rfi_kernel.cpp
	      output_rfi.cpp
              )

add_custom_target(kernel_copy
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${KOTEKAN_SOURCE_DIR}/lib/opencl/kernels"
        "kernels/")
add_dependencies( kotekan_opencl kernel_copy )

target_link_libraries( kotekan_opencl kotekan_libs )
