cmake_minimum_required( VERSION 2.8 )
project( libkotekan_opencl )

include_directories(${OPENCL_INCLUDE_DIRS})

include_directories (${CHIME_SOURCE_DIR}/include)

include_directories (${CHIME_SOURCE_DIR}/lib)
include_directories (${CHIME_SOURCE_DIR}/lib/opencl)

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
              )

#file(GLOB opencl_kernels "${CHIME_SOURCE_DIR}/opencl/kernels/*.cl")

#add_executable(libkotekan_opencl ${SOURCE_EXE} ${open_cl_kernels})

add_custom_target(kernel_copy
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CHIME_SOURCE_DIR}/lib/opencl/kernels"
        "kernels")
add_dependencies( kotekan_opencl kernel_copy )

target_link_libraries( kotekan_opencl kotekan_libs )
