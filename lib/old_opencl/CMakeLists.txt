cmake_minimum_required( VERSION 2.8 )
project( libkotekan_old_opencl )
set(CMAKE_MODULE_PATH ${CHIME_SOURCE_DIR}/cmake)
find_package( OpenCL REQUIRED )


include_directories( ${OPENCL_INCLUDE_DIRS} )

include_directories (${CHIME_SOURCE_DIR}/include)

include_directories (${CHIME_SOURCE_DIR}/lib)
include_directories (${CHIME_SOURCE_DIR}/lib/old_opencl)

add_library ( kotekan_old_opencl
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
              beamform_incoherent_kernel.cpp
              output_beamform_incoh_result.cpp
              gpu_thread.cpp
              gpu_command.cpp
              gpu_command_factory.cpp
              )

file(GLOB opencl_kernels "${CHIME_SOURCE_DIR}/kernels/*.cl")

#add_executable(kotekan ${SOURCE_EXE} ${open_cl_kernels})

add_custom_target(kernel_copy
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CHIME_SOURCE_DIR}/kernels"
        "kernels")
add_dependencies( kotekan_old_opencl kernel_copy )
