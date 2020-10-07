project(kotekan_opencl)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    add_definitions(-DCL_SILENCE_DEPRECATION)
endif()

add_library(
    kotekan_opencl
    clCommand.cpp
    clDeviceInterface.cpp
    clEventContainer.cpp
    clProcess.cpp
    # Copy-in & general-purpose:
    clInputData.cpp
    # CHIME N2 Kernels & copy-out:
    clOutputDataZero.cpp
    clPresumZero.cpp
    clPresumKernel.cpp
    clPreseedKernel.cpp
    clCorrelatorKernel.cpp
    clOutputData.cpp
    clKVCorr.cpp
    # CHIME/Pulsar Kernels & copy-out:
    clBeamformPhaseData.cpp
    clBeamformKernel.cpp
    clOutputBeamformResult.cpp
    # TODO: if these are not compiled, should we remove these RFI Kernels: clRfiTimeSum.cpp
    # clRfiInputSum.cpp clRfiOutput.cpp
)

target_link_libraries(
    kotekan_opencl
    INTERFACE ${OpenCL_LIBRARY}
    PRIVATE libexternal kotekan_libs)
target_include_directories(kotekan_opencl PUBLIC .)
target_include_directories(kotekan_opencl SYSTEM INTERFACE ${OpenCL_INCLUDE_DIR})

add_custom_target(
    kernel_copy
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${KOTEKAN_SOURCE_DIR}/lib/opencl/kernels" "kernels/"
    COMMENT "Copying OpenCL kernels")

add_dependencies(kotekan_opencl kernel_copy kotekan_gpu)
