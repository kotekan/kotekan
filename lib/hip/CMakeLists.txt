project(kotekan_hip)

add_definitions(-D__HIP_PLATFORM_HCC__)

hip_add_library(
    kotekan_hip
    hipCommand.cpp
    hipDeviceInterface.cpp
    hipEventContainer.cpp
    hipProcess.cpp
    # Copy-in & general-purpose:
    hipInputData.cpp
    hipOutputData.cpp)

target_link_libraries(
    kotekan_hip
    PRIVATE libexternal kotekan_libs
    INTERFACE ${HIP_LIBRARIES})

target_include_directories(kotekan_hip PUBLIC . ${HIP_INCLUDE_DIR} ${HIP_INCLUDE_DIRS})

add_dependencies(kotekan_hip kotekan_gpu)
