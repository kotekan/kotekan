project(kotekan_hip)

include_directories(${HIP_INCLUDE_DIR})

include_directories (${KOTEKAN_SOURCE_DIR}/external)

include_directories (${KOTEKAN_SOURCE_DIR}/lib)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/gpu)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/hip)

add_definitions(-D__HIP_PLATFORM_HCC__)

HIP_ADD_LIBRARY(kotekan_hip ${KOTEKAN_HIP_SOURCES}
        hipCommand.cpp
        hipDeviceInterface.cpp
        hipEventContainer.cpp
        hipProcess.cpp
        # Copy-in & general-purpose:
        hipInputData.cpp
        hipOutputData.cpp)

target_link_libraries(kotekan_hip libexternal)

add_dependencies(kotekan_hip kotekan_gpu)
