project(kotekan_utils)

add_library(
    kotekan_utils
    fpga_header_functions.c
    gateSpec.cpp
    gpsTime.c
    nt_memcpy.c
    nt_memset.c
    output_formating.c
    time_tracking.c
    type.cpp
    util.c
    visBuffer.cpp
    visUtil.cpp
    visFile.cpp
    visFileRaw.cpp
    visFileRing.cpp
    tx_utils.cpp
    datasetManager.cpp
    dataset.cpp
    pulsarTiming.cpp
    datasetState.cpp
    restClient.cpp
    BipBuffer.cpp
    Hash.cpp
    network_functions.cpp
    Stack.cpp)

target_link_libraries(kotekan_utils PRIVATE libexternal kotekan_core)

# HDF5 stuff
if(${USE_HDF5})
    target_sources(kotekan_utils PRIVATE visFileH5.cpp visFileArchive.cpp)
    include_directories(${HDF5_INCLUDE_DIRS})
    include_directories(SYSTEM /kotekan/HighFive/include/)
    target_link_libraries(kotekan_utils PRIVATE ${HDF5_HL_LIBRARIES} ${HDF5_LIBRARIES})
    add_dependencies(kotekan_utils highfive)
endif()

# Libevent base&pthreads is required for the restClient
set(CMAKE_MODULE_PATH ${KOTEKAN_SOURCE_DIR}/cmake)
find_package(LIBEVENT REQUIRED)
target_link_libraries(kotekan_utils PUBLIC ${LIBEVENT_BASE} ${LIBEVENT_PTHREADS})

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    target_sources(kotekan_utils PRIVATE osxBindCPU.cpp)
endif()
