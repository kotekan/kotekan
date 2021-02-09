project(kotekan_utils)

add_library(
    kotekan_utils
    gateSpec.cpp
    nt_memcpy.c
    nt_memset.c
    output_formating.c
    type.cpp
    util.c
    FrameView.cpp
    HFBFrameView.cpp
    visBuffer.cpp
    visUtil.cpp
    visFile.cpp
    visFileRaw.cpp
    hfbFileRaw.cpp
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
    Stack.cpp
    Telescope.cpp
    ICETelescope.cpp
    CHIMETelescope.cpp
    SystemInterface.cpp)

target_link_libraries(kotekan_utils PRIVATE libexternal kotekan_libs)
target_include_directories(kotekan_utils PUBLIC .)

# HDF5 stuff
if(${USE_HDF5})
    target_sources(kotekan_utils PRIVATE visFileH5.cpp visFileArchive.cpp HFBFileArchive.cpp)
    target_include_directories(kotekan_utils SYSTEM INTERFACE ${HDF5_INCLUDE_DIRS}
                                                              ${HIGHFIVE_PATH}/include)
    target_link_libraries(kotekan_utils PRIVATE ${HDF5_HL_LIBRARIES} ${HDF5_LIBRARIES})
    add_dependencies(kotekan_utils highfive)
endif()

# Libevent base&pthreads is required for the restClient
find_package(LIBEVENT REQUIRED)
target_link_libraries(kotekan_utils PUBLIC ${LIBEVENT_BASE} ${LIBEVENT_PTHREADS})

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    target_sources(kotekan_utils PRIVATE osxBindCPU.cpp)
endif()
