project(kotekan_core)

set(KOTEKAN_CORE_SOURCES
    basebandReadoutManager.cpp
    basebandApiManager.cpp
    buffer.c
    bufferContainer.cpp
    bufferFactory.cpp
    Config.cpp
    configUpdater.cpp
    errors.c
    kotekanLogging.cpp
    kotekanMode.cpp
    metadata.c
    metadataFactory.cpp
    prometheusMetrics.cpp
    restServer.cpp
    Stage.cpp
    StageFactory.cpp)

add_library(kotekan_core ${KOTEKAN_CORE_SOURCES})

target_link_libraries(kotekan_core ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(kotekan_core libexternal)

if(${NUMA_FOUND})
    target_link_libraries(kotekan_core numa)
endif()

if(OPENSSL_FOUND)
    target_link_libraries(kotekan_core ${OPENSSL_LIBRARIES})
endif()
