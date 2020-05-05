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

target_link_libraries(kotekan_core PRIVATE ${CMAKE_THREAD_LIBS_INIT} libexternal)

if(${NUMA_FOUND})
    target_link_libraries(kotekan_core PRIVATE numa)
endif()

if(OPENSSL_FOUND)
    target_link_libraries(kotekan_core PRIVATE ${OPENSSL_LIBRARIES})
endif()

# Libevent is required for the restServer
set(CMAKE_MODULE_PATH ${KOTEKAN_SOURCE_DIR}/cmake)
find_package(LIBEVENT REQUIRED)
target_link_libraries(kotekan_core PRIVATE ${LIBEVENT_LIBRARIES})
