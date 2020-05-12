project(kotekan_core)

add_library(
    kotekan_core
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

target_link_libraries(kotekan_core PRIVATE ${CMAKE_THREAD_LIBS_INIT} libexternal)

if(${NUMA_FOUND})
    target_link_libraries(kotekan_core PRIVATE numa)
endif()

# Optionally include OpenSSL crypto lib for hash function support (needed by Config and kotekanMode)
set(CMAKE_MODULE_PATH ${KOTEKAN_SOURCE_DIR}/cmake)
find_package(OpenSSL)
if(OPENSSL_FOUND)
    target_link_libraries(kotekan_core PRIVATE ${OPENSSL_CRYPTO_LIBRARY})
    target_compile_definitions(kotekan_core PUBLIC WITH_SSL)
endif()

# Libevent base & pthreads is required for the restServer
find_package(LIBEVENT REQUIRED)
target_link_libraries(
    kotekan_core
    PUBLIC ${LIBEVENT_BASE}
    PRIVATE ${LIBEVENT_PTHREADS})
