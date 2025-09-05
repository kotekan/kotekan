project(kotekan_core)

add_library(
    kotekan_core
    basebandReadoutManager.cpp
    basebandApiManager.cpp
    buffer.cpp
    ringbuffer.cpp
    bufferContainer.cpp
    bufferFactory.cpp
    Config.cpp
    configUpdater.cpp
    cpuMonitor.cpp
    errors.c
    kotekanLogging.cpp
    kotekanMode.cpp
    kotekanTrackers.cpp
    metadata.cpp
    metadataFactory.cpp
    prometheusMetrics.cpp
    restServer.cpp
    Stage.cpp
    StageFactory.cpp
    $<$<BOOL:${Julia_FOUND}>:juliaManager.cpp>)

target_include_directories(kotekan_core PUBLIC .)

if(${Julia_FOUND})
    target_include_directories(kotekan_core INTERFACE ${JULIA_INCLUDE_DIRS})
    target_link_libraries(kotekan_core INTERFACE ${JULIA_LIBRARY})
    add_definitions(-DWITH_JULIA)
endif()

# Libnuma is optionally used by buffer.c
find_package(NUMA)
if(${NUMA_FOUND})
    add_definitions(-DWITH_NUMA)
    set(NUMA_ENABLED
        ON
        CACHE BOOL "NUMA available")
    set(NUMA_REASON
        "enabled"
        CACHE STRING "")
    target_link_libraries(kotekan_core PRIVATE numa)
else()
    set(NUMA_ENABLED
        OFF
        CACHE BOOL "NUMA available")
    set(NUMA_REASON
        "not found"
        CACHE STRING "")
endif()

# Optionally include OpenSSL crypto lib for hash function support (needed by Config and kotekanMode)
find_package(OpenSSL)
if(OPENSSL_FOUND)
    set(OPENSSL_ENABLED
        ON
        CACHE BOOL "OpenSSL available")
    set(OPENSSL_REASON
        "found"
        CACHE STRING "")
    target_link_libraries(kotekan_core PRIVATE ${OPENSSL_CRYPTO_LIBRARY})
    target_compile_definitions(kotekan_core PUBLIC WITH_SSL)
else()
    set(OPENSSL_ENABLED
        OFF
        CACHE BOOL "OpenSSL available")
    set(OPENSSL_REASON
        "not found"
        CACHE STRING "")
endif()

# Libevent base & pthreads is required for the restServer
find_package(LIBEVENT REQUIRED)

target_link_libraries(
    kotekan_core
    PRIVATE libexternal ${CMAKE_THREAD_LIBS_INIT} kotekan_libs
    PUBLIC ${LIBEVENT_BASE} ${LIBEVENT_CORE} ${LIBEVENT_PTHREADS} ${LIBEVENT_EXTRA} event)

target_include_directories(kotekan_core INTERFACE ${LIBEVENT_INCLUDE_DIR})
