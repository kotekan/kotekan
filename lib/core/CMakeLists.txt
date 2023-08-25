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
    cpuMonitor.cpp
    errors.c
    kotekanLogging.cpp
    kotekanMode.cpp
    kotekanTrackers.cpp
    metadata.c
    metadataFactory.cpp
    prometheusMetrics.cpp
    juliaManager.cpp
    restServer.cpp
    Stage.cpp
    StageFactory.cpp)
target_include_directories(kotekan_core PUBLIC .)

# Julia is a covnenient language to implement stages for testing and development
find_package(Julia)
if(${Julia_FOUND})
    message("Julia support enabled")
    target_include_directories(kotekan_core PRIVATE ${Julia_INCLUDE_DIRS})
    target_link_libraries(kotekan_core PUBLIC ${Julia_LIBRARY})
    add_definitions(-DWITH_JULIA)
else()
    message("Julia not found, support disabled in kotekan_core")
endif()

# Libnuma is optionally used by buffer.c
find_package(NUMA)
if(${NUMA_FOUND})
    add_definitions(-DWITH_NUMA)
    message("NUMA support enabled")
    target_link_libraries(kotekan_core PRIVATE numa)
else()
    message("libnuma not found, NUMA support will be disabled in kotekan_core")
endif()

# Optionally include OpenSSL crypto lib for hash function support (needed by Config and kotekanMode)
find_package(OpenSSL)
if(OPENSSL_FOUND)
    target_link_libraries(kotekan_core PRIVATE ${OPENSSL_CRYPTO_LIBRARY})
    target_compile_definitions(kotekan_core PUBLIC WITH_SSL)
endif()

# Libevent base & pthreads is required for the restServer
find_package(LIBEVENT REQUIRED)

target_link_libraries(
    kotekan_core
    PRIVATE libexternal ${CMAKE_THREAD_LIBS_INIT} kotekan_libs
    PUBLIC ${LIBEVENT_BASE} ${LIBEVENT_CORE} ${LIBEVENT_PTHREADS} ${LIBEVENT_EXTRA} event)

target_include_directories(kotekan_core INTERFACE ${LIBEVENT_INCLUDE_DIR})
