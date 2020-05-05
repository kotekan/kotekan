project(kotekan)

set(SOURCE_EXE kotekan.cpp)

include_directories(${KOTEKAN_SOURCE_DIR}/lib)
include_directories(${KOTEKAN_SOURCE_DIR}/lib/core)
include_directories(${KOTEKAN_SOURCE_DIR}/lib/version)
include_directories(${KOTEKAN_SOURCE_DIR}/lib/utils)

link_directories(${KOTEKAN_BINARY_DIR}/lib)

add_definitions(-mssse3 -D__STDC_LIMIT_MACROS)

add_executable(kotekan ${SOURCE_EXE})

if(${NUMA_FOUND})
    target_link_libraries(kotekan PRIVATE numa)
endif()

target_link_libraries(kotekan PRIVATE libexternal m pthread kotekan_libs)

install(
    TARGETS kotekan
    DESTINATION /usr/local/bin
    COMPONENT binaries)
