PROJECT (kotekan)

SET(SOURCE_EXE kotekan.cpp)

include_directories (${KOTEKAN_SOURCE_DIR}/lib)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/core)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/version)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/utils)

link_directories (${KOTEKAN_BINARY_DIR}/lib)

ADD_DEFINITIONS(-mssse3 -D__STDC_LIMIT_MACROS)

add_executable(kotekan ${SOURCE_EXE})

if (${NUMA_FOUND})
    target_link_libraries( kotekan numa )
endif()

target_link_libraries( kotekan libexternal )
target_link_libraries( kotekan m dl pthread )
target_link_libraries( kotekan kotekan_libs )

install(TARGETS kotekan DESTINATION /usr/local/bin COMPONENT binaries)
