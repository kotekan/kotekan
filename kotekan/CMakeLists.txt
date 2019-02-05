cmake_minimum_required(VERSION 2.8)

PROJECT (kotekan)

SET(SOURCE_EXE kotekan.cpp)

include_directories (${KOTEKAN_SOURCE_DIR}/lib)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/core)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/version)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/utils)
include_directories (${KOTEKAN_SOURCE_DIR}/include)

link_directories (${KOTEKAN_BINARY_DIR}/lib)

ADD_DEFINITIONS(-mssse3 -D__STDC_LIMIT_MACROS)

add_executable(kotekan ${SOURCE_EXE})

target_link_libraries( kotekan m dl pthread numa )
target_link_libraries( kotekan kotekan_libs )

install(TARGETS kotekan DESTINATION /usr/local/bin COMPONENT binaries)
