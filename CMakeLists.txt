cmake_minimum_required (VERSION 2.6)
project (CHIME)

find_package(Threads REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99 -O3 -D_GNU_SOURCE -Wall -msse4.2 -MMD -march=corei7-avx -mtune=corei7-avx")

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -ggdb ")

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/")

SET(CMAKE_INSTALL_PREFIX "/root/kotekan/")

install(DIRECTORY kernels DESTINATION . )

add_subdirectory(lib)
add_subdirectory(kotekan)
add_subdirectory(tools)
add_subdirectory(scripts)
add_subdirectory(vdif_dump)
