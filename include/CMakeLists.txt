cmake_minimum_required( VERSION 2.8 )
project( libinclude )

include_directories (${KOTEKAN_SOURCE_DIR}/include)
include_directories (${KOTEKAN_SOURCE_DIR}/generated)

set (LIBINCLUDE_LIB_SOURCES
              mongoose.c
              timing_mach.c)

#if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
#    set (LIBINCLUDE_LIB_SOURCES ${LIBINCLUDE_LIB_SOURCES} timing_mach.c)
#endif()

add_library( libinclude
             ${LIBINCLUDE_LIB_SOURCES} )

target_link_libraries( libinclude m )