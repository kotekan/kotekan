cmake_minimum_required( VERSION 2.8 )
project( libinclude )

include_directories (${KOTEKAN_SOURCE_DIR}/include)
include_directories (${KOTEKAN_SOURCE_DIR}/generated)

set (LIBINCLUDE_LIB_SOURCES
              mongoose.c)

add_library( libinclude
             ${LIBINCLUDE_LIB_SOURCES} )

target_link_libraries( libinclude m )