cmake_minimum_required( VERSION 2.8 )
project( libinclude )

include_directories (${KOTEKAN_SOURCE_DIR}/include)
include_directories (${KOTEKAN_SOURCE_DIR}/generated)

add_library( libinclude
             mongoose.c )

target_link_libraries( libinclude m )