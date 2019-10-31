# There are no symbols in this entire library, and so it is created as an
# INTERFACE library. If any symbols are ever added it will need to be turned
# back into a standard CMake library

cmake_minimum_required( VERSION 3.0 )
project( kotekan_metadata )

include_directories (${KOTEKAN_SOURCE_DIR}/include)

add_library ( kotekan_metadata INTERFACE )
