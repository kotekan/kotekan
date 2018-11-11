cmake_minimum_required( VERSION 2.8 )
project( libinclude )

include_directories (${KOTEKAN_SOURCE_DIR}/include)
include_directories (${KOTEKAN_SOURCE_DIR}/generated)

# Currently we don't have any builds in `/include` if this changes uncomment the line
# below and other commented references to libinclude
#add_library( libinclude )
