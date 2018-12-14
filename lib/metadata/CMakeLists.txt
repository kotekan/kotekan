cmake_minimum_required( VERSION 2.8 )
project( kotekan_metadata )

include_directories (${KOTEKAN_SOURCE_DIR}/include)

set ( KOTEKAN_METADATA_SOURCES
      chimeMetadata.c
    )

# There are no symbols in this entire library,
# the following lines prevent warnings on MacOS.
# If someone ever adds symbols this library,
# the lines below should be removed!
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    SET(CMAKE_C_ARCHIVE_CREATE   "<CMAKE_AR> Scr <TARGET> <LINK_FLAGS> <OBJECTS>")
    SET(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> Scr <TARGET> <LINK_FLAGS> <OBJECTS>")
    SET(CMAKE_C_ARCHIVE_FINISH   "")
    SET(CMAKE_CXX_ARCHIVE_FINISH "")
endif ()

add_library ( kotekan_metadata
              ${KOTEKAN_METADATA_SOURCES})
