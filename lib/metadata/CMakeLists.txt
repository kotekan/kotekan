cmake_minimum_required( VERSION 2.8 )
project( kotekan_metadata )

include_directories (${KOTEKAN_SOURCE_DIR}/include)

set (KOTEKAN_METADATA_SOURCES
              chimeMetadata.c
       )

add_library ( kotekan_metadata
              ${KOTEKAN_METADATA_SOURCES})
