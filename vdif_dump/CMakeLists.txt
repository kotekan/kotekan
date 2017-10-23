include_directories (${KOTEKAN_SOURCE_DIR}/lib)

link_directories (${CHIME_BINARY_DIR}/lib)

# TODO re-enable this.
include_directories( ${OPENCL_INCLUDE_DIRS} )
add_executable(vdif_dump vdif_dump.c)

target_link_libraries( vdif_dump chime )
target_link_libraries( vdif_dump ${CMAKE_THREAD_LIBS_INIT} )
target_link_libraries( vdif_dump m )
target_link_libraries( vdif_dump rt )