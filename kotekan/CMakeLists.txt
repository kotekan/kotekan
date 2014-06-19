include_directories (${CHIME_SOURCE_DIR}/lib)

link_directories (${CHIME_BINARY_DIR}/lib)

include_directories( ${OPENCL_INCLUDE_DIRS} )

file(GLOB open_cl_kernels "../kernels/*.cl")

add_executable(kotekan kotekan.c ${open_cl_kernels})

target_link_libraries( kotekan chime )
target_link_libraries( kotekan ${CMAKE_THREAD_LIBS_INIT} )
target_link_libraries( kotekan m )
target_link_libraries( kotekan rt )
target_link_libraries( kotekan pfring.a )

add_custom_target(kernel_copy 
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${PROJECT_SOURCE_DIR}/kernels"
        "kernels")

add_dependencies( kotekan kernel_copy )
