include_directories (${CHIME_SOURCE_DIR}/lib)

link_directories (${CHIME_BINARY_DIR}/lib)

include_directories( ${OPENCL_INCLUDE_DIRS} )

file(GLOB open_cl_kernels "../kernels/*.cl")

add_executable(correlator correlator.c ${open_cl_kernels})

target_link_libraries( correlator chime )
target_link_libraries( correlator ${CMAKE_THREAD_LIBS_INIT} )
target_link_libraries( correlator m )
target_link_libraries( correlator rt )
target_link_libraries( correlator pfring.a )

add_custom_target(kernel_copy 
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${PROJECT_SOURCE_DIR}/kernels"
        "kernels")

add_dependencies( correlator kernel_copy )
