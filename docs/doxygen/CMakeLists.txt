set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/kotekan.doxy.in)
set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/kotekan.doxy)

configure_file(${doxyfile_in} ${doxyfile} @ONLY)

add_custom_target(doc ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM)
