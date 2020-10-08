set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/kotekan.doxy.in)
set(DOXYFILE ${CMAKE_CURRENT_BINARY_DIR}/kotekan.doxy)

configure_file(${DOXYFILE_IN} ${DOXYFILE} @ONLY)

add_custom_target(
    doc ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM)
