set(CMAKE_MODULE_PATH ${KOTEKAN_SOURCE_DIR}/cmake)
find_package(Sphinx REQUIRED)

if(NOT DEFINED SPHINX_THEME)
    set(SPHINX_THEME default)
endif()
 
if(NOT DEFINED SPHINX_THEME_DIR)
    set(SPHINX_THEME_DIR)
endif()
 
# configured documentation tools and intermediate build results
set(BINARY_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/sphinx")
 
# Sphinx cache with pickled ReST documents
set(SPHINX_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/sphinx/_doctrees")
 
# HTML output directory
set(SPHINX_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/sphinx/html")
 
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/sphinx/conf.py"
    "${BINARY_BUILD_DIR}/conf.py"
    @ONLY)
 
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/sphinx/_static")

add_custom_target(sphinx ALL
    ${SPHINX_EXECUTABLE}
        -q -b html
        -c "${BINARY_BUILD_DIR}"
        -d "${SPHINX_CACHE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/sphinx"
        "${SPHINX_HTML_DIR}"
    COMMENT "Building HTML documentation with Sphinx")
