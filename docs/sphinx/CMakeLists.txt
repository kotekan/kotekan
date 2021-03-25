if(NOT DEFINED SPHINX_THEME)
    set(SPHINX_THEME default)
endif()

if(NOT DEFINED SPHINX_THEME_DIR)
    set(SPHINX_THEME_DIR)
endif()

# configured documentation tools and intermediate build results
set(BINARY_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}")

# Sphinx cache with pickled ReST documents
set(SPHINX_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_doctrees")

# HTML output directory
set(SPHINX_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")

if(NOT DEFINED PLANTUML_PATH)
    message("PLANTUML_PATH not specified: Downloading plantuml.jar.")
    include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
    ExternalProject_Add(
        plantuml
        URL https://downloads.sourceforge.net/project/plantuml/plantuml.jar
        PREFIX "plantuml"
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        DOWNLOAD_NO_EXTRACT 1)
    set(PLANTUML_PATH "${CMAKE_CURRENT_BINARY_DIR}/plantuml/src")
    set(PLANTUML_DIR "${CMAKE_CURRENT_BINARY_DIR}/plantuml/src")
    set(DOWNLOAD_PLANTUML 1)
else()
    set(PLANTUML_DIR ${PLANTUML_PATH})
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in" "${BINARY_BUILD_DIR}/conf.py" @ONLY)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/_static")

add_custom_target(
    sphinx ALL
    ${SPHINX_EXECUTABLE}
    -W
    -q
    -b
    html
    -c
    "${BINARY_BUILD_DIR}"
    -d
    "${SPHINX_CACHE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${SPHINX_HTML_DIR}"
    COMMENT "Building HTML documentation with Sphinx")

if(DEFINED DOWNLOAD_PLANTUML)
    add_dependencies(sphinx plantuml)
endif()

message("Using plantUML from ${PLANTUML_PATH}")
