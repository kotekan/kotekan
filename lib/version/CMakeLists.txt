# This file is full of awful cmake workarounds... The summary is this file generates the file
# version.c compiles it into the library kotekan_version, links it against the kotekan binary, and
# then deletes the version.c file so that it is regenerated each time.
#
# However because of cmake limitations, you cannot make the `configure_file` command run as a part
# of the make process.  You need to add a custom command which runs as a part of the make process to
# generate any files which the library requires.   So you put the `configure_file` command into a
# cmake script and then call that with a custom command.  i.e. you don't rerun the entire cmake
# configure process, just a small cmake helper script. However the external script cannot easily
# grab all the variables from the cmake build options you might want, so we generate the script to
# generate the version.c file on the inital cmake run, and then we use that to update the version.c
# file each time we run make.
#
# There are other solutions to this problem, but they all involve some kind of ulgy hack like this,
# or the cause the system to take forever every time you run make because they do too much work.

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/make_version.cmake.in
               ${CMAKE_CURRENT_BINARY_DIR}/make_version.cmake @ONLY)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/version.c
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/make_version.cmake
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/make_version.cmake ${CMAKE_CURRENT_SOURCE_DIR}/version.c.in
    COMMENT "Generating file: version.c")

add_library(kotekan_version ${CMAKE_CURRENT_BINARY_DIR}/version.c)

add_custom_command(
    TARGET kotekan_version
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/version.c
    COMMENT "Removing generated file: version.c")
