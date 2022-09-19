# There are no symbols in this entire library, and so it is created as an INTERFACE library. If any
# symbols are ever added it will need to be turned back into a standard CMake library
project(kotekan_metadata)

find_package(Python COMPONENTS Interpreter REQUIRED)
execute_process(COMMAND ${Python_EXECUTABLE} enum_metadatas.py
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE PY_ENUM_METADATA_RESULT)
if(NOT PY_ENUM_METADATA_RESULT EQUAL "0")
    message(FATAL_ERROR "python enum_metadatas.py failed with ${PY_ENUM_METADATA_RESULT}, please run it manually before making the package.")
endif()

add_library(kotekan_metadata INTERFACE)
target_include_directories(kotekan_metadata INTERFACE .)
