# Set the version of CMake required for *all* CMake files in the project. 3.8.2 to match C++17.
cmake_minimum_required(VERSION 3.8.2)
project(
    KOTEKAN
    DESCRIPTION "A high performance radio data processing pipeline"
    LANGUAGES C CXX)
set(CMAKE_MODULE_PATH ${KOTEKAN_SOURCE_DIR}/cmake ${KOTEKAN_SOURCE_DIR}/cmake/Modules)
include(cmake/Color.cmake)
include(cmake/Options.cmake)
include(cmake/Toolchain.cmake)

# Toolchain and options now handled by included modules

# Fix a problem with homebrew linking on MacOS >= 10.14 See
# https://stackoverflow.com/questions/54068035
if(APPLE)
    link_directories(/usr/local/lib)
endif()

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -Wall")

include(cmake/Features/FeatureGPU.cmake)
include(cmake/Features/FeatureDPDK.cmake)

if(${USE_AIRSPY})
    find_package(LIBAIRSPY)
    set(USE_AIRSPY ${LIBAIRSPY_FOUND})
endif()

include(cmake/Features/FeatureMath.cmake)

# Math features handled in FeatureMath

# Toolchain covers build types, IWYU, and GPU summary

if(${USE_AIRSPY})
    add_definitions(-DWITH_AIRSPY)
endif()

include(cmake/Features/FeatureIO.cmake)

# Julia runtime and bindings
include(cmake/Features/FeatureJulia.cmake)

# Final feature summary (printed after all configuration)

find_package(Threads REQUIRED)
# Flags and link options handled in Toolchain; OpenCL install handled in FeatureGPU

if(${COMPILE_DOCS})
    add_subdirectory(docs EXCLUDE_FROM_ALL)
endif()

# Prevent memory locking and setting any memory policies. This is useful when running in restricted
# environments, e.g. Docker containers.  Note this can also be done with the configuration file for
# systems which do support memory locking, but where some memory shouldn't be memory locked.
if(${NO_MEMLOCK})
    add_definitions(-DWITH_NO_MEMLOCK)
endif()

add_subdirectory(lib)
add_subdirectory(kotekan)
add_subdirectory(scripts)
add_subdirectory(external)
add_subdirectory(config)

# Testing with boost
if(${WITH_BOOST_TESTS})
    set(_BOOST_TESTS_DIR
        tests/boost/
        CACHE INTERNAL "Path to boost test sources")
    message(STATUS "BOOST_TESTS enabled: building tests in ${_BOOST_TESTS_DIR}")
    add_subdirectory(${_BOOST_TESTS_DIR})
else()
    message(STATUS "BOOST_TESTS disabled")
endif()

# Enforce code formatting rules with clang-format.
find_program(
    CLANG_FORMAT_PATH
    NAMES "clang-format-8" "clang-format"
    DOC "Path to clang-format executable")
if(NOT CLANG_FORMAT_PATH)
    message(STATUS "clang-format not found.")
else()
    message(STATUS "clang-format found: ${CLANG_FORMAT_PATH}")
    set(DO_CLANG_FORMAT "${CLANG_FORMAT_PATH}" "-i -style=file")
endif()

# Final feature summary (last)
include(cmake/Summary.cmake)

# Remove files containing substring from a list of file names.
function(exclude_files_containing var exclude_string)
    set(listVar "")
    foreach(file ${ARGN})
        string(FIND ${file} ${exclude_string} EXCLUDE_STRING_FOUND)
        if(${EXCLUDE_STRING_FOUND} EQUAL -1)
            list(APPEND listVar ${file})
        endif()
    endforeach()
    set(${var}
        "${listVar}"
        PARENT_SCOPE)
endfunction()

if(CLANG_FORMAT_PATH)
    # Find all .c(pp) and .h(pp) files.
    file(GLOB_RECURSE KOTEKAN_ALL_CPP_FILES *.cpp)
    file(GLOB_RECURSE KOTEKAN_ALL_HPP_FILES *.hpp)
    file(GLOB_RECURSE KOTEKAN_ALL_C_FILES *.c)
    file(GLOB_RECURSE KOTEKAN_ALL_H_FILES *.h)

    list(APPEND FILES_TO_FORMAT ${KOTEKAN_ALL_CPP_FILES})
    list(APPEND FILES_TO_FORMAT ${KOTEKAN_ALL_HPP_FILES})
    list(APPEND FILES_TO_FORMAT ${KOTEKAN_ALL_C_FILES})
    list(APPEND FILES_TO_FORMAT ${KOTEKAN_ALL_H_FILES})

    # Exclude all code we don't want auto-formatted (external libs).
    set(EXCLUDE_DIR "external")
    exclude_files_containing(FILES_TO_FORMAT ${EXCLUDE_DIR} ${FILES_TO_FORMAT})
    set(EXCLUDE_DIR "build")
    exclude_files_containing(FILES_TO_FORMAT ${EXCLUDE_DIR} ${FILES_TO_FORMAT})

    # Use .clang-format file to auto format.
    add_custom_target(
        clang-format
        COMMAND ${CLANG_FORMAT_PATH} -style=file -i ${FILES_TO_FORMAT}
        COMMENT "Running clang-format")
endif()
