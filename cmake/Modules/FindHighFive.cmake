# FindHighFive.cmake - locate HighFive (header-only HDF5 C++ wrapper)

# Try CMake package first (if HighFive installed with a config file)
find_package(HighFive QUIET CONFIG)

if(HighFive_FOUND)
    # Prefer includes from exported target if available
    if(TARGET HighFive::HighFive)
        get_target_property(_hf_includes HighFive::HighFive INTERFACE_INCLUDE_DIRECTORIES)
        if(_hf_includes)
            set(HighFive_INCLUDE_DIRS ${_hf_includes})
        endif()
    endif()
    # Some packages define HIGHFIVE_INCLUDE_DIR; normalize to HighFive_INCLUDE_DIRS
    if(NOT HighFive_INCLUDE_DIRS AND DEFINED HIGHFIVE_INCLUDE_DIR)
        set(HighFive_INCLUDE_DIRS ${HIGHFIVE_INCLUDE_DIR})
    endif()
endif()

if(NOT HighFive_INCLUDE_DIRS)
    # Fallback: search common include locations for header
    find_path(
        HighFive_INCLUDE_DIRS
        NAMES HighFive/HighFive.hpp highfive/highfive.hpp highfive/H5Easy.hpp
        PATHS /usr/local/include /usr/include)
endif()

if(HighFive_INCLUDE_DIRS)
    set(HighFive_FOUND TRUE)
else()
    set(HighFive_FOUND FALSE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HighFive REQUIRED_VARS HighFive_INCLUDE_DIRS)

mark_as_advanced(HighFive_INCLUDE_DIRS)
