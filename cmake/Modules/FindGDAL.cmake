# FindGDAL.cmake - locate GDAL via pkg-config or fallbacks

find_package(PkgConfig QUIET)

if(PKG_CONFIG_FOUND)
    pkg_check_modules(GDAL gdal QUIET)
endif()

if(NOT GDAL_FOUND)
    # Fallback: try to find headers and library in common locations
    find_path(
        GDAL_INCLUDE_DIRS gdal.h
        PATHS /usr/include /usr/local/include
        PATH_SUFFIXES gdal)
    find_library(
        GDAL_LIBRARIES
        NAMES gdal
        PATHS /usr/lib /usr/local/lib)

    if(GDAL_INCLUDE_DIRS AND GDAL_LIBRARIES)
        set(GDAL_FOUND TRUE)
    else()
        set(GDAL_FOUND FALSE)
    endif()

    set(GDAL_LDFLAGS "")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GDAL REQUIRED_VARS GDAL_INCLUDE_DIRS GDAL_LIBRARIES)

mark_as_advanced(GDAL_INCLUDE_DIRS GDAL_LIBRARIES GDAL_LDFLAGS)
