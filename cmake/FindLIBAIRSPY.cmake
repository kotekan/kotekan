# Finds airspy include path and libraries Sets the following if airspy is found: LibAirSpy_FOUND,
# LIBAIRSPY_INCLUDE_DIR, LIBAIRSPY_LIBRARIES

include(FindPackageHandleStandardArgs)

set(LIBAIRSPY_SEARCH_PATHS /usr/include /usr/local/include /usr/local/Cellar)

find_path(
    LIBAIRSPY_INCLUDE_DIR
    NAMES airspy.h
    PATHS ${LIBAIRSPY_SEARCH_PATHS}
    PATH_SUFFIXES libairspy)

find_library(
    LIBAIRSPY_LIBRARIES
    NAMES airspy
    PATHS /usr/local/lib /usr/lib)

find_package_handle_standard_args(LIBAIRSPY DEFAULT_MSG LIBAIRSPY_LIBRARIES LIBAIRSPY_INCLUDE_DIR)

mark_as_advanced(LIBAIRSPY_INCLUDE_DIR LIBAIRSPY_LIBRARIES)
