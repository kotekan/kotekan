# Finds libgputils include path and libraries Sets the following if libgputils is found: GPUTILS_FOUND,
# GPUTILS_INCLUDE_DIR, GPUTILS_LIBRARY

include(FindPackageHandleStandardArgs)

set(GPUTILS_SEARCH_PATHS /usr/include /usr/local/include)

find_path(
    GPUTILS_INCLUDE_DIR
    NAMES gputils.hpp
    PATHS ${GPUTILS_SEARCH_PATHS})

find_library(GPUTILS_LIBRARY NAMES gputils)

find_package_handle_standard_args(GPUTILS DEFAULT_MSG GPUTILS_LIBRARY GPUTILS_INCLUDE_DIR)

mark_as_advanced(GPUTILS_INCLUDE_DIR GPUTILS_LIBRARY)
