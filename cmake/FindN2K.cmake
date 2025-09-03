# Finds libn2k include path and libraries Sets the following if libn2k is found: N2K_FOUND,
# N2K_INCLUDE_DIR, N2K_LIBRARY

include(FindPackageHandleStandardArgs)

set(N2K_SEARCH_PATHS /usr/include /usr/local/include)

find_path(
    N2K_INCLUDE_DIR
    NAMES n2k.hpp
    PATHS ${N2K_SEARCH_PATHS})

find_library(N2K_LIBRARY NAMES n2k)

find_package_handle_standard_args(N2K DEFAULT_MSG N2K_LIBRARY N2K_INCLUDE_DIR)

mark_as_advanced(N2K_INCLUDE_DIR N2K_LIBRARY)
