# Finds libnuma include path and libraries Sets the following if libnuma is found: NUMA_FOUND,
# NUMA_INCLUDE_DIR, NUMA_LIBRARY

include(FindPackageHandleStandardArgs)

set(NUMA_SEARCH_PATHS /usr/include /usr/local/include)

find_path(
    NUMA_INCLUDE_DIR
    NAMES numa.h
    PATHS ${NUMA_SEARCH_PATHS})

find_library(NUMA_LIBRARY NAMES numa)

find_package_handle_standard_args(NUMA DEFAULT_MSG NUMA_LIBRARY NUMA_INCLUDE_DIR)

mark_as_advanced(NUMA_INCLUDE_DIR NUMA_LIBRARY)
