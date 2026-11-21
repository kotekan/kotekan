# Finds the libbladeRF include path and library. Sets the following if libbladeRF is found:
# LIBBLADERF_FOUND, LIBBLADERF_INCLUDE_DIR, LIBBLADERF_LIBRARIES

include(FindPackageHandleStandardArgs)

find_path(
    LIBBLADERF_INCLUDE_DIR
    NAMES libbladeRF.h
    PATHS /usr/include /usr/local/include)

find_library(
    LIBBLADERF_LIBRARIES
    NAMES bladeRF
    PATHS /usr/local/lib /usr/lib)

find_package_handle_standard_args(LIBBLADERF DEFAULT_MSG LIBBLADERF_LIBRARIES
                                  LIBBLADERF_INCLUDE_DIR)

mark_as_advanced(LIBBLADERF_INCLUDE_DIR LIBBLADERF_LIBRARIES)
