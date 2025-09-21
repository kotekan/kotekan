# * Find FFTW Find the native FFTW includes and library
#
# FFTW_INCLUDES    - where to find fftw3.h FFTW_LIBRARIES   - List of libraries when using FFTW.
# FFTW_FOUND       - True if FFTW found.

if(FFTW_INCLUDES)
    # Already in cache, be silent
    set(FFTW_FIND_QUIETLY TRUE)
endif(FFTW_INCLUDES)

find_path(
    FFTW_INCLUDES fftw3.h
    PATHS /usr/local/Cellar /usr/local/include
    PATH_SUFFIXES fftw)

find_library(
    FFTW_LIBRARY
    NAMES fftw3
    PATHS /usr/local/Cellar PATHS_SUFFIXES fftw)

find_library(
    FFTWF_LIBRARY
    NAMES fftw3f
    PATHS /usr/local/Cellar PATHS_SUFFIXES fftw)

set(FFTW_LIBRARIES ${FFTW_LIBRARY} ${FFTWF_LIBRARY})

# handle the QUIETLY and REQUIRED arguments and set FFTW_FOUND to TRUE if all listed variables are
# TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW DEFAULT_MSG FFTW_LIBRARIES FFTW_INCLUDES)

mark_as_advanced(FFTW_LIBRARY FFTW_INCLUDES FFTWF_LIBRARY)
