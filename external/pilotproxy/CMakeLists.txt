# Vendored PilotProxy ATSC DTV pilot-tone F-statistic detector core (see README.md and VENDOR.json
# in this directory; the sources are byte-identical copies of https://github.com/WVURAIL/pilot-proxy
# cuda/).
#
# This CMakeLists.txt is kotekan-side integration glue rather than part of the vendored upstream
# file set.
project(pilotproxy CXX CUDA)

# Same default as lib/cuda: A40 / sm_86. The variable set inside lib/cuda is directory-scoped and
# does not reach external/, so repeat the guard; a -DCMAKE_CUDA_ARCHITECTURES=... cache value still
# wins.
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

add_library(
    pilotproxy STATIC f_statistic.cu # CUDA kernel + extern "C" API (FStat_*)
                      f_statistic_reference.cpp # bit-accurate CPU reference (fstat_ref_*)
)

# Expose the parent (external/) as the include root so consumers keep the `#include
# "pilotproxy/f_statistic.h"` spelling they used when the core lived under lib/cuda/. SYSTEM matches
# the ksgpu/n2k treatment: vendored headers are exempt from kotekan's -Werror surface (the copies
# are warning-clean under -Wall -Wextra, but they must stay byte-identical to upstream, so kotekan's
# diagnostics may not force edits here).
target_include_directories(pilotproxy SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
