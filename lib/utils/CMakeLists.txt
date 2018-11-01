cmake_minimum_required( VERSION 2.8 )
project( kotekan_utils )

include_directories (${KOTEKAN_SOURCE_DIR}/include)

set (KOTEKAN_UTIL_SOURCES
              fpga_header_functions.c
              gpsTime.c
              nt_memcpy.c
              nt_memset.c
              output_formating.c
              time_tracking.c
              util.c
              visBuffer.cpp
              visUtil.cpp
              visFile.cpp
              visFileRaw.cpp
              visFileRing.cpp
              tx_utils.cpp
              datasetManager.cpp
              datasetState.cpp
              restClient.cpp
       )

if (${USE_HDF5})
    set (KOTEKAN_UTIL_SOURCES ${KOTEKAN_UTIL_SOURCES} visFileH5.cpp visFileArchive.cpp)
endif ()

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
       set (KOTEKAN_UTIL_SOURCES ${KOTEKAN_UTIL_SOURCES} osxBindCPU.cpp)
endif ()

add_library ( kotekan_utils
              ${KOTEKAN_UTIL_SOURCES})

# HDF5 stuff
if (${USE_HDF5})
  include_directories (${HDF5_INCLUDE_DIRS})
  include_directories (/kotekan/HighFive/include/)
  target_link_libraries ( kotekan_utils ${HDF5_HL_LIBRARIES} ${HDF5_LIBRARIES} )
  add_dependencies (kotekan_utils highfive)
endif ()

