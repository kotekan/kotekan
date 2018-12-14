cmake_minimum_required( VERSION 2.8 )
project( kotekan_utils )

include_directories (${KOTEKAN_SOURCE_DIR}/include)

include ( utils.cmake )

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

