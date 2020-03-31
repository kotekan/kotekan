project( kotekan_utils )

include_directories (${KOTEKAN_SOURCE_DIR}/include)

set ( KOTEKAN_UTIL_SOURCES
      fpga_header_functions.c
      gateSpec.cpp
      gpsTime.c
      nt_memcpy.c
      nt_memset.c
      output_formating.c
      time_tracking.c
      type.cpp
      util.c
      FrameView.cpp
      HfbFrameView.cpp
      visBuffer.cpp
      visUtil.cpp
      visFile.cpp
      visFileRaw.cpp
      hfbFileRaw.cpp
      visFileRing.cpp
      tx_utils.cpp
      datasetManager.cpp
      dataset.cpp
      pulsarTiming.cpp
      datasetState.cpp
      restClient.cpp
      BipBuffer.cpp
      Hash.cpp
      MurmurHash3.cpp
      network_functions.cpp
    )

if (${USE_HDF5})
    set (KOTEKAN_UTIL_SOURCES ${KOTEKAN_UTIL_SOURCES} visFileH5.cpp hfbFileH5.cpp visFileArchive.cpp hfbFileArchive.cpp)
endif ()

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
       set (KOTEKAN_UTIL_SOURCES ${KOTEKAN_UTIL_SOURCES} osxBindCPU.cpp)
endif ()

add_library ( kotekan_utils
              ${KOTEKAN_UTIL_SOURCES})

# HDF5 stuff
if (${USE_HDF5})
  include_directories (${HDF5_INCLUDE_DIRS})
  include_directories (SYSTEM /kotekan/HighFive/include/)
  target_link_libraries ( kotekan_utils ${HDF5_HL_LIBRARIES} ${HDF5_LIBRARIES} )
  add_dependencies (kotekan_utils highfive)
endif ()

