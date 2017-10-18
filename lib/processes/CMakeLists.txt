cmake_minimum_required( VERSION 2.8 )
project( kotekan_libs )

include_directories (${CHIME_SOURCE_DIR}/lib)

set (KOTEKAN_PROCESS_LIB_SOURCES
              beamformingPostProcess.cpp
              chrxUplink.cpp
              computeDualpolPower.cpp
              fullPacketDump.cpp
              gpuPostProcess.cpp
              nDiskFileWrite.cpp
              nDiskFileRead.cpp
              networkPowerStream.cpp
              integratePowerStream.cpp
              networkInputPowerStream.cpp
              pyPlotResult.cpp
              rawFileRead.cpp
              rawFileWrite.cpp
              vdifStream.cpp
              recvSingleDishVDIF.cpp
              streamSingleDishVDIF.cpp
              accumulate.cpp
              bufferStatus.cpp
              hdf5Writer.cpp
              )

if (${USE_DPDK})
       set (KOTEKAN_PROCESS_LIB_SOURCES ${KOTEKAN_PROCESS_LIB_SOURCES} dpdkWrapper.cpp)
endif ()

add_library (kotekan_processes ${KOTEKAN_PROCESS_LIB_SOURCES})


# HDF5 stuff
set(HDF5_USE_STATIC_LIBRARIES ON)
find_package(HDF5 REQUIRED)
include_directories (${HDF5_INCLUDE_DIRS})
include_directories (/kotekan/HighFive/include/)
target_link_libraries ( kotekan_processes ${HDF5_HL_LIBRARIES} ${HDF5_LIBRARIES} )

# This seems like a circular reference, seems like there should be a
# way to avoid this.
target_link_libraries( kotekan_processes kotekan_libs )
