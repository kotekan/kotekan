cmake_minimum_required( VERSION 2.8 )
project( kotekan_libs )

include_directories (${KOTEKAN_SOURCE_DIR}/lib)

set (KOTEKAN_PROCESS_LIB_SOURCES
              beamformingPostProcess.cpp
              chrxUplink.cpp
              computeDualpolPower.cpp
              fullPacketDump.cpp
              gpuPostProcess.cpp
              pulsarPostProcess.cpp
              nDiskFileWrite.cpp
              nDiskFileRead.cpp
              networkPowerStream.cpp
              integratePowerStream.cpp
              networkInputPowerStream.cpp
              pyPlotN2.cpp
              rawFileRead.cpp
              rawFileWrite.cpp
              vdifStream.cpp
              recvSingleDishVDIF.cpp
              streamSingleDishVDIF.cpp
              rfiVDIF.cpp
              rfiBroadcastVDIF.cpp
              rfiBroadcast.cpp
              accumulate.cpp
              bufferStatus.cpp
              bufferSend.cpp
              bufferRecv.cpp
              simpleAutocorr.cpp
              freqSlicer.cpp
              prodSubset.cpp
              visProcess.cpp
              timeDownsample.cpp
              restInspectFrame.cpp
              frbNetworkProcess.cpp
              pulsarNetworkProcess.cpp
              )

if (NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
       set (KOTEKAN_PROCESS_LIB_SOURCES ${KOTEKAN_PROCESS_LIB_SOURCES} frbNetworkProcess.cpp pulsarNetworkProcess.cpp frbPostProcess.cpp)
endif ()

if (${USE_DPDK})
       set (KOTEKAN_PROCESS_LIB_SOURCES ${KOTEKAN_PROCESS_LIB_SOURCES} dpdkWrapper.cpp)
endif ()

if (${USE_AIRSPY})
       set (KOTEKAN_PROCESS_LIB_SOURCES ${KOTEKAN_PROCESS_LIB_SOURCES} airspyInput.cpp)
endif ()

if (${USE_FFTW})
       set (KOTEKAN_PROCESS_LIB_SOURCES ${KOTEKAN_PROCESS_LIB_SOURCES} fftwEngine.cpp)
endif ()

# This is temporarily disabled because it requires a newer version of hdf5
# not available in most computers
# HDF5 stuff
if (${USE_HDF5})
  # Disable for the moment as requires too modern HDF5
  #set (KOTEKAN_PROCESS_LIB_SOURCES ${KOTEKAN_PROCESS_LIB_SOURCES} visWriter.cpp calibration_io.cpp)
  set (KOTEKAN_PROCESS_LIB_SOURCES ${KOTEKAN_PROCESS_LIB_SOURCES} visWriter.cpp)
endif ()

if (${USE_LAPACK})
  set (KOTEKAN_PROCESS_LIB_SOURCES ${KOTEKAN_PROCESS_LIB_SOURCES} eigenVis.cpp)
endif ()


add_library (kotekan_processes ${KOTEKAN_PROCESS_LIB_SOURCES})

# HDF5 stuff
if (${USE_HDF5})
  include_directories (${HDF5_INCLUDE_DIRS})
  target_link_libraries ( kotekan_processes ${HDF5_HL_LIBRARIES} ${HDF5_LIBRARIES} )
endif ()

if (${USE_LAPACK})
  include_directories (${OpenBLAS_INCLUDE_DIRS} ${LAPACK_INCLUDE_DIRS})
  target_link_libraries ( kotekan_processes  ${OpenBLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif ()

target_link_libraries( kotekan_processes kotekan_utils kotekan_core )
