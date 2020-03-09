project( kotekan_libs )

include_directories (${KOTEKAN_SOURCE_DIR}/lib)

set ( KOTEKAN_PROCESS_LIB_SOURCES
      beamformingPostProcess.cpp
      chrxUplink.cpp
      computeDualpolPower.cpp
      fullPacketDump.cpp
      gpuPostProcess.cpp
      pulsarPostProcess.cpp
      ReadGain.cpp
      nDiskFileWrite.cpp
      nDiskFileRead.cpp
      networkPowerStream.cpp
      integratePowerStream.cpp
      monitorBuffer.cpp
      networkInputPowerStream.cpp
      pyPlotN2.cpp
      rawFileRead.cpp
      rawFileWrite.cpp
      vdifStream.cpp
      recvSingleDishVDIF.cpp
      streamSingleDishVDIF.cpp
      accumulate.cpp
      bufferStatus.cpp
      bufferSend.cpp
      bufferRecv.cpp
      simpleAutocorr.cpp
      freqSplit.cpp
      freqSubset.cpp
      prodSubset.cpp
      countCheck.cpp
      visAccumulate.cpp
      visCompression.cpp
      timeDownsample.cpp
      frbPostProcess.cpp
      visWriter.cpp
      visRawReader.cpp
      restInspectFrame.cpp
      frbNetworkProcess.cpp
      pulsarNetworkProcess.cpp
      bufferMerge.cpp
      bufferCopy.cpp
      bufferSwitch.cpp
      #RFI Pipeline Processes
      rfiVDIF.cpp
      rfiAVXVDIF.cpp
      rfiBroadcast.cpp
      rfiRecord.cpp
      rfiBadInputFinder.cpp
      rfiUpdateMetadata.cpp
      ReceiveFlags.cpp
      valve.cpp
      visTransform.cpp
      visTestPattern.cpp
      visDebug.cpp
      removeEv.cpp
      integrateHFBData.cpp
      compressLostSamples.cpp
      compressData.cpp
      bufferBadInputs.cpp
      RfiFrameDrop.cpp
    )

if (${USE_AIRSPY})
  include_directories(${LIBAIRSPY_INCLUDE_DIR})
  set (KOTEKAN_PROCESS_LIB_SOURCES ${KOTEKAN_PROCESS_LIB_SOURCES} airspyInput.cpp)
endif ()
if (${USE_FFTW})
  set (KOTEKAN_PROCESS_LIB_SOURCES ${KOTEKAN_PROCESS_LIB_SOURCES} fftwEngine.cpp)
endif ()

# HDF5 stuff
if (${USE_HDF5})
  # Disable for the moment as requires too modern HDF5 not available in most computers
  #set (KOTEKAN_PROCESS_LIB_SOURCES ${KOTEKAN_PROCESS_LIB_SOURCES} calibration_io.cpp)
  set (KOTEKAN_PROCESS_LIB_SOURCES ${KOTEKAN_PROCESS_LIB_SOURCES} basebandReadout.cpp applyGains.cpp)
endif ()

# HDF5 and OpenMP
if (${USE_HDF5} AND ${USE_OMP})
  set (KOTEKAN_PROCESS_LIB_SOURCES ${KOTEKAN_PROCESS_LIB_SOURCES} visTranspose.cpp)
endif()

# OpenMP
if (${USE_OMP})
  set (KOTEKAN_PROCESS_LIB_SOURCES ${KOTEKAN_PROCESS_LIB_SOURCES} visTruncate.cpp)
endif()

if (${USE_LAPACK})
  set (KOTEKAN_PROCESS_LIB_SOURCES ${KOTEKAN_PROCESS_LIB_SOURCES} eigenVis.cpp)
  set (KOTEKAN_PROCESS_LIB_SOURCES ${KOTEKAN_PROCESS_LIB_SOURCES} EigenVisIter.cpp)
endif ()

add_library (kotekan_stages ${KOTEKAN_PROCESS_LIB_SOURCES})
target_link_libraries( kotekan_stages ${EVENT_LIBRARIES} )

if (${USE_AIRSPY})
  target_link_libraries(kotekan_stages ${LIBAIRSPY_LIBRARIES})
endif ()
if (${USE_FFTW})
  target_link_libraries(kotekan_stages ${FFTW_LIBRARIES})
endif ()

if (${USE_HDF5})
  include_directories (${HDF5_INCLUDE_DIRS})
  target_link_libraries ( kotekan_stages ${HDF5_HL_LIBRARIES} ${HDF5_LIBRARIES} )
endif ()
if (${USE_EVENT})
    target_link_libraries( kotekan_stages ${EVENT_LIBRARIES} )
endif ()

if (${USE_LAPACK})
  include_directories (${OpenBLAS_INCLUDE_DIRS} ${LAPACK_INCLUDE_DIRS})
  target_link_libraries ( kotekan_stages  ${OpenBLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif ()

# Add atomic for building against GCC
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_link_libraries ( kotekan_stages -latomic )
endif()

target_link_libraries( kotekan_stages kotekan_utils kotekan_core )
