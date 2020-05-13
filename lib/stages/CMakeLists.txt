project(kotekan_libs)

include_directories(${KOTEKAN_SOURCE_DIR}/lib)

add_library(
    kotekan_stages
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
    InputSubset.cpp
    # RFI Pipeline Processes
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
    VisSharedMemWriter.cpp)

if(${USE_AIRSPY})
    target_sources(kotekan_stages PRIVATE airspyInput.cpp)
    include_directories(${LIBAIRSPY_INCLUDE_DIR})
    target_link_libraries(kotekan_stages PRIVATE ${LIBAIRSPY_LIBRARIES})
endif()

if(${USE_FFTW})
    set(KOTEKAN_PROCESS_LIB_SOURCES ${KOTEKAN_PROCESS_LIB_SOURCES} fftwEngine.cpp)
    target_link_libraries(kotekan_stages PRIVATE ${FFTW_LIBRARIES})
endif()

if(${USE_HDF5})
    target_sources(kotekan_stages PRIVATE basebandReadout.cpp applyGains.cpp)
    include_directories(${HDF5_INCLUDE_DIRS})
    target_link_libraries(kotekan_stages PRIVATE ${HDF5_HL_LIBRARIES} ${HDF5_LIBRARIES})
endif()

if(${USE_HDF5} AND ${USE_OMP})
    target_sources(kotekan_stages PRIVATE visTranspose.cpp)
endif()

if(${USE_OMP})
    target_sources(kotekan_stages PRIVATE visTruncate.cpp)
endif()

if(${USE_LAPACK})
    target_sources(kotekan_stages PRIVATE eigenVis.cpp RingMapMaker.cpp EigenVisIter.cpp)
    include_directories(${BLAS_INCLUDE_DIRS})
    include_directories(${LAPACKE_INCLUDE_DIRS})
    target_link_libraries(kotekan_stages PRIVATE ${BLAS_LIBRARIES})
    target_link_libraries(kotekan_stages PRIVATE ${LAPACK_LIBRARIES})
    target_link_libraries(kotekan_stages PRIVATE ${LAPACKE_LIBRARIES})
endif()

# Libevent is required for bufferRecv
set(CMAKE_MODULE_PATH ${KOTEKAN_SOURCE_DIR}/cmake)
find_package(LIBEVENT REQUIRED)
target_link_libraries(
    kotekan_stages
    PUBLIC ${LIBEVENT_BASE}
    PRIVATE ${LIBEVENT_PTHREAD})

target_link_libraries(kotekan_stages PRIVATE kotekan_utils kotekan_core libexternal)

# -lrt is needed for shm_open in VisSharedMemWriter on linux but not Clang
if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_link_libraries(kotekan_stages PRIVATE rt)
endif()
