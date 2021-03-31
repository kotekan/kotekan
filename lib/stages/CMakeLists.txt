project(kotekan_libs)

add_library(
    kotekan_stages
    beamformingPostProcess.cpp
    chrxUplink.cpp
    computeDualpolPower.cpp
    fullPacketDump.cpp
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
    FreqSubset.cpp
    prodSubset.cpp
    countCheck.cpp
    visAccumulate.cpp
    visCompression.cpp
    timeDownsample.cpp
    frbPostProcess.cpp
    BaseWriter.cpp
    VisWriter.cpp
    HFBWriter.cpp
    RawReader.cpp
    VisRawReader.cpp
    HFBRawReader.cpp
    restInspectFrame.cpp
    frbNetworkProcess.cpp
    pulsarNetworkProcess.cpp
    bufferMerge.cpp
    bufferCopy.cpp
    bufferSwitch.cpp
    bufferDelay.cpp
    BufferSplit.cpp
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
    HFBAccumulate.cpp
    compressLostSamples.cpp
    bufferBadInputs.cpp
    RfiFrameDrop.cpp
    VisSharedMemWriter.cpp
    DataQuality.cpp
    BeamExtract.cpp
    BeamInspect.cpp
    testDataGen.cpp
    DPDKShuffleSimulate.cpp
    DotProduct.cpp
    BadInputFlag.cpp)

target_include_directories(kotekan_stages PRIVATE .)

if(${USE_AIRSPY})
    target_sources(kotekan_stages PRIVATE airspyInput.cpp)
    target_include_directories(kotekan_stages SYSTEM PRIVATE ${LIBAIRSPY_INCLUDE_DIR})
    target_include_directories(kotekan_stages SYSTEM PRIVATE ${FFTW_INCLUDES})
    target_link_libraries(kotekan_stages PRIVATE ${LIBAIRSPY_LIBRARIES})
endif()

if(${USE_FFTW})
    target_sources(kotekan_stages PRIVATE fftwEngine.cpp)
    target_link_libraries(kotekan_stages PRIVATE ${FFTW_LIBRARIES})
endif()

if(${USE_HDF5})
    target_sources(kotekan_stages PRIVATE basebandReadout.cpp applyGains.cpp)
    target_include_directories(kotekan_stages SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(kotekan_stages PRIVATE ${HDF5_HL_LIBRARIES} ${HDF5_LIBRARIES})
endif()

if(${USE_HDF5} AND ${USE_OMP})
    target_sources(kotekan_stages PRIVATE Transpose.cpp VisTranspose.cpp HFBTranspose.cpp)
endif()

if(${USE_OMP})
    target_sources(kotekan_stages PRIVATE VisTruncate.cpp HFBTruncate.cpp)
endif()

if(${USE_LAPACK})
    target_sources(kotekan_stages PRIVATE eigenVis.cpp RingMapMaker.cpp EigenVisIter.cpp)
    target_include_directories(kotekan_stages SYSTEM PRIVATE ${BLAS_INCLUDE_DIRS}
                                                             ${LAPACKE_INCLUDE_DIRS} ${BLAZE_PATH})
    target_link_libraries(kotekan_stages PRIVATE ${BLAS_LIBRARIES})
    target_link_libraries(kotekan_stages PRIVATE ${LAPACK_LIBRARIES})
    target_link_libraries(kotekan_stages PRIVATE ${LAPACKE_LIBRARIES})
endif()

# Libevent is required for bufferRecv
find_package(LIBEVENT REQUIRED)
target_link_libraries(
    kotekan_stages
    PUBLIC ${LIBEVENT_BASE}
    PRIVATE ${LIBEVENT_PTHREAD})

target_link_libraries(kotekan_stages PRIVATE kotekan_libs libexternal)

# -lrt is needed for shm_open in VisSharedMemWriter on linux but not Clang
if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_link_libraries(kotekan_stages PRIVATE rt)
endif()

# -latomic is needed for atomic operations in applyGains, but it doesn't need to be included
# separately for clang on macOS
if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" OR NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_link_libraries(kotekan_stages PRIVATE atomic)
endif()
