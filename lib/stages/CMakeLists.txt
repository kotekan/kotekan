project(kotekan_libs)

set(HDF5_CONDITION
    $<AND:$<BOOL:${HDF5_FOUND}>,$<BOOL:${HighFive_FOUND}>,$<BOOL:${HDF5Plugin_PLUGIN_DIR}>>)

add_library(
    kotekan_stages
    beamformingPostProcess.cpp
    chrxUplink.cpp
    computeDualpolPower.cpp
    fullPacketDump.cpp
    pulsarPostProcess.cpp
    loadFeedGains.cpp
    processFeedGains.cpp
    nDiskFileWrite.cpp
    nDiskFileRead.cpp
    networkPowerStream.cpp
    integratePowerStream.cpp
    monitorBuffer.cpp
    networkInputPowerStream.cpp
    parseReorderDefault.cpp
    pyPlotN2.cpp
    rawFileRead.cpp
    rawFileWrite.cpp
    waitingForMaxFrames.cpp
    $<$<BOOL:${ASDF_CXX_FOUND}>:asdfFileRead.cpp>
    $<$<BOOL:${ASDF_CXX_FOUND}>:asdfFileWrite.cpp>
    $<$<BOOL:${GDAL_FOUND}>:gdalFileRead.cpp>
    $<$<BOOL:${GDAL_FOUND}>:gdalFileWrite.cpp>
    $<$<BOOL:${GDAL_FOUND}>:gdalFiles.cpp>
    $<${HDF5_CONDITION}:hdf5N2Write.cpp>
    $<${HDF5_CONDITION}:hdf5FileRead.cpp>
    $<${HDF5_CONDITION}:hdf5FileReadSingleFile.cpp>
    $<${HDF5_CONDITION}:hdf5FileWrite.cpp>
    $<${HDF5_CONDITION}:hdf5Files.cpp>
    vdifStream.cpp
    recvSingleDishVDIF.cpp
    streamSingleDishVDIF.cpp
    accumulate.cpp
    bufferStatus.cpp
    bufferSend.cpp
    bufferRecv.cpp
    simpleAutocorr.cpp
    SampleAutocorr.cpp
    SimpleCrosscorr.cpp
    SpawnProcess.cpp
    freqSplit.cpp
    FreqSubset.cpp
    prodSubset.cpp
    N2Subset.cpp
    countCheck.cpp
    visAccumulate.cpp
    visCompression.cpp
    timeDownsample.cpp
    frbPostProcess.cpp
    basebandReadout.cpp
    BasebandWriter.cpp
    BaseWriter.cpp
    HFBWriter.cpp
    RawReader.cpp
    VisRawReader.cpp
    VisWriter.cpp
    HFBRawReader.cpp
    restInspectFrame.cpp
    frbNetworkProcess.cpp
    pulsarNetworkProcess.cpp
    bufferMerge.cpp
    bufferCopy.cpp
    bufferSwitch.cpp
    bufferDelay.cpp
    bufferQuiet.cpp
    BufferSplit.cpp
    InputSubset.cpp
    setScatterIndices.cpp
    UpchannelizationSchedule.cpp
    setBBBeams.cpp
    calcBBPhase.cpp
    setUpchanGain.cpp
    setFRB1Phase.cpp
    setFRBBeams.cpp
    calcFRB2Weights.cpp
    # RFI Pipeline Processes
    rfiVDIF.cpp
    rfiAVXVDIF.cpp
    ReceiveFlags.cpp
    valve.cpp
    visTransform.cpp
    visDebug.cpp
    removeEv.cpp
    HFBAccumulate.cpp
    compressLostSamples.cpp
    lostSamplesToPLMask.cpp
    lostSamplesToN2Counts.cpp
    rfiBroadcast.cpp
    bufferBadInputs.cpp
    RfiFrameDrop.cpp
    VisSharedMemWriter.cpp
    DataQuality.cpp
    BeamExtract.cpp
    BeamInspect.cpp
    configTrackerWriter.cpp
    givenDataGen.cpp
    testDataGen.cpp
    DPDKShuffleSimulate.cpp
    BadInputFlag.cpp
    hexDump.cpp
    ProcessPacketMask.cpp
    printSparse.cpp
    ExampleProducer.cpp
    ExampleConsumer.cpp
    ExampleDotProduct.cpp
    N2Accumulate.cpp
    N2TimeDownsample.cpp
    TransposeBasebandArray.cpp
    dropAllFrames.cpp
    N2FrameToVisFrame.cpp
    RfiMaskSum.cpp
    RfiFrameMask.cpp
    CountLostPLSamplesScalar.cpp
    PLMaskExpandedToCompact.cpp
    # Julia
    $<$<BOOL:${Julia_FOUND}>:juliaHelloWorld.cpp>)

target_include_directories(kotekan_stages PRIVATE .)

if(LIBAIRSPY_FOUND)
    target_sources(kotekan_stages PRIVATE airspyInput.cpp)
    target_include_directories(kotekan_stages SYSTEM PRIVATE ${LIBAIRSPY_INCLUDE_DIR})
    target_include_directories(kotekan_stages SYSTEM PRIVATE ${FFTW_INCLUDES})
    target_link_libraries(kotekan_stages PRIVATE ${LIBAIRSPY_LIBRARIES})
    if(FFTW_FOUND)
        target_sources(kotekan_stages PRIVATE AirspyAlign.cpp)
        target_link_libraries(kotekan_stages PRIVATE ${FFTW_LIBRARIES})
    endif()
endif()

if(FFTW_FOUND)
    target_sources(kotekan_stages PRIVATE fftwEngine.cpp)
    target_link_libraries(kotekan_stages PRIVATE ${FFTW_LIBRARIES})
endif()

if(ASDF_CXX_FOUND)
    target_include_directories(kotekan_stages PRIVATE ${ASDF_CXX_INCLUDE_DIRS})
    target_link_options(kotekan_stages PUBLIC ${ASDF_CXX_LDFLAGS})
    target_link_libraries(kotekan_stages PUBLIC ${ASDF_CXX_LIBRARIES})
endif()

if(GDAL_FOUND)
    target_include_directories(kotekan_stages PRIVATE ${GDAL_INCLUDE_DIRS})
    target_link_options(kotekan_stages PUBLIC ${GDAL_LDFLAGS})
    target_link_libraries(kotekan_stages PUBLIC ${GDAL_LIBRARIES})
endif()

if(HDF5_FOUND
   AND HighFive_FOUND
   AND HDF5Plugin_PLUGIN_DIR)
    target_sources(kotekan_stages PRIVATE applyGains.cpp)
    target_include_directories(kotekan_stages SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(kotekan_stages PRIVATE ${HDF5_HL_LIBRARIES} ${HDF5_LIBRARIES}
                                                 HighFive::HighFive)
    # applyGains.cpp uses std::atomic<timespec>, which relies on libatomic on some platforms
    target_link_libraries(kotekan_stages PRIVATE atomic)
endif()

if(${USE_ZARR})
    target_include_directories(kotekan_stages PRIVATE ${ZARR_CXX_INCLUDE_DIRS})
    target_link_options(kotekan_stages PUBLIC ${ZARR_CXX_LDFLAGS})
    target_link_libraries(kotekan_stages PUBLIC ${ZARR_CXX_LIBRARIES})
endif()

if(HDF5_FOUND
   AND HighFive_FOUND
   AND HDF5Plugin_PLUGIN_DIR
   AND OpenMP_FOUND)
    target_sources(kotekan_stages PRIVATE Transpose.cpp VisTranspose.cpp HFBTranspose.cpp)
endif()

if(${Julia_FOUND})
    target_include_directories(kotekan_stages INTERFACE ${JULIA_INCLUDE_DIRS})
    target_link_libraries(kotekan_stages INTERFACE ${JULIA_LIBRARY})
    add_definitions(-DWITH_JULIA)
endif()

if(OpenMP_FOUND)
    target_sources(kotekan_stages PRIVATE VisTruncate.cpp HFBTruncate.cpp)
endif()

if(LAPACKE_FOUND AND BLAZE_FOUND)
    target_sources(kotekan_stages PRIVATE eigenVis.cpp RingMapMaker.cpp EigenVisIter.cpp
                                          EigenN2Iter.cpp)
    target_include_directories(kotekan_stages SYSTEM PRIVATE ${LAPACKE_INCLUDE_DIRS} ${BLAZE_PATH})
    target_link_libraries(kotekan_stages PRIVATE ${LAPACKE_LIBRARIES})
    if(DEFINED OPENBLAS_EXTRA_LIB AND OPENBLAS_EXTRA_LIB)
        target_link_libraries(kotekan_stages PRIVATE ${OPENBLAS_EXTRA_LIB})
    endif()
endif()

# Libevent core&pthreads is required for bufferRecv
find_package(LIBEVENT REQUIRED)
target_link_libraries(
    kotekan_stages
    PUBLIC ${LIBEVENT_CORE}
    PRIVATE ${LIBEVENT_PTHREAD})

target_link_libraries(kotekan_stages PRIVATE kotekan_libs libexternal)

# -lrt is needed for shm_open in VisSharedMemWriter on linux but not Clang
if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_link_libraries(kotekan_stages PRIVATE rt)
endif()

# -latomic was historically required for applyGains on some platforms, but newer toolchains on
# x86_64 provide the necessary builtins without an extra library.  Add it only when the build system
# requests it explicitly via KOTEKAN_FORCE_LIBATOMIC.
if(KOTEKAN_FORCE_LIBATOMIC)
    target_link_libraries(kotekan_stages PRIVATE atomic)
endif()

if(CUDAToolkit_FOUND)
    target_include_directories(kotekan_stages PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    message(STATUS "kotekan_stages include dirs: ")
    get_target_property(STAGES_INCLUDES kotekan_stages INCLUDE_DIRECTORIES)
    message(STATUS "${STAGES_INCLUDES}")
endif()
