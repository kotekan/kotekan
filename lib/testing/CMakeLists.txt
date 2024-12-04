project(libkotekan_testing)

add_library(
    kotekan_testing
    chimeMetadataDump.cpp
    constDataCheck.cpp
    fakeGpu.cpp
    fakeGpuPattern.cpp
    FakeVis.cpp
    FakeHFB.cpp
    FakeVisPattern.cpp
    gpuBeamformSimulate.cpp
    gpuTrackingBeamformSimulate.cpp
    gpuSimulate.cpp
    gpuSimulateCudaBasebandBeamformer.cpp
    gpuSimulateCudaFRBBeamformer.cpp
    gpuSimulateCudaUpchannelize.cpp
    gpuSimulateN2k.cpp
    networkOutputSim.cpp
    simVdifData.cpp
    testDataCheck.cpp
    testDataGenQuad.cpp
    testDataGenFloat.cpp
    TestDropFrames.cpp
    testCHORDTelescope.cpp
    GenHFBFrames.cpp
    pulsarSimProcess.cpp
    visNoise.cpp
    visDrop.cpp
    zeroLowerTriangle.cpp
    visTestPattern.cpp)

target_include_directories(kotekan_testing PUBLIC .)

target_link_libraries(kotekan_testing PRIVATE libexternal kotekan_libs)

if(${USE_LAPACK})
    target_sources(kotekan_testing PRIVATE FakeVis.cpp FakeVisPattern.cpp)
    target_include_directories(kotekan_testing SYSTEM PRIVATE ${BLAS_INCLUDE_DIRS}
                                                             ${LAPACKE_INCLUDE_DIRS} ${BLAZE_PATH})
    target_link_libraries(kotekan_testing PRIVATE ${BLAS_LIBRARIES})
    target_link_libraries(kotekan_testing PRIVATE ${LAPACK_LIBRARIES})
    target_link_libraries(kotekan_testing PRIVATE ${LAPACKE_LIBRARIES})
endif()
