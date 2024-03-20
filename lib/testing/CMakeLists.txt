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
    GenHFBFrames.cpp
    pulsarSimProcess.cpp
    visNoise.cpp
    visDrop.cpp
    zeroLowerTriangle.cpp
    visTestPattern.cpp)

target_include_directories(kotekan_testing PUBLIC .)

target_link_libraries(kotekan_testing PRIVATE libexternal kotekan_libs)
