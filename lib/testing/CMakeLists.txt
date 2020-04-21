project(libkotekan_testing)

set(KOTEKAN_TESTING_SOURCES
    chimeMetadataDump.cpp
    constDataCheck.cpp
    fakeGpu.cpp
    fakeGpuPattern.cpp
    FakeVis.cpp
    FakeVisPattern.cpp
    gpuBeamformSimulate.cpp
    gpuBeamformPulsarSimulate.cpp
    gpuSimulate.cpp
    hexDump.cpp
    networkOutputSim.cpp
    simVdifData.cpp
    testDataCheck.cpp
    testDataGen.cpp
    testDataGenQuad.cpp
    testDataGenFloat.cpp
    TestDropFrames.cpp
    test_data_generation.c
    pulsarSimProcess.cpp
    visNoise.cpp
    visDrop.cpp)

add_library(kotekan_testing ${KOTEKAN_TESTING_SOURCES})

target_link_libraries(kotekan_testing libexternal)
