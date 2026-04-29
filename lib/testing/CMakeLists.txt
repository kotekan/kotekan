project(libkotekan_testing)

add_library(
    kotekan_testing
    applyGenPL.cpp
    chimeMetadataDump.cpp
    constDataCheck.cpp
    fakeGpu.cpp
    fakeGpuPattern.cpp
    FakeVis.cpp
    FakeN2.cpp
    FakeHFB.cpp
    FakeVisPattern.cpp
    gpuSimulate.cpp
    gpuSimulateCudaBasebandBeamformer.cpp
    gpuSimulateCudaFRBBeamformer.cpp
    gpuSimulateCudaUpchannelize.cpp
    gpuSimulateN2kCorr.cpp
    gpuSimulateN2kPLExpand.cpp
    gpuSimulateN2kPL1bitCorr.cpp
    gpuSimulatePLMaskAccumulator.cpp
    gpuSimulateRFIS012.cpp
    gpuSimulateRFIS012bar.cpp
    gpuSimulateRFIS012tilde.cpp
    gpuSimulateRFISK.cpp
    inventBFMask.cpp
    inventVoltage.cpp
    inventVoltageCHIME.cpp
    inventPLMask.cpp
    networkOutputSim.cpp
    simVdifData.cpp
    testDataCheck.cpp
    testDataGenQuad.cpp
    testDataGenFloat.cpp
    testLostCountsGen.cpp
    testLostSamplesToPLMask.cpp
    testN2kGen.cpp
    testRFIS012Gen.cpp
    testRFImaskGen.cpp
    testRFIFrameMaskGen.cpp
    TestDropFrames.cpp
    testCHORDTelescope.cpp
    testXpose.cpp
    TimeUtilDump.cpp
    GenHFBFrames.cpp
    pulsarSimProcess.cpp
    visNoise.cpp
    visDrop.cpp
    waitingForAllTests.cpp
    zeroLowerTriangle.cpp
    visTestPattern.cpp)

target_include_directories(kotekan_testing PUBLIC .)

target_link_libraries(kotekan_testing PRIVATE libexternal kotekan_libs)
