cmake_minimum_required( VERSION 2.8 )
project( libkotekan_testing )

add_library ( kotekan_testing
              chimeMetadataDump.cpp
              constDataCheck.cpp
              fakeGpuBuffer.cpp
              gpuBeamformSimulate.cpp
              gpuBeamformPulsarSimulate.cpp
              gpuSimulate.cpp
              hexDump.cpp
              networkOutputSim.cpp
              simVdifData.cpp
              testDataCheck.cpp
              testDataGen.cpp
              test_data_generation.c
              )
