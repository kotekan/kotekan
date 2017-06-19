cmake_minimum_required( VERSION 2.8 )
project( libkotekan_testing )

add_library ( kotekan_testing
              constDataCheck.cpp
              gpuBeamformSimulate.cpp
              gpuSimulate.cpp
              networkOutputSim.cpp
              simVdifData.cpp
              testDataCheck.cpp
              testDataGen.cpp
              test_data_generation.c
              )

