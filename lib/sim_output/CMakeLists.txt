cmake_minimum_required( VERSION 2.8 )
project( libkotekan_sim_output )

#include_directories (${CHIME_SOURCE_DIR}/include)
#include_directories (${CHIME_SOURCE_DIR}/lib)
#include_directories (${CHIME_SOURCE_DIR}/lib/hsa)

add_library ( kotekan_sim_output
              gpuSimulate.cpp
              gpuBeamformSimulate.cpp
              )

