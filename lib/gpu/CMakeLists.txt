cmake_minimum_required( VERSION 2.8 )
project( kotekan_gpu )

include_directories (${KOTEKAN_SOURCE_DIR}/include)

include_directories (${KOTEKAN_SOURCE_DIR}/lib)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/gpu)

add_library ( kotekan_gpu
              gpuDeviceInterface.cpp
              gpuEventContainer.cpp
#              gpuCommand.cpp
#              gpuProcess.cpp
              )
