project(kotekan_gpu)

add_library(kotekan_gpu
            gpuBufferHandler.cpp
            gpuDeviceInterface.cpp
            gpuEventContainer.cpp
            gpuProcess.cpp
            gpuCommand.cpp)

target_link_libraries(kotekan_gpu PRIVATE libexternal kotekan_libs)
target_include_directories(kotekan_gpu PUBLIC .)
