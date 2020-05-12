project(kotekan_gpu)

include_directories(${KOTEKAN_SOURCE_DIR}/lib)
include_directories(${KOTEKAN_SOURCE_DIR}/lib/gpu)

add_library(kotekan_gpu gpuDeviceInterface.cpp gpuEventContainer.cpp gpuProcess.cpp gpuCommand.cpp)

target_link_libraries(kotekan_gpu libexternal)
