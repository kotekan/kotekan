project(kotekan_gpu)

include_directories(${KOTEKAN_SOURCE_DIR}/lib)
include_directories(${KOTEKAN_SOURCE_DIR}/lib/gpu)

set(KOTEKAN_GPU_SOURCES gpuDeviceInterface.cpp gpuEventContainer.cpp gpuProcess.cpp gpuCommand.cpp)

add_library(kotekan_gpu ${KOTEKAN_GPU_SOURCES})

target_link_libraries(kotekan_gpu libexternal)
