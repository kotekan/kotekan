project(kotekan_metadata)

add_library(
    kotekan_metadata
    chordMetadata.cpp
    HFBMetadata.cpp
    BasebandMetadata.cpp
    BeamMetadata.cpp
    DataType.cpp
    Metadata.cpp
    NDArray.cpp
    Symbol.cpp)

target_link_libraries(kotekan_metadata PRIVATE kotekan_utils)
target_include_directories(kotekan_metadata PUBLIC .)

if(${USE_CUDA})
    target_include_directories(kotekan_metadata INTERFACE ${CUDAToolkit_INCLUDE_DIRS})
endif()
