project(kotekan_metadata)

add_library(
    kotekan_metadata
    chordMetadata.cpp
    chimeMetadata.cpp
    HFBMetadata.cpp
    BasebandMetadata.cpp
    BeamMetadata.cpp
    DataType.cpp
    Metadata.cpp
    NDArray.cpp
    Symbol.cpp)

target_link_libraries(kotekan_metadata PRIVATE kotekan_utils)
target_include_directories(kotekan_metadata PUBLIC .)

add_executable(testDataType testDataType.cpp)
target_link_libraries(testDataType kotekan_metadata)

add_executable(testMetadata testMetadata.cpp)
target_link_libraries(testMetadata kotekan_metadata)

add_executable(testNDArray testNDArray.cpp)
target_link_libraries(testNDArray kotekan_metadata)

add_executable(testSymbol testSymbol.cpp)
target_link_libraries(testSymbol kotekan_metadata)
