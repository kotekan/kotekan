project(kotekan_metadata)

add_library(kotekan_metadata chordMetadata.cpp chimeMetadata.cpp HFBMetadata.cpp
                             BasebandMetadata.cpp BeamMetadata.cpp
                             Metadata.cpp NDArray.cpp
                             # Struct.cpp
                             Symbol.cpp)

target_link_libraries(kotekan_metadata PRIVATE kotekan_utils)
target_include_directories(kotekan_metadata PUBLIC .)

add_executable(testMetadata testMetadata.cpp)
target_link_libraries(testMetadata kotekan_metadata)

#TODO add_executable(testNDArray testNDArray.cpp)
#TODO target_link_libraries(testNDArray kotekan_metadata)

add_executable(testSymbol testSymbol.cpp)
target_link_libraries(testSymbol kotekan_metadata)
