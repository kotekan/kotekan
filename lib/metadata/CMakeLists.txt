project(kotekan_metadata)

add_library(kotekan_metadata chordMetadata.cpp chimeMetadata.cpp HFBMetadata.cpp
                             BasebandMetadata.cpp BeamMetadata.cpp N2Metadata.cpp)

target_link_libraries(kotekan_metadata PRIVATE kotekan_utils)
target_include_directories(kotekan_metadata PUBLIC .)
