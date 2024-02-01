project(kotekan_metadata)

add_library(kotekan_metadata chordMetadata.cpp)

target_link_libraries(kotekan_metadata PRIVATE kotekan_utils)
target_include_directories(kotekan_metadata PUBLIC .)
