project(kotekan_N2)

add_library(
    kotekan_N2
    N2kAccumulate.cpp
    N2Metadata.cpp
    N2FrameView.cpp)

target_include_directories(kotekan_stages PRIVATE .)

target_link_libraries(kotekan_N2 PRIVATE libexternal kotekan_libs)
