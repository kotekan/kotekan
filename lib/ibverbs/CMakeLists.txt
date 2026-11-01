project(kotekan_ibverbs)

# Assumes FeatureIbverbs.cmake already ran at the top level and populated IBVERBS_* variables via
# pkg-config when enabled and detected.

add_library(kotekan_ibverbs ibvUtils.cpp ibvUdpSend.cpp ibvUdpRecv.cpp)

target_include_directories(kotekan_ibverbs SYSTEM PRIVATE ${IBVERBS_INCLUDE_DIRS})
target_link_libraries(kotekan_ibverbs PRIVATE ${IBVERBS_LDFLAGS} libexternal kotekan_libs)
target_include_directories(kotekan_ibverbs PUBLIC .)
