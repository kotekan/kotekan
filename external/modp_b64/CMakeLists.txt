project(MurmurHash3)

add_library(modp_b64 modp_b64.cpp)
target_include_directories(modp_b64 SYSTEM INTERFACE .)
