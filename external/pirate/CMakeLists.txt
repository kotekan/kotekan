cmake_minimum_required(VERSION 3.5)
project(pirate)

message(STATUS "PIRATE Current source directory: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "PIRATE Current binary directory: ${CMAKE_CURRENT_BINARY_DIR}")

set(HFILES
    include/pirate/XEngineMetadata.hpp
    include/pirate/YamlFile.hpp)

set(CFILES
    src_lib/XEngineMetadata.cpp
    src_lib/YamlFile.cpp
    src_lib/file_utils.cpp)

add_library(pirate SHARED ${CFILES} ${HFILES})
target_link_libraries(pirate ksgpu)
target_include_directories(pirate PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Install the shared library so installed kotekan can find at runtime.
install(TARGETS pirate LIBRARY DESTINATION lib)


if(CUDAToolkit_FOUND)
    target_include_directories(pirate PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
endif()


# # Set architecture flags
# set_target_properties(n2k PROPERTIES CUDA_ARCHITECTURES "86;89")
# 
# # Set language for CUDA files
# set_target_properties(n2k PROPERTIES LANGUAGE CUDA)
# set_target_properties(n2k PROPERTIES LINKER_LANGUAGE CUDA)
# 
# # Compiler options
# set_property(TARGET n2k PROPERTY CUDA_STANDARD 17)
# set(CMAKE_CUDA_FLAGS
#     "${CMAKE_CUDA_FLAGS} -m64 -O3 --compiler-options -Wall,-fPIC --forward-unknown-to-host-compiler"
# )
