cmake_minimum_required(VERSION 3.5)
project(pirate)

message(STATUS "PIRATE Current source directory: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "PIRATE Current binary directory: ${CMAKE_CURRENT_BINARY_DIR}")

set(HFILES include/pirate/XEngineMetadata.hpp include/pirate/YamlFile.hpp)

set(CFILES src_lib/XEngineMetadata.cpp src_lib/YamlFile.cpp src_lib/file_utils.cpp)

add_library(pirate SHARED ${CFILES} ${HFILES})
target_link_libraries(pirate ksgpu)
target_include_directories(pirate PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Install the shared library so installed kotekan can find at runtime.
install(TARGETS pirate LIBRARY DESTINATION lib)

if(CUDAToolkit_FOUND)
    target_include_directories(pirate PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
endif()
