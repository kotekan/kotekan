set(CMAKE_MODULE_PATH ${KOTEKAN_SOURCE_DIR}/cmake)

find_package(Doxygen REQUIRED)
add_subdirectory(doxygen)

find_package(Sphinx REQUIRED)
find_package(Breathe REQUIRED)
add_subdirectory(sphinx)
