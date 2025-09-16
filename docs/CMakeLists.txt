set(CMAKE_MODULE_PATH ${KOTEKAN_SOURCE_DIR}/cmake ${KOTEKAN_SOURCE_DIR}/cmake/Modules)

find_package(Doxygen REQUIRED)
add_subdirectory(doxygen)

find_package(SPHINX REQUIRED)
find_package(BREATHE REQUIRED)
add_subdirectory(sphinx)
