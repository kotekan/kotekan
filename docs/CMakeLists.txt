set(CMAKE_MODULE_PATH ${KOTEKAN_SOURCE_DIR}/cmake ${KOTEKAN_SOURCE_DIR}/cmake/Modules)

find_package(Doxygen REQUIRED)
add_subdirectory(doxygen)

# find_package(SPHINX REQUIRED)
find_package(BREATHE REQUIRED)
add_subdirectory(sphinx)

# Aggregate documentation targets
add_custom_target(
    docs
    DEPENDS sphinx doxygen
    COMMENT "Building all documentation")
add_custom_target(
    doc
    DEPENDS docs
    COMMENT "Building all documentation (alias)")
