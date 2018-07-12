project( kotekan_boost_tests )

# Prep ourselves for compiling boost
find_package(Boost COMPONENTS unit_test_framework REQUIRED)
include_directories (${Boost_INCLUDE_DIRS} $BOOST_TESTS_DIR)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/utils)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/processes)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/core)
include_directories (${KOTEKAN_SOURCE_DIR}/lib)

# test source files are located in tests/boost/
file(GLOB TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${BOOST_TESTS_DIR}*.cpp)

# Run through each source
foreach(testSrc ${TEST_SRCS})
	get_filename_component(testName ${testSrc} NAME_WE)
    add_executable(${testName} ${testSrc})

    # testing binaries are located in /tests and executed by pytest
    # if pytest-cpp is installed
    set_target_properties(${testName} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach(testSrc)

