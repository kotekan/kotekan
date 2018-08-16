project( kotekan_boost_tests )

# Prep ourselves for compiling boost
find_package(Boost COMPONENTS unit_test_framework REQUIRED)
include_directories (${Boost_INCLUDE_DIRS} $BOOST_TESTS_DIR)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/utils)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/processes)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/core)
include_directories (${KOTEKAN_SOURCE_DIR}/lib)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/metadata)
include_directories (${KOTEKAN_SOURCE_DIR}/include)

# test source files are located in tests/boost/
# list them all here:
set(KOTEKAN_BOOST_TEST_SOURCES
                test_truncate.cpp
                test_dataset_manager.cpp)

# list test source files that need HDF5 here:
if(${USE_HDF5})
    set(KOTEKAN_BOOST_TEST_SOURCES ${KOTEKAN_BOOST_TEST_SOURCES} test_transpose.cpp)
endif()

# Run through each source
foreach(testSrc ${KOTEKAN_BOOST_TEST_SOURCES})
	get_filename_component(testName ${testSrc} NAME_WE)
    add_executable(${testName} ${testSrc})
    target_link_libraries(${testName} kotekan_utils kotekan_core)

    if(${USE_HDF5})
        include_directories(${HDF5_INCLUDE_DIRS})
        target_link_libraries(${testName} ${HDF5_HL_LIBRARIES} ${HDF5_LIBRARIES})
    endif()

    # testing binaries are located in /tests and executed by pytest
    # if pytest-cpp is installed
    set_target_properties(${testName} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/..)
endforeach(testSrc)

