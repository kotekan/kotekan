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
    test_dataset_manager.cpp
    test_dataset_manager_rest.cpp
    test_updatequeue.cpp
    test_truncate.cpp
    test_restclient.cpp
    test_chime_stacking.cpp
)

# list test source files that need HDF5 here:
if(${USE_HDF5})
    set(KOTEKAN_BOOST_TEST_SOURCES ${KOTEKAN_BOOST_TEST_SOURCES} test_transpose.cpp)
endif()

# Run through each source
foreach(testSrc ${KOTEKAN_BOOST_TEST_SOURCES})
	get_filename_component(testName ${testSrc} NAME_WE)
    add_executable(${testName} ${testSrc})
    target_link_libraries(${testName} kotekan_libs libinclude dl m pthread)

    if(${USE_HDF5})
        include_directories(${HDF5_INCLUDE_DIRS})
        target_link_libraries(${testName} ${HDF5_HL_LIBRARIES} ${HDF5_LIBRARIES})
    endif()

    # testing binaries are located in /tests and executed by pytest
    # if pytest-cpp is installed
    set_target_properties(${testName} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/..)
endforeach(testSrc)

# compile source files for broker test
set(BROKER_BOOST_TEST_SOURCES
    dataset_broker_producer.cpp
    dataset_broker_producer2.cpp
    dataset_broker_consumer.cpp
)

# Run through each source
foreach(testSrc ${BROKER_BOOST_TEST_SOURCES})
	get_filename_component(testName ${testSrc} NAME_WE)
    add_executable(${testName} ${testSrc})
    target_link_libraries(${testName} kotekan_libs libinclude dl m pthread)

    # those binaries will be in a seperate folder and are run by
    # test_dataset_broker.py
    set_target_properties(${testName} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/dataset-broker)

    # they only work together with the broker
    add_dependencies(${testName} ch_acq)
endforeach(testSrc)

find_package(Git REQUIRED)
include(ExternalProject)
ExternalProject_Add(ch_acq
                    PREFIX            ${CMAKE_BINARY_DIR}/ext
                    GIT_REPOSITORY    git@bitbucket.org:chime/ch_acq.git
                    GIT_TAG           dataset-broker
                    BUILD_COMMAND     ""
                    CONFIGURE_COMMAND ""
                    INSTALL_COMMAND   ""
                    UPDATE_COMMAND    ${GIT_EXECUTABLE} pull
                    UPDATE_DISCONNECTED 0
                    LOG_DOWNLOAD        ON
                    LOG_UPDATE          ON
)

