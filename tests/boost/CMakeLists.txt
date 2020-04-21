project(kotekan_boost_tests)

# Prep ourselves for compiling boost
find_package(
    Boost
    COMPONENTS unit_test_framework
    REQUIRED)
include_directories(${Boost_INCLUDE_DIRS} $_BOOST_TESTS_DIR)

include_directories(${KOTEKAN_SOURCE_DIR}/lib/stages)
include_directories(${KOTEKAN_SOURCE_DIR}/lib/utils)
include_directories(${KOTEKAN_SOURCE_DIR}/lib/core)
include_directories(${KOTEKAN_SOURCE_DIR}/lib)
include_directories(${KOTEKAN_SOURCE_DIR}/lib/metadata)

set(KOTEKAN_BOOST_TEST_SOURCES
    test_bip_buffer.cpp
    test_dataset_manager.cpp
    test_dataset_manager_rest.cpp
    test_updatequeue.cpp
    test_truncate.cpp
    test_restclient.cpp
    test_chime_stacking.cpp
    test_config.cpp
    test_prometheus_metrics.cpp
    test_synchronized_queue.cpp
    test_hash.cpp)

# source files for broker test
set(KOTEKAN_BROKER_TEST_SOURCES dataset_broker_producer.cpp dataset_broker_producer2.cpp
                                dataset_broker_consumer.cpp)

# list test source files that need HDF5 here:
if(${USE_HDF5})
    set(KOTEKAN_BOOST_TEST_SOURCES ${KOTEKAN_BOOST_TEST_SOURCES} test_transpose.cpp)
endif()

# Run through each source
foreach(testSrc ${KOTEKAN_BOOST_TEST_SOURCES})
    get_filename_component(testName ${testSrc} NAME_WE)
    add_executable(${testName} ${testSrc})

    # TODO: fix cmake build for buffer.c : It uses hsa_host_malloc etc. TODO: remove chimeFeed from
    # visCompression files. After that it should be enough to link against kotekan_utils
    # kotekan_core
    target_link_libraries(${testName} kotekan_libs dl m pthread)

    if(${USE_HDF5})
        include_directories(${HDF5_INCLUDE_DIRS})
        target_link_libraries(${testName} ${HDF5_HL_LIBRARIES} ${HDF5_LIBRARIES})
    endif()
endforeach(testSrc)

# Run through each dataset broker test source
foreach(testSrc ${KOTEKAN_BROKER_TEST_SOURCES})
    get_filename_component(testName ${testSrc} NAME_WE)
    add_executable(${testName} ${testSrc})
    target_link_libraries(${testName} libexternal)
    target_link_libraries(${testName} kotekan_utils kotekan_core dl m pthread)
endforeach(testSrc)
