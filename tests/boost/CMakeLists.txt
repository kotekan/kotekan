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

add_executable(test_truncate test_truncate.cpp)

add_executable(test_synchronized_queue test_synchronized_queue.cpp)
target_link_libraries(test_synchronized_queue PRIVATE pthread)

# needs MurmurHash3
add_executable(test_hash test_hash.cpp)
target_link_libraries(test_hash PRIVATE libexternal kotekan_utils)

# test_updatequeue needs fmt
add_executable(test_updatequeue test_updatequeue.cpp)
target_link_libraries(test_updatequeue PRIVATE libexternal)

# test_dataset_manager_rest needs fmt
add_executable(test_dataset_manager_rest test_dataset_manager_rest.cpp)
target_link_libraries(test_dataset_manager_rest PRIVATE libexternal kotekan_core kotekan_utils)

# test_dataset_manager needs fmt
add_executable(test_dataset_manager test_dataset_manager.cpp)
target_link_libraries(test_dataset_manager PRIVATE libexternal kotekan_utils)

# test_restclient needs fmt
add_executable(test_restclient test_restclient.cpp)
target_link_libraries(test_restclient PRIVATE libexternal kotekan_core kotekan_utils)

# test_bip_buffer needs fmt
add_executable(test_bip_buffer test_bip_buffer.cpp)
target_link_libraries(test_bip_buffer PRIVATE libexternal kotekan_utils)

# test_prometheus_metrics needs fmt and prometheusMetrics
add_executable(test_prometheus_metrics test_prometheus_metrics.cpp)
target_link_libraries(test_prometheus_metrics PRIVATE libexternal kotekan_core)

# test_config needs fmt
add_executable(test_config test_config.cpp)
target_link_libraries(test_config PRIVATE libexternal kotekan_core)

# test_chime_stacking needs MurmurHash3 and VisUtil
add_executable(test_chime_stacking test_chime_stacking.cpp)
target_link_libraries(test_chime_stacking PRIVATE libexternal kotekan_utils)

# source files for broker test
add_executable(dataset_broker_producer dataset_broker_producer.cpp)
add_executable(dataset_broker_producer2 dataset_broker_producer2.cpp)
add_executable(dataset_broker_consumer dataset_broker_consumer.cpp)

# dataset_broker executables need fmt
target_link_libraries(dataset_broker_producer2 PRIVATE libexternal kotekan_utils kotekan_core)
target_link_libraries(dataset_broker_producer PRIVATE libexternal kotekan_utils)
target_link_libraries(dataset_broker_consumer PRIVATE libexternal kotekan_utils)

# list test source files that need HDF5 here:
if(${USE_HDF5})
    add_executable(test_transpose test_transpose.cpp)
    include_directories(${HDF5_INCLUDE_DIRS})
    target_link_libraries(test_transpose PRIVATE ${HDF5_HL_LIBRARIES} ${HDF5_LIBRARIES})
    # test_transpose needs fmt
    target_link_libraries(test_transpose PRIVATE libexternal)
endif()
