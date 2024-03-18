project(kotekan_boost_tests)

# Prep ourselves for compiling boost
find_package(
    Boost
    COMPONENTS unit_test_framework
    REQUIRED)

include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

add_executable(test_truncate test_truncate.cpp)
target_link_libraries(test_truncate PRIVATE kotekan_libs kotekan_utils)

add_executable(test_synchronized_queue test_synchronized_queue.cpp)
target_link_libraries(test_synchronized_queue PRIVATE pthread kotekan_libs kotekan_utils)

add_executable(test_stat_tracker test_stat_tracker.cpp)
target_link_libraries(test_stat_tracker PRIVATE libexternal kotekan_libs kotekan_utils kotekan_core)

# needs MurmurHash3
add_executable(test_hash test_hash.cpp)
target_link_libraries(test_hash PRIVATE libexternal kotekan_libs kotekan_utils)

# test_updatequeue needs fmt
add_executable(test_updatequeue test_updatequeue.cpp)
target_link_libraries(test_updatequeue PRIVATE libexternal kotekan_libs kotekan_utils kotekan_core)

# test_dataset_manager_rest needs fmt
add_executable(test_dataset_manager_rest test_dataset_manager_rest.cpp)
target_link_libraries(test_dataset_manager_rest PRIVATE libexternal kotekan_libs kotekan_core
                                                        kotekan_utils)

# test_dataset_manager needs fmt
add_executable(test_dataset_manager test_dataset_manager.cpp)
target_link_libraries(test_dataset_manager PRIVATE libexternal kotekan_libs kotekan_utils
                                                   kotekan_core)

# test_ringbuffer
add_executable(test_ringbuffer test_ringbuffer.cpp)
target_link_libraries(test_ringbuffer PRIVATE libexternal kotekan_core kotekan_libs kotekan_utils)

# test_fmt
add_executable(test_fmt test_fmt.cpp)
# target_link_libraries(test_fmt PRIVATE libexternal kotekan_utils)
target_link_libraries(test_fmt PRIVATE kotekan_libs kotekan_utils)

# test_restclient needs fmt
add_executable(test_restclient test_restclient.cpp)
target_link_libraries(test_restclient PRIVATE libexternal kotekan_core kotekan_libs kotekan_utils)

# test_bip_buffer needs fmt
add_executable(test_bip_buffer test_bip_buffer.cpp)
target_link_libraries(test_bip_buffer PRIVATE libexternal kotekan_libs kotekan_utils kotekan_core)

# test_prometheus_metrics needs fmt and prometheusMetrics
add_executable(test_prometheus_metrics test_prometheus_metrics.cpp)
target_link_libraries(test_prometheus_metrics PRIVATE libexternal kotekan_libs kotekan_core)

# test_config needs fmt
add_executable(test_config test_config.cpp)
target_link_libraries(test_config PRIVATE libexternal kotekan_core kotekan_stages)

# test_chime_stacking needs MurmurHash3 and VisUtil
add_executable(test_chime_stacking test_chime_stacking.cpp)
target_link_libraries(test_chime_stacking PRIVATE libexternal kotekan_utils kotekan_stages
                                                  kotekan_core)
target_include_directories(test_chime_stacking PRIVATE ${KOTEKAN_SOURCE_DIR}/lib/stages)

# source files for broker test
add_executable(dataset_broker_producer dataset_broker_producer.cpp)
add_executable(dataset_broker_producer2 dataset_broker_producer2.cpp)
add_executable(dataset_broker_consumer dataset_broker_consumer.cpp)

# dataset_broker executables need fmt
target_link_libraries(dataset_broker_producer2 PRIVATE libexternal kotekan_libs kotekan_utils
                                                       kotekan_core)
target_link_libraries(dataset_broker_producer PRIVATE libexternal kotekan_libs kotekan_utils
                                                      kotekan_core)
target_link_libraries(dataset_broker_consumer PRIVATE libexternal kotekan_libs kotekan_utils
                                                      kotekan_core)

# Many tests now need the Cuda include directory for float16_t via visUtil.hpp
target_include_directories(dataset_broker_producer2 PUBLIC . ${CUDA_INCLUDE_DIR}
                                                           ${CUDA_INCLUDE_DIRS})
target_include_directories(dataset_broker_producer PUBLIC . ${CUDA_INCLUDE_DIR}
                                                          ${CUDA_INCLUDE_DIRS})
target_include_directories(dataset_broker_consumer PUBLIC . ${CUDA_INCLUDE_DIR}
                                                          ${CUDA_INCLUDE_DIRS})
target_include_directories(test_chime_stacking PUBLIC . ${CUDA_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
target_include_directories(test_stat_tracker PUBLIC . ${CUDA_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
target_include_directories(test_updatequeue PUBLIC . ${CUDA_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
target_include_directories(test_dataset_manager PUBLIC . ${CUDA_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
target_include_directories(test_dataset_manager_rest PUBLIC . ${CUDA_INCLUDE_DIR}
                                                            ${CUDA_INCLUDE_DIRS})

# list test source files that need HDF5 here:
if(${USE_HDF5})
    add_executable(test_transpose test_transpose.cpp)
    target_link_libraries(test_transpose PRIVATE ${HDF5_HL_LIBRARIES} ${HDF5_LIBRARIES})
    # test_transpose needs fmt and VisTranspose
    target_link_libraries(test_transpose PRIVATE libexternal kotekan_stages kotekan_core
                                                 kotekan_utils kotekan_metadata)
    target_include_directories(test_transpose PRIVATE ${KOTEKAN_SOURCE_DIR}/lib/stages)
    target_include_directories(test_transpose SYSTEM PUBLIC ${CUDA_INCLUDE_DIR}
                                                            ${CUDA_INCLUDE_DIRS})
endif()
