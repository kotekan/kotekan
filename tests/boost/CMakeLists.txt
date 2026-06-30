project(kotekan_boost_tests)

# Prep ourselves for compiling boost
find_package(Boost REQUIRED COMPONENTS unit_test_framework filesystem)

include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

add_executable(test_truncate test_truncate.cpp)
target_link_libraries(test_truncate PRIVATE kotekan_libs kotekan_utils)

add_executable(test_synchronized_queue test_synchronized_queue.cpp)
target_link_libraries(test_synchronized_queue PRIVATE pthread kotekan_libs kotekan_utils)

# Pure C/A code generator: compile the source directly so the test pulls in no
# FFTW / libairspy dependency from kotekan_stages.
add_executable(test_gps_ca_code test_gps_ca_code.cpp ../../lib/stages/gpsCACode.cpp)
target_include_directories(test_gps_ca_code PRIVATE ../../lib/stages)

add_executable(test_gps_l2c_code test_gps_l2c_code.cpp ../../lib/stages/gpsL2CCode.cpp)
target_include_directories(test_gps_l2c_code PRIVATE ../../lib/stages)

add_executable(test_gps_l5_code test_gps_l5_code.cpp ../../lib/stages/gpsL5Code.cpp)
target_include_directories(test_gps_l5_code PRIVATE ../../lib/stages)

add_executable(test_gps_l1c_code test_gps_l1c_code.cpp ../../lib/stages/gpsL1CCode.cpp)
target_include_directories(test_gps_l1c_code PRIVATE ../../lib/stages)

# Pure band-plan logic: compile the source directly, no Telescope/FFTW needed.
add_executable(test_gnss_band_plan test_gnss_band_plan.cpp ../../lib/stages/gnssBandPlan.cpp)
target_include_directories(test_gnss_band_plan PRIVATE ../../lib/stages)

# Broker assignment logic: pure, no FFTW.
add_executable(test_gnss_broker test_gnss_broker.cpp ../../lib/stages/gnssBroker.cpp
                                ../../lib/stages/gnssBandPlan.cpp)
target_include_directories(test_gnss_broker PRIVATE ../../lib/stages)

# Pure PFB core (prototype design + polyphase fold): no FFTW; the test uses a
# reference DFT to check channel isolation against a straight-FFT channelizer.
add_executable(test_pfb_prototype test_pfb_prototype.cpp ../../lib/stages/pfbPrototype.cpp)
target_include_directories(test_pfb_prototype PRIVATE ../../lib/stages)

# Channelized despread: sim signal -> PFB analysis (reference DFT) -> per-channel
# correlate-and-sum, checked against the full-band matched filter. No FFTW.
add_executable(
    test_gnss_channelized_despread test_gnss_channelized_despread.cpp
    ../../lib/stages/gnssChannelizedDespread.cpp ../../lib/stages/gnssSimSignal.cpp
    ../../lib/stages/pfbPrototype.cpp ../../lib/stages/gpsCACode.cpp
    ../../lib/stages/gpsL5Code.cpp ../../lib/stages/gpsL1CCode.cpp)
target_include_directories(test_gnss_channelized_despread PRIVATE ../../lib/stages)

# Channelized acquisition: FFT-domain coarse correlation + cross-channel fine-lag
# DFT + incoherent multi-window accumulation, checked against injected code
# phase / Doppler. Gated on FFTW (matches lib/stages, which only compiles
# gnssChannelizedAcquire.cpp under FFTW_FOUND): the coarse-correlation workspace
# needs fftwf. FFTW_FOUND is set by FeatureMath.cmake before this subdirectory.
if(FFTW_FOUND)
    add_executable(
        test_gnss_channelized_acquire test_gnss_channelized_acquire.cpp
        ../../lib/stages/gnssChannelizedAcquire.cpp ../../lib/stages/pfbPrototype.cpp
        ../../lib/stages/gpsCACode.cpp
        # The L2C cases drive the real ChannelizedReplicaBank (time-mux combined code) so
        # the bug hunt exercises the production replica + acquire path, not a test stand-in.
        ../../lib/stages/gnssChannelizedReplica.cpp ../../lib/stages/gnssBandPlan.cpp
        ../../lib/stages/gnssChannelizedDespread.cpp
        ../../lib/stages/gpsL2CCode.cpp ../../lib/stages/gpsL5Code.cpp)
    target_include_directories(test_gnss_channelized_acquire PRIVATE ../../lib/stages)
    target_include_directories(test_gnss_channelized_acquire SYSTEM PRIVATE ${FFTW_INCLUDES})
    target_link_libraries(test_gnss_channelized_acquire PRIVATE ${FFTW_LIBRARY} ${FFTWF_LIBRARY})

    # Channelized replica bank: per-signal code dispatch (L1 C/A, L5 I/Q, L2C),
    # checked by self / cross-PRN / cross-signal despread. Needs FFTW (r2c replica).
    add_executable(
        test_gnss_channelized_replica test_gnss_channelized_replica.cpp
        ../../lib/stages/gnssChannelizedReplica.cpp ../../lib/stages/gnssChannelizedDespread.cpp
        ../../lib/stages/gnssBandPlan.cpp ../../lib/stages/pfbPrototype.cpp
        ../../lib/stages/gpsCACode.cpp ../../lib/stages/gpsL1CCode.cpp ../../lib/stages/gpsL2CCode.cpp
        ../../lib/stages/gpsL5Code.cpp)
    target_include_directories(test_gnss_channelized_replica PRIVATE ../../lib/stages)
    target_include_directories(test_gnss_channelized_replica SYSTEM PRIVATE ${FFTW_INCLUDES})
    target_link_libraries(test_gnss_channelized_replica PRIVATE ${FFTW_LIBRARY} ${FFTWF_LIBRARY})
endif()

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
target_link_libraries(test_dataset_manager_rest PRIVATE libexternal kotekan_libs kotekan_utils
                                                        kotekan_core)

# test_dataset_manager needs fmt
add_executable(test_dataset_manager test_dataset_manager.cpp)
target_link_libraries(test_dataset_manager PRIVATE libexternal kotekan_libs kotekan_utils
                                                   kotekan_core)

# test_ringbuffer
add_executable(test_ringbuffer test_ringbuffer.cpp)
target_link_libraries(test_ringbuffer PRIVATE libexternal kotekan_libs kotekan_utils kotekan_core)

# test_fmt
add_executable(test_fmt test_fmt.cpp)
# target_link_libraries(test_fmt PRIVATE libexternal kotekan_utils)
target_link_libraries(test_fmt PRIVATE kotekan_libs kotekan_utils)

# test_restclient needs fmt
add_executable(test_restclient test_restclient.cpp)
target_link_libraries(test_restclient PRIVATE libexternal kotekan_libs kotekan_utils kotekan_core)

# test_bip_buffer needs fmt
add_executable(test_bip_buffer test_bip_buffer.cpp)
target_link_libraries(test_bip_buffer PRIVATE libexternal kotekan_libs kotekan_utils kotekan_core)

# test_prometheus_metrics needs fmt and prometheusMetrics
add_executable(test_prometheus_metrics test_prometheus_metrics.cpp)
target_link_libraries(test_prometheus_metrics PRIVATE libexternal kotekan_libs kotekan_core)

# test_config needs Config, json
add_executable(test_config test_config.cpp)
target_link_libraries(test_config PRIVATE libexternal kotekan_core kotekan_stages)

# test_configTracker needs
add_executable(test_configTracker test_configTracker.cpp)
target_link_libraries(test_configTracker PRIVATE libexternal kotekan_core kotekan_stages
                                                 kotekan_metadata kotekan_utils Boost::filesystem)

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
if(HDF5_FOUND
   AND HighFive_FOUND
   AND HDF5Plugin_PLUGIN_DIR)
    add_executable(test_transpose test_transpose.cpp)
    target_link_libraries(test_transpose PRIVATE ${HDF5_HL_LIBRARIES} ${HDF5_LIBRARIES})
    # test_transpose needs fmt and VisTranspose
    target_link_libraries(test_transpose PRIVATE libexternal kotekan_stages kotekan_core
                                                 kotekan_utils kotekan_metadata)
    target_include_directories(test_transpose PRIVATE ${KOTEKAN_SOURCE_DIR}/lib/stages)
    target_include_directories(test_transpose SYSTEM PUBLIC ${CUDA_INCLUDE_DIR}
                                                            ${CUDA_INCLUDE_DIRS})
endif()

if(LAPACKE_FOUND AND BLAZE_FOUND)
    add_executable(test_eigenStages test_eigenStages.cpp)
    target_link_libraries(test_eigenStages PRIVATE libexternal kotekan_stages kotekan_libs
                                                   kotekan_utils kotekan_core)
    target_include_directories(test_eigenStages PRIVATE ${KOTEKAN_SOURCE_DIR}/lib/stages
                                                        ${KOTEKAN_SOURCE_DIR}/lib/testing)
endif()

add_executable(test_DataType test_DataType.cpp)
target_link_libraries(test_DataType kotekan_metadata libexternal)

add_executable(test_Metadata test_Metadata.cpp)
target_link_libraries(test_Metadata kotekan_metadata)

add_executable(test_metadataObject test_metadataObject.cpp)
target_link_libraries(test_metadataObject kotekan_metadata kotekan_utils libexternal)

add_executable(test_NDArray test_NDArray.cpp)
target_link_libraries(test_NDArray libexternal kotekan_metadata)

add_executable(test_Symbol test_Symbol.cpp)
target_link_libraries(test_Symbol libexternal kotekan_metadata)

add_executable(test_div test_div.cpp)
target_link_libraries(test_div PRIVATE kotekan_utils)

add_executable(test_timeUtil test_timeUtil.cpp)
target_link_libraries(test_timeUtil PRIVATE kotekan_utils)
target_compile_definitions(test_timeUtil PRIVATE TEST_SCRIPT_DIR="${PROJECT_BINARY_DIR}")

add_custom_command(
    OUTPUT ${PROJECT_BINARY_DIR}/timeUtil.py
    DEPENDS ${PROJECT_SOURCE_DIR}/timeUtil.py
    COMMAND ${CMAKE_COMMAND} ARGS -E copy ${PROJECT_SOURCE_DIR}/timeUtil.py
            ${PROJECT_BINARY_DIR}/timeUtil.py
    COMMENT "Copying timeUtil.py script into destination")
add_custom_target(
    test_timeUtil_script ALL
    DEPENDS ${PROJECT_BINARY_DIR}/timeUtil.py
    COMMENT "Generating timeUtil.py script")

# visFileData unit test + end-to-end writer test (HDF5 backend)
if(HDF5_FOUND AND HighFive_FOUND)
    add_executable(test_hdf5N2Write test_hdf5N2Write.cpp)
    target_link_libraries(
        test_hdf5N2Write
        PRIVATE libexternal
                kotekan_stages
                kotekan_libs
                kotekan_utils
                kotekan_core
                ${HDF5_HL_LIBRARIES}
                ${HDF5_LIBRARIES}
                HighFive::HighFive)
    target_include_directories(
        test_hdf5N2Write PRIVATE ${KOTEKAN_SOURCE_DIR}/lib/stages ${KOTEKAN_SOURCE_DIR}/lib/utils
                                 ${HDF5_INCLUDE_DIRS})
    target_include_directories(test_hdf5N2Write PUBLIC . ${CUDA_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
    target_compile_definitions(test_hdf5N2Write
                               PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/testdata")
endif()

add_executable(test_Telescope test_Telescope.cpp)
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    target_link_libraries(test_Telescope PRIVATE -Wl,-all_load kotekan_utils kotekan_testing)
else()
    target_link_libraries(test_Telescope PRIVATE -Wl,--whole-archive kotekan_utils kotekan_testing
                                                 -Wl,--no-whole-archive)
endif()

add_executable(test_CHORDTelescope test_CHORDTelescope.cpp)
target_link_libraries(test_CHORDTelescope PRIVATE kotekan_utils)

# test_N2FrameDesc - unit tests for N2FrameDesc product list features
add_executable(test_N2FrameDesc test_N2FrameDesc.cpp)
target_link_libraries(test_N2FrameDesc PRIVATE kotekan_metadata kotekan_utils libexternal)

# test_N2Subset - tests for N2Subset stage
add_executable(test_N2Subset test_N2Subset.cpp)
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    target_link_libraries(
        test_N2Subset
        PRIVATE libexternal
                kotekan_stages
                kotekan_libs
                kotekan_utils
                kotekan_core
                -Wl,-all_load
                kotekan_metadata)
else()
    target_link_libraries(
        test_N2Subset
        PRIVATE libexternal
                kotekan_stages
                kotekan_libs
                kotekan_utils
                kotekan_core
                -Wl,--whole-archive
                kotekan_metadata
                -Wl,--no-whole-archive)
endif()
target_include_directories(test_N2Subset PRIVATE ${KOTEKAN_SOURCE_DIR}/lib/stages
                                                 ${KOTEKAN_SOURCE_DIR}/lib/utils)
