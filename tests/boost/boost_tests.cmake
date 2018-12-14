set ( KOTEKAN_BOOST_TEST_SOURCES
    test_dataset_manager.cpp
    # TODO: disabled until we can fix the Libevent issues
    #test_dataset_manager_rest.cpp
    test_updatequeue.cpp
    test_truncate.cpp
    test_restclient.cpp
    test_chime_stacking.cpp
    test_config.cpp
)

# source files for broker test
set ( KOTEKAN_BROKER_TEST_SOURCES
    dataset_broker_producer.cpp
    dataset_broker_producer2.cpp
    dataset_broker_consumer.cpp
)
