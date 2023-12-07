#define BOOST_TEST_MODULE "test_ringbuffer"

#include "Config.hpp" // for Config
#include "ringbuffer.hpp"
#include "metadataFactory.hpp"

#include <memory>

#undef WITH_CUDA

#include "json.hpp" // for basic_json<>::object_t, basic_json<>::value...

#include <boost/test/included/unit_test.hpp>
#include <iostream>

using kotekan::Config;
using json = nlohmann::json;
using namespace kotekan;
using namespace std;

BOOST_AUTO_TEST_CASE(test1) {
    // Create metadata pool
    json json_config = json::parse(
        R"({"type": "config", "log_level": "info", "main_pool": {"kotekan_metadata_pool": "oneHotMetadata", "num_metadata_objects": 100}})");
    Config conf2;
    conf2.update_config(json_config);
    metadataFactory mfac(conf2);
    auto facs = mfac.build_pools();
    struct metadataPool* pool = facs["main_pool"];
    BOOST_CHECK(pool != nullptr);

    std::cout << "Hello world" << std::endl;
    std::cerr << "Hello error" << std::endl;
    __enable_syslog = 0;
    _global_log_level = 4;

    RingBuffer rb(100, pool, "rb1", "ring");
    rb.set_log_level("debug");
    rb.print_full_status();

    rb.register_producer("A");
    rb.register_producer("B");
    rb.register_consumer("C");
    rb.register_consumer("D");

    rb.print_full_status();

    std::optional<size_t> oa = rb.wait_for_writable("A", 10);
    std::optional<size_t> ob = rb.wait_for_writable("B", 10);
    BOOST_CHECK(oa.value_or(99) == 0);
    BOOST_CHECK(ob.value_or(99) == 0);

    rb.finish_write("A", 5);
    rb.finish_write("B", 10);

    std::optional<size_t> oc = rb.wait_and_claim_readable("C", 1);
    std::optional<size_t> od = rb.wait_and_claim_readable("D", 5);
    
    BOOST_CHECK(oc.value_or(99) == 0);
    BOOST_CHECK(od.value_or(99) == 0);

    rb.finish_read("C", 1);
    rb.finish_read("D", 5);

    INFO_NON_OO("Finished reading 1/5 items");
    rb.print_full_status();

    oa = rb.wait_for_writable("A", 10);
    ob = rb.wait_for_writable("B", 10);

    INFO_NON_OO("oa: {:d}, ob: {:d}", oa.value_or(99), oa.value_or(99));
    BOOST_CHECK(oa.value_or(99) == 5);
    BOOST_CHECK(ob.value_or(99) == 10);

    rb.finish_write("A", 5);
    rb.finish_write("B", 10);

    INFO_NON_OO("Finished writing 5/10 items");
    rb.print_full_status();

    oc = rb.wait_and_claim_readable("C", 1);
    od = rb.wait_and_claim_readable("D", 5);
    
    BOOST_CHECK(oc.value_or(99) == 1);
    BOOST_CHECK(od.value_or(99) == 5);

}
