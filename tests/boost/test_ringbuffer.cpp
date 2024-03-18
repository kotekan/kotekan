#define BOOST_TEST_MODULE "test_ringbuffer"

#include "Config.hpp" // for Config
#include "metadataFactory.hpp"
#include "ringbuffer.hpp"

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
    std::shared_ptr<metadataPool> pool = facs["main_pool"];
    BOOST_CHECK(pool != nullptr);

    std::cout << "Hello world" << std::endl;
    std::cerr << "Hello error" << std::endl;
    __enable_syslog = 0;
    _global_log_level = 4;

    const int inst = 0; // instance (only used for debug messages)
    RingBuffer rb(20, pool, "rb1", "ring");
    rb.set_log_level("debug");
    rb.print_full_status();

    rb.register_producer("A");
    rb.register_producer("B");
    rb.register_consumer("C");
    rb.register_consumer("D");

    rb.print_full_status();

    std::optional<size_t> oa = rb.wait_for_writable("A", inst, 10);
    std::optional<size_t> ob = rb.wait_for_writable("B", inst, 10);
    BOOST_CHECK(oa.value_or(99) == 0);
    BOOST_CHECK(ob.value_or(99) == 0);

    rb.finish_write("A", inst, 5);
    rb.finish_write("B", inst, 10);

    std::optional<size_t> oc = rb.wait_and_claim_readable("C", inst, 1);
    std::optional<size_t> od = rb.wait_and_claim_readable("D", inst, 5);

    BOOST_CHECK(oc.value_or(99) == 0);
    BOOST_CHECK(od.value_or(99) == 0);

    rb.finish_read("C", inst, 1);
    rb.finish_read("D", inst, 5);

    INFO_NON_OO("Finished reading 1/5 items");
    rb.print_full_status();

    oa = rb.wait_for_writable("A", inst, 10);
    ob = rb.wait_for_writable("B", inst, 10);

    INFO_NON_OO("oa: {:d}, ob: {:d}", oa.value_or(99), oa.value_or(99));
    BOOST_CHECK(oa.value_or(99) == 10);
    BOOST_CHECK(ob.value_or(99) == 10);

    rb.finish_write("A", inst, 5);
    rb.finish_write("B", inst, 10);

    INFO_NON_OO("Finished writing 5/10 items");
    rb.print_full_status();

    oc = rb.wait_and_claim_readable("C", inst, 1);
    od = rb.wait_and_claim_readable("D", inst, 5);

    BOOST_CHECK(oc.value_or(99) == 1);
    BOOST_CHECK(od.value_or(99) == 5);

    rb.finish_read("C", inst, 1);
    rb.finish_read("D", inst, 5);

    auto owa = rb.get_writable("A", inst);
    auto owb = rb.get_writable("B", inst);
    auto orc = rb.peek_readable("C", inst);
    auto ord = rb.peek_readable("D", inst);

    BOOST_CHECK(owa.has_value());
    BOOST_CHECK(owb.has_value());
    BOOST_CHECK(orc.has_value());
    BOOST_CHECK(ord.has_value());

    auto wa = owa.value();
    auto wb = owb.value();
    auto rc = orc.value();
    auto rd = ord.value();

    INFO_NON_OO("A: writable: offset {:d}, n {:d}", wa.first, wa.second);
    INFO_NON_OO("B: writable: offset {:d}, n {:d}", wb.first, wb.second);
    INFO_NON_OO("C: readable: offset {:d}, n {:d}", rc.first, rc.second);
    INFO_NON_OO("D: readable: offset {:d}, n {:d}", rd.first, rd.second);

    BOOST_CHECK(wa.first == 0);
    BOOST_CHECK(wa.second == 2);
    BOOST_CHECK(wb.first == 0); // wrapped
    BOOST_CHECK(wb.second == 2);
    BOOST_CHECK(rc.first == 2);
    BOOST_CHECK(rc.second == 8);
    BOOST_CHECK(rd.first == 10);
    BOOST_CHECK(rd.second == 0);

    oc = rb.wait_and_claim_readable("C", inst, 1);
    oc = rb.wait_and_claim_readable("C", inst, 1);
    oc = rb.wait_and_claim_readable("C", inst, 1);
    BOOST_CHECK(oc.value_or(99) == 4);

    rb.finish_read("C", inst, 1);
    rb.finish_read("C", inst, 1);
    rb.finish_read("C", inst, 1);

    owa = rb.get_writable("A", inst);
    wa = owa.value();

    INFO_NON_OO("A: writable: offset {:d}, n {:d}", wa.first, wa.second);
    BOOST_CHECK(wa.first == 20);
    BOOST_CHECK(wa.second == 5);
}
