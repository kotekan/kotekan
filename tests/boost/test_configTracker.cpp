#define BOOST_TEST_MODULE "test_configTracker"

#include <boost/test/included/unit_test.hpp> // for BOOST_PP_IIF_1, BOOST_PP_IIF_0, BOOST_PP_BO...

// the code to test:
#include "configTracker.hpp" // for configTracker

#include "json.hpp" // for json_ref, basic_json<>::object_t, json

using namespace kotekan;
using json = nlohmann::json;

// Wrapper structure for ConfigTracker to ensure it is reset/cleared after each test.
struct ConfigFixture {
    ConfigFixture() {
        auto& tracker = ConfigTracker::instance();
        tracker.reset();
    }

    ~ConfigFixture() {
        auto& tracker = ConfigTracker::instance();
        tracker.reset();
    }
};

BOOST_FIXTURE_TEST_SUITE(MyFeatureTests, ConfigFixture)

BOOST_AUTO_TEST_CASE(test_add_json) {

    auto& tracker = ConfigTracker::instance();

    // Example json
    json j = {{"key1", "value1"},
              {"key2", {{"subkey1", "subvalue1"}, {"subkey2", "subvalue2"}}},
              {"key3", "value3"}};
    // The hash of this should be ...

    tracker.insertConfig("localhost", 8080, j, "1.0.0", "main", "abcdef1234567890",
                         "CMAKE_BUILD_TYPE=Release");

    // Check hashing
    BOOST_TEST_MESSAGE("Config hash from getTrackerHash: " << tracker.getTrackerHash());

    BOOST_CHECK_EQUAL(tracker.getTrackerHash(), "16f3adf520a3a7cca9831c7952c4749f");
    BOOST_CHECK_EQUAL(tracker.n_configs(), 1);
}

BOOST_AUTO_TEST_CASE(test_add_two_jsons) {

    auto& tracker = ConfigTracker::instance();

    // Some json
    json j1 = {{"key1", "value1"},
               {"key2", {{"subkey1", "subvalue1"}, {"subkey2", "subvalue2"}}},
               {"key3", "value3"}};

    // Another json
    json j2 = {{"key4", "value4"},
               {"key5", {{"subkey3", "subvalue3"}, {"subkey4", "subvalue4"}}},
               {"key6", "value6"}};

    tracker.insertConfig("localhost", 8080, j1, "1.0.0", "main", "abcdef1234567890",
                         "CMAKE_BUILD_TYPE=Release");

    // different port so no conflict
    tracker.insertConfig("localhost", 9090, j2, "1.0.0", "main", "abcdef1234567890",
                         "CMAKE_BUILD_TYPE=Release");

    // Check hashing
    BOOST_TEST_MESSAGE("Config hash from getTrackerHash: " << tracker.getTrackerHash());

    BOOST_CHECK_EQUAL(tracker.n_configs(), 2);
}

BOOST_AUTO_TEST_CASE(test_add_same_jsons) {

    auto& tracker = ConfigTracker::instance();

    // Some json
    json j1 = {{"key4", "value4"},
               {"key5", {{"subkey3", "subvalue3"}, {"subkey4", "subvalue4"}}},
               {"key6", "value6"}};

    // Another identical json with updatable config
    json j2 = {{"key4", "value4"},
               {"key5", {{"subkey3", "subvalue3"}, {"subkey4", "subvalue4"}}},
               {"key6", "value6"},
               {"updatable_config", {{"key7", "value7"}, {"key8", "value8"}}}};

    tracker.insertConfig("localhost", 8080, j1, "1.0.0", "main", "abcdef1234567890",
                         "CMAKE_BUILD_TYPE=Release");

    // same port, same config (updatable_config should be ignored!), so no conflict expected
    tracker.insertConfig("localhost", 8080, j2, "1.0.0", "main", "abcdef1234567890",
                         "CMAKE_BUILD_TYPE=Release");

    BOOST_CHECK_EQUAL(tracker.n_configs(), 1);
}

BOOST_AUTO_TEST_CASE(test_add_same_jsons_bad) {

    auto& tracker = ConfigTracker::instance();

    // Some json
    json j1 = {{"key1", "value1"},
               {"key2", {{"subkey1", "subvalue1"}, {"subkey2", "subvalue2"}}},
               {"key3", "value3"}};

    // Another json
    json j2 = {{"key4", "value4"},
               {"key5", {{"subkey3", "subvalue3"}, {"subkey4", "subvalue4"}}},
               {"key6", "value6"}};

    tracker.insertConfig("localhost", 8080, j1, "1.0.0", "main", "abcdef1234567890",
                         "CMAKE_BUILD_TYPE=Release");

    // same port, different config, so conflict *is* expected
    BOOST_CHECK_THROW(tracker.insertConfig("localhost", 8080, j2, "1.0.0", "main",
                                           "abcdef1234567890", "CMAKE_BUILD_TYPE=Release"),
                      std::runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()
