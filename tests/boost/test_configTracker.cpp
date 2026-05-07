#define BOOST_TEST_MODULE "test_configTracker"

#include "configTracker.hpp" // for ConfigTracker
#include "restServer.hpp"    // for restServer

#include "json.hpp" // for json_ref, basic_json<>::object_t, json

#include <boost/filesystem.hpp>
#include <boost/test/included/unit_test.hpp>
#include <fstream>
#include <stdexcept>
#include <string>

using namespace kotekan;
using json = nlohmann::json;

// Wrapper structure for ConfigTracker to ensure it is reset/cleared after each test.
struct ConfigFixture {
    ConfigFixture() {
        ConfigTracker::instance().reset();
    }
    ~ConfigFixture() {
        ConfigTracker::instance().reset();
    }
};

namespace {
const json kSampleJson1 = {{"key1", "value1"},
                           {"key2", {{"subkey1", "subvalue1"}, {"subkey2", "subvalue2"}}},
                           {"key3", "value3"}};
const json kSampleJson2 = {{"key4", "value4"},
                           {"key5", {{"subkey3", "subvalue3"}, {"subkey4", "subvalue4"}}},
                           {"key6", "value6"}};
constexpr const char* kVersion = "1.0.0";
constexpr const char* kBranch = "main";
constexpr const char* kCommit = "abcdef1234567890";
constexpr const char* kCmake = "CMAKE_BUILD_TYPE=Release";
} // namespace

BOOST_FIXTURE_TEST_SUITE(ConfigTrackerTests, ConfigFixture)

BOOST_AUTO_TEST_CASE(set_local_config) {
    auto& tracker = ConfigTracker::instance();

    tracker.setLocalConfig(kSampleJson1, kVersion, kBranch, kCommit, kCmake);

    BOOST_CHECK(tracker.hasLocalConfig());
    BOOST_CHECK_EQUAL(tracker.n_configs(), 1u);

    const auto local = tracker.getLocalConfigInfo();
    BOOST_REQUIRE(local.has_value());
    BOOST_CHECK_EQUAL(local->config, kSampleJson1);
    BOOST_CHECK_EQUAL(local->kotekan_version, kVersion);

    // Local hash is the JSON-only md5 (no host:port prefix), so it equals
    // getLocalConfigHash() and is 32 hex chars.
    BOOST_CHECK_EQUAL(tracker.getLocalConfigHash(), local->json_hash);
    BOOST_CHECK_EQUAL(local->json_hash.size(), 32u);

    // Tracker hash is non-empty and printed for the curious.
    BOOST_TEST_MESSAGE("Local hash: " << local->json_hash);
    BOOST_TEST_MESSAGE("Tracker hash: " << tracker.getTrackerHash());
    BOOST_CHECK_EQUAL(tracker.getTrackerHash().size(), 32u);
}

BOOST_AUTO_TEST_CASE(local_and_upstream_coexist) {
    auto& tracker = ConfigTracker::instance();

    tracker.setLocalConfig(kSampleJson1, kVersion, kBranch, kCommit, kCmake);
    tracker.insertUpstreamConfig("10.0.0.1", 9090, kSampleJson2, kVersion, kBranch, kCommit,
                                 kCmake);

    BOOST_CHECK_EQUAL(tracker.n_configs(), 2u);
    BOOST_CHECK(tracker.hasLocalConfig());
    BOOST_CHECK(tracker.hasUpstreamConfig("10.0.0.1", 9090));
    BOOST_CHECK(!tracker.hasUpstreamConfig("10.0.0.2", 9090));

    // The upstream and local are independent slots: even if their content
    // were identical, they wouldn't collide.
    tracker.reset();
    tracker.setLocalConfig(kSampleJson1, kVersion, kBranch, kCommit, kCmake);
    tracker.insertUpstreamConfig("10.0.0.1", 9090, kSampleJson1, kVersion, kBranch, kCommit,
                                 kCmake);
    BOOST_CHECK_EQUAL(tracker.n_configs(), 2u);
}

BOOST_AUTO_TEST_CASE(set_local_config_idempotent) {
    auto& tracker = ConfigTracker::instance();

    // Same content (modulo a kotekan_update_endpoint block, which is stripped
    // before hashing) -> second call is a no-op.
    json j_with_endpoint = kSampleJson1;
    j_with_endpoint["block"] = {{"kotekan_update_endpoint", "value"}, {"key8", "value8"}};

    tracker.setLocalConfig(kSampleJson1, kVersion, kBranch, kCommit, kCmake);
    tracker.setLocalConfig(j_with_endpoint, kVersion, kBranch, kCommit, kCmake);

    BOOST_CHECK_EQUAL(tracker.n_configs(), 1u);
}

BOOST_AUTO_TEST_CASE(set_local_config_conflict_throws) {
    auto& tracker = ConfigTracker::instance();

    tracker.setLocalConfig(kSampleJson1, kVersion, kBranch, kCommit, kCmake);
    BOOST_CHECK_THROW(tracker.setLocalConfig(kSampleJson2, kVersion, kBranch, kCommit, kCmake),
                      std::runtime_error);
}

BOOST_AUTO_TEST_CASE(insert_upstream_idempotent) {
    auto& tracker = ConfigTracker::instance();

    json j_with_endpoint = kSampleJson1;
    j_with_endpoint["block"] = {{"kotekan_update_endpoint", "value"}, {"key8", "value8"}};

    tracker.insertUpstreamConfig("127.0.0.1", 8080, kSampleJson1, kVersion, kBranch, kCommit,
                                 kCmake);
    tracker.insertUpstreamConfig("127.0.0.1", 8080, j_with_endpoint, kVersion, kBranch, kCommit,
                                 kCmake);

    BOOST_CHECK_EQUAL(tracker.n_configs(), 1u);
    BOOST_CHECK(tracker.hasUpstreamConfig("127.0.0.1", 8080));
}

BOOST_AUTO_TEST_CASE(insert_upstream_conflict_throws) {
    auto& tracker = ConfigTracker::instance();

    tracker.insertUpstreamConfig("127.0.0.1", 8080, kSampleJson1, kVersion, kBranch, kCommit,
                                 kCmake);
    BOOST_CHECK_THROW(tracker.insertUpstreamConfig("127.0.0.1", 8080, kSampleJson2, kVersion,
                                                   kBranch, kCommit, kCmake),
                      std::runtime_error);
}

BOOST_AUTO_TEST_CASE(insert_upstream_distinct_ports) {
    auto& tracker = ConfigTracker::instance();

    tracker.insertUpstreamConfig("127.0.0.1", 8080, kSampleJson1, kVersion, kBranch, kCommit,
                                 kCmake);
    tracker.insertUpstreamConfig("127.0.0.1", 9090, kSampleJson2, kVersion, kBranch, kCommit,
                                 kCmake);

    BOOST_CHECK_EQUAL(tracker.n_configs(), 2u);
    BOOST_CHECK(tracker.hasUpstreamConfig("127.0.0.1", 8080));
    BOOST_CHECK(tracker.hasUpstreamConfig("127.0.0.1", 9090));
}

BOOST_AUTO_TEST_CASE(upstream_hash_lookup) {
    auto& tracker = ConfigTracker::instance();

    tracker.insertUpstreamConfig("127.0.0.1", 8080, kSampleJson1, kVersion, kBranch, kCommit,
                                 kCmake);

    // Two upstream entries with identical JSON but different (host, port)
    // hash to distinct values, so both are findable by hash.
    tracker.insertUpstreamConfig("127.0.0.1", 9090, kSampleJson1, kVersion, kBranch, kCommit,
                                 kCmake);

    BOOST_CHECK_EQUAL(tracker.n_configs(), 2u);
}

BOOST_AUTO_TEST_CASE(write_configs) {
    auto& tracker = ConfigTracker::instance();

    tracker.setLocalConfig(kSampleJson1, kVersion, kBranch, kCommit, kCmake);
    tracker.insertUpstreamConfig("127.0.0.1", 8080, kSampleJson2, kVersion, kBranch, kCommit,
                                 kCmake);

    boost::filesystem::path temp_dir = boost::filesystem::temp_directory_path();
    boost::filesystem::path local_path = temp_dir / "local.json";
    boost::filesystem::path upstream_path = temp_dir / "127.0.0.1_8080.json";

    // Pre-clean in case a previous test run left files behind.
    boost::filesystem::remove(local_path);
    boost::filesystem::remove(upstream_path);

    size_t n = tracker.writeConfigsToDisk(temp_dir.string());
    BOOST_CHECK_EQUAL(n, 2u);
    BOOST_CHECK(boost::filesystem::exists(local_path));
    BOOST_CHECK(boost::filesystem::exists(upstream_path));

    {
        std::ifstream f(local_path.string());
        std::string line;
        std::getline(f, line);
        BOOST_TEST(!line.empty());
    }

    boost::filesystem::remove(local_path);
    boost::filesystem::remove(upstream_path);
}

BOOST_AUTO_TEST_SUITE_END()
