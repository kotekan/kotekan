#define BOOST_TEST_MODULE "test_gnss_fleet_trim"

#include "Config.hpp"                // for Config
#include "GnssFleetTrim.hpp"         // for GnssFleetTrim
#include "buffer.hpp"                // for Buffer
#include "bufferContainer.hpp"       // for bufferContainer
#include "kotekanLoggingFixture.hpp" // for kotekan_logging_fixture
#include "metadata.hpp"              // for metadataPool
#include "restClient.hpp"            // for restClient
#include "restServer.hpp"            // for restServer

#include "json.hpp" // for json

#include <boost/test/included/unit_test.hpp>
#include <chrono> // for seconds, milliseconds, steady_clock
#include <set>    // for set
#include <string> // for string
#include <thread> // for sleep_for
#include <vector> // for vector

using json = nlohmann::json;
using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::restServer;

BOOST_GLOBAL_FIXTURE(kotekan_logging_fixture);

namespace {

const std::string STAGE = "/fleet_trim";
constexpr double TTL_S = 5.0; // the smallest policy_ttl_s the stage accepts

json policy_for(const std::string& chain, const std::vector<int>& armed,
                const std::string& target) {
    return {{"chains",
             {{chain,
               {{"armed", armed},
                {"gain_per_s", 2.5},
                {"leak_per_s", 0.5},
                {"clamp", 3.0},
                {"spacing", 0.5},
                {"targets", {target}}}}}}};
}

void post_policy(unsigned short port, const json& payload) {
    const auto reply = restClient::instance().make_request_blocking(STAGE + "/set_policy", payload,
                                                                    "127.0.0.1", port);
    BOOST_REQUIRE_MESSAGE(reply.first, "set_policy failed: " + reply.second);
}

bool policy_taken(const std::string& stage, unsigned short port, const json& payload) {
    return restClient::instance()
        .make_request_blocking(stage + "/set_policy", payload, "127.0.0.1", port)
        .first;
}

json get_stats(unsigned short port, const std::string& stage = STAGE) {
    const auto reply =
        restClient::instance().make_request_blocking(stage + "/get_stats", {}, "127.0.0.1", port);
    BOOST_REQUIRE_MESSAGE(reply.first, "get_stats failed: " + reply.second);
    return json::parse(reply.second);
}

std::set<int> armed_in_fleet_dll(const json& stats, const std::string& chain) {
    return stats.at("policy").at(chain).at("armed").get<std::set<int>>();
}

unsigned short rest_port() {
    if (restServer::instance().port() == 0)
        restServer::instance().start("127.0.0.1", 0);
    return restServer::instance().port();
}

} // namespace

// A chain that stops posting /set_policy for policy_ttl_s is disarmed in FleetDll, so its trims
// leak away, and its post targets are dropped; a chain that keeps posting is untouched. No
// frames flow, so the other chain's posts are what run the sweep.
BOOST_AUTO_TEST_CASE(silent_chain_is_disarmed_and_its_targets_dropped) {
    const unsigned short port = rest_port();
    BOOST_REQUIRE(port != 0);

    auto pool = metadataPool::create(2, 64, "test_pool", "GnssChanMetadata");
    Buffer telem_buf(2, 4096, pool, "telem_buf", "standard", 0, false, false, std::vector<int>{},
                     true);
    bufferContainer bc;
    bc.add_buffer("telem_buf", &telem_buf);

    json cfg;
    cfg["log_level"] = "warn";
    cfg["fleet_trim"] = {{"kotekan_stage", "GnssFleetTrim"},
                         {"log_level", "warn"},
                         {"cpu_affinity", std::vector<int>{}},
                         {"in_buf", "telem_buf"},
                         {"trim_state_file", "/tmp/test_gnss_fleet_trim_unused.json"},
                         {"policy_ttl_s", TTL_S},
                         {"post_threads", 1}};
    Config config;
    config.update_config(cfg);

    GnssFleetTrim stage(config, STAGE, bc);

    post_policy(port, policy_for("gal_e5a", {6, 12}, "http://127.0.0.1:9/gal_e5a/set_trim"));
    post_policy(port, policy_for("gps_l5", {1}, "http://127.0.0.1:9/gps_l5/set_trim"));
    json st = get_stats(port);
    BOOST_CHECK_EQUAL(st.at("post_targets").get<int>(), 2);
    BOOST_CHECK(armed_in_fleet_dll(st, "gal_e5a") == (std::set<int>{6, 12}));
    const double leak = st.at("policy").at("gal_e5a").at("leak_per_update").get<double>();

    // gps_l5 keeps posting past gal_e5a's TTL.
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - t0 < std::chrono::duration<double>(TTL_S + 1.0)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        post_policy(port, policy_for("gps_l5", {1}, "http://127.0.0.1:9/gps_l5/set_trim"));
    }

    st = get_stats(port);
    BOOST_CHECK_EQUAL(st.at("policy_expired").get<int>(), 1);
    BOOST_CHECK(!st.at("policy_armed_requested").contains("gal_e5a"));
    BOOST_CHECK(armed_in_fleet_dll(st, "gal_e5a").empty());
    // Disarmed under its last loop constants, so its trims leak at the rate the broker set.
    BOOST_CHECK_EQUAL(st.at("policy").at("gal_e5a").at("leak_per_update").get<double>(), leak);
    BOOST_CHECK(armed_in_fleet_dll(st, "gps_l5") == (std::set<int>{1}));
    BOOST_CHECK_EQUAL(st.at("post_targets").get<int>(), 1);

    // The posters exit on stop_thread, which the destructor does not set.
    stage.stop();
}

// A GPU half's target, {"url", "chain"}, is taken beside a tracker's own url. A policy is refused
// whole, and changes nothing, when a target names another chain, is malformed, or names a url
// another target names in the other form.
BOOST_AUTO_TEST_CASE(half_targets_are_taken_and_mixed_forms_refused) {
    const std::string stage = "/fleet_trim_half";
    const unsigned short port = rest_port();
    BOOST_REQUIRE(port != 0);

    auto pool = metadataPool::create(2, 64, "half_pool", "GnssChanMetadata");
    Buffer telem_buf(2, 4096, pool, "half_telem_buf", "standard", 0, false, false,
                     std::vector<int>{}, true);
    bufferContainer bc;
    bc.add_buffer("telem_buf", &telem_buf);
    json cfg;
    cfg["log_level"] = "warn";
    cfg["fleet_trim_half"] = {{"kotekan_stage", "GnssFleetTrim"},
                              {"log_level", "warn"},
                              {"cpu_affinity", std::vector<int>{}},
                              {"in_buf", "telem_buf"},
                              {"trim_state_file", "/tmp/test_gnss_fleet_trim_unused.json"},
                              {"post_threads", 1}};
    Config config;
    config.update_config(cfg);
    GnssFleetTrim trim(config, stage, bc);

    const std::string half = "http://127.0.0.1:9/gnss0_inject/set_trim";
    json pol = policy_for("gps_l5", {1}, "http://127.0.0.1:9/gnss0_gps_l5_inject/set_trim");
    pol["chains"]["gps_l5"]["targets"].push_back({{"url", half}, {"chain", "gps_l5"}});
    BOOST_CHECK(policy_taken(stage, port, pol));
    pol = policy_for("gal_e5a", {6}, "http://127.0.0.1:9/gnss1_gal_e5a_inject/set_trim");
    pol["chains"]["gal_e5a"]["targets"].push_back({{"url", half}, {"chain", "gal_e5a"}});
    BOOST_CHECK(policy_taken(stage, port, pol));
    BOOST_CHECK_EQUAL(get_stats(port, stage).at("post_targets").get<int>(), 4);

    for (const json& bad : {json{{"url", half}, {"chain", "gps_l5"}}, json{{"url", half}},
                            json{{"url", half}, {"chain", "bds_b2a"}, {"x", 1}}, json(7)}) {
        pol = policy_for("bds_b2a", {2}, "http://127.0.0.1:9/gnss0_bds_b2a_inject/set_trim");
        pol["chains"]["bds_b2a"]["targets"].push_back(bad);
        BOOST_CHECK_MESSAGE(!policy_taken(stage, port, pol), "taken: " + bad.dump());
    }
    BOOST_CHECK(!policy_taken(stage, port, policy_for("bds_b2a", {2}, half)));
    const json st = get_stats(port, stage);
    BOOST_CHECK_EQUAL(st.at("post_targets").get<int>(), 4);
    BOOST_CHECK(!st.at("policy_armed_requested").contains("bds_b2a"));

    trim.stop();
}
