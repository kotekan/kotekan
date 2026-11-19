#define BOOST_TEST_MODULE "test_gnss_fleet_trim"

#include "Config.hpp"                // for Config
#include "GnssFleetTrim.hpp"         // for GnssFleetTrim
#include "buffer.hpp"                // for Buffer
#include "bufferContainer.hpp"       // for bufferContainer
#include "gnssRecord.hpp"            // for REC_PRN, CHAN_*
#include "gnssTelem.hpp"             // for TelemHeader, telem_frame_bytes, telem_rows
#include "kotekanLoggingFixture.hpp" // for kotekan_logging_fixture
#include "metadata.hpp"              // for metadataPool
#include "restClient.hpp"            // for restClient
#include "restServer.hpp"            // for restServer

#include "json.hpp" // for json

#include <boost/test/included/unit_test.hpp>
#include <chrono>             // for seconds, milliseconds, steady_clock
#include <cmath>              // for nan
#include <condition_variable> // for condition_variable
#include <cstdio>             // for remove
#include <cstring>            // for memcpy
#include <functional>         // for bind
#include <map>                // for map
#include <mutex>              // for mutex, lock_guard, unique_lock
#include <set>                // for set
#include <string>             // for string
#include <thread>             // for sleep_for
#include <unistd.h>           // for getpid
#include <vector>             // for vector

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

json get_stats(unsigned short port, const std::string& stage = STAGE,
               const std::string& verb = "/get_stats") {
    const auto reply =
        restClient::instance().make_request_blocking(stage + verb, {}, "127.0.0.1", port);
    BOOST_REQUIRE_MESSAGE(reply.first, verb + " failed: " + reply.second);
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

constexpr int BURST_PRN = 7;

/// One sender's frame for window `win` of `chain`, laid out as in test_gnss_fleet_dll: BURST_PRN
/// in every record, with E above L, so FleetDll steps an armed trim on every close.
std::vector<uint8_t> burst_frame(const std::string& chain, const std::string& inst, uint64_t win) {
    constexpr int n_rec = 4, n_prn = 2, n_chan = 2;
    std::vector<uint8_t> buf(gnss::telem_frame_bytes(n_rec, n_prn, n_chan), 0);
    auto* h = (gnss::TelemHeader*)buf.data();
    h->magic = gnss::TELEM_MAGIC;
    h->version = gnss::TELEM_VERSION;
    h->n_rec = n_rec;
    h->n_prn = n_prn;
    h->n_row = gnss::RECORD_FLOATS;
    h->n_chan = n_chan;
    h->n_elem = 1;
    h->hops_per_record = 2048;
    h->fft_len = 16384;
    h->win = win;
    h->seq = win;
    h->wstart0 = (int64_t)win * n_rec * h->hops_per_record * h->fft_len;
    h->present = 0xF;
    h->max_chan = n_chan;
    h->n_row_total = gnss::telem_row_floats(n_chan);
    gnss::telem_set_name(h->chain, chain);
    gnss::telem_set_name(h->inst, inst);
    for (int c = 0; c < n_chan; ++c)
        h->chan_id[c] = (uint16_t)(6000 + c);
    float* rows = gnss::telem_rows(buf.data());
    for (int r = 0; r < n_rec; ++r) {
        float* row = rows + gnss::telem_row_offset(r, 0, n_prn, gnss::telem_row_floats(n_chan));
        row[gnss::REC_PRN] = (float)BURST_PRN;
        for (int c = 0; c < n_chan; ++c) {
            float* cc = row + gnss::telem_chan_offset(c);
            cc[gnss::CHAN_RE] = 1.0f;
            cc[gnss::CHAN_ENERGY] = 1.0f;
            cc[gnss::CHAN_E_RE] = 1.2f;
            cc[gnss::CHAN_E_ENERGY] = 1.0f;
            cc[gnss::CHAN_L_RE] = 0.8f;
            cc[gnss::CHAN_L_ENERGY] = 1.0f;
        }
    }
    return buf;
}

/// Window `win` of `chain` from both senders (min_instances is 2). Its first frame closes the
/// chain's previous window. False if the buffer is shutting down.
bool send_window(Buffer& buf, int& id, const std::string& chain, uint64_t win) {
    for (const char* inst : {"s0.0", "s1.0"}) {
        const std::vector<uint8_t> f = burst_frame(chain, inst, win);
        uint8_t* dst = buf.wait_for_empty_frame("burst_sender", id);
        if (dst == nullptr)
            return false;
        std::memcpy(dst, f.data(), f.size());
        buf.mark_frame_full("burst_sender", id);
        id = (id + 1) % buf.num_frames;
    }
    return true;
}

/// The trackers' set_trim endpoints, served by this process's REST server. While `hold` is set,
/// a post of chain `first` blocks the server, and the poster waiting on its reply, until the
/// test clears it.
struct Trackers {
    std::mutex mtx;
    std::condition_variable cv;
    std::map<std::string, int> posts;   ///< chain -> set_trim requests taken
    std::map<std::string, double> trim; ///< chain -> BURST_PRN's trim_chips in its newest post
    std::string first;
    bool hold = false, held = false;

    void take(const std::string& chain, kotekan::connectionInstance& conn, json& request) {
        std::unique_lock<std::mutex> lk(mtx);
        posts[chain]++;
        double t = std::nan(""); // a post without BURST_PRN matches no trim
        for (const auto& r : request)
            if (r.value("prn", -1) == BURST_PRN)
                t = r.value("trim_chips", t);
        trim[chain] = t;
        if (hold && chain == first) {
            held = true;
            cv.notify_all();
            cv.wait(lk, [this] { return !hold; });
        }
        cv.notify_all();
        lk.unlock();
        conn.send_empty_reply(kotekan::HTTP_RESPONSE::OK);
    }

    bool each_posted(const std::map<std::string, int>& want) {
        for (const auto& w : want)
            if (posts[w.first] < w.second)
                return false;
        return true;
    }
};

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

// Every sender ships all its chains' frames back to back, so the chains' windows close in a burst
// while the poster is still sending the first chain's trims. Each chain's newest trims must
// still reach its tracker once per window, and none twice.
//
// On demand, not in CI: --run_test=chains_closed_while_the_poster_sends_are_each_posted
BOOST_AUTO_TEST_CASE(chains_closed_while_the_poster_sends_are_each_posted,
                     *boost::unit_test::disabled()) {
    const std::string stage = "/fleet_trim_burst";
    const unsigned short port = rest_port();
    BOOST_REQUIRE(port != 0);
    const std::vector<std::string> chains{"bds_b2a", "gal_e5a", "gal_e6", "gps_l5"};

    Trackers trk;
    trk.first = chains[0];
    for (const auto& c : chains)
        restServer::instance().register_post_callback(
            stage + "_trk/" + c + "/set_trim",
            std::bind(&Trackers::take, &trk, c, std::placeholders::_1, std::placeholders::_2));

    auto pool = metadataPool::create(2, 64, "burst_pool", "GnssChanMetadata");
    Buffer telem_buf(16, burst_frame(chains[0], "s0.0", 0).size(), pool, "burst_telem_buf",
                     "standard", 0, false, false, std::vector<int>{}, true);
    telem_buf.register_producer("burst_sender");
    bufferContainer bc;
    bc.add_buffer("telem_buf", &telem_buf);
    const std::string store =
        "/tmp/test_gnss_fleet_trim_burst_" + std::to_string(::getpid()) + ".json";
    json cfg;
    cfg["log_level"] = "warn";
    cfg["fleet_trim_burst"] = {{"kotekan_stage", "GnssFleetTrim"},
                               {"log_level", "warn"},
                               {"cpu_affinity", std::vector<int>{0}},
                               {"in_buf", "telem_buf"},
                               {"trim_state_file", store},
                               {"post_threads", 1}};
    Config config;
    config.update_config(cfg);
    GnssFleetTrim trim(config, stage, bc);
    // The posters exit on stop_thread, which the destructor does not set: stop on every exit,
    // or a failed check would hang the test.
    struct Stopper {
        GnssFleetTrim& s;
        ~Stopper() {
            s.stop();
        }
    } stopper{trim};
    for (const auto& c : chains) {
        json pol = policy_for(c, {BURST_PRN},
                              "http://127.0.0.1:" + std::to_string(port) + stage + "_trk/" + c
                                  + "/set_trim");
        pol["chains"][c]["p_floor_abs"] = 1e-30; // below every power here, so every close steps
        BOOST_CHECK(policy_taken(stage, port, pol));
    }
    trim.start();

    // Warm-up, one chain at a time, until every tracker has a kept-alive connection and a post.
    int id = 0;
    uint64_t win = 5000;
    std::map<std::string, int> want;
    for (const auto& c : chains)
        want[c] = 1;
    bool ok = true, warm = false;
    for (int i = 0; i < 40 && ok && !warm; ++i, ++win) {
        for (const auto& c : chains) {
            ok = ok && send_window(telem_buf, id, c, win);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        std::lock_guard<std::mutex> lk(trk.mtx);
        warm = trk.each_posted(want);
    }
    BOOST_CHECK_MESSAGE(warm, "a tracker got no trims with its chain's windows closing alone");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const json st0 = get_stats(port, stage);
    {
        std::lock_guard<std::mutex> lk(trk.mtx);
        want = trk.posts;
    }
    const std::map<std::string, int> base = want;

    // Bursts: the first chain's post is held at its tracker while the other chains close.
    constexpr int n_burst = 5;
    std::map<std::string, int> stale; // chain -> bursts whose newest post was not FleetDll's trim
    std::map<std::string, json> stale_last; // chain -> [posted, held] in its last such burst
    for (int k = 0; k < n_burst && ok && warm; ++k, ++win) {
        {
            std::lock_guard<std::mutex> lk(trk.mtx);
            trk.hold = true;
            trk.held = false;
        }
        ok = send_window(telem_buf, id, chains[0], win);
        {
            std::unique_lock<std::mutex> lk(trk.mtx);
            ok = ok && trk.cv.wait_for(lk, std::chrono::seconds(2), [&] { return trk.held; });
        }
        for (size_t i = 1; i < chains.size(); ++i)
            ok = ok && send_window(telem_buf, id, chains[i], win);
        std::this_thread::sleep_for(std::chrono::milliseconds(30)); // the fold closes them
        {
            std::lock_guard<std::mutex> lk(trk.mtx);
            trk.hold = false;
        }
        trk.cv.notify_all();
        for (auto& w : want)
            w.second++;
        std::map<std::string, double> sent;
        {
            std::unique_lock<std::mutex> lk(trk.mtx);
            trk.cv.wait_for(lk, std::chrono::seconds(1), [&] { return trk.each_posted(want); });
            sent = trk.trim;
        }
        // No window closes until the next burst, so each chain's newest post must carry the
        // trim FleetDll holds now: the trims step on every close, so an older payload differs.
        const json dll = get_stats(port, stage, "/get_dll");
        for (const auto& c : chains) {
            const double now =
                dll.at(c).at(std::to_string(BURST_PRN)).at("trim_chips").get<double>();
            if (sent[c] != now) {
                stale[c]++;
                stale_last[c] = {sent[c], now};
            }
        }
    }
    BOOST_CHECK_MESSAGE(ok,
                        "a burst did not run: the first chain's post never reached its tracker");
    std::this_thread::sleep_for(std::chrono::milliseconds(200)); // time for a stray repeat

    const json st1 = get_stats(port, stage);
    for (const auto& c : chains) {
        const int closed = st1.at("chains").at(c).at("windows_closed").get<int>()
                           - st0.at("chains").at(c).at("windows_closed").get<int>();
        std::lock_guard<std::mutex> lk(trk.mtx);
        const int posted = trk.posts[c] - base.at(c);
        BOOST_CHECK_MESSAGE(posted == closed && closed == n_burst,
                            c + ": " + std::to_string(posted) + " posts for "
                                + std::to_string(closed) + " window closes");
        BOOST_CHECK_MESSAGE(stale[c] == 0, c + ": " + std::to_string(stale[c])
                                               + " bursts posted a trim FleetDll no longer held, "
                                               + "[posted, held] " + stale_last[c].dump());
    }

    trim.stop();
    trim.join();
    for (const auto& c : chains)
        restServer::instance().remove_json_callback(stage + "_trk/" + c + "/set_trim");
    std::remove(store.c_str());
}
