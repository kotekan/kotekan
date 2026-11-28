// BeiDou-3S (M2-S, C58) codes: the generators against the chips recovered from the sky, and the
// replica bank's per-PRN selection of them.
#define BOOST_TEST_MODULE "test_gnss_bds3s"

#include "Config.hpp"
#include "beidou3sCode.hpp"
#include "beidouB1ICode.hpp"
#include "beidouB2aCode.hpp"
#include "beidouB3ICode.hpp"
#include "gnssChannelizedReplica.hpp"
#include "gnssSignal.hpp"

#include "json.hpp"

#include <algorithm>
#include <boost/test/included/unit_test.hpp>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using nlohmann::json;

namespace {

std::vector<int8_t> read_fixture(const std::string& name, size_t n) {
    std::ifstream f(std::string(TEST_DATA_DIR) + "/bds3s_m2s/" + name, std::ios::binary);
    BOOST_REQUIRE_MESSAGE(f.good(), "missing fixture " << name);
    std::vector<int8_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)n);
    BOOST_REQUIRE_EQUAL((size_t)f.gcount(), n);
    return v;
}

// Sum of a[k] * b[k] over the chips of one parity (2k + phase).
long parity_dot(const std::vector<int8_t>& a, const std::vector<int8_t>& b, int phase) {
    long s = 0;
    for (size_t k = (size_t)phase; k < a.size(); k += 2)
        s += a[k] * b[k];
    return s;
}

template<typename A>
std::vector<int8_t> vec(const A& a) {
    return std::vector<int8_t>(a.begin(), a.end());
}

std::vector<int8_t> tile(const std::vector<int8_t>& a, const std::vector<int8_t>& s) {
    std::vector<int8_t> t;
    for (int8_t sk : s)
        for (int8_t c : a)
            t.push_back((int8_t)(c * sk));
    return t;
}

std::vector<int8_t> bank_code(const char* signal, std::vector<int> prns, std::vector<int> bds3s,
                              int p) {
    const gnss::SignalDescriptor* sig = gnss::signal_by_name(signal);
    BOOST_REQUIRE(sig != nullptr);
    gnss::ChannelizedReplicaBank bank(*sig, 40.0e6, 10.0e6, 10, 4, dsp::Window::Hamming, prns,
                                      bds3s);
    return bank.full_code(p);
}

} // namespace

// Each component matches the sky chips at its parity, up to sign, and is zero at the other.
BOOST_AUTO_TEST_CASE(generators_match_the_sky_chips) {
    const auto b2a = read_fixture("b2a_sky.i8", 10230);
    const auto b2b = read_fixture("b2b_sky.i8", 10230);
    const auto ad = vec(beidou::generate_b2ad_3s_code(58));
    const auto ap = vec(beidou::generate_b2ap_3s_code(58));
    const auto bp = vec(beidou::generate_b2bp_3s_code(58));
    BOOST_CHECK_EQUAL(std::labs(parity_dot(ad, b2a, 0)), 5115);
    BOOST_CHECK_EQUAL(std::labs(parity_dot(ap, b2a, 1)), 5115);
    BOOST_CHECK_EQUAL(std::labs(parity_dot(bp, b2b, 1)), 5115);
    for (size_t k = 0; k < 10230; k += 2) {
        BOOST_REQUIRE_EQUAL(ap[k], 0);
        BOOST_REQUIRE_EQUAL(bp[k], 0);
        BOOST_REQUIRE_EQUAL(ad[k + 1], 0);
    }
    BOOST_CHECK_THROW(beidou::generate_b2ap_3s_code(57), std::out_of_range);
}

// M2-S's B3I is the ICD code with G1 left free-running: the same first 8190 chips, a different
// tail.
BOOST_AUTO_TEST_CASE(b3i_g1_runs_free) {
    const auto sky = read_fixture("b3i_sky.i8", 10230);
    const auto free_g1 = vec(beidou::generate_b3i_code(58, false));
    const auto icd = vec(beidou::generate_b3i_code(58));
    long d_free = 0, d_head = 0, d_tail = 0;
    for (size_t k = 0; k < 10230; ++k) {
        d_free += free_g1[k] * sky[k];
        (k < 8190 ? d_head : d_tail) += icd[k] * free_g1[k];
    }
    BOOST_CHECK_EQUAL(std::labs(d_free), 10230);
    BOOST_CHECK_EQUAL(d_head, 8190);
    BOOST_CHECK(std::labs(d_tail) < 200);
}

// The B2a pilot overlay is the ICD secondary of PRN 58 at some cyclic phase.
BOOST_AUTO_TEST_CASE(b2a_pilot_overlay_is_icd_prn58) {
    const auto sky = read_fixture("b2a_pilot_secondary_sky.i8", 100);
    const auto icd = beidou::b2ap_secondary(58);
    int best = 0;
    for (int s = 0; s < 100; ++s) {
        int d = 0;
        for (int k = 0; k < 100; ++k)
            d += sky[(size_t)k] * icd[(size_t)((k + s) % 100)];
        best = std::max(best, std::abs(d));
    }
    BOOST_CHECK_EQUAL(best, 100);
}

// A listed PRN gets its BeiDou-3S code on every signal that has one, with the ICD overlays, at
// construction and after set_prn; the other PRNs and other signals are unchanged.
BOOST_AUTO_TEST_CASE(bank_selects_bds3s_codes_for_listed_prns) {
    const auto ap = vec(beidou::generate_b2ap_3s_code(58));
    const auto ad = vec(beidou::generate_b2ad_3s_code(58));
    const std::vector<std::pair<const char*, std::vector<int8_t>>> cases = {
        {"BDS_B2A_P", ap},
        {"BDS_B2A_P_CS", tile(ap, vec(beidou::b2ap_secondary(58)))},
        {"BDS_B2A_D", ad},
        {"BDS_B2A_D_CS", tile(ad, vec(beidou::b2ad_secondary()))},
        {"BDS_B2B_I", vec(beidou::generate_b2bp_3s_code(58))},
        {"BDS_B3I", vec(beidou::generate_b3i_code(58, false))},
        {"BDS_B3I_NH", tile(vec(beidou::generate_b3i_code(58, false)), vec(beidou::B3I_NH20))},
    };
    for (const auto& c : cases) {
        BOOST_TEST_CONTEXT(c.first) {
            BOOST_CHECK(bank_code(c.first, {57, 58}, {58}, 1) == c.second);
            BOOST_CHECK(bank_code(c.first, {57, 58}, {}, 1) != c.second);
            BOOST_CHECK(bank_code(c.first, {57, 58}, {58}, 0) == bank_code(c.first, {57}, {}, 0));

            const gnss::SignalDescriptor* sig = gnss::signal_by_name(c.first);
            gnss::ChannelizedReplicaBank bank(*sig, 40.0e6, 10.0e6, 10, 4, dsp::Window::Hamming,
                                              {57}, {58});
            BOOST_REQUIRE(bank.set_prn(0, 58));
            BOOST_CHECK(bank.full_code(0) == c.second);
        }
    }
    BOOST_CHECK(bank_code("BDS_B1I", {58}, {58}, 0) == vec(beidou::generate_b1i_code(58)));
}

// The stages read bds3s_prns with get_default from their own path, so one root-level key reaches
// every stage, nested or not.
BOOST_AUTO_TEST_CASE(stages_inherit_the_root_key) {
    json cfg;
    cfg["log_level"] = "warn";
    cfg["bds3s_prns"] = {58};
    cfg["b2a_search"] = {{"kotekan_stage", "GnssChannelizedSearch"}};
    cfg["gpu"] = {{"b2a_track", {{"data", {{"signal", "BDS_B2A_D_CS"}}}}}};
    kotekan::Config config;
    config.update_config(cfg);
    const std::vector<int> want = {58};
    BOOST_CHECK(config.get_default<std::vector<int>>("/b2a_search", "bds3s_prns", {}) == want);
    BOOST_CHECK(config.get_default<std::vector<int>>("/gpu/b2a_track/data", "bds3s_prns", {})
                == want);

    cfg.erase("bds3s_prns");
    kotekan::Config bare;
    bare.update_config(cfg);
    BOOST_CHECK(bare.get_default<std::vector<int>>("/b2a_search", "bds3s_prns", {}).empty());
}
