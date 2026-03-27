#define BOOST_TEST_MODULE "test_Telescope"

#include "CHORDTelescope.hpp"
#include "Config.hpp"
#include "Telescope.hpp"
#include "configUpdater.hpp"
#include "errors.h" // _global_log_level
#include "restServer.hpp"
#include "test_logging.hpp"
#include "timeUtil.hpp"

#include "fmt.hpp"
#include "json.hpp"

#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/output_test_stream.hpp>
#include <complex>
#include <filesystem>
#include <inttypes.h>
#include <iostream>
#include <sys/wait.h>
#include <time.h>
#include <vector>

using kotekan::Config;
using json = nlohmann::json;
using kotekan::configUpdater;
using kotekan::restServer;
using namespace std::complex_literals;

const std::string default_tel_config_str = R"tel_config_str({
"type": "config",
"log_level": "debug",
"telescope": {
    "name": "TestTelescope",
    "require_eop": false,
    "eop_updatable_config": ""
    }
})tel_config_str";

// Just a hollow Telescope for testing
class TestTelescope : public Telescope {
public:
    TestTelescope(const Config& config, const std::string& path) :
        Telescope(path, config.get<std::string>(path, "log_level"),
                  config.get<bool>(path, "require_eop"),
                  config.get<std::string>(path, "eop_updatable_config")) {}

    // These functions don't have to do anything, they're not being tested.
    freq_id_t to_freq_id([[maybe_unused]] stream_t stream,
                         [[maybe_unused]] uint32_t ind) const override {
        return 0;
    }
    double to_freq_MHz([[maybe_unused]] freq_id_t freq_id) const override {
        return 0.0;
    }
    size_t num_freq_per_stream() const override {
        return 0;
    }
    size_t num_freq() const override {
        return 0;
    }
    double freq_width_MHz([[maybe_unused]] freq_id_t freq_id) const override {
        return 0.0;
    };
    uint8_t nyquist_zone() const override {
        return 0;
    }
    bool gps_time_enabled() const override {
        return false;
    }
    timespec to_time([[maybe_unused]] uint64_t seq) const override {
        return {.tv_sec = 0, .tv_nsec = 0};
    };
    int64_t to_time_ns([[maybe_unused]] uint64_t seq) const override {
        return 0;
    }
    uint64_t to_seq([[maybe_unused]] timespec time) const override {
        return 0;
    }
    uint64_t seq_length_nsec() const override {
        return 0;
    }
};

REGISTER_TELESCOPE(TestTelescope, "TestTelescope");

/******************
 *
 * Fixtures: Logging, RestServer, and Telescope(s)
 *
 ******************/


struct LoggingFixture {
    LoggingFixture() {
        kotekan_test_logging::configure();
    }
};

struct RestServerFixture {
    RestServerFixture() {
        try {
            kotekan::restServer::instance().start("127.0.0.1", 0);
        } catch (...) {
        }
    }
};

struct TelescopeFixture {
    TelescopeFixture() {
        json json_config = json::parse(default_tel_config_str);
        Config conf;
        conf.update_config(json_config);
        configUpdater::instance().apply_config(conf);
        Telescope::instance(conf);
    }
};

struct TelescopeEOPFixture {
    TelescopeEOPFixture() {
        json json_config = json::parse(default_tel_config_str);

        N = 5;

        int64_t giga = 1'000'000'000;
        t = {1, 0, giga, giga * giga, giga * giga + 100'000 * giga};
        dut1 = {0.0, -1.5, 9.8765e-3, 18, 0.123456};
        x = {0.0, 0.1, -1.0, -987, 1.234e-5};
        y = {0.0, 0.2, -0.5, 2.345e-5, 123};

        BOOST_REQUIRE(t.size() == N);
        BOOST_REQUIRE(dut1.size() == N);
        BOOST_REQUIRE(x.size() == N);
        BOOST_REQUIRE(y.size() == N);

        for (size_t i = 0; i < N; i++) {
            BareEOP beop = {
                .t_inst_ns = t[i], .delta_UT1_inst = dut1[i], .xp_as = x[i], .yp_as = y[i]};
            fix_bare_eop_table.push_back(beop);
            fix_eop_table.push_back(beop.to_EOP());
        }

        json_config["telescope"]["require_eop"] = true;
        json_config["telescope"]["eop_updatable_config"] = "/eop_update";
        json_config["eop_update"] = {{"kotekan_update_endpoint", "json"},
                                     {"earth_orientation_parameter_table", fix_bare_eop_table}};

        Config conf;
        conf.update_config(json_config);
        configUpdater::instance().apply_config(conf);
        Telescope::instance(conf);
    }

    size_t N;
    std::vector<int64_t> t;
    std::vector<double> dut1;
    std::vector<double> x;
    std::vector<double> y;

    std::vector<EOP> fix_eop_table;
    std::vector<BareEOP> fix_bare_eop_table;
};

BOOST_TEST_GLOBAL_FIXTURE(LoggingFixture);
BOOST_TEST_GLOBAL_FIXTURE(RestServerFixture);

/******************
 *
 * TESTS
 *
 ******************/


BOOST_FIXTURE_TEST_CASE(_name_tel, TelescopeFixture) {
    const Telescope& tel = Telescope::instance();

    BOOST_CHECK_EQUAL(tel.get_name(), "TestTelescope");
}

BOOST_FIXTURE_TEST_CASE(_get_eop_table, TelescopeEOPFixture) {
    const Telescope& tel = Telescope::instance();

    std::vector<EOP> eop_table = tel.get_current_EOP_table();

    // Check table is right size
    BOOST_CHECK_EQUAL(eop_table.size(), N);

    // Check table is ordered in instrument time
    for (size_t i = 0; i < N - 1; i++)
        BOOST_CHECK_LT(eop_table[i].t_inst_ns, eop_table[i + 1].t_inst_ns);

    // Check table is also ordered in UT1
    for (size_t i = 0; i < N - 1; i++)
        BOOST_CHECK_LT(eop_table[i].t_ut1_ns, eop_table[i + 1].t_ut1_ns);

    // Check table entries are identical to what we sent, they might not be in same order though!
    std::set<size_t> found_indices;
    for (size_t i = 0; i < N; i++) {
        const auto it = std::find(fix_eop_table.begin(), fix_eop_table.end(), eop_table[i]);
        BOOST_CHECK_MESSAGE(it != fix_eop_table.end(),
                            fmt::format("EOP entry {:d} {} not found", i, eop_table[i]));

        size_t index = std::distance(fix_eop_table.begin(), it);

        BOOST_CHECK_MESSAGE(found_indices.count(index) == 0,
                            fmt::format("EOP entry {:d} {} occured again!", i, eop_table[i]));
        found_indices.insert(index);
    }
}
