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

#define ERA_TOL 2e-12 // <~ 0.5 nanoseconds of rotation in degrees
#define NS_TOL 0
#define DUT1_TOL 1.0e-10 // 0.5 ns
#define XP_TOL 7.5e-9    // 0.5 nanoseconds of rotation in arcseconds

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
    "eop_updatable_config": "",
    "inst_lat": 50.123,
    "inst_long": -120.456
    }
})tel_config_str";

// Just a hollow Telescope for testing
class TestTelescope : public Telescope {
public:
    TestTelescope(const Config& config, const std::string& path) :
        Telescope(path, config.get<std::string>(path, "log_level"),
                  config.get<bool>(path, "require_eop"),
                  config.get<std::string>(path, "eop_updatable_config"),
                  GeoFrame(config.get<std::string>(path, "log_level"), "grid",
                           config.get_default<double>(path, "inst_lat", 0.0),
                           config.get_default<double>(path, "inst_long", 0.0), {0, 0, 0}, {1, 0, 0},
                           {0, 1, 0}, {0, 0, 1}),
                  20.0, 10.0) {}

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
    station_id_t element_index_to_station_id(uint64_t el_idx, [[maybe_unused]] ElementOrder ord) const override {
        return el_idx;
    }
    uint64_t station_id_to_element_index(station_id_t st_id, [[maybe_unused]] ElementOrder ord) const override {
        return st_id;
    }
    grid_idx_2d_t station_id_to_grid_indices([[maybe_unused]] station_id_t st_id) const override {
        return grid_idx_2d_t{-1, -1};
    }
    vec3d_t station_id_to_feed_position_m([[maybe_unused]] station_id_t st_id) const override {
        return vec3d_t{0.0, 0.0, 0.0};
    }
};

REGISTER_TELESCOPE(TestTelescope, "TestTelescope");

/******************
 *
 * Comparison functions
 *
 ******************/


void check_ns_close(int64_t t1, int64_t t2, const std::string& name) {
    BOOST_CHECK_MESSAGE(abs(t1 - t2) <= NS_TOL,
                        fmt::format("|{0:s}1 - {0:s}2| = |{1:d} - {2:d}| = {3:d} < {4:d} failed",
                                    name, t1, t2, abs(t1 - t2), NS_TOL));
}

void check_double_close(double x, double y, double tol, const std::string& name) {
    BOOST_CHECK_MESSAGE(
        fabs(x - y) < tol,
        fmt::format("|{0:s}1 - {0:s}2| = |{1:.14f} - {2:.14f}| = {3:.3e} < {4:.3e} failed", name, x,
                    y, fabs(x - y), tol));
}

void check_eop_close(const EOP& eop1, const EOP& eop2) {
    check_ns_close(eop1.t_inst_ns, eop2.t_inst_ns, "inst");
    check_ns_close(eop1.t_ut1_ns, eop2.t_ut1_ns, "ut1");
    check_double_close(eop1.delta_UT1_inst, eop2.delta_UT1_inst, DUT1_TOL, "dut1");
    check_double_close(eop1.ERA_deg, eop2.ERA_deg, ERA_TOL, "era");
    check_double_close(eop1.xp_as, eop2.xp_as, XP_TOL, "xp");
    check_double_close(eop1.yp_as, eop2.yp_as, XP_TOL, "yp");
}


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
        // When we test interpolation later, we test it at 1/4
        // the distance between table entries. So the smallest
        // separation in our test table is 4 nanoseconds.
        //
        // Deliberately out of order to test Telescope's reordering.
        t = {4, 0, giga, giga * giga, giga * giga + 100'000 * giga};
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

// Uncomment to show kotekan logs to stdout during test run.
// BOOST_TEST_GLOBAL_FIXTURE(LoggingFixture);
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

/*
 * @brief   Test position getters.
 */
BOOST_FIXTURE_TEST_CASE(_instrument_position, TelescopeFixture) {
    BOOST_TEST_MESSAGE(fmt::format("Testing telescope position."));

    const Telescope& tel = Telescope::instance();

    BOOST_CHECK_EQUAL(tel.get_itrs_lat_deg(), 50.123);
    BOOST_CHECK_EQUAL(tel.get_itrs_lon_deg(), -120.456);
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

BOOST_FIXTURE_TEST_CASE(_get_eop_at_t, TelescopeEOPFixture) {
    const Telescope& tel = Telescope::instance();

    int64_t giga = 1'000'000'000;

    std::vector<EOP> eop_table = tel.get_current_EOP_table();

    // Loop over intervals between times in tables (include times before/after
    for (size_t i = 0; i <= N; i++) {

        // For each interval, grab the left and right values (just copy for before/after)
        int64_t ta, tb;
        double dut1a, dut1b, xa, xb, ya, yb;
        if (i > 0) {
            ta = eop_table[i - 1].t_inst_ns;
            dut1a = eop_table[i - 1].delta_UT1_inst;
            xa = eop_table[i - 1].xp_as;
            ya = eop_table[i - 1].yp_as;
        } else {
            ta = eop_table[0].t_inst_ns - giga;
            dut1a = eop_table[0].delta_UT1_inst;
            xa = eop_table[0].xp_as;
            ya = eop_table[0].yp_as;
        }

        if (i < N) {
            tb = eop_table[i].t_inst_ns;
            dut1b = eop_table[i].delta_UT1_inst;
            xb = eop_table[i].xp_as;
            yb = eop_table[i].yp_as;
        } else {
            tb = eop_table[N - 1].t_inst_ns + giga;
            dut1b = eop_table[N - 1].delta_UT1_inst;
            xb = eop_table[N - 1].xp_as;
            yb = eop_table[N - 1].yp_as;
        }

        BOOST_TEST_MESSAGE(fmt::format("eop_a - t: {} dut1: {} x: {} y: {}", ta, dut1a, xa, ya));
        BOOST_TEST_MESSAGE(fmt::format("eop_b - t: {} dut1: {} x: {} y: {}", tb, dut1b, xb, yb));

        // Test a few points on this interval.
        for (double w : {0.0, 0.25, 0.5, 0.75, 1.0}) {

            BOOST_TEST_MESSAGE(fmt::format("w: {}", w));

            // Make  the target EOP
            BareEOP test_beop = {.t_inst_ns = ta + static_cast<int64_t>(w * (tb - ta)),
                                 .delta_UT1_inst = (1 - w) * dut1a + w * dut1b,
                                 .xp_as = (1 - w) * xa + w * xb,
                                 .yp_as = (1 - w) * ya + w * yb};
            EOP test_eop = test_beop.to_EOP();

            // Check the _at_time_ns function.
            EOP tel_eop = tel.get_EOP_at_time_ns(test_eop.t_inst_ns);

            check_eop_close(tel_eop, test_eop);

            // Check the _at_function while we're here.
            EOP tel_eop2 = tel.get_EOP_at_time(nanosec_i64_to_timespec(test_eop.t_inst_ns));

            check_eop_close(tel_eop2, test_eop);
        }
    }
}

BOOST_FIXTURE_TEST_CASE(_get_eop_at_ut1, TelescopeEOPFixture) {
    const Telescope& tel = Telescope::instance();

    int64_t giga = 1'000'000'000;

    std::vector<EOP> eop_table = tel.get_current_EOP_table();

    // Loop over intervals between times in tables (include times before/after
    for (size_t i = 0; i <= N; i++) {

        // For each interval, grab the left and right values (just copy for before/after)
        int64_t ut1a, ut1b;
        double dut1a, dut1b, xa, xb, ya, yb;
        if (i > 0) {
            ut1a = eop_table[i - 1].t_ut1_ns;
            dut1a = eop_table[i - 1].delta_UT1_inst;
            xa = eop_table[i - 1].xp_as;
            ya = eop_table[i - 1].yp_as;
        } else {
            ut1a = eop_table[0].t_ut1_ns - giga;
            dut1a = eop_table[0].delta_UT1_inst;
            xa = eop_table[0].xp_as;
            ya = eop_table[0].yp_as;
        }

        if (i < N) {
            ut1b = eop_table[i].t_ut1_ns;
            dut1b = eop_table[i].delta_UT1_inst;
            xb = eop_table[i].xp_as;
            yb = eop_table[i].yp_as;
        } else {
            ut1b = eop_table[N - 1].t_ut1_ns + giga;
            dut1b = eop_table[N - 1].delta_UT1_inst;
            xb = eop_table[N - 1].xp_as;
            yb = eop_table[N - 1].yp_as;
        }

        BOOST_TEST_MESSAGE(
            fmt::format("eop_a - ut1: {} dut1: {} x: {} y: {}", ut1a, dut1a, xa, ya));
        BOOST_TEST_MESSAGE(
            fmt::format("eop_b - ut1: {} dut1: {} x: {} y: {}", ut1b, dut1b, xb, yb));

        // Test a few points on this interval.
        for (double w : {0.0, 0.25, 0.5, 0.75, 1.0}) {

            BOOST_TEST_MESSAGE(fmt::format("w: {}", w));

            // Make  the target EOP
            EOP test_eop = {.t_inst_ns = 0,
                            .t_ut1_ns = ut1a + static_cast<int64_t>(w * (ut1b - ut1a)),
                            .delta_UT1_inst = (1 - w) * dut1a + w * dut1b,
                            .ERA_deg = 0.0,
                            .xp_as = (1 - w) * xa + w * xb,
                            .yp_as = (1 - w) * ya + w * yb};
            test_eop.t_inst_ns = get_time_ns_from_UT1(test_eop.t_ut1_ns, test_eop.delta_UT1_inst);
            test_eop.ERA_deg = get_ERA_from_UT1(test_eop.t_ut1_ns, nullptr);

            // Check the _at_UT1 function.
            EOP tel_eop = tel.get_EOP_at_UT1(test_eop.t_ut1_ns);

            check_eop_close(tel_eop, test_eop);
        }
    }
}
