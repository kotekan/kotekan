#define BOOST_TEST_MODULE "test_Telescope"

#include "CHORDTelescope.hpp"
#include "Config.hpp"
#include "Telescope.hpp"
#include "configUpdater.hpp"
#include "errors.h" // _global_log_level
#include "restServer.hpp"
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
    "name": "fake",
    "require_eop": false
    }
})tel_config_str";

const std::string default_eop_config_str = R"eop_config_str({
"earth_rotation_data": {
    "kotekan_update_endpoint": "json",
    "earth_orientation_parameter_table": [
        {
            "time_inst_ns": 1761883200000000000,
            "delta_UT1_inst": 0.0,
            "x_pm": 0.0,
            "y_pm": 0.0
        }, {
            "time_inst_ns": 1761969600000000000,
            "delta_UT1_inst": 0.0,
            "x_pm": 0.0,
            "y_pm": 0.0
        }]
    }
})eop_config_str";

const Telescope& get_telescope(json& json_config) {
    Config conf;
    conf.update_config(json_config);
    configUpdater& config_updater = configUpdater::instance();
    config_updater.apply_config(conf);
    Telescope::instance(conf);
    return Telescope::instance();
}

/******************
 *
 * HELPERS
 *
 ******************/


std::vector<EOP> make_full_eop_table(const std::vector<int64_t> &t, const std::vector<double> &dut1,
                                     const std::vector<double> &xpm, const std::vector<double> &ypm) {

    int N = t.size();
    
    std::vector<EOP> eop(N);

    for(int i = 0; i < N; i++) {
        int64_t ut1 = get_UT1_from_time(nanosec_i64_to_timespec(t[i]), dut1[i]);
        eop[i] = {.t_inst=t[i], .t_ut1 = ut1,
                  .delta_UT1_inst=dut1[i], .ERA_deg=get_ERA_from_UT1(ut1, nullptr),
                  .xp_as=xpm[i], .yp_as=ypm[i]};
    }

    return eop;
}

json make_conf_eop_table(const std::vector<int64_t> &t, const std::vector<double> &dut1,
                                       const std::vector<double> &xpm, const std::vector<double> &ypm) {

    int N = t.size();
    
    BOOST_TEST_MESSAGE("make table");
    json eop_update_table = json::array();

    for(int i = 0; i < N; i++) {
        BOOST_TEST_MESSAGE("add elem");
        eop_update_table.push_back({{"time_inst_ns", t[i]}, {"delta_UT1_inst", dut1[i]},
                                    {"x_pm", xpm[i]}, {"y_pm", ypm[i]}});
    }

    BOOST_TEST_MESSAGE("make final");
    json eop_conf_json = {};
    BOOST_TEST_MESSAGE("add endpoint");
    eop_conf_json["kotekan_update_endpoint"] = "json";
    BOOST_TEST_MESSAGE("add table");
    eop_conf_json["earth_orientation_parameter_table"] = eop_update_table;
    BOOST_TEST_MESSAGE("return");

    return eop_conf_json;
}


/******************
 *
 * TESTS
 *
 ******************/


BOOST_AUTO_TEST_CASE(_name_tel) {
    json json_config = json::parse(default_tel_config_str);

    const Telescope& tel = get_telescope(json_config);

    BOOST_CHECK_EQUAL(tel.get_name(), "fake");
}

BOOST_AUTO_TEST_CASE(_BareEOP_to_json) {
   
    BareEOP eop(12345678901234, -1.7, 1.23, -3.6e-8);
    json jeop = json::parse(R"({"time_inst_ns": 12345678901234, "delta_UT1_inst": -1.7, "x_pm": 1.23, "y_pm": -3.6e-8})");

    json jeop_conv = eop;

    BOOST_CHECK_MESSAGE(jeop_conv == jeop, "to_json(BareEOP)");
}

BOOST_AUTO_TEST_CASE(_json_to_BareEOP) {
   
    BareEOP eop(12345678901234, -1.7, 1.23, -3.6e-8);
    json jeop = json::parse(R"({"time_inst_ns": 12345678901234, "delta_UT1_inst": -1.7, "x_pm": 1.23, "y_pm": -3.6e-8})");

    BareEOP eop_conv = jeop;

    BOOST_CHECK_MESSAGE(eop_conv == eop, "from_json(BareEOP)");
}

/*
BOOST_AUTO_TEST_CASE(_eop_table) {
    _global_log_level = 5;
    BOOST_TEST_MESSAGE("parse");
    json json_config = json::parse(default_tel_config_str);

    BOOST_TEST_MESSAGE("add field");
    json_config["telescope"]["eop_updatable_config"] = "eop_update";
    json_config["telescope"]["require_eop"] = true;

    int64_t giga = 1'000'000'000L;

    BOOST_TEST_MESSAGE("init vecs");
    std::vector<int64_t> t{100 * giga, giga * giga, 2 * giga * giga + 2};
    std::vector<double> dut1{0.0, 0.5, -1.7};
    std::vector<double> xpm{0.0, -1.0, 2.34};
    std::vector<double> ypm{0.0, 0.00001, -8.99999};

    std::vector<EOP> eop_true = make_full_eop_table(t, dut1, xpm, ypm);

    BOOST_TEST_MESSAGE("make conf");
    json eop_update = make_conf_eop_table(t, dut1, xpm, ypm);

    BOOST_TEST_MESSAGE("assign eop");
    json_config["eop_update"] = eop_update;

    BOOST_TEST_MESSAGE(json_config);
    std::vector<EOP> eop;
    BOOST_TEST_MESSAGE("Make tel");
    boost::test_tools::output_test_stream my_cout;
    boost::test_tools::output_test_stream my_cerr;
    {
        cout_redirect guard_out(my_cout.rdbuf());
        cerr_redirect guard_err(my_cerr.rdbuf());
        try {
            const Telescope& tel = get_telescope(json_config);
            eop = tel.get_current_EOP_table();
        } catch (const std::runtime_error &e ) {
            BOOST_TEST_MESSAGE(fmt::format("ERROR: {}", e.what()));

            BOOST_TEST_MESSAGE("STDOUT");
            BOOST_TEST_MESSAGE(my_cout.str());
            BOOST_TEST_MESSAGE("STDERR");
            BOOST_TEST_MESSAGE(my_cerr.str());
        }
    }

    BOOST_TEST_MESSAGE("STDOUT");
    BOOST_TEST_MESSAGE(my_cout.str());
    BOOST_TEST_MESSAGE("STDERR");
    BOOST_TEST_MESSAGE(my_cerr.str());


    BOOST_CHECK_EQUAL(eop.size(), eop_true.size());
}
*/
