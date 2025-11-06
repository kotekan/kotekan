#define BOOST_TEST_MODULE "test_CHORDTelescope"

#include "CHORDTelescope.hpp"
#include "Config.hpp"
#include "Telescope.hpp"
#include "configUpdater.hpp"
#include "errors.h" // _global_log_level
#include "restServer.hpp"

#include "fmt.hpp"
#include "json.hpp"

#include <boost/test/included/unit_test.hpp>
#include <filesystem>
#include <inttypes.h>
#include <sys/wait.h>
#include <time.h>
#include <vector>

using kotekan::Config;
using json = nlohmann::json;
using kotekan::configUpdater;
using kotekan::restServer;

const std::string default_config_str = R"config_str({
"type": "config",
"log_level": "debug",
"num_dishes": 2,
"telescope": {
    "name": "CHORDTelescope",
    "inst_long_deg":  -119.621,
    "inst_lat_deg":   49.321,
    "inst_coelev_deg":   100.0,
    "inst_grid_x_axis":   [1.0, 0.0, 0.0],
    "inst_grid_y_axis":   [0.0, 1.0, 0.0],
    "inst_dish_elev_axis": [1.0, 0.0, 0.0],
    "inst_dish_vert_axis":    [0.0, 0.0, 1.0],
    "require_gps":        false,
    "updatable_config":   "/earth_rotation_data"
    },
"dish_inputs" : [
    {"dish_idx": 0, "ew_idx": 0, "ns_idx": 0, "pos_disp_m": [0.0, 0.0, 0.0],
     "coelev_disp_deg": 0.0, "type": 0, "label": "D00"},
    {"dish_idx": 1, "ew_idx": 1, "ns_idx": 0, "pos_disp_m": [0.0, 0.0, 0.0],
     "coelev_disp_deg": 0.0, "type": 0, "label": "D01"}],
"gps_time": {
    "frame0_nano": 1761926400000000000
    },
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
})config_str";

const CHORDTelescope& get_telescope(json& json_config) {
    Config conf;
    conf.update_config(json_config);
    configUpdater& config_updater = configUpdater::instance();
    config_updater.apply_config(conf);
    Telescope::instance(conf);
    return Telescope::instance().cast<CHORDTelescope>();
}


BOOST_AUTO_TEST_CASE(_instrument_position) {
    BOOST_TEST_MESSAGE(fmt::format("Testing telescope position."));

    double lon = -119.621123;
    double lat = 49.321123;

    json json_config = json::parse(default_config_str);
    json_config["telescope"]["inst_lat_deg"] = lat;
    json_config["telescope"]["inst_long_deg"] = lon;

    const CHORDTelescope& tel = get_telescope(json_config);

    BOOST_CHECK_EQUAL(tel.get_inst_long_deg(), lon);
    BOOST_CHECK_EQUAL(tel.get_inst_lat_deg(), lat);
}

BOOST_AUTO_TEST_CASE(_instrument_orientation) {
    BOOST_TEST_MESSAGE(fmt::format("Testing telescope orientation."));

    double coelev = -70;

    json json_config = json::parse(default_config_str);
    json_config["telescope"]["inst_coelev_deg"] = coelev;

    const CHORDTelescope& tel = get_telescope(json_config);

    BOOST_CHECK_EQUAL(tel.get_inst_coelev_deg(), coelev);
}

void check_dishes(const dishInfo& d1, const dishInfo& d2) {
    BOOST_CHECK_MESSAGE(d1 == d2,
            fmt::format("Expected dish (({:s})) == (({:s}))", json(d1).dump(), json(d2).dump()));
}

void check_equal_vec3d(const std::array<double, 3>& v1, const std::array<double, 3>& v2) {
    BOOST_CHECK_MESSAGE(v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2],
            fmt::format("Expected dish ({:g}, {:g}, {:g}) == ({:g}, {:g}, {:g})",
                v1[0], v1[1], v1[2], v2[0], v2[1], v2[2]));
}

BOOST_AUTO_TEST_CASE(_dish_num) {
    dishInfo d0 = make_dishInfo(0, 0, 0, {0.0, 0.0, 0.0}, 0.0,   0, "D1");
    dishInfo d1 = make_dishInfo(1, 0, 1, {0.0, 0.0, 0.0}, 35.0,  0, "D2");
    dishInfo d2 = make_dishInfo(2, 1, 0, {0.1, 0.0, 0.0}, 0.0,   0, "D3");
    dishInfo d3 = dish_null;
    dishInfo d4 = dish_null;
    dishInfo d5 = make_dishInfo(5, -5, 814, {-0.3, 1.0, 0.5}, -9.0, 0, "D4");
    dishInfo d6 = dish_null;
    dishInfo d7 = dish_null;
    d3.idx = 3;
    d4.idx = 4;
    d6.idx = 6;
    d7.idx = 7;
    
    json json_config = json::parse(default_config_str);
    json_config["num_dishes"] = 8;
    json_config["dish_separation_ew_m"] = 1.0;
    json_config["dish_separation_ns_m"] = 2.0;
    json_config["dish_inputs"] = std::vector<dishInfo>({d5, d0, d2, d1});
    const CHORDTelescope& tel = get_telescope(json_config);

    // Check the number of dishes is correct
    BOOST_CHECK_EQUAL(tel.get_num_dishes(), 8);
}

BOOST_AUTO_TEST_CASE(_dish_info) {
    dishInfo d0 = make_dishInfo(0, 0, 0, {0.0, 0.0, 0.0}, 0.0,   0, "D1");
    dishInfo d1 = make_dishInfo(1, 0, 1, {0.0, 0.0, 0.0}, 35.0,  0, "D2");
    dishInfo d2 = make_dishInfo(2, 1, 0, {0.1, 0.0, 0.0}, 0.0,   0, "D3");
    dishInfo d3 = dish_null;
    dishInfo d4 = dish_null;
    dishInfo d5 = make_dishInfo(5, -5, 814, {-0.3, 1.0, 0.5}, -9.0, 0, "D4");
    dishInfo d6 = dish_null;
    dishInfo d7 = dish_null;
    d3.idx = 3;
    d4.idx = 4;
    d6.idx = 6;
    d7.idx = 7;
    
    json json_config = json::parse(default_config_str);
    json_config["num_dishes"] = 8;
    json_config["dish_separation_ew_m"] = 1.0;
    json_config["dish_separation_ns_m"] = 2.0;
    json_config["dish_inputs"] = std::vector<dishInfo>({d5, d0, d2, d1});
    const CHORDTelescope& tel = get_telescope(json_config);

    // Check dish info is correct for all dishes
    check_dishes(tel.get_dish_at_idx(0), d0);
    check_dishes(tel.get_dish_at_idx(1), d1);
    check_dishes(tel.get_dish_at_idx(2), d2);
    check_dishes(tel.get_dish_at_idx(3), d3);
    check_dishes(tel.get_dish_at_idx(4), d4);
    check_dishes(tel.get_dish_at_idx(5), d5);
    check_dishes(tel.get_dish_at_idx(6), d6);
    check_dishes(tel.get_dish_at_idx(7), d7);
}

BOOST_AUTO_TEST_CASE(_dish_position) {
    dishInfo d0 = make_dishInfo(0, 0, 0, {0.0, 0.0, 0.0}, 0.0,   0, "D1");
    dishInfo d1 = make_dishInfo(1, 0, 1, {0.0, 0.0, 0.0}, 35.0,  0, "D2");
    dishInfo d2 = make_dishInfo(2, 1, 0, {0.1, 0.0, 0.0}, 0.0,   0, "D3");
    dishInfo d3 = dish_null;
    dishInfo d4 = dish_null;
    dishInfo d5 = make_dishInfo(5, -5, 814, {-0.3, 1.0, 0.5}, -9.0, 0, "D4");
    dishInfo d6 = dish_null;
    dishInfo d7 = dish_null;
    d3.idx = 3;
    d4.idx = 4;
    d6.idx = 6;
    d7.idx = 7;
    
    json json_config = json::parse(default_config_str);
    json_config["num_dishes"] = 8;
    json_config["dish_separation_ew_m"] = 1.0;
    json_config["dish_separation_ns_m"] = 2.0;
    json_config["dish_inputs"] = std::vector<dishInfo>({d5, d0, d2, d1});
    const CHORDTelescope& tel = get_telescope(json_config);

    // Check dish positions.
    // With weirder input these might fail floating point equality
    check_equal_vec3d(tel.get_dish_position(0), std::array<double, 3>({0.0, 0.0, 0.0}));
    check_equal_vec3d(tel.get_dish_position(1), std::array<double, 3>({0.0, 2.0, 0.0}));
    check_equal_vec3d(tel.get_dish_position(2), std::array<double, 3>({1.1, 0.0, 0.0}));
    check_equal_vec3d(tel.get_dish_position(3), std::array<double, 3>({0.0, 0.0, 0.0}));
    check_equal_vec3d(tel.get_dish_position(4), std::array<double, 3>({0.0, 0.0, 0.0}));
    check_equal_vec3d(tel.get_dish_position(5), std::array<double, 3>({-5.0-0.3, 814*2.0 + 1.0, 0.5}));
    check_equal_vec3d(tel.get_dish_position(6), std::array<double, 3>({0.0, 0.0, 0.0}));
    check_equal_vec3d(tel.get_dish_position(7), std::array<double, 3>({0.0, 0.0, 0.0}));
}

BOOST_AUTO_TEST_CASE(_dish_input_fields) {
    dishInfo d0 = make_dishInfo(0, 0, 0, {0.0, 0.0, 0.0}, 0.0,   0, "D1");
    dishInfo d1 = make_dishInfo(1, 0, 1, {0.0, 0.0, 0.0}, 35.0,  0, "D2");
    dishInfo d2 = make_dishInfo(2, 1, 0, {0.1, 0.0, 0.0}, 0.0,   0, "D3");
    dishInfo d3 = dish_null;
    dishInfo d4 = dish_null;
    dishInfo d5 = make_dishInfo(5, -5, 814, {-0.3, 1.0, 0.5}, -9.0, 0, "D4");
    dishInfo d6 = dish_null;
    dishInfo d7 = dish_null;
    d3.idx = 3;
    d4.idx = 4;
    d6.idx = 6;
    d7.idx = 7;
    
    json json_config = json::parse(default_config_str);
    json_config["num_dishes"] = 8;
    json_config["dish_separation_ew_m"] = 1.0;
    json_config["dish_separation_ns_m"] = 2.0;
    json_config["dish_inputs"] = std::vector<dishInfo>({d5, d0, d2, d1});
    const CHORDTelescope& tel = get_telescope(json_config);

    std::vector<dishInfo> d({d0, d1, d2, d3, d4, d5, d6, d7});

    dishInputFields buf;

    tel.get_dish_inputs(buf);

    for(int i = 0; i < 8; i++) {
        BOOST_CHECK_EQUAL(buf.ew_idx[i], d[i].ew_idx);
        BOOST_CHECK_EQUAL(buf.ns_idx[i], d[i].ns_idx);
        check_equal_vec3d(buf.pos_disp_m[i], d[i].pos_disp_m);
        BOOST_CHECK_EQUAL(buf.coelev_disp_deg[i], d[i].coelev_disp_deg);
        BOOST_CHECK_EQUAL(buf.type[i], d[i].type);
        BOOST_CHECK_EQUAL(buf.label[i], d[i].label);
    }
}
