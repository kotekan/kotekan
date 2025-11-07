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
    BOOST_CHECK_MESSAGE(d1 == d2, fmt::format("Expected dish (({:s})) == (({:s}))", json(d1).dump(),
                                              json(d2).dump()));
}

void check_equal_vec3d(const std::array<double, 3>& v1, const std::array<double, 3>& v2) {
    BOOST_CHECK_MESSAGE(v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2],
                        fmt::format("Expected ({:g}, {:g}, {:g}) == ({:g}, {:g}, {:g})", v1[0],
                                    v1[1], v1[2], v2[0], v2[1], v2[2]));
}

void check_close_vec3d(const std::array<double, 3>& v1, const std::array<double, 3>& v2, double atol, double rtol, const std::string &label) {

    double diff[3] = {v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]};
    double tol[3] = {atol + rtol * fabs(0.5 * (v1[0] + v2[0])),
                     atol + rtol * fabs(0.5 * (v1[1] + v2[1])),
                     atol + rtol * fabs(0.5 * (v1[2] + v2[2]))};

    bool pass[3] = {fabs(diff[0]) <= tol[0], fabs(diff[1]) <= tol[1], fabs(diff[2]) <= tol[2]};
    BOOST_CHECK_MESSAGE(pass[0] && pass[1] && pass[2],
                        fmt::format("Expected |{:s}| = |{:g}, {:g}, {:g}| <= ({:g}, {:g}, {:g})",
                            label, diff[0], diff[1], diff[2], tol[0], tol[1], tol[2]));
}

BOOST_AUTO_TEST_CASE(_dish_num) {
    dishInfo d0 = make_dishInfo(0, 0, 0, {0.0, 0.0, 0.0}, 0.0, 0, "D1");
    dishInfo d1 = make_dishInfo(1, 0, 1, {0.0, 0.0, 0.0}, 35.0, 0, "D2");
    dishInfo d2 = make_dishInfo(2, 1, 0, {0.1, 0.0, 0.0}, 0.0, 0, "D3");
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
    dishInfo d0 = make_dishInfo(0, 0, 0, {0.0, 0.0, 0.0}, 0.0, 0, "D1");
    dishInfo d1 = make_dishInfo(1, 0, 1, {0.0, 0.0, 0.0}, 35.0, 0, "D2");
    dishInfo d2 = make_dishInfo(2, 1, 0, {0.1, 0.0, 0.0}, 0.0, 0, "D3");
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
    dishInfo d0 = make_dishInfo(0, 0, 0, {0.0, 0.0, 0.0}, 0.0, 0, "D1");
    dishInfo d1 = make_dishInfo(1, 0, 1, {0.0, 0.0, 0.0}, 35.0, 0, "D2");
    dishInfo d2 = make_dishInfo(2, 1, 0, {0.1, 0.0, 0.0}, 0.0, 0, "D3");
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
    check_equal_vec3d(tel.get_dish_position(5),
                      std::array<double, 3>({-5.0 - 0.3, 814 * 2.0 + 1.0, 0.5}));
    check_equal_vec3d(tel.get_dish_position(6), std::array<double, 3>({0.0, 0.0, 0.0}));
    check_equal_vec3d(tel.get_dish_position(7), std::array<double, 3>({0.0, 0.0, 0.0}));
}

BOOST_AUTO_TEST_CASE(_dish_input_fields) {
    dishInfo d0 = make_dishInfo(0, 0, 0, {0.0, 0.0, 0.0}, 0.0, 0, "D1");
    dishInfo d1 = make_dishInfo(1, 0, 1, {0.0, 0.0, 0.0}, 35.0, 0, "D2");
    dishInfo d2 = make_dishInfo(2, 1, 0, {0.1, 0.0, 0.0}, 0.0, 0, "D3");
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

    // Set up an object to receive the dish input info.
    dishInputFields buf;
    // Fill the object with the info.
    tel.get_dish_inputs(buf);

    // Make sure its correct.
    for (int i = 0; i < 8; i++) {
        BOOST_CHECK_EQUAL(buf.ew_idx[i], d[i].ew_idx);
        BOOST_CHECK_EQUAL(buf.ns_idx[i], d[i].ns_idx);
        check_equal_vec3d(buf.pos_disp_m[i], d[i].pos_disp_m);
        BOOST_CHECK_EQUAL(buf.coelev_disp_deg[i], d[i].coelev_disp_deg);
        BOOST_CHECK_EQUAL(buf.type[i], d[i].type);
        BOOST_CHECK_EQUAL(buf.label[i], d[i].label);
    }
}

BOOST_AUTO_TEST_CASE(_pointing_vec_dish) {

    // Test co-elevation
    double coelev_deg = 20.0;
    
    // Compute correct pointing vec.
    double coelev = M_PI * coelev_deg / 180.0;
    std::array<double, 3> point{0.0, sin(coelev), cos(coelev)};

    // Build telescope
    json json_config = json::parse(default_config_str);
    json_config["telescope"]["inst_coelev_deg"] = coelev_deg;
    const CHORDTelescope& tel = get_telescope(json_config);

    // Check (may have FP equality issues)
    check_equal_vec3d(tel.get_pointing_vec_in_dish_coords(), point);

    // Test value 2
    coelev_deg = -40.0;

    // Compute truth
    coelev = M_PI * coelev_deg / 180.0;
    std::array<double, 3> point2{0.0, sin(coelev), cos(coelev)};

    // construct telescope
    json_config["telescope"]["inst_coelev_deg"] = coelev_deg;
    const CHORDTelescope &tel2 = get_telescope(json_config);

    // check
    check_equal_vec3d(tel2.get_pointing_vec_in_dish_coords(), point2);
}

BOOST_AUTO_TEST_CASE(_vec_topocen_to_dish) {
    //Make test frame
    double dphi = -0.5;
    double dtheta = 0.2;

    std::array<double, 3> z({cos(dphi)*sin(dtheta), sin(dphi)*sin(dtheta), cos(dtheta)});
    std::array<double, 3> x({cos(dphi)*cos(dtheta), sin(dphi)*cos(dtheta), -sin(dtheta)});
    std::array<double, 3> y({-sin(dphi), cos(dphi), 0.0});

    // Make telescope
    json json_config = json::parse(default_config_str);
    json_config["telescope"]["inst_dish_elev_axis"] = x;
    json_config["telescope"]["inst_dish_vert_axis"] = z;
    const CHORDTelescope& tel = get_telescope(json_config);

    // test vectors
    std::array<double, 3> n1({1.0, 0.0, 0.0});
    std::array<double, 3> n2({0.0, 1.0, 0.0});
    std::array<double, 3> n3({0.0, 0.0, 1.0});

    // Should just pick out basis vectors.
    check_close_vec3d(tel.vec_topocen_to_dish(x), n1, 1.0e-14, 1.0e-14, "x_dish - n1"); 
    check_close_vec3d(tel.vec_topocen_to_dish(y), n2, 1.0e-14, 1.0e-14, "y_dish - n2"); 
    check_close_vec3d(tel.vec_topocen_to_dish(z), n3, 1.0e-14, 1.0e-14, "z_dish - n3"); 
}

BOOST_AUTO_TEST_CASE(_vec_dish_to_topocen) {
    //Make test frame
    double dphi = -0.5;
    double dtheta = 0.2;

    std::array<double, 3> z({cos(dphi)*sin(dtheta), sin(dphi)*sin(dtheta), cos(dtheta)});
    std::array<double, 3> x({cos(dphi)*cos(dtheta), sin(dphi)*cos(dtheta), -sin(dtheta)});
    std::array<double, 3> y({-sin(dphi), cos(dphi), 0.0});

    // Make telescope
    json json_config = json::parse(default_config_str);
    json_config["telescope"]["inst_dish_elev_axis"] = x;
    json_config["telescope"]["inst_dish_vert_axis"] = z;
    const CHORDTelescope& tel = get_telescope(json_config);

    // test vectors
    std::array<double, 3> n1({1.0, 0.0, 0.0});
    std::array<double, 3> n2({0.0, 1.0, 0.0});
    std::array<double, 3> n3({0.0, 0.0, 1.0});

    // Should just pick out basis vectors.
    check_close_vec3d(tel.vec_dish_to_topocen(n1), x, 1.0e-14, 1.0e-14, "n1_topo - x"); 
    check_close_vec3d(tel.vec_dish_to_topocen(n2), y, 1.0e-14, 1.0e-14, "n2_topo - y"); 
    check_close_vec3d(tel.vec_dish_to_topocen(n3), z, 1.0e-14, 1.0e-14, "n3_topo - z"); 
}

BOOST_AUTO_TEST_CASE(_vec_topocen_to_tel) {
    //Make test frame
    double dphi = -0.5;
    double dtheta = 0.2;

    std::array<double, 3> z({cos(dphi)*sin(dtheta), sin(dphi)*sin(dtheta), cos(dtheta)});
    std::array<double, 3> x({cos(dphi)*cos(dtheta), sin(dphi)*cos(dtheta), -sin(dtheta)});
    std::array<double, 3> y({-sin(dphi), cos(dphi), 0.0});

    // Make telescope
    json json_config = json::parse(default_config_str);
    json_config["telescope"]["inst_grid_x_axis"] = x;
    json_config["telescope"]["inst_grid_y_axis"] = y;
    const CHORDTelescope& tel = get_telescope(json_config);

    // test vectors
    std::array<double, 3> n1({1.0, 0.0, 0.0});
    std::array<double, 3> n2({0.0, 1.0, 0.0});
    std::array<double, 3> n3({0.0, 0.0, 1.0});

    // Should just pick out basis vectors.
    check_close_vec3d(tel.vec_topocen_to_tel(x), n1, 1.0e-14, 1.0e-14, "x_tel - n1"); 
    check_close_vec3d(tel.vec_topocen_to_tel(y), n2, 1.0e-14, 1.0e-14, "y_tel - n2"); 
    check_close_vec3d(tel.vec_topocen_to_tel(z), n3, 1.0e-14, 1.0e-14, "z_tel - n3"); 
}

BOOST_AUTO_TEST_CASE(_vec_tel_to_topocen) {
    //Make test frame
    double dphi = -0.5;
    double dtheta = 0.2;

    std::array<double, 3> z({cos(dphi)*sin(dtheta), sin(dphi)*sin(dtheta), cos(dtheta)});
    std::array<double, 3> x({cos(dphi)*cos(dtheta), sin(dphi)*cos(dtheta), -sin(dtheta)});
    std::array<double, 3> y({-sin(dphi), cos(dphi), 0.0});

    // Make telescope
    json json_config = json::parse(default_config_str);
    json_config["telescope"]["inst_grid_x_axis"] = x;
    json_config["telescope"]["inst_grid_y_axis"] = y;
    const CHORDTelescope& tel = get_telescope(json_config);

    // test vectors
    std::array<double, 3> n1({1.0, 0.0, 0.0});
    std::array<double, 3> n2({0.0, 1.0, 0.0});
    std::array<double, 3> n3({0.0, 0.0, 1.0});

    // Should just pick out basis vectors.
    check_close_vec3d(tel.vec_tel_to_topocen(n1), x, 1.0e-14, 1.0e-14, "n1_topo - x"); 
    check_close_vec3d(tel.vec_tel_to_topocen(n2), y, 1.0e-14, 1.0e-14, "n2_topo - y"); 
    check_close_vec3d(tel.vec_tel_to_topocen(n3), z, 1.0e-14, 1.0e-14, "n3_topo - z"); 
}

