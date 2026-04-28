#define BOOST_TEST_MODULE "test_CHORDTelescope"

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
#include <complex>
#include <filesystem>
#include <inttypes.h>
#include <sys/wait.h>
#include <time.h>
#include <vector>

using kotekan::Config;
using json = nlohmann::json;
using kotekan::configUpdater;
using kotekan::restServer;
using namespace std::complex_literals;

const std::string default_config_str = R"config_str({
"type": "config",
"log_level": "debug",
"num_dishes": 2,
"telescope": {
    "name": "CHORDTelescope",
    "origin_itrs_lon_deg":  -119.621,
    "origin_itrs_lat_deg":   49.321,
    "dish_coelev_deg":   100.0,
    "grid_x_axis":   [1.0, 0.0, 0.0],
    "grid_y_axis":   [0.0, 1.0, 0.0],
    "dish_elev_axis": [1.0, 0.0, 0.0],
    "dish_vert_axis":    [0.0, 0.0, 1.0],
    "require_gps":        false,
    "eop_updatable_config":   "/earth_rotation_data"
    },
    "num_dishes_x" : 22,
    "num_dishes_y" : 24,
    "dish_ew_separation_m": 6.3,
    "dish_ns_separation_m": 8.5,
    "dish_inputs" : [
        {"dish_idx": 0, "grid_x_idx": 0, "grid_y_idx": 0, "feed_pos_disp_m": [0.0, 0.0, 0.0],
         "coelev_disp_deg": 0.0, "type": "ArrayDish", "label": "D00"},
        {"dish_idx": 1, "grid_x_idx": 1, "grid_y_idx": 0, "feed_pos_disp_m": [0.0, 0.0, 0.0],
         "coelev_disp_deg": 0.0, "type": "ArrayDish", "label": "D01"}],
"gps_time": {
    "frame0_nano": 1761926400000000000
    },
"earth_rotation_data": {
    "kotekan_update_endpoint": "json",
    "earth_orientation_parameter_table": [
        {
            "t_inst_ns": 1761883200000000000,
            "delta_UT1_inst": 0.0,
            "xp_as": 0.0,
            "yp_as": 0.0
        }, {
            "t_inst_ns": 1761969600000000000,
            "delta_UT1_inst": 0.0,
            "xp_as": 0.0,
            "yp_as": 0.0
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

/******************
 *
 * HELPERS
 *
 ******************/


/*
 * @brief   Helper to compare dishes
 */
void check_dishes(const dishInfo& d1, const dishInfo& d2) {
    BOOST_CHECK_MESSAGE(d1 == d2, fmt::format("Expected dish (({:s})) == (({:s}))", json(d1).dump(),
                                              json(d2).dump()));
}

/*
 * @brief   Helper to test equality for double vec3s
 */
void check_equal_vec3d(const std::array<double, 3>& v1, const std::array<double, 3>& v2) {
    BOOST_CHECK_MESSAGE(v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2],
                        fmt::format("Expected ({:g}, {:g}, {:g}) == ({:g}, {:g}, {:g})", v1[0],
                                    v1[1], v1[2], v2[0], v2[1], v2[2]));
}

/*
 * @brief   Helper to test equality for float vec3s within tolerance.
 */
void check_close_float(float v1, float v2, float atol, float rtol, const std::string& label1,
                       const std::string& label2) {

    float diff = v1 - v2;
    float tol = atol + rtol * fabs(0.5 * (v1 + v2));

    BOOST_CHECK_MESSAGE(
        fabs(diff) <= tol,
        fmt::format("Expected |{:s} - {:s}| = |{:g} - {:g}| <= {:g}", label1, label2, v1, v2, tol));
}

/*
 * @brief   Helper to test equality for double vec3s within tolerance.
 */
void check_close_double(double v1, double v2, double atol, double rtol, const std::string& label1,
                        const std::string& label2) {

    double diff = v1 - v2;
    double tol = atol + rtol * fabs(0.5 * (v1 + v2));

    BOOST_CHECK_MESSAGE(
        fabs(diff) <= tol,
        fmt::format("Expected |{:s} - {:s}| = |{:g} - {:g}| <= {:g}", label1, label2, v1, v2, tol));
}

/*
 * @brief   Helper to test equality for double vec3s within tolerance.
 */
void check_close_vec3d(const std::array<double, 3>& v1, const std::array<double, 3>& v2,
                       double atol, double rtol, const std::string& label) {

    double diff[3] = {v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]};
    double tol[3] = {atol + rtol * fabs(0.5 * (v1[0] + v2[0])),
                     atol + rtol * fabs(0.5 * (v1[1] + v2[1])),
                     atol + rtol * fabs(0.5 * (v1[2] + v2[2]))};

    bool pass[3] = {fabs(diff[0]) <= tol[0], fabs(diff[1]) <= tol[1], fabs(diff[2]) <= tol[2]};
    BOOST_CHECK_MESSAGE(pass[0] && pass[1] && pass[2],
                        fmt::format("Expected |{:s}| = |{:g}, {:g}, {:g}| <= ({:g}, {:g}, {:g})",
                                    label, diff[0], diff[1], diff[2], tol[0], tol[1], tol[2]));
}

/*
 * @brief   helper to calculate cirs -> itrs conversion
 */
std::array<double, 3> cirs_to_itrs(const CHORDTelescope& tel, std::array<double, 3> v,
                                   double ERA_deg, double xp_as, double yp_as) {
    double ERA = M_PI * ERA_deg / 180.0;
    double xp = M_PI * xp_as / (3600 * 180.0);
    double yp = M_PI * yp_as / (3600 * 180.0);

    std::array<double, 3> Rv = tel.vec_axes_rotation_R3(v, ERA);
    std::array<double, 3> W2Rv = tel.vec_axes_rotation_R2(Rv, -xp);
    std::array<double, 3> WRv = tel.vec_axes_rotation_R1(W2Rv, -yp);

    return WRv;
}

/*
 * @brief   helper to calculate itrs -> cirs conversion
 */
std::array<double, 3> itrs_to_cirs(const CHORDTelescope& tel, std::array<double, 3> v,
                                   double ERA_deg, double xp_as, double yp_as) {
    double ERA = M_PI * ERA_deg / 180.0;
    double xp = M_PI * xp_as / (3600 * 180.0);
    double yp = M_PI * yp_as / (3600 * 180.0);

    std::array<double, 3> W1v = tel.vec_axes_rotation_R1(v, yp);
    std::array<double, 3> Wv = tel.vec_axes_rotation_R2(W1v, xp);
    std::array<double, 3> RWv = tel.vec_axes_rotation_R3(Wv, -ERA);

    return RWv;
}

/******************
 *
 * TESTS
 *
 ******************/

BOOST_AUTO_TEST_CASE(_DishType_to_json) {

    json fake = DishType::Fake;
    BOOST_CHECK_MESSAGE(fake == "Fake", "to_json(Fake)");

    json arrayDish = DishType::ArrayDish;
    BOOST_CHECK_MESSAGE(arrayDish == "ArrayDish", "to_json(ArrayDish)");

    json j;
    BOOST_CHECK_THROW(to_json(j, static_cast<DishType>(-2)), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(_DishType_from_json) {

    json fake = "Fake";
    BOOST_CHECK_MESSAGE(fake.get<DishType>() == DishType::Fake, "from_json(Fake)");

    json array_dish = "ArrayDish";
    BOOST_CHECK_MESSAGE(array_dish.get<DishType>() == DishType::ArrayDish, "from_json(ArrayDish)");

    json bad = "UnKnOwN";
    BOOST_CHECK_THROW(bad.get<DishType>(), std::runtime_error);

    json fake_int = -1;
    BOOST_CHECK_THROW(fake_int.get<DishType>(), std::runtime_error);
}

/*
 * @brief   Test position getters.
 */
BOOST_AUTO_TEST_CASE(_query_gps_time_config) {
    BOOST_TEST_MESSAGE(fmt::format("Testing time0 query from config."));

    uint64_t time0_ns = 1234567890123;

    json json_config = json::parse(default_config_str);
    json_config["gps_time"]["frame0_nano"] = time0_ns;

    const CHORDTelescope& tel = get_telescope(json_config);

    uint64_t test_time0_val = 0;
    tel.query_gps_time0_ns(test_time0_val, 30);
    BOOST_CHECK_EQUAL(test_time0_val, time0_ns);
}


/*
 * @brief   Test position getters.
 */
BOOST_AUTO_TEST_CASE(_instrument_position) {
    BOOST_TEST_MESSAGE(fmt::format("Testing telescope position."));

    double lon = -119.621123;
    double lat = 49.321123;

    json json_config = json::parse(default_config_str);
    json_config["telescope"]["origin_itrs_lat_deg"] = lat;
    json_config["telescope"]["origin_itrs_lon_deg"] = lon;

    const CHORDTelescope& tel = get_telescope(json_config);

    BOOST_CHECK_EQUAL(tel.get_origin_itrs_lon_deg(), lon);
    BOOST_CHECK_EQUAL(tel.get_origin_itrs_lat_deg(), lat);
}

/*
 * @brief   Test coelev getters.
 */
BOOST_AUTO_TEST_CASE(_dish_coelev) {
    BOOST_TEST_MESSAGE(fmt::format("Testing telescope coelevation."));

    double coelev = -70;

    json json_config = json::parse(default_config_str);
    json_config["telescope"]["dish_coelev_deg"] = coelev;

    const CHORDTelescope& tel = get_telescope(json_config);

    BOOST_CHECK_EQUAL(tel.get_dish_coelev_deg(), coelev);
}


/*
 * @brief   Test dish number getter
 */
BOOST_AUTO_TEST_CASE(_dish_num) {
    dishInfo d0 = dishInfo(0, 0, 0, {0.0, 0.0, 0.0}, 0.0, DishType::ArrayDish, "D1");
    dishInfo d1 = dishInfo(1, 0, 1, {0.0, 0.0, 0.0}, 35.0, DishType::ArrayDish, "D2");
    dishInfo d2 = dishInfo(2, 1, 0, {0.1, 0.0, 0.0}, 0.0, DishType::ArrayDish, "D3");
    dishInfo d3 = dishInfo(3);
    dishInfo d4 = dishInfo(4);
    dishInfo d5 = dishInfo(5, 21, 23, {-0.3, 1.0, 0.5}, -9.0, DishType::ArrayDish, "D4");
    dishInfo d6 = dishInfo(6);
    dishInfo d7 = dishInfo(7);

    json json_config = json::parse(default_config_str);
    json_config["num_dishes"] = 8;
    json_config["dish_separation_x_m"] = 1.0;
    json_config["dish_separation_y_m"] = 2.0;
    json_config["dish_inputs"] = std::vector<dishInfo>({d5, d0, d2, d1});
    const CHORDTelescope& tel = get_telescope(json_config);

    // Check the number of dishes is correct
    BOOST_CHECK_EQUAL(tel.get_num_dishes(), 8);
}

/*
 * @brief   Test dish_info retrieval
 */
BOOST_AUTO_TEST_CASE(_dish_info) {
    dishInfo d0 = dishInfo(0, 0, 0, {0.0, 0.0, 0.0}, 0.0, DishType::ArrayDish, "D1");
    dishInfo d1 = dishInfo(1, 0, 1, {0.0, 0.0, 0.0}, 35.0, DishType::ArrayDish, "D2");
    dishInfo d2 = dishInfo(2, 1, 0, {0.1, 0.0, 0.0}, 0.0, DishType::ArrayDish, "D3");
    dishInfo d3 = dishInfo(3);
    dishInfo d4 = dishInfo(4);
    dishInfo d5 = dishInfo(5, 21, 23, {-0.3, 1.0, 0.5}, -9.0, DishType::ArrayDish, "D4");
    dishInfo d6 = dishInfo(6);
    dishInfo d7 = dishInfo(7);

    json json_config = json::parse(default_config_str);
    json_config["num_dishes"] = 8;
    json_config["dish_separation_x_m"] = 1.0;
    json_config["dish_separation_y_m"] = 2.0;
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

/*
 * @brief   Test dish_position calculation
 */
BOOST_AUTO_TEST_CASE(_dish_position) {
    dishInfo d0 = dishInfo(0, 0, 0, {0.0, 0.0, 0.0}, 0.0, DishType::ArrayDish, "D1");
    dishInfo d1 = dishInfo(1, 0, 1, {0.0, 0.0, 0.0}, 35.0, DishType::ArrayDish, "D2");
    dishInfo d2 = dishInfo(2, 1, 0, {0.1, 0.0, 0.0}, 0.0, DishType::ArrayDish, "D3");
    dishInfo d3 = dishInfo(3);
    dishInfo d4 = dishInfo(4);
    dishInfo d5 = dishInfo(5, 21, 23, {-0.3, 1.0, 0.5}, -9.0, DishType::ArrayDish, "D4");
    dishInfo d6 = dishInfo(6);
    dishInfo d7 = dishInfo(7);

    json json_config = json::parse(default_config_str);
    json_config["num_dishes"] = 8;
    json_config["dish_separation_x_m"] = 1.0;
    json_config["dish_separation_y_m"] = 2.0;
    json_config["dish_inputs"] = std::vector<dishInfo>({d5, d0, d2, d1});
    const CHORDTelescope& tel = get_telescope(json_config);

    // Check dish positions.
    // With weirder input these might fail floating point equality
    check_equal_vec3d(tel.get_dish_position_in_grid_coords(0),
                      std::array<double, 3>({0.0, 0.0, 0.0}));
    check_equal_vec3d(tel.get_dish_position_in_grid_coords(1),
                      std::array<double, 3>({0.0, 2.0, 0.0}));
    check_equal_vec3d(tel.get_dish_position_in_grid_coords(2),
                      std::array<double, 3>({1.1, 0.0, 0.0}));
    check_equal_vec3d(tel.get_dish_position_in_grid_coords(3),
                      std::array<double, 3>({0.0, 0.0, 0.0}));
    check_equal_vec3d(tel.get_dish_position_in_grid_coords(4),
                      std::array<double, 3>({0.0, 0.0, 0.0}));
    check_equal_vec3d(tel.get_dish_position_in_grid_coords(5),
                      std::array<double, 3>({21.0 - 0.3, 23.0 * 2.0 + 1.0, 0.5}));
    check_equal_vec3d(tel.get_dish_position_in_grid_coords(6),
                      std::array<double, 3>({0.0, 0.0, 0.0}));
    check_equal_vec3d(tel.get_dish_position_in_grid_coords(7),
                      std::array<double, 3>({0.0, 0.0, 0.0}));
}

/*
 * @brief   Test dish_input fields retrieval
 */
BOOST_AUTO_TEST_CASE(_get_input_maps) {
    dishInfo d0 = dishInfo(0, 0, 0, {0.0, 0.0, 0.0}, 0.0, DishType::ArrayDish, "D1");
    dishInfo d1 = dishInfo(1, 0, 1, {0.0, 0.0, 0.0}, 35.0, DishType::ArrayDish, "D2");
    dishInfo d2 = dishInfo(2, 1, 0, {0.1, 0.0, 0.0}, 0.0, DishType::ArrayDish, "D3");
    dishInfo d3 = dishInfo(3);
    dishInfo d4 = dishInfo(4);
    dishInfo d5 = dishInfo(5, 21, 23, {-0.3, 1.0, 0.5}, -9.0, DishType::ArrayDish, "D4");
    dishInfo d6 = dishInfo(6);
    dishInfo d7 = dishInfo(7);

    json json_config = json::parse(default_config_str);
    json_config["num_dishes"] = 8;
    json_config["num_dishes_x"] = 22;
    json_config["num_dishes_y"] = 24;
    json_config["dish_separation_x_m"] = 1.0;
    json_config["dish_separation_y_m"] = 2.0;
    json_config["dish_inputs"] = std::vector<dishInfo>({d5, d0, d2, d1});
    const CHORDTelescope& tel = get_telescope(json_config);

    std::vector<dishInfo> d({d0, d1, d2, d3, d4, d5, d6, d7});

    // Set up an object to receive the dish input info.
    dishInputFields buf;
    // Fill the object with the info.
    tel.fill_input_maps(buf);

    // Make sure it's correct.
    for (int i = 0; i < 8; i++) {
        BOOST_CHECK_EQUAL(buf.grid_x_idx[i], d[i].grid_x_idx);
        BOOST_CHECK_EQUAL(buf.grid_y_idx[i], d[i].grid_y_idx);
        check_equal_vec3d(buf.feed_pos_disp_m[i], d[i].feed_pos_disp_m);
        BOOST_CHECK_EQUAL(buf.coelev_disp_deg[i], d[i].coelev_disp_deg);
        BOOST_CHECK_EQUAL(static_cast<int32_t>(buf.type[i]), static_cast<int32_t>(d[i].type));
        BOOST_CHECK_EQUAL(buf.label[i], d[i].label);
    }
}

/*
 * @brief   Test dish_grid population
 */
BOOST_AUTO_TEST_CASE(_dish_grid_population) {
    dishInfo d0 = dishInfo(0, 0, 0, {0.0, 0.0, 0.0}, 0.0, DishType::ArrayDish, "D1");
    dishInfo d1 = dishInfo(1, 0, 1, {0.0, 0.0, 0.0}, 35.0, DishType::ArrayDish, "D2");
    dishInfo d2 = dishInfo(2, 1, 0, {0.1, 0.0, 0.0}, 0.0, DishType::ArrayDish, "D3");
    dishInfo d3 = dishInfo(3);
    dishInfo d4 = dishInfo(4);
    dishInfo d5 = dishInfo(5, 21, 23, {-0.3, 1.0, 0.5}, -9.0, DishType::ArrayDish, "D4");
    dishInfo d6 = dishInfo(6);
    dishInfo d7 = dishInfo(7);

    json json_config = json::parse(default_config_str);
    json_config["num_dishes"] = 8;
    json_config["num_dishes_x"] = 22;
    json_config["num_dishes_y"] = 24;
    json_config["dish_inputs"] = std::vector<dishInfo>({d5, d0, d2, d1});
    const CHORDTelescope& tel = get_telescope(json_config);

    const dishGrid& grid = tel.get_dish_grid();
    BOOST_CHECK_EQUAL(grid.dish_index(0, 0), 0);
    BOOST_CHECK_EQUAL(grid.dish_index(1, 0), 2);
    BOOST_CHECK_EQUAL(grid.dish_index(0, 1), 1);
    BOOST_CHECK_EQUAL(grid.dish_index(21, 23), 5);
    BOOST_CHECK_EQUAL(grid.dish_index(2, 0), -1);
    BOOST_CHECK_EQUAL(grid.dish_index(0, 2), -1);
}

/*
 * @brief   Test pointing vector retrieval
 */
BOOST_AUTO_TEST_CASE(_pointing_vec_dish) {

    // Test co-elevation
    double coelev_deg = 20.0;

    // Compute correct pointing vec.
    double coelev = M_PI * coelev_deg / 180.0;
    std::array<double, 3> point{0.0, sin(coelev), cos(coelev)};

    // Build telescope
    json json_config = json::parse(default_config_str);
    json_config["telescope"]["dish_coelev_deg"] = coelev_deg;
    const CHORDTelescope& tel = get_telescope(json_config);

    // Check (may have FP equality issues)
    check_equal_vec3d(tel.get_pointing_vec_in_dish_coords(), point);

    // Test value 2
    coelev_deg = -40.0;

    // Compute truth
    coelev = M_PI * coelev_deg / 180.0;
    std::array<double, 3> point2{0.0, sin(coelev), cos(coelev)};

    // construct telescope
    json_config["telescope"]["dish_coelev_deg"] = coelev_deg;
    const CHORDTelescope& tel2 = get_telescope(json_config);

    // check
    check_equal_vec3d(tel2.get_pointing_vec_in_dish_coords(), point2);
}

/*
 * @brief   Test topocen -> dish conversion
 */
BOOST_AUTO_TEST_CASE(_vec_topocen_to_dish) {
    // Make test frame
    double dphi = -0.5;
    double dtheta = 0.2;

    std::array<double, 3> z({cos(dphi) * sin(dtheta), sin(dphi) * sin(dtheta), cos(dtheta)});
    std::array<double, 3> x({cos(dphi) * cos(dtheta), sin(dphi) * cos(dtheta), -sin(dtheta)});
    std::array<double, 3> y({-sin(dphi), cos(dphi), 0.0});

    // Make telescope
    json json_config = json::parse(default_config_str);
    json_config["telescope"]["dish_elev_axis"] = x;
    json_config["telescope"]["dish_vert_axis"] = z;
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

/*
 * @brief   Test topocen <- dish conversion
 */
BOOST_AUTO_TEST_CASE(_vec_dish_to_topocen) {
    // Make test frame
    double dphi = -0.5;
    double dtheta = 0.2;

    std::array<double, 3> z({cos(dphi) * sin(dtheta), sin(dphi) * sin(dtheta), cos(dtheta)});
    std::array<double, 3> x({cos(dphi) * cos(dtheta), sin(dphi) * cos(dtheta), -sin(dtheta)});
    std::array<double, 3> y({-sin(dphi), cos(dphi), 0.0});

    // Make telescope
    json json_config = json::parse(default_config_str);
    json_config["telescope"]["dish_elev_axis"] = x;
    json_config["telescope"]["dish_vert_axis"] = z;
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

/*
 * @brief   Test topocen -> grid conversion
 */
BOOST_AUTO_TEST_CASE(_vec_topocen_to_grid) {
    // Make test frame
    double dphi = -0.5;
    double dtheta = 0.2;

    std::array<double, 3> z({cos(dphi) * sin(dtheta), sin(dphi) * sin(dtheta), cos(dtheta)});
    std::array<double, 3> x({cos(dphi) * cos(dtheta), sin(dphi) * cos(dtheta), -sin(dtheta)});
    std::array<double, 3> y({-sin(dphi), cos(dphi), 0.0});

    // Make telescope
    json json_config = json::parse(default_config_str);
    json_config["telescope"]["grid_x_axis"] = x;
    json_config["telescope"]["grid_y_axis"] = y;
    const CHORDTelescope& tel = get_telescope(json_config);

    // test vectors
    std::array<double, 3> n1({1.0, 0.0, 0.0});
    std::array<double, 3> n2({0.0, 1.0, 0.0});
    std::array<double, 3> n3({0.0, 0.0, 1.0});

    // Should just pick out basis vectors.
    check_close_vec3d(tel.vec_topocen_to_grid(x), n1, 1.0e-14, 1.0e-14, "x_grid - n1");
    check_close_vec3d(tel.vec_topocen_to_grid(y), n2, 1.0e-14, 1.0e-14, "y_grid - n2");
    check_close_vec3d(tel.vec_topocen_to_grid(z), n3, 1.0e-14, 1.0e-14, "z_grid - n3");
}

/*
 * @brief   Test topocen <- grid conversion
 */
BOOST_AUTO_TEST_CASE(_vec_grid_to_topocen) {
    // Make test frame
    double dphi = -0.5;
    double dtheta = 0.2;

    std::array<double, 3> z({cos(dphi) * sin(dtheta), sin(dphi) * sin(dtheta), cos(dtheta)});
    std::array<double, 3> x({cos(dphi) * cos(dtheta), sin(dphi) * cos(dtheta), -sin(dtheta)});
    std::array<double, 3> y({-sin(dphi), cos(dphi), 0.0});

    // Make telescope
    json json_config = json::parse(default_config_str);
    json_config["telescope"]["grid_x_axis"] = x;
    json_config["telescope"]["grid_y_axis"] = y;
    const CHORDTelescope& tel = get_telescope(json_config);

    // test vectors
    std::array<double, 3> n1({1.0, 0.0, 0.0});
    std::array<double, 3> n2({0.0, 1.0, 0.0});
    std::array<double, 3> n3({0.0, 0.0, 1.0});

    // Should just pick out basis vectors.
    check_close_vec3d(tel.vec_grid_to_topocen(n1), x, 1.0e-14, 1.0e-14, "n1_topo - x");
    check_close_vec3d(tel.vec_grid_to_topocen(n2), y, 1.0e-14, 1.0e-14, "n2_topo - y");
    check_close_vec3d(tel.vec_grid_to_topocen(n3), z, 1.0e-14, 1.0e-14, "n3_topo - z");
}

/*
 * @brief   Test frequency ID clamping across Nyquist zones
 */
BOOST_AUTO_TEST_CASE(_freq_ids_clamping_nyquist_zones) {
    struct Case {
        nyquist_zone_t nz;
        double min_freq;
        double max_freq;
        freq_id_t expected_min_id;
        freq_id_t expected_max_id;
    };

    // fft_length=16384, fs=3200 MHz -> df=0.1953125 MHz, num_freqs=8192 (Nyquist bin excluded)
    const std::vector<Case> cases = {
        {1, 300.0, 1500.0, 1536, 7680},  // df>0
        {2, 1800.0, 2500.0, 3584, 7168}, // df<0
        {3, 3800.0, 4500.0, 3072, 6656}, // df>0
    };

    for (const auto& c : cases) {
        json json_config = json::parse(default_config_str);
        json_config["telescope"]["nyquist_zone"] = c.nz;
        json_config["telescope"]["min_science_freq_MHz"] = c.min_freq;
        json_config["telescope"]["max_science_freq_MHz"] = c.max_freq;

        const CHORDTelescope& tel = get_telescope(json_config);

        auto min_id = tel.min_science_freq_id();
        auto max_id = tel.max_science_freq_id();

        BOOST_CHECK_EQUAL(min_id, c.expected_min_id);
        BOOST_CHECK_EQUAL(max_id, c.expected_max_id);

        double f_min_id = tel.to_freq_MHz(min_id);
        double f_max_id = tel.to_freq_MHz(max_id);

        double expected_low = (c.nz % 2u == 1u) ? c.min_freq : c.max_freq;
        double expected_high = (c.nz % 2u == 1u) ? c.max_freq : c.min_freq;

        check_close_double(f_min_id, expected_low, 1.0e-12, 1.0e-12, "f_min_id", "expected_low");
        check_close_double(f_max_id, expected_high, 1.0e-12, 1.0e-12, "f_max_id", "expected_high");
    }
}

/*
 * @brief   Test itrs -> topocen conversion
 */
BOOST_AUTO_TEST_CASE(_vec_itrs_to_topocen) {
    // test vectors
    std::array<double, 3> n1({1.0, 0.0, 0.0});
    std::array<double, 3> n2({0.0, 1.0, 0.0});
    std::array<double, 3> n3({0.0, 0.0, 1.0});

    //"easy" tests
    // test lat 0 lon 0 -- itrs x
    json json_config = json::parse(default_config_str);
    json_config["telescope"]["origin_itrs_lon_deg"] = 0;
    json_config["telescope"]["origin_itrs_lat_deg"] = 0;
    const CHORDTelescope& tel_x = get_telescope(json_config);
    check_close_vec3d(tel_x.vec_itrs_to_topocen(n1), std::array<double, 3>({0, 0, 1}), 1.0e-14,
                      1.0e-14, "n1_topo - x");
    check_close_vec3d(tel_x.vec_itrs_to_topocen(n2), std::array<double, 3>({1, 0, 0}), 1.0e-14,
                      1.0e-14, "n2_topo - y");
    check_close_vec3d(tel_x.vec_itrs_to_topocen(n3), std::array<double, 3>({0, 1, 0}), 1.0e-14,
                      1.0e-14, "n3_topo - z");

    // test lat 0 lon 90 -- itrs y
    json_config["telescope"]["origin_itrs_lon_deg"] = 90;
    json_config["telescope"]["origin_itrs_lat_deg"] = 0;
    const CHORDTelescope& tel_y = get_telescope(json_config);
    check_close_vec3d(tel_y.vec_itrs_to_topocen(n1), std::array<double, 3>({-1, 0, 0}), 1.0e-14,
                      1.0e-14, "n1_topo - x");
    check_close_vec3d(tel_y.vec_itrs_to_topocen(n2), std::array<double, 3>({0, 0, 1}), 1.0e-14,
                      1.0e-14, "n2_topo - y");
    check_close_vec3d(tel_y.vec_itrs_to_topocen(n3), std::array<double, 3>({0, 1, 0}), 1.0e-14,
                      1.0e-14, "n3_topo - z");

    // test lat 0 lon 90 -- itrs y
    json_config["telescope"]["origin_itrs_lon_deg"] = -90;
    json_config["telescope"]["origin_itrs_lat_deg"] = 90;
    const CHORDTelescope& tel_z = get_telescope(json_config);
    check_close_vec3d(tel_z.vec_itrs_to_topocen(n1), std::array<double, 3>({1, 0, 0}), 1.0e-14,
                      1.0e-14, "n1_topo - x");
    check_close_vec3d(tel_z.vec_itrs_to_topocen(n2), std::array<double, 3>({0, 1, 0}), 1.0e-14,
                      1.0e-14, "n2_topo - y");
    check_close_vec3d(tel_z.vec_itrs_to_topocen(n3), std::array<double, 3>({0, 0, 1}), 1.0e-14,
                      1.0e-14, "n3_topo - z");

    // "harder" test
    double lat_deg = 45;
    double lon_deg = -120;

    double lat = M_PI * lat_deg / 180.0;
    double lon = M_PI * lon_deg / 180.0;

    // Make telescope
    json_config["telescope"]["origin_itrs_lon_deg"] = lon_deg;
    json_config["telescope"]["origin_itrs_lat_deg"] = lat_deg;
    const CHORDTelescope& tel = get_telescope(json_config);


    // in itrs
    //
    // x_topo = cos(lon) * (0, 1, 0) + sin(lon) * (-1, 0, 0)
    //        = (-sin(lon), cos(lon), 0)
    // y_topo = cos(lon) * (-sin(lat), 0, cos(lat)/cos(lon)) + sin(lon) * (0, -sin(lat),
    // cos(lat)/sin(lon))
    //        = (-cos(lon) * sin(lat), -sin(lon) * sin(lat), cos(lat))
    // z_topo = cos(lon) * (cos(lat), 0, sin(lat)/cos(lon)) + sin(lon) * (0, cos(lat),
    // sin(lat)/sin(lon)
    //        = (cos(lon) * cos(lat), sin(lon) * cos(lat), sin(lat))

    // Should just pick out basis vectors.
    check_close_vec3d(tel.vec_itrs_to_topocen(n1),
                      std::array<double, 3>({-sin(lon), -cos(lon) * sin(lat), cos(lon) * cos(lat)}),
                      1.0e-14, 1.0e-14, "n1_topo - x");
    check_close_vec3d(tel.vec_itrs_to_topocen(n2),
                      std::array<double, 3>({cos(lon), -sin(lon) * sin(lat), sin(lon) * cos(lat)}),
                      1.0e-14, 1.0e-14, "n2_topo - y");
    check_close_vec3d(tel.vec_itrs_to_topocen(n3), std::array<double, 3>({0, cos(lat), sin(lat)}),
                      1.0e-14, 1.0e-14, "n3_topo - z");
}

/*
 * @brief   Test itrs <- topocen conversion
 */
BOOST_AUTO_TEST_CASE(_vec_topocen_to_itrs) {
    // test vectors
    std::array<double, 3> n1({1.0, 0.0, 0.0});
    std::array<double, 3> n2({0.0, 1.0, 0.0});
    std::array<double, 3> n3({0.0, 0.0, 1.0});

    //"easy" tests
    // test lat 0 lon 0 -- itrs x
    json json_config = json::parse(default_config_str);
    json_config["telescope"]["origin_itrs_lon_deg"] = 0;
    json_config["telescope"]["origin_itrs_lat_deg"] = 0;
    const CHORDTelescope& tel_x = get_telescope(json_config);
    check_close_vec3d(tel_x.vec_topocen_to_itrs(n1), std::array<double, 3>({0, 1, 0}), 1.0e-14,
                      1.0e-14, "n1_topo - x");
    check_close_vec3d(tel_x.vec_topocen_to_itrs(n2), std::array<double, 3>({0, 0, 1}), 1.0e-14,
                      1.0e-14, "n2_topo - y");
    check_close_vec3d(tel_x.vec_topocen_to_itrs(n3), std::array<double, 3>({1, 0, 0}), 1.0e-14,
                      1.0e-14, "n3_topo - z");

    // test lat 0 lon 90 -- itrs y
    json_config["telescope"]["origin_itrs_lon_deg"] = 90;
    json_config["telescope"]["origin_itrs_lat_deg"] = 0;
    const CHORDTelescope& tel_y = get_telescope(json_config);
    check_close_vec3d(tel_y.vec_topocen_to_itrs(n1), std::array<double, 3>({-1, 0, 0}), 1.0e-14,
                      1.0e-14, "n1_topo - x");
    check_close_vec3d(tel_y.vec_topocen_to_itrs(n2), std::array<double, 3>({0, 0, 1}), 1.0e-14,
                      1.0e-14, "n2_topo - y");
    check_close_vec3d(tel_y.vec_topocen_to_itrs(n3), std::array<double, 3>({0, 1, 0}), 1.0e-14,
                      1.0e-14, "n3_topo - z");

    // test lat 0 lon 90 -- itrs y
    json_config["telescope"]["origin_itrs_lon_deg"] = -90;
    json_config["telescope"]["origin_itrs_lat_deg"] = 90;
    const CHORDTelescope& tel_z = get_telescope(json_config);
    check_close_vec3d(tel_z.vec_topocen_to_itrs(n1), std::array<double, 3>({1, 0, 0}), 1.0e-14,
                      1.0e-14, "n1_topo - x");
    check_close_vec3d(tel_z.vec_topocen_to_itrs(n2), std::array<double, 3>({0, 1, 0}), 1.0e-14,
                      1.0e-14, "n2_topo - y");
    check_close_vec3d(tel_z.vec_topocen_to_itrs(n3), std::array<double, 3>({0, 0, 1}), 1.0e-14,
                      1.0e-14, "n3_topo - z");

    // "harder" test
    double lat_deg = 45;
    double lon_deg = -120;

    double lat = M_PI * lat_deg / 180.0;
    double lon = M_PI * lon_deg / 180.0;

    // Make telescope
    json_config["telescope"]["origin_itrs_lon_deg"] = lon_deg;
    json_config["telescope"]["origin_itrs_lat_deg"] = lat_deg;
    const CHORDTelescope& tel = get_telescope(json_config);


    // in itrs
    //
    // x_topo = cos(lon) * (0, 1, 0) + sin(lon) * (-1, 0, 0)
    //        = (-sin(lon), cos(lon), 0)
    // y_topo = cos(lon) * (-sin(lat), 0, cos(lat)/cos(lon)) + sin(lon) * (0, -sin(lat),
    // cos(lat)/sin(lon))
    //        = (-cos(lon) * sin(lat), -sin(lon) * sin(lat), cos(lat))
    // z_topo = cos(lon) * (cos(lat), 0, sin(lat)/cos(lon)) + sin(lon) * (0, cos(lat),
    // sin(lat)/sin(lon)
    //        = (cos(lon) * cos(lat), sin(lon) * cos(lat), sin(lat))

    // Should just pick out basis vectors.
    check_close_vec3d(tel.vec_topocen_to_itrs(n1), std::array<double, 3>({-sin(lon), cos(lon), 0}),
                      1.0e-14, 1.0e-14, "n1_topo - x");
    check_close_vec3d(tel.vec_topocen_to_itrs(n2),
                      std::array<double, 3>({-cos(lon) * sin(lat), -sin(lon) * sin(lat), cos(lat)}),
                      1.0e-14, 1.0e-14, "n2_topo - y");
    check_close_vec3d(tel.vec_topocen_to_itrs(n3),
                      std::array<double, 3>({cos(lon) * cos(lat), sin(lon) * cos(lat), sin(lat)}),
                      1.0e-14, 1.0e-14, "n3_topo - z");
}

/*
 * @brief   Test R1 rotation
 */
BOOST_AUTO_TEST_CASE(_vec_axes_rotation_R1) {
    // test vectors
    std::array<double, 3> p1({1.0, 0.0, 0.0});
    std::array<double, 3> p2({0.0, 1.0, 0.0});
    std::array<double, 3> p3({0.0, 0.0, 1.0});
    std::array<double, 3> m3({0.0, 0.0, -1.0});

    //"easy" tests
    // test lat 0 lon 0 -- itrs x
    json json_config = json::parse(default_config_str);
    const CHORDTelescope& tel = get_telescope(json_config);

    check_close_vec3d(tel.vec_axes_rotation_R1(p1, 0.0), p1, 1.0e-14, 1.0e-14, "R(x) - x");
    check_close_vec3d(tel.vec_axes_rotation_R1(p2, 0.0), p2, 1.0e-14, 1.0e-14, "R(y) - y");
    check_close_vec3d(tel.vec_axes_rotation_R1(p3, 0.0), p3, 1.0e-14, 1.0e-14, "R(z) - z");
    check_close_vec3d(tel.vec_axes_rotation_R1(p1, 0.5 * M_PI), p1, 1.0e-14, 1.0e-14, "R(x) - x");
    check_close_vec3d(tel.vec_axes_rotation_R1(p2, 0.5 * M_PI), m3, 1.0e-14, 1.0e-14, "R(y) - -z");
    check_close_vec3d(tel.vec_axes_rotation_R1(p3, 0.5 * M_PI), p2, 1.0e-14, 1.0e-14, "R(z) - y");
}

/*
 * @brief   Test R2 rotation
 */
BOOST_AUTO_TEST_CASE(_vec_axes_rotation_R2) {
    // test vectors
    std::array<double, 3> p1({1.0, 0.0, 0.0});
    std::array<double, 3> p2({0.0, 1.0, 0.0});
    std::array<double, 3> p3({0.0, 0.0, 1.0});
    std::array<double, 3> m1({-1.0, 0.0, 0.0});

    //"easy" tests
    // test lat 0 lon 0 -- itrs x
    json json_config = json::parse(default_config_str);
    const CHORDTelescope& tel = get_telescope(json_config);

    check_close_vec3d(tel.vec_axes_rotation_R2(p1, 0.0), p1, 1.0e-14, 1.0e-14, "R(x) - x");
    check_close_vec3d(tel.vec_axes_rotation_R2(p2, 0.0), p2, 1.0e-14, 1.0e-14, "R(y) - y");
    check_close_vec3d(tel.vec_axes_rotation_R2(p3, 0.0), p3, 1.0e-14, 1.0e-14, "R(z) - z");
    check_close_vec3d(tel.vec_axes_rotation_R2(p1, 0.5 * M_PI), p3, 1.0e-14, 1.0e-14, "R(x) - x");
    check_close_vec3d(tel.vec_axes_rotation_R2(p2, 0.5 * M_PI), p2, 1.0e-14, 1.0e-14, "R(y) - -z");
    check_close_vec3d(tel.vec_axes_rotation_R2(p3, 0.5 * M_PI), m1, 1.0e-14, 1.0e-14, "R(z) - y");
}

/*
 * @brief   Test R3 rotation
 */
BOOST_AUTO_TEST_CASE(_vec_axes_rotation_R3) {
    // test vectors
    std::array<double, 3> p1({1.0, 0.0, 0.0});
    std::array<double, 3> p2({0.0, 1.0, 0.0});
    std::array<double, 3> p3({0.0, 0.0, 1.0});
    std::array<double, 3> m2({0.0, -1.0, 0.0});

    //"easy" tests
    // test lat 0 lon 0 -- itrs x
    json json_config = json::parse(default_config_str);
    const CHORDTelescope& tel = get_telescope(json_config);

    check_close_vec3d(tel.vec_axes_rotation_R3(p1, 0.0), p1, 1.0e-14, 1.0e-14, "R(x) - x");
    check_close_vec3d(tel.vec_axes_rotation_R3(p2, 0.0), p2, 1.0e-14, 1.0e-14, "R(y) - y");
    check_close_vec3d(tel.vec_axes_rotation_R3(p3, 0.0), p3, 1.0e-14, 1.0e-14, "R(z) - z");
    check_close_vec3d(tel.vec_axes_rotation_R3(p1, 0.5 * M_PI), m2, 1.0e-14, 1.0e-14, "R(x) - x");
    check_close_vec3d(tel.vec_axes_rotation_R3(p2, 0.5 * M_PI), p1, 1.0e-14, 1.0e-14, "R(y) - -z");
    check_close_vec3d(tel.vec_axes_rotation_R3(p3, 0.5 * M_PI), p3, 1.0e-14, 1.0e-14, "R(z) - y");
}

/*
 * @brief   Test itrs <- cirs conversion
 */
BOOST_AUTO_TEST_CASE(_vec_cirs_to_itrs) {
    // test vectors
    std::array<double, 3> n1({1.0, 0.0, 0.0});
    std::array<double, 3> n2({0.0, 1.0, 0.0});
    std::array<double, 3> n3({0.0, 0.0, 1.0});

    //"easy" tests
    // test lat 0 lon 0 -- itrs x
    json json_config = json::parse(default_config_str);
    const CHORDTelescope& tel = get_telescope(json_config);

    // test EOP
    EOP eop = {.t_inst_ns = 0,
               .t_ut1_ns = 0,
               .delta_UT1_inst = 0,
               .ERA_deg = 0.0,
               .xp_as = 0.0,
               .yp_as = 0.0};

    check_close_vec3d(tel.vec_cirs_to_itrs(n1, eop),
                      cirs_to_itrs(tel, n1, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel0_x - test0_x");
    check_close_vec3d(tel.vec_cirs_to_itrs(n2, eop),
                      cirs_to_itrs(tel, n2, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel0_y - test0_y");
    check_close_vec3d(tel.vec_cirs_to_itrs(n3, eop),
                      cirs_to_itrs(tel, n3, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel0_z - test0_z");

    eop.ERA_deg = -279.6;
    eop.xp_as = 0.0;
    eop.yp_as = 0.0;

    check_close_vec3d(tel.vec_cirs_to_itrs(n1, eop),
                      cirs_to_itrs(tel, n1, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel1_x - test1_x");
    check_close_vec3d(tel.vec_cirs_to_itrs(n2, eop),
                      cirs_to_itrs(tel, n2, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel1_y - test1_y");
    check_close_vec3d(tel.vec_cirs_to_itrs(n3, eop),
                      cirs_to_itrs(tel, n3, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel1_z - test1_z");

    eop.ERA_deg = 0.0;
    eop.xp_as = 123.45;
    eop.yp_as = 0.0;

    check_close_vec3d(tel.vec_cirs_to_itrs(n1, eop),
                      cirs_to_itrs(tel, n1, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel2_x - test2_x");
    check_close_vec3d(tel.vec_cirs_to_itrs(n2, eop),
                      cirs_to_itrs(tel, n2, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel2_y - test2_y");
    check_close_vec3d(tel.vec_cirs_to_itrs(n3, eop),
                      cirs_to_itrs(tel, n3, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel2_z - test2_z");

    eop.ERA_deg = 0.0;
    eop.xp_as = 0.0;
    eop.yp_as = -987.654;

    check_close_vec3d(tel.vec_cirs_to_itrs(n1, eop),
                      cirs_to_itrs(tel, n1, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel3_x - test3_x");
    check_close_vec3d(tel.vec_cirs_to_itrs(n2, eop),
                      cirs_to_itrs(tel, n2, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel3_y - test3_y");
    check_close_vec3d(tel.vec_cirs_to_itrs(n3, eop),
                      cirs_to_itrs(tel, n3, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel3_z - test3_z");

    eop.ERA_deg = 234.56;
    eop.xp_as = 789.012;
    eop.yp_as = 3456.78;

    check_close_vec3d(tel.vec_cirs_to_itrs(n1, eop),
                      cirs_to_itrs(tel, n1, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel4_x - test4_x");
    check_close_vec3d(tel.vec_cirs_to_itrs(n2, eop),
                      cirs_to_itrs(tel, n2, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel4_y - test4_y");
    check_close_vec3d(tel.vec_cirs_to_itrs(n3, eop),
                      cirs_to_itrs(tel, n3, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel4_z - test4_z");
}

/*
 * @brief   Test itrs -> cirs conversion
 */
BOOST_AUTO_TEST_CASE(_vec_itrs_to_cirs) {
    // test vectors
    std::array<double, 3> n1({1.0, 0.0, 0.0});
    std::array<double, 3> n2({0.0, 1.0, 0.0});
    std::array<double, 3> n3({0.0, 0.0, 1.0});

    //"easy" tests
    // test lat 0 lon 0 -- itrs x
    json json_config = json::parse(default_config_str);
    const CHORDTelescope& tel = get_telescope(json_config);

    // test EOP
    EOP eop = {.t_inst_ns = 0,
               .t_ut1_ns = 0,
               .delta_UT1_inst = 0,
               .ERA_deg = 0.0,
               .xp_as = 0.0,
               .yp_as = 0.0};

    check_close_vec3d(tel.vec_itrs_to_cirs(n1, eop),
                      itrs_to_cirs(tel, n1, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel0_x - test0_x");
    check_close_vec3d(tel.vec_itrs_to_cirs(n2, eop),
                      itrs_to_cirs(tel, n2, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel0_y - test0_y");
    check_close_vec3d(tel.vec_itrs_to_cirs(n3, eop),
                      itrs_to_cirs(tel, n3, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel0_z - test0_z");

    eop.ERA_deg = -279.6;
    eop.xp_as = 0.0;
    eop.yp_as = 0.0;

    check_close_vec3d(tel.vec_itrs_to_cirs(n1, eop),
                      itrs_to_cirs(tel, n1, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel1_x - test1_x");
    check_close_vec3d(tel.vec_itrs_to_cirs(n2, eop),
                      itrs_to_cirs(tel, n2, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel1_y - test1_y");
    check_close_vec3d(tel.vec_itrs_to_cirs(n3, eop),
                      itrs_to_cirs(tel, n3, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel1_z - test1_z");

    eop.ERA_deg = 0.0;
    eop.xp_as = 123.45;
    eop.yp_as = 0.0;

    check_close_vec3d(tel.vec_itrs_to_cirs(n1, eop),
                      itrs_to_cirs(tel, n1, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel2_x - test2_x");
    check_close_vec3d(tel.vec_itrs_to_cirs(n2, eop),
                      itrs_to_cirs(tel, n2, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel2_y - test2_y");
    check_close_vec3d(tel.vec_itrs_to_cirs(n3, eop),
                      itrs_to_cirs(tel, n3, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel2_z - test2_z");

    eop.ERA_deg = 0.0;
    eop.xp_as = 0.0;
    eop.yp_as = -987.654;

    check_close_vec3d(tel.vec_itrs_to_cirs(n1, eop),
                      itrs_to_cirs(tel, n1, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel3_x - test3_x");
    check_close_vec3d(tel.vec_itrs_to_cirs(n2, eop),
                      itrs_to_cirs(tel, n2, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel3_y - test3_y");
    check_close_vec3d(tel.vec_itrs_to_cirs(n3, eop),
                      itrs_to_cirs(tel, n3, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel3_z - test3_z");

    eop.ERA_deg = 234.56;
    eop.xp_as = 789.012;
    eop.yp_as = 3456.78;

    check_close_vec3d(tel.vec_itrs_to_cirs(n1, eop),
                      itrs_to_cirs(tel, n1, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel4_x - test4_x");
    check_close_vec3d(tel.vec_itrs_to_cirs(n2, eop),
                      itrs_to_cirs(tel, n2, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel4_y - test4_y");
    check_close_vec3d(tel.vec_itrs_to_cirs(n3, eop),
                      itrs_to_cirs(tel, n3, eop.ERA_deg, eop.xp_as, eop.yp_as), 1.0e-14, 1.0e-14,
                      "tel4_z - test4_z");
}

/*
 * @brief   Test fringestopping phases
 */
BOOST_AUTO_TEST_CASE(_fringestop_phases_1d) {
    /*
     * Thompson, Moran, and Swenson (3ed), Eq. 4.9 gives observed the fringe rate as:
     *
     * dw/dt = -w_e * u * cos(delta),
     *
     * and their visibility is defined as:
     *
     * V = \int_dOmega A(\hat{n}) I(\hat{n}) exp(-2*pi*i * (u, v, w)*\hat{n})
     *
     * Therefore the observed visibility phase evolves as:
     *
     * Thompson, obs -- dphi/dt = -2*pi * dw/dt = 2 * pi * w_e * u * cos(delta)
     *
     * delta = declination of source
     * u = E/W baseline in wavelengths
     * w_e = 7.292115e-5 rad/s = Earth's rotation rate (correcting a typo, the written
     *        value is 7.29115e-15 rad/s.)
     *
     * In CHORD we define the visibility with the opposite sign of the phase w.r.t.
     * Thompson, so we expect:
     *
     * CHORD, obs -- dphi/dt = + 2 * pi * w_e * u * cos(delta)
     *
     * to first order in time.
     */
    double freq_MHz = 1.0;

    double C = 2.99792458e8;

    double lambda = C / (1.0e6 * freq_MHz);

    double w_e = 7.292115e-5; // = 2pi * 1.0027378... / 86400 s

    // target straight overhead
    double target_dec_deg = 45.0;
    double tel_lat_deg = 45.0;

    double dec = target_dec_deg * M_PI / 180;

    dishInfo d0 = dishInfo(0, 0, 0, {0.0, 0.0, 0.0}, 0.0, DishType::ArrayDish, "D00");
    dishInfo d1 = dishInfo(1, 1, 0, {0.0, 0.0, 0.0}, 0.0, DishType::ArrayDish, "D01");
    dishInfo d2 = dishInfo(2, 0, 1, {0.0, 0.0, 0.0}, 0.0, DishType::ArrayDish, "D10");
    dishInfo d3 = dishInfo(3, 2, 2, {-lambda, -lambda, 0.0}, 0.0, DishType::ArrayDish, "D11");

    double t = 1.0; // A short time to get only 1st order effects
    EOP eop0 = eop_null;
    EOP eop = eop_null;
    eop.ERA_deg = t * w_e * 180.0 / M_PI;

    json json_config = json::parse(default_config_str);
    json_config["num_elements"] = 8;
    json_config["num_dishes"] = 4;
    json_config["num_dishes_x"] = 3;
    json_config["num_dishes_y"] = 3;
    json_config["telescope"]["origin_itrs_lat_deg"] = tel_lat_deg;
    json_config["telescope"]["origin_itrs_lon_deg"] = 0.0;
    json_config["telescope"]["dish_separation_x_m"] = lambda;
    json_config["telescope"]["dish_separation_y_m"] = lambda;
    json_config["telescope"]["dish_inputs"] = {d0, d1, d2, d3};
    json_config["telescope"]["dish_elev_axis"] = {1.0, 0.0, 0.0};
    json_config["telescope"]["dish_vert_axis"] = {0.0, 0.0, 1.0};
    json_config["telescope"]["dish_coelev_deg"] = target_dec_deg - tel_lat_deg;
    json_config["telescope"]["grid_x_axis"] = {1.0, 0.0, 0.0};
    json_config["telescope"]["grid_y_axis"] = {0.0, 1.0, 0.0};

    std::vector<std::complex<float>> tel_phases(4, 0);

    float test_phase_val = 2 * M_PI * w_e * t * cos(dec);

    std::vector<float> test_phases({0.0, test_phase_val, 0.0, test_phase_val});

    const CHORDTelescope& tel = get_telescope(json_config);

    tel.fringestop_phases_1d(freq_MHz, eop, eop0, tel_phases);

    for (int i = 0; i < 4; i++) {
        check_close_float(std::norm(tel_phases[i]), 1.0, 1.0e-12, 1.0e-12, "|e^i*phase|", "1");
        check_close_float(atan2(tel_phases[i].imag(), tel_phases[i].real()), test_phases[i], 1.0e-8,
                          1.0e-5, "tel_phase1", "test_phase1");
    }

    // Position the array at a different latitude (tests that we're converting
    // topo -> itrs coords)
    tel_lat_deg = 60.0;
    target_dec_deg = 30.0;
    json_config["telescope"]["origin_itrs_lat_deg"] = tel_lat_deg;
    json_config["telescope"]["dish_coelev_deg"] = target_dec_deg - tel_lat_deg;

    test_phase_val = 2 * M_PI * w_e * t * cos(target_dec_deg * M_PI / 180);
    std::vector<double> test_phases2({0.0, test_phase_val, 0.0, test_phase_val});

    const CHORDTelescope& tel2 = get_telescope(json_config);
    tel2.fringestop_phases_1d(freq_MHz, eop, eop0, tel_phases);

    for (int i = 0; i < 4; i++) {
        check_close_float(std::norm(tel_phases[i]), 1.0, 1.0e-12, 1.0e-12, "|e^i*phase|", "1");
        check_close_float(atan2(tel_phases[i].imag(), tel_phases[i].real()), test_phases2[i],
                          1.0e-7, 1.0e-5, "tel_phase2", "test_phase2");
    }

    // Rotate the dish orientation (tests that we're converting dish -> topo coords)
    double deflection_deg = 20.0;
    double deflection = deflection_deg * M_PI / 180;
    target_dec_deg = 70.0;
    json_config["telescope"]["dish_vert_axis"] = {0.0, sin(deflection), cos(deflection)};
    json_config["telescope"]["dish_coelev_deg"] = target_dec_deg - tel_lat_deg - deflection_deg;

    test_phase_val = 2 * M_PI * w_e * t * cos(target_dec_deg * M_PI / 180);
    std::vector<double> test_phases3({0.0, test_phase_val, 0.0, test_phase_val});

    const CHORDTelescope& tel3 = get_telescope(json_config);
    tel3.fringestop_phases_1d(freq_MHz, eop, eop0, tel_phases);

    for (int i = 0; i < 4; i++) {
        check_close_float(std::norm(tel_phases[i]), 1.0, 1.0e-12, 1.0e-12, "|e^i*phase|", "1");
        check_close_float(atan2(tel_phases[i].imag(), tel_phases[i].real()), test_phases3[i],
                          1.0e-7, 1.0e-5, "tel_phase3", "test_phase3");
    }

    // Rotate the grid orientation (tests that we've converting tel -> topo
    // coords)
    //    |2 3      3 1|
    //    |0 1  --> 2 0|
    // ---+---      ---+---
    json_config["telescope"]["grid_x_axis"] = {0.0, 1.0, 0.0};
    json_config["telescope"]["grid_y_axis"] = {-1.0, 0.0, 0.0};

    test_phase_val = 2 * M_PI * w_e * t * cos(target_dec_deg * M_PI / 180);
    std::vector<double> test_phases4({0.0, 0.0, -test_phase_val, -test_phase_val});

    const CHORDTelescope& tel4 = get_telescope(json_config);
    tel4.fringestop_phases_1d(freq_MHz, eop, eop0, tel_phases);

    for (int i = 0; i < 4; i++) {
        check_close_float(std::norm(tel_phases[i]), 1.0, 1.0e-12, 1.0e-12, "|e^i*phase|", "1");
        check_close_float(atan2(tel_phases[i].imag(), tel_phases[i].real()), test_phases4[i],
                          1.0e-7, 1.0e-5, "tel_phase4", "test_phase4");
    }
}

/*
 * @brief   Test local ERA calculation
 */
BOOST_AUTO_TEST_CASE(_eral) {
    BOOST_TEST_MESSAGE(fmt::format("Testing telescope position."));

    double lon = -123.45678;
    double lat = 49.321123;

    json json_config = json::parse(default_config_str);
    json_config["telescope"]["origin_itrs_lat_deg"] = lat;
    json_config["telescope"]["origin_itrs_lon_deg"] = lon;

    const CHORDTelescope& tel = get_telescope(json_config);

    EOP eop = eop_null;
    eop.ERA_deg = 3.45678;

    BOOST_CHECK_EQUAL(tel.get_ERAL_deg(eop), 240.0);
}
