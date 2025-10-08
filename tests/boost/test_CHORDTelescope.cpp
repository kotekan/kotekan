#define BOOST_TEST_MODULE "test_CHORDTelescope"

#include "CHORDTelescope.hpp"
#include "Config.hpp"
#include "configUpdater.hpp"
#include "Telescope.hpp"
#include "restServer.hpp"
#include "errors.h" // _global_log_level

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
using kotekan::restServer;
using kotekan::configUpdater;

BOOST_AUTO_TEST_CASE(_instrument_position) {
    BOOST_TEST_MESSAGE(fmt::format("Testing telescope position."));

    double lon = -119.621;
    double lat = 49.321;

    json json_config_tel;
    json_config_tel["name"] = "CHORDTelescope";
    json_config_tel["inst_long_deg"] = -119.621;
    json_config_tel["inst_lat_deg"] = 49.321;
    json_config_tel["inst_alt_deg"] = 100.0;
    json_config_tel["inst_grid_x_axis"] = {1.0, 0.0, 0.0};
    json_config_tel["inst_grid_y_axis"] = {0.0, 1.0, 0.0};
    json_config_tel["inst_dish_alt_axis"] = {1.0, 0.0, 0.0};
    json_config_tel["inst_dish_vert_axis"] = {0.0, 0.0, 1.0};
    json_config_tel["dish_positions"] = {
        {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}};
    json_config_tel["updatable_config"] = "/earth_rotation_data";

    json json_config_eop_1;
    json_config_eop_1["time_inst_ns"] = 0;
    json_config_eop_1["delta_UT1_inst"] = 0.1;
    json_config_eop_1["x_pm"] = 0.2;
    json_config_eop_1["y_pm"] = 0.3;

    json json_config_erd;
    json_config_erd["kotekan_update_endpoint"] = "json";
    json_config_erd["earth_orientation_parameter_table"] = {json_config_eop_1};

    json json_config;
    json_config["telescope"] = json_config_tel;
    json_config["log_level"] = "DEBUG";
    json_config["earth_rotation_data"] = json_config_erd;
   
    Config conf;
    std::cout << "Setting config" << std::endl;
    conf.update_config(json_config);
    std::cout << "Config set!" << std::endl;

    std::cout << "Making the configUpdater!" << std::endl;
    configUpdater& config_updater = configUpdater::instance();
    config_updater.apply_config(conf);

    //std::cout << conf.get_full_config_json().dump(4) << std::endl;

    std::cout << "Building telescope." << std::endl;
    Telescope::instance(conf);

    std::cout << "Getting telescope." << std::endl;
    const CHORDTelescope& tel = Telescope::instance().cast<CHORDTelescope>();
    std::cout << "Have telescope, let's go!" << std::endl;

    BOOST_CHECK_EQUAL(tel.get_inst_long_deg(), lon);
    BOOST_CHECK_EQUAL(tel.get_inst_lat_deg(), lat);
}
