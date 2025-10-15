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

const std::string default_config_str = "{\
\"type\": \"config\",\
\"log_level\": \"debug\",\
\"telescope\": {\
    \"name\": \"CHORDTelescope\",\
    \"inst_long_deg\":  -119.621,\
    \"inst_lat_deg\":   49.321,\
    \"inst_alt_deg\":   100.0,\
    \"inst_grid_x_axis\":   [1.0, 0.0, 0.0],\
    \"inst_grid_y_axis\":   [0.0, 1.0, 0.0],\
    \"inst_dish_alt_axis\": [1.0, 0.0, 0.0],\
    \"inst_dish_vert_axis\":    [0.0, 0.0, 1.0],\
    \"dish_positions\": [\
        [0.0, 0.0, 0.0],\
        [1.0, 0.0, 0.0],\
        [0.0, 1.0, 0.0],\
        [1.0, 1.0, 0.0],\
        [100.0, 0.0, 0.0],\
        [0.0, 100.0, 0.0],\
        [100.0, 100.0, 0.0]],\
    \"require_gps\":        false,\
    \"updatable_config\":   \"/earth_rotation_data\"\
    },\
\"gps_time\": {\
    \"frame0_nano\": 1761926400000000000\
    },\
\"earth_rotation_data\": {\
    \"kotekan_update_endpoint\": \"json\",\
    \"earth_orientation_parameter_table\": [\
        {\
            \"time_inst_ns\": 1761883200000000000,\
            \"delta_UT1_inst\": 0.0,\
            \"x_pm\": 0.0,\
            \"y_pm\": 0.0\
        }, {\
            \"time_inst_ns\": 1761969600000000000,\
            \"delta_UT1_inst\": 0.0,\
            \"x_pm\": 0.0,\
            \"y_pm\": 0.0\
        }]\
    }\
}";

const CHORDTelescope& get_telescope(json &json_config) {
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

    double alt = 160;

    json json_config = json::parse(default_config_str);
    json_config["telescope"]["inst_alt_deg"] = alt;
   
    const CHORDTelescope& tel = get_telescope(json_config);

    BOOST_CHECK_EQUAL(tel.get_inst_alt_deg(), alt);
}
