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

const std::string default_tel_config_str = R"tel_config_str({
"type": "config",
"log_level": "debug",
"telescope": {
    "name": "fake",
    "require_eop": false,
    "eop_updatable_config":   "/earth_rotation_data"
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
 * TESTS
 *
 ******************/


BOOST_AUTO_TEST_CASE(_name_tel) {
    json json_config = json::parse(default_tel_config_str);

    const Telescope& tel = get_telescope(json_config);

    BOOST_CHECK_EQUAL(tel.get_name(), "fake");
}
