#define BOOST_TEST_MODULE "test_dataset_broker_producer2"

#include "Config.hpp"         // for Config
#include "Hash.hpp"           // for operator<<
#include "dataset.hpp"        // for dataset
#include "datasetManager.hpp" // for state_id_t, datasetManager, dset_id_t
#include "datasetState.hpp"   // for freqState, inputState, prodState, datasetState
#include "errors.h"           // for __enable_syslog, _global_log_level
#include "restServer.hpp"     // for restServer
#include "test_utils.hpp"     // for CompareCTypes
#include "visUtil.hpp"        // for input_ctype, prod_ctype, freq_ctype

#include "json.hpp" // for basic_json<>::object_t, basic_json<>::value...

#include <algorithm>                         // for copy
#include <boost/test/included/unit_test.hpp> // for master_test_suite, master_test_suite_t, BOO...
#include <exception>                         // for exception
#include <iostream>                          // for operator<<, ostream, endl, basic_ostream, cout
#include <map>                               // for map
#include <stdexcept>                         // for out_of_range
#include <stdint.h>                          // for uint32_t
#include <stdlib.h>                          // for atoi
#include <string>                            // for string, operator<<, getline, string_literals
#include <unistd.h>                          // for usleep
#include <utility>                           // for pair
#include <vector>                            // for vector


#define WAIT_TIME 4000000

using kotekan::Config;

using json = nlohmann::json;

using namespace std::string_literals;

int read_from_argv() {
    // The randomly chosen port for the dataset broker is passed to this test as a command line
    // argument.
    // At some point boost stopped requiring the `--` to pass command line arguments, so we
    // should be ready for both `--` being there or not...
    BOOST_CHECK(boost::unit_test::framework::master_test_suite().argc >= 2);
    int broker_port;
    if (!std::string("--").compare(boost::unit_test::framework::master_test_suite().argv[1])) {
        BOOST_CHECK(boost::unit_test::framework::master_test_suite().argc == 3);
        broker_port = atoi(boost::unit_test::framework::master_test_suite().argv[2]);
    } else {
        BOOST_CHECK(boost::unit_test::framework::master_test_suite().argc == 2);
        broker_port = atoi(boost::unit_test::framework::master_test_suite().argv[1]);
    }
    BOOST_CHECK(broker_port);
    return broker_port;
}

BOOST_FIXTURE_TEST_CASE(_dataset_manager_general, CompareCTypes) {
    int broker_port = read_from_argv();

    _global_log_level = 5;
    __enable_syslog = 0;

    // We have to start the restServer here, because the datasetManager uses it (for forced-update).
    kotekan::restServer::instance().start("127.0.0.1", 0);

    json json_config;
    json json_config_dm;
    json_config_dm["use_dataset_broker"] = true;
    json_config_dm["ds_broker_port"] = broker_port;
    json_config["dataset_manager"] = json_config_dm;

    Config conf;
    conf.update_config(json_config);
    datasetManager& dm = datasetManager::instance(conf);

    // reproduce states:
    std::vector<input_ctype> old_inputs = {input_ctype(1, "1"), input_ctype(2, "2"),
                                           input_ctype(3, "3")};
    std::vector<prod_ctype> old_prods = {{1, 1}, {2, 2}, {3, 3}};
    std::vector<std::pair<uint32_t, freq_ctype>> old_freqs = {
        {1, {1.1, 1}}, {2, {2, 2.2}}, {3, {3, 3}}};

    // read ds_id from file
    std::string line;
    std::ifstream f("DS_ID.txt");
    if (f.is_open()) {
        if (!std::getline(f, line))
            std::cout << "Unable to read from file DS_ID.txt\n";
        f.close();
    } else
        std::cout << "Unable to open file DS_D.txt\n";
    dset_id_t ds_id;
    ds_id.set_from_string(line);

    // Add a meaningless state in here. This is designed to trigger a failure
    // case in older versions where if the local manager already knew about a
    // state, it didn't bother checking to see if it knew about earlier states
    // auto state_id = dm.create_state<flagState>("flag_id1").first;
    // ds_id = dm.add_dataset(state_id, ds_id);

    auto freq_state = dm.dataset_state<freqState>(ds_id);
    check_equal(old_freqs, freq_state->get_freqs());

    auto prod_state = dm.dataset_state<prodState>(ds_id);
    check_equal(old_prods, prod_state->get_prods());

    auto input_state = dm.dataset_state<inputState>(ds_id);
    check_equal(old_inputs, input_state->get_inputs());

    // change states:
    std::vector<input_ctype> new_inputs = {input_ctype(1, "1"), input_ctype(3, "3")};
    std::vector<std::pair<uint32_t, freq_ctype>> new_freqs = {{1, {1.1, 1}}, {3, {3, 3}}};

    // add new input state and new freq state
    std::pair<state_id_t, const freqState*> new_freq_state = dm.create_state<freqState>(new_freqs);
    std::pair<state_id_t, const inputState*> new_input_state =
        dm.create_state<inputState>(new_inputs);

    dset_id_t ds_id2 = dm.add_dataset({new_freq_state.first, new_input_state.first}, ds_id);

    // write ID to disk for consumer
    std::ofstream o("DS_ID2.txt");
    o << ds_id2.to_string();
    o.close();

    std::cout << dm.summary() << std::endl;

    for (auto s : dm.states())
        std::cout << s.first << " - " << s.second->data_to_json().dump() << std::endl;

    for (auto s : dm.datasets())
        std::cout << s.second.state() << " - " << s.second.base_dset() << std::endl;

    // wait a bit, to make sure we see errors in any late callbacks
    usleep(WAIT_TIME);
}
