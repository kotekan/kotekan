#define BOOST_TEST_MODULE "test_dataset_broker_producer"

#include "Config.hpp"         // for Config
#include "Hash.hpp"           // for operator<<
#include "dataset.hpp"        // for dataset
#include "datasetManager.hpp" // for state_id_t, datasetManager, dset_id_t
#include "datasetState.hpp"   // for freqState, inputState, prodState, datasetState
#include "errors.h"           // for __enable_syslog, _global_log_level
#include "restClient.hpp"     // for restClient, restClient::restReply
#include "restServer.hpp"     // for restServer
#include "visUtil.hpp"        // for input_ctype, prod_ctype, freq_ctype

#include "json.hpp" // for basic_json<>::object_t, basic_json<>::value...

#include <algorithm>                         // for max
#include <boost/test/included/unit_test.hpp> // for master_test_suite, BOOST_PP_IIF_1, BOOST_CHECK
#include <chrono>                            // for milliseconds
#include <iostream>                          // for ofstream, operator<<, basic_ostream, ostream
#include <map>                               // for map
#include <stdint.h>                          // for uint32_t
#include <stdlib.h>                          // for atoi
#include <string>                            // for allocator, string, operator<<, operator==
#include <thread>                            // for sleep_for
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

BOOST_AUTO_TEST_CASE(_dataset_manager_general) {
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

    // generate datasets:
    std::vector<input_ctype> inputs = {input_ctype(1, "1"), input_ctype(2, "2"),
                                       input_ctype(3, "3")};
    std::vector<prod_ctype> prods = {{1, 1}, {2, 2}, {3, 3}};
    std::vector<std::pair<uint32_t, freq_ctype>> freqs = {
        {1, {1.1, 1}}, {2, {2, 2.2}}, {3, {3, 3}}};

    // wait for the restServer to start
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Force the dM to update while it knows of nothing yet.
    restClient::restReply reply = restClient::instance().make_request_blocking(
        "/dataset-manager/force-update", {}, "127.0.0.1", kotekan::restServer::instance().port);
    BOOST_CHECK(reply.first == true);
    BOOST_CHECK(reply.second == "");

    std::vector<state_id_t> states1;
    states1.push_back(dm.create_state<freqState>(freqs).first);
    states1.push_back(dm.create_state<prodState>(prods).first);
    states1.push_back(dm.create_state<inputState>(inputs).first);

    dset_id_t init_ds_id = dm.add_dataset(states1);

    // register same state
    std::vector<state_id_t> states2;
    states2.push_back(dm.create_state<freqState>(freqs).first);
    states2.push_back(dm.create_state<prodState>(prods).first);
    states2.push_back(dm.create_state<inputState>(inputs).first);

    // register new dataset with the twin state
    dset_id_t ds_id = dm.add_dataset(states2, init_ds_id);

    // write ID to disk for producer2
    std::ofstream o("DS_ID.txt");
    o << ds_id.to_string();
    o.close();

    for (auto s : dm.states())
        std::cout << s.first << " - " << s.second->data_to_json().dump() << std::endl;

    for (auto s : dm.datasets())
        std::cout << s.second.state() << " - " << s.second.base_dset() << std::endl;

    usleep(WAIT_TIME);
}


BOOST_AUTO_TEST_CASE(_dataset_manager_state_known_to_broker) {
    int broker_port = read_from_argv();

    _global_log_level = 5;
    __enable_syslog = 0;

    json json_config;
    json json_config_dm;
    json_config_dm["use_dataset_broker"] = true;
    json_config_dm["ds_broker_port"] = broker_port;
    json_config["dataset_manager"] = json_config_dm;

    Config conf;
    conf.update_config(json_config);
    datasetManager& dm = datasetManager::instance(conf);

    // generate datasets:
    std::vector<input_ctype> inputs = {input_ctype(1, "1"), input_ctype(2, "2"),
                                       input_ctype(3, "3")};
    std::vector<prod_ctype> prods = {{1, 1}, {2, 2}, {3, 3}};
    std::vector<std::pair<uint32_t, freq_ctype>> freqs = {
        {1, {1.1, 1}}, {2, {2, 2.2}}, {3, {3, 3}}};

    std::vector<state_id_t> states1;
    states1.push_back(dm.create_state<freqState>(freqs).first);
    states1.push_back(dm.create_state<prodState>(prods).first);
    states1.push_back(dm.create_state<inputState>(inputs).first);

    dm.add_dataset(states1);

    // wait a bit, to make sure we see errors in any late callbacks
    usleep(WAIT_TIME);
}

BOOST_AUTO_TEST_CASE(_dataset_manager_second_root) {
    int broker_port = read_from_argv();

    _global_log_level = 5;
    __enable_syslog = 0;

    json json_config;
    json json_config_dm;
    json_config_dm["use_dataset_broker"] = true;
    json_config_dm["ds_broker_port"] = broker_port;
    json_config["dataset_manager"] = json_config_dm;

    Config conf;
    conf.update_config(json_config);
    datasetManager& dm = datasetManager::instance(conf);

    // generate datasets:
    std::vector<input_ctype> inputs = {input_ctype(1, "4"), input_ctype(3, "3")};
    std::vector<prod_ctype> prods = {{4, 1}, {2, 2}, {3, 3}};
    std::vector<std::pair<uint32_t, freq_ctype>> freqs = {{4, {1.1, 1}}, {3, {3, 3}}};

    std::vector<state_id_t> states1;
    states1.push_back(dm.create_state<freqState>(freqs).first);
    states1.push_back(dm.create_state<prodState>(prods).first);
    states1.push_back(dm.create_state<inputState>(inputs).first);

    dset_id_t second_root = dm.add_dataset(states1);

    // write ID to disk for consumer
    std::ofstream o("SECOND_ROOT.txt");
    o << second_root.to_string();
    o.close();

    // wait a bit, to make sure we see errors in any late callbacks
    usleep(WAIT_TIME);
}
