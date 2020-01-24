#define BOOST_TEST_MODULE "test_dataset_broker_consumer"

#include "Config.hpp"         // for Config
#include "Hash.hpp"           // for operator<<
#include "dataset.hpp"        // for dataset
#include "datasetManager.hpp" // for datasetManager, state_id_t, dset_id_t
#include "datasetState.hpp"   // for freqState, inputState, prodState, datasetState
#include "errors.h"           // for __enable_syslog, _global_log_level
#include "restClient.hpp"     // for restClient, restClient::restReply
#include "restServer.hpp"     // for restServer
#include "test_utils.hpp"     // for CompareCTypes
#include "visUtil.hpp"        // for input_ctype, prod_ctype, freq_ctype

#include "json.hpp" // for basic_json<>::object_t, basic_json<>::value...

#include <algorithm>                         // for max
#include <boost/test/included/unit_test.hpp> // for master_test_suite, BOOST_PP_IIF_1, BOOST_CHECK
#include <exception>                         // for exception
#include <iostream>                          // for operator<<, ostream, endl, basic_ostream, cout
#include <map>                               // for map
#include <stdexcept>                         // for out_of_range
#include <stdint.h>                          // for uint32_t
#include <stdlib.h>                          // for atoi
#include <string>                            // for operator<<, allocator, string, getline, ope...
#include <unistd.h>                          // for usleep
#include <utility>                           // for pair
#include <vector>                            // for vector


#define WAIT_TIME 4000000

using kotekan::Config;

using json = nlohmann::json;

using namespace std::string_literals;

dset_id_t second_root_update;

int read_from_argv() {
    // The randomly chosen port for the dataset broker is passed to this test as a command line
    // argument.
    // At some point boost stopped requiring the `--` to pass command line arguments, so we
    // should be ready for both.
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

BOOST_FIXTURE_TEST_CASE(_ask_broker_for_ancestors, CompareCTypes) {
    int broker_port = read_from_argv();

    _global_log_level = 4;
    __enable_syslog = 0;

    // We have to start the restServer here, because the datasetManager uses it.
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
    std::vector<input_ctype> inputs = {input_ctype(1, "1"), input_ctype(3, "3")};
    std::vector<prod_ctype> prods = {{1, 1}, {2, 2}, {3, 3}};
    std::vector<std::pair<uint32_t, freq_ctype>> freqs = {{1, {1.1, 1}}, {3, {3, 3}}};

    std::cout << dm.summary() << std::endl;

    for (auto s : dm.states())
        std::cout << s.first << " - " << s.second->data_to_json().dump() << std::endl;

    for (auto s : dm.datasets())
        std::cout << s.second.state() << " - " << s.second.base_dset() << std::endl;

    // read ds_id from file
    std::string line;
    std::ifstream file("DS_ID2.txt");
    if (file.is_open()) {
        if (!std::getline(file, line))
            std::cout << "Unable to read from file DS_ID2.txt\n";
        file.close();
    } else
        std::cout << "Unable to open file DS_ID2.txt\n";
    dset_id_t ds_id;
    ds_id.set_from_string(line);

    auto i = dm.dataset_state<inputState>(ds_id);
    check_equal(i->get_inputs(), inputs);

    auto f = dm.dataset_state<freqState>(ds_id);
    check_equal(f->get_freqs(), freqs);

    // for this ancestor it will have to ask the broker again!
    auto p = dm.dataset_state<prodState>(ds_id);
    std::cout << dm.summary() << std::endl;

    for (auto s : dm.states())
        std::cout << s.first << " - " << s.second->data_to_json().dump() << std::endl;

    for (auto s : dm.datasets())
        std::cout << s.second.state() << " - " << s.second.base_dset() << std::endl;

    check_equal(p->get_prods(), prods);

    // wait a bit, to make sure we see errors in any late callbacks
    usleep(WAIT_TIME);
}

BOOST_AUTO_TEST_CASE(_dataset_manager_second_root_update) {
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
    std::vector<input_ctype> inputs = {input_ctype(1, "4")};
    std::vector<prod_ctype> prods = {{4, 1}};
    std::vector<std::pair<uint32_t, freq_ctype>> freqs = {{4, {1.1, 1}}};

    std::vector<state_id_t> states;
    states.push_back(dm.create_state<freqState>(freqs).first);
    states.push_back(dm.create_state<prodState>(prods).first);
    states.push_back(dm.create_state<inputState>(inputs).first);

    // read second_root from file
    std::string line;
    std::ifstream f("SECOND_ROOT.txt");
    if (f.is_open()) {
        if (!std::getline(f, line))
            std::cout << "Unable to read from file SECOND_ROOT.txt\n";
        f.close();
    } else
        std::cout << "Unable to open file SECOND_ROOT.txt\n";
    dset_id_t second_root;
    second_root.set_from_string(line);

    second_root_update = dm.add_dataset(states, second_root);

    // wait a bit, to make sure we see errors in any late callbacks
    usleep(WAIT_TIME);
}

BOOST_FIXTURE_TEST_CASE(_ask_broker_for_second_root, CompareCTypes) {
    int broker_port = read_from_argv();

    _global_log_level = 4;
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

    std::cout << dm.summary() << std::endl;

    for (auto s : dm.states())
        std::cout << s.first << " - " << s.second->data_to_json().dump() << std::endl;

    for (auto s : dm.datasets())
        std::cout << s.second.state() << " - " << s.second.base_dset() << std::endl;

    // read second_root from file
    std::string line;
    std::ifstream file("SECOND_ROOT.txt");
    if (file.is_open()) {
        if (!std::getline(file, line))
            std::cout << "Unable to read from file SECOND_ROOT.txt\n";
        file.close();
    } else
        std::cout << "Unable to open file SECOND_ROOT.txt\n";
    dset_id_t second_root;
    second_root.set_from_string(line);

    auto i = dm.dataset_state<inputState>(second_root);
    check_equal(i->get_inputs(), inputs);

    auto f = dm.dataset_state<freqState>(second_root);
    check_equal(f->get_freqs(), freqs);

    auto p = dm.dataset_state<prodState>(second_root);
    check_equal(p->get_prods(), prods);

    std::cout << dm.summary() << std::endl;

    for (auto s : dm.states())
        std::cout << s.first << " - " << s.second->data_to_json().dump() << std::endl;

    for (auto s : dm.datasets())
        std::cout << s.second.state() << " - " << s.second.base_dset() << std::endl;

    // wait a bit, to make sure we see errors in any late callbacks
    usleep(WAIT_TIME);
}

BOOST_FIXTURE_TEST_CASE(_ask_broker_for_second_root_update, CompareCTypes) {
    int broker_port = read_from_argv();

    _global_log_level = 4;
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
    std::vector<input_ctype> inputs = {input_ctype(1, "4")};
    std::vector<prod_ctype> prods = {{4, 1}};
    std::vector<std::pair<uint32_t, freq_ctype>> freqs = {{4, {1.1, 1}}};

    std::cout << dm.summary() << std::endl;

    for (auto s : dm.states())
        std::cout << s.first << " - " << s.second->data_to_json().dump() << std::endl;

    for (auto s : dm.datasets())
        std::cout << s.second.state() << " - " << s.second.base_dset() << std::endl;

    auto i = dm.dataset_state<inputState>(second_root_update);
    check_equal(i->get_inputs(), inputs);

    auto f = dm.dataset_state<freqState>(second_root_update);
    check_equal(f->get_freqs(), freqs);

    auto p = dm.dataset_state<prodState>(second_root_update);
    check_equal(p->get_prods(), prods);

    std::cout << dm.summary() << std::endl;

    for (auto s : dm.states())
        std::cout << s.first << " - " << s.second->data_to_json().dump() << std::endl;

    for (auto s : dm.datasets())
        std::cout << s.second.state() << " - " << s.second.base_dset() << std::endl;

    // Force the dM to register everything again.
    restClient::restReply reply = restClient::instance().make_request_blocking(
        "/dataset-manager/force-update", {}, "127.0.0.1", kotekan::restServer::instance().port);
    BOOST_CHECK(reply.first == true);
    BOOST_CHECK(reply.second == "");


    // wait a bit, to make sure we see errors in any late callbacks
    usleep(WAIT_TIME);
}
