#define BOOST_TEST_MODULE "test_dataset_broker_producer"

#include "restClient.hpp"
#include "restServer.hpp"
#include "visCompression.hpp"
#include "visUtil.hpp"

#include "json.hpp"

#include <boost/test/included/unit_test.hpp>
#include <fstream>
#include <iostream>
#include <string>

// the code to test:
#include "datasetManager.hpp"

using kotekan::Config;

using json = nlohmann::json;

using namespace std::string_literals;

BOOST_AUTO_TEST_CASE(_dataset_manager_general) {
    BOOST_CHECK(boost::unit_test::framework::master_test_suite().argc == 2);
    int broker_port = atoi(boost::unit_test::framework::master_test_suite().argv[1]);
    BOOST_CHECK(broker_port);

    _global_log_level = 5;
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
    std::vector<input_ctype> inputs = {input_ctype(1, "1"), input_ctype(2, "2"),
                                       input_ctype(3, "3")};
    std::vector<prod_ctype> prods = {{1, 1}, {2, 2}, {3, 3}};
    std::vector<std::pair<uint32_t, freq_ctype>> freqs = {
        {1, {1.1, 1}}, {2, {2, 2.2}}, {3, {3, 3}}};

    // Force the dM to update while it knows of nothing yet.
    restReply reply = restClient::instance().make_request_blocking(
        "/dataset-manager/force-update", {}, "127.0.0.1", kotekan::restServer::instance().port);
    BOOST_CHECK(reply.first == true);
    BOOST_CHECK(reply.second == "");

    std::pair<state_id_t, const inputState*> input_state =
        dm.add_state(std::make_unique<inputState>(
            inputs, std::make_unique<prodState>(prods, std::make_unique<freqState>(freqs))));

    dset_id_t init_ds_id = dm.add_dataset(input_state.first);

    // register same state
    std::pair<state_id_t, const inputState*> input_state2 =
        dm.add_state(std::make_unique<inputState>(
            inputs, std::make_unique<prodState>(prods, std::make_unique<freqState>(freqs))));
    // register new dataset with the twin state
    dset_id_t ds_id = dm.add_dataset(init_ds_id, input_state2.first);

    // write ID to disk for producer2
    std::ofstream o("DS_ID.txt");
    o << ds_id;
    o.close();

    for (auto s : dm.states())
        std::cout << s.first << " - " << s.second->data_to_json().dump() << std::endl;

    for (auto s : dm.datasets())
        std::cout << s.second.state() << " - " << s.second.base_dset() << std::endl;

    usleep(1000000);
}


BOOST_AUTO_TEST_CASE(_dataset_manager_state_known_to_broker) {
    BOOST_CHECK(boost::unit_test::framework::master_test_suite().argc == 2);
    int broker_port = atoi(boost::unit_test::framework::master_test_suite().argv[1]);
    BOOST_CHECK(broker_port);

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

    std::pair<state_id_t, const inputState*> input_state =
        dm.add_state(std::make_unique<inputState>(
            inputs, std::make_unique<prodState>(prods, std::make_unique<freqState>(freqs))));

    dm.add_dataset(input_state.first);

    // wait a bit, to make sure we see errors in any late callbacks
    usleep(500000);
}

BOOST_AUTO_TEST_CASE(_dataset_manager_second_root) {
    BOOST_CHECK(boost::unit_test::framework::master_test_suite().argc == 2);
    int broker_port = atoi(boost::unit_test::framework::master_test_suite().argv[1]);
    BOOST_CHECK(broker_port);

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

    std::pair<state_id_t, const inputState*> input_state =
        dm.add_state(std::make_unique<inputState>(
            inputs, std::make_unique<prodState>(prods, std::make_unique<freqState>(freqs))));

    dset_id_t second_root = dm.add_dataset(input_state.first);

    // write ID to disk for consumer
    std::ofstream o("SECOND_ROOT.txt");
    o << second_root;
    o.close();

    // wait a bit, to make sure we see errors in any late callbacks
    usleep(1000000);
}
