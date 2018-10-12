#define BOOST_TEST_MODULE "test_dataset_broker_producer"

#include <boost/test/included/unit_test.hpp>
#include <string>
#include <iostream>
#include "json.hpp"
#include "visUtil.hpp"
#include "restClient.hpp"
#include "visCompression.hpp"

// the code to test:
#include "datasetManager.hpp"

using json = nlohmann::json;

using namespace std::string_literals;

BOOST_AUTO_TEST_CASE( _dataset_manager_general ) {
    __log_level = 4;
    __enable_syslog = 0;

    json json_config;
    json_config["use_ds_broker"] = true;
    json_config["ds_broker_port"] = 12050;
    json_config["ds_broker_host"] = "localhost";
    json_config["register_state_path"] = "/register-state";
    json_config["send_state_path"] = "/send-state";
    json_config["register_dataset_path"] = "/register-dataset";
    json_config["request_ancestor_path"] = "/request-ancestor";

    datasetManager& dm = datasetManager::instance();
    Config conf;
    conf.update_config(json_config);
    dm.apply_config(conf);

    // generate datasets:
    std::vector<input_ctype> inputs = {input_ctype(1, "1"),
                                       input_ctype(2, "2"),
                                       input_ctype(3, "3")};
    std::vector<prod_ctype> prods = {{1, 1},
                                     {2, 2},
                                     {3, 3}};
    std::vector<std::pair<uint32_t, freq_ctype>> freqs = {{1, {1.1, 1}},
                                                          {2, {2, 2.2}},
                                                          {3, {3, 3}}};

    std::pair<state_id_t, const inputState*> input_state =
            dm.add_state(std::make_unique<inputState>(inputs,
                                               make_unique<prodState>(prods,
                                               make_unique<freqState>(freqs))));

    dset_id_t init_ds_id = dm.add_dataset(dataset(input_state.first, 0, true));

    // register same state
    std::pair<state_id_t, const inputState*>input_state2 =
            dm.add_state(std::make_unique<inputState>(inputs,
                              make_unique<prodState>(prods,
                              make_unique<freqState>(freqs))));
    // register new dataset with the twin state
    dset_id_t init_ds_id2 = dm.add_dataset(dataset(input_state2.first,
                                                   init_ds_id));

    std::cout << dm.summary() << std::endl;

    for (auto s : dm.states())
        std::cout << s.first << " - " << s.second->data_to_json().dump()
                  << std::endl;

    for (auto s : dm.datasets())
        std::cout << s.second.state() << " - " << s.second.base_dset() <<
                     std::endl;

    for (auto s : dm.ancestors(init_ds_id2))
        std::cout << s.first << " - " << s.second->data_to_json().dump()
                  << std::endl;

    usleep(1000000);
}


BOOST_AUTO_TEST_CASE( _dataset_manager_state_known_to_broker ) {
    __log_level = 4;
    __enable_syslog = 0;

    json json_config;
    json_config["use_ds_broker"] = true;
    json_config["ds_broker_port"] = 12050;
    json_config["ds_broker_host"] = "localhost";
    json_config["register_state_path"] = "/register-state";
    json_config["send_state_path"] = "/send-state";
    json_config["register_dataset_path"] = "/register-dataset";
    json_config["request_ancestor_path"] = "/request-ancestor";

    datasetManager& dm = datasetManager::instance();
    Config conf;
    conf.update_config(json_config);
    dm.apply_config(conf);

    // generate datasets:
    std::vector<input_ctype> inputs = {input_ctype(1, "1"),
                                       input_ctype(2, "2"),
                                       input_ctype(3, "3")};
    std::vector<prod_ctype> prods = {{1, 1},
                                     {2, 2},
                                     {3, 3}};
    std::vector<std::pair<uint32_t, freq_ctype>> freqs = {{1, {1.1, 1}},
                                                          {2, {2, 2.2}},
                                                          {3, {3, 3}}};

    std::pair<state_id_t, const inputState*> input_state =
            dm.add_state(std::make_unique<inputState>(inputs,
                                               make_unique<prodState>(prods,
                                               make_unique<freqState>(freqs))));

    dm.add_dataset(dataset(input_state.first, 0, true));

    // wait a bit, to make sure we see errors in any late callbacks
    usleep(500000);
}