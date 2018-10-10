#define BOOST_TEST_MODULE "test_dataset_broker_producer2"

#include <boost/test/included/unit_test.hpp>
#include <string>
#include <iostream>
#include "json.hpp"
#include "visUtil.hpp"
#include "restClient.hpp"
#include "visCompression.hpp"
#include "test_utils.hpp"

// the code to test:
#include "datasetManager.hpp"

// dataset id from producer
#define DSET_ID 10014618525185544200UL

using json = nlohmann::json;

using namespace std::string_literals;

BOOST_FIXTURE_TEST_CASE( _dataset_manager_general, CompareCTypes ) {
    __log_level = 4;
    __enable_syslog = 0;

    json json_config;
    json_config["use_ds_broker"] = true;
    json_config["ds_broker_port"] = 12050;
    json_config["ds_broker_host"] = "localhost";
    json_config["register_state_path"] = "/register-state";
    json_config["send_state_path"] = "/send-state";
    json_config["register_dataset_path"] = "/register-dataset";
    json_config["request_ancestors_path"] = "/request-ancestors";

    datasetManager& dm = datasetManager::instance();
    Config conf;
    conf.update_config(json_config);
    dm.apply_config(conf);

    // reproduce states:
    std::vector<input_ctype> old_inputs = {input_ctype(1, "1"),
                                           input_ctype(2, "2"),
                                           input_ctype(3, "3")};
    std::vector<prod_ctype> old_prods = {{1, 1},
                                         {2, 2},
                                         {3, 3}};
    std::vector<std::pair<uint32_t, freq_ctype>> old_freqs = {{1, {1.1, 1}},
                                                              {2, {2, 2.2}},
                                                              {3, {3, 3}}};

    auto freq_state = dm.closest_ancestor_of_type<freqState>(DSET_ID);
    check_equal(old_freqs, freq_state.second->get_freqs());

    auto prod_state = dm.closest_ancestor_of_type<prodState>(DSET_ID);
    check_equal(old_prods, prod_state.second->get_prods());

    auto input_state = dm.closest_ancestor_of_type<inputState>(DSET_ID);
    check_equal(old_inputs, input_state.second->get_inputs());

    // change states:
    std::vector<input_ctype> new_inputs = {input_ctype(1, "1"),
                                           input_ctype(3, "3")};
    std::vector<std::pair<uint32_t, freq_ctype>> new_freqs = {{1, {1.1, 1}},
                                                              {3, {3, 3}}};


    std::pair<state_id_t, const inputState*> new_input_state =
            dm.add_state(std::make_unique<inputState>(new_inputs,
                                           make_unique<prodState>(old_prods,
                                           make_unique<freqState>(new_freqs))));

    dset_id_t init_ds_id = dm.add_dataset(dataset(new_input_state.first,
                                                  DSET_ID));

    std::cout << dm.summary() << std::endl;

    for (auto s : dm.states())
        std::cout << s.first << " - " << s.second->data_to_json().dump()
                  << std::endl;

    for (auto s : dm.datasets())
        std::cout << s.second.state() << " - " << s.second.base_dset() <<
                     std::endl;

    for (auto s : dm.ancestors(init_ds_id))
        std::cout << s.first << " - " << s.second->data_to_json().dump()
                  << std::endl;

    // wait a bit, to make sure we see errors in any late callbacks
    usleep(1000000);
}