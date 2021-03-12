#define BOOST_TEST_MODULE "test_datasetManager"

#include "Config.hpp"         // for Config
#include "Hash.hpp"           // for operator<<
#include "dataset.hpp"        // for dataset
#include "datasetManager.hpp" // for state_id_t, datasetManager, dset_id_t
#include "datasetState.hpp"   // for inputState, prodState, freqState, datasetState
#include "errors.h"           // for _global_log_level, __enable_syslog
#include "test_utils.hpp"     // for CompareCTypes
#include "visUtil.hpp"        // for input_ctype, prod_ctype, freq_ctype

#include <algorithm>                         // for max
#include <boost/test/included/unit_test.hpp> // for BOOST_PP_IIF_1, BOOST_PP_IIF_0, BOOST_PP_BO...
#include <exception>                         // for exception
#include <iostream>                          // for endl, operator<<, ostream, basic_ostream, cout
#include <json.hpp>                          // for basic_json<>::object_t, basic_json<>::value...
#include <map>                               // for map
#include <memory>                            // for allocator, make_unique, unique_ptr
#include <stdexcept>                         // for out_of_range
#include <stdint.h>                          // for uint32_t
#include <string>                            // for string, operator<<, string_literals
#include <utility>                           // for pair
#include <vector>                            // for vector


using kotekan::Config;

using json = nlohmann::json;

using namespace std::string_literals;

// The datasetManager uses the restServer, but it's not started by this test.
// That leads to a warnings message from the restServer on exit we can ignore.


BOOST_FIXTURE_TEST_CASE(_general, CompareCTypes) {
    _global_log_level = 5;
    __enable_syslog = 0;
    json json_config;
    json json_config_dm;
    json_config_dm["use_dataset_broker"] = false;
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
    dset_id_t init_ds_id = dm.add_dataset(states1);

    inputs = {input_ctype(1, "1"), input_ctype(2, "2")};
    prods = {{1, 1}, {2, 2}};
    freqs = {{1, {1.1, 1}}, {2, {2, 2.2}}};

    std::vector<state_id_t> states2;
    states2.push_back(dm.create_state<freqState>(freqs).first);
    states2.push_back(dm.create_state<prodState>(prods).first);
    states2.push_back(dm.create_state<inputState>(inputs).first);
    dset_id_t init_ds_id2 = dm.add_dataset(states2, init_ds_id);


    // transform that data:
    std::vector<input_ctype> new_inputs = {input_ctype(1, "1"), input_ctype(2, "2"),
                                           input_ctype(3, "3"), input_ctype(4, "4")};
    std::vector<prod_ctype> new_prods = {{1, 1}, {2, 2}, {3, 3}, {4, 4}};
    std::vector<std::pair<uint32_t, freq_ctype>> new_freqs = {
        {1, {1.1, 1}}, {2, {2, 2.2}}, {3, {3, 3}}, {4, {4, 4}}};
    const inputState* old_state = dm.dataset_state<inputState>(init_ds_id2);
    BOOST_CHECK(old_state);
    const std::vector<input_ctype>& old_inputs = old_state->get_inputs();
    check_equal(old_inputs, inputs);
    const prodState* old_state2 = dm.dataset_state<prodState>(init_ds_id2);
    BOOST_CHECK(old_state2);
    const std::vector<prod_ctype>& old_prods = old_state2->get_prods();
    check_equal(old_prods, prods);

    const freqState* old_state3 = dm.dataset_state<freqState>(init_ds_id2);
    BOOST_CHECK(old_state3);
    const std::vector<std::pair<uint32_t, freq_ctype>>& old_freqs = old_state3->get_freqs();
    check_equal(old_freqs, freqs);

    std::vector<state_id_t> transformed_states;
    transformed_states.push_back(dm.create_state<freqState>(new_freqs).first);
    transformed_states.push_back(dm.create_state<prodState>(new_prods).first);
    transformed_states.push_back(dm.create_state<inputState>(new_inputs).first);
    dset_id_t transformed_ds_id = dm.add_dataset(transformed_states, init_ds_id2);

    // get state
    const inputState* final_state = dm.dataset_state<inputState>(transformed_ds_id);
    const std::vector<input_ctype>& final_inputs = final_state->get_inputs();
    check_equal(new_inputs, final_inputs);

    std::cout << dm.summary() << std::endl;

    for (auto s : dm.states())
        std::cout << s.first << " - " << s.second->data_to_json().dump() << std::endl;

    for (auto s : dm.datasets())
        std::cout << s.second.state() << " - " << s.second.base_dset() << std::endl;
}

BOOST_AUTO_TEST_CASE(_serialization_input) {
    _global_log_level = 4;
    json json_config;
    json json_config_dm;
    json_config_dm["use_dataset_broker"] = false;
    json_config["dataset_manager"] = json_config_dm;
    Config conf;
    conf.update_config(json_config);
    datasetManager& dm = datasetManager::instance(conf);

    // serialize and deserialize
    std::vector<input_ctype> inputs = {input_ctype(1, "1"), input_ctype(2, "2"),
                                       input_ctype(3, "3")};
    std::vector<prod_ctype> prods = {{1, 1}, {2, 2}, {3, 3}};
    std::vector<std::pair<uint32_t, freq_ctype>> freqs = {
        {1, {1.1, 1}}, {2, {2, 2.2}}, {3, {3, 3}}};
    std::pair<state_id_t, const inputState*> input_state =
        dm.add_state(std::make_unique<inputState>(inputs));
    json j = input_state.second->to_json();
    state_uptr s = datasetState::from_json(j);
    json j2 = s->to_json();
    BOOST_CHECK_EQUAL(j, j2);

    // serialize 2 states with the same data
    std::pair<state_id_t, const inputState*> input_state3 =
        dm.add_state(std::make_unique<inputState>(inputs));
    json j3 = input_state3.second->to_json();
    BOOST_CHECK_EQUAL(j, j3);

    // check that different data leads to different json
    std::vector<input_ctype> diff_inputs = {input_ctype(1, "1"), input_ctype(7, "7")};
    std::pair<state_id_t, const inputState*> diff_input_state =
        dm.add_state(std::make_unique<inputState>(diff_inputs));
    json j_diff = diff_input_state.second->to_json();
    BOOST_CHECK(j != j_diff);
}

BOOST_AUTO_TEST_CASE(_serialization_prod) {
    _global_log_level = 4;
    json json_config;
    json json_config_dm;
    json_config_dm["use_dataset_broker"] = false;
    json_config["dataset_manager"] = json_config_dm;
    Config conf;
    conf.update_config(json_config);
    datasetManager& dm = datasetManager::instance(conf);

    // serialize and deserialize
    std::vector<prod_ctype> prods = {{2, 2}, {1, 1}, {3, 3}};
    std::pair<state_id_t, const prodState*> state =
        dm.add_state(std::make_unique<prodState>(prods));
    json j = state.second->to_json();
    state_uptr s = datasetState::from_json(j);
    json j2 = s->to_json();
    BOOST_CHECK_EQUAL(j, j2);

    // serialize 2 states with the same data
    std::pair<state_id_t, const prodState*> state3 =
        dm.add_state(std::make_unique<prodState>(prods));
    json j3 = state3.second->to_json();
    BOOST_CHECK_EQUAL(j, j3);

    // check that different data leads to different json
    std::vector<prod_ctype> diff_prods = {{1, 2}, {7, 3}};
    std::pair<state_id_t, const prodState*> diff_state =
        dm.add_state(std::make_unique<prodState>(diff_prods));
    json j_diff = diff_state.second->to_json();
    BOOST_CHECK(j != j_diff);
}

BOOST_AUTO_TEST_CASE(_serialization_freq) {
    _global_log_level = 4;
    json json_config;
    json json_config_dm;
    json_config_dm["use_dataset_broker"] = false;
    json_config["dataset_manager"] = json_config_dm;
    Config conf;
    conf.update_config(json_config);
    datasetManager& dm = datasetManager::instance(conf);

    // serialize and deserialize
    std::vector<std::pair<uint32_t, freq_ctype>> freqs = {
        {1, {1.1, 1}}, {2, {2, 2.2}}, {3, {3, 3}}};
    std::pair<state_id_t, const freqState*> state =
        dm.add_state(std::make_unique<freqState>(freqs));
    json j = state.second->to_json();
    state_uptr s = datasetState::from_json(j);
    json j2 = s->to_json();
    BOOST_CHECK_EQUAL(j, j2);

    // serialize 2 states with the same data
    std::pair<state_id_t, const freqState*> state3 =
        dm.add_state(std::make_unique<freqState>(freqs));
    json j3 = state3.second->to_json();
    BOOST_CHECK_EQUAL(j, j3);

    // check that different data leads to different json
    std::vector<std::pair<uint32_t, freq_ctype>> diff_freqs = {{1, {1.2, 2}}, {3, {7, 3}}};
    std::pair<state_id_t, const freqState*> diff_state =
        dm.add_state(std::make_unique<freqState>(diff_freqs));
    json j_diff = diff_state.second->to_json();
    BOOST_CHECK(j != j_diff);
}

BOOST_AUTO_TEST_CASE(_no_state_of_type_found) {
    _global_log_level = 4;
    json json_config;
    json json_config_dm;
    json_config_dm["use_dataset_broker"] = false;
    json_config["dataset_manager"] = json_config_dm;
    Config conf;
    conf.update_config(json_config);
    datasetManager& dm = datasetManager::instance(conf);

    std::vector<input_ctype> inputs = {input_ctype(1, "1")};
    std::pair<state_id_t, const inputState*> input_state =
        dm.add_state(std::make_unique<inputState>(inputs));
    dset_id_t init_ds_id = dm.add_dataset(input_state.first);

    const prodState* not_found = dm.dataset_state<prodState>(init_ds_id);

    BOOST_CHECK_EQUAL(not_found, // == nullptr
                      static_cast<decltype(not_found)>(nullptr));
}

BOOST_FIXTURE_TEST_CASE(_equal_states, CompareCTypes) {
    _global_log_level = 4;
    json json_config;
    json json_config_dm;
    json_config_dm["use_dataset_broker"] = false;
    json_config["dataset_manager"] = json_config_dm;
    Config conf;
    conf.update_config(json_config);
    datasetManager& dm = datasetManager::instance(conf);

    // inputStates
    std::vector<input_ctype> inputs = {input_ctype(24, "4"), input_ctype(11, "19")};
    std::pair<state_id_t, const inputState*> input_state =
        dm.add_state(std::make_unique<inputState>(inputs));
    BOOST_CHECK_EQUAL(input_state.second->to_json().dump(),
                      std::make_unique<inputState>(inputs)->to_json().dump());
}
