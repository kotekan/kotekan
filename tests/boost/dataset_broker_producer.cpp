#define BOOST_TEST_MODULE "test_dataset_broker_producer"

#include <inttypes.h>
#include "gateSpec.hpp"
#include "restClient.hpp"
#include "visCompression.hpp"
#include "visUtil.hpp"

#include "json.hpp"

#include <boost/test/included/unit_test.hpp>
#include <iostream>
#include <string>

// the code to test:
#include "datasetManager.hpp"
#include "gateSpec.hpp"

using json = nlohmann::json;

using namespace std::string_literals;

BOOST_AUTO_TEST_CASE( _dataset_manager_general ) {
    __log_level = 4;
    __enable_syslog = 0;

    json json_config;
    json json_config_dm;
    json_config_dm["use_dataset_broker"] = true;
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

    dset_id_t init_ds_id = dm.add_dataset(input_state.first);

    // register same state
    std::pair<state_id_t, const inputState*> input_state2 =
        dm.add_state(std::make_unique<inputState>(
            inputs, std::make_unique<prodState>(prods, std::make_unique<freqState>(freqs))));
    // register new dataset with the twin state
    dset_id_t ds_id2 = dm.add_dataset(init_ds_id, input_state2.first);

    // register a pulsarGatingState to test polymorphic dM
    pulsarSpec pspec = pulsarSpec("test_pulsar");
    auto pspec2 = gateSpec::create("pulsar", "test_pulsar");
    INFO("TEST: Created pulsar spec: name: %s type: %s", pspec2->name().c_str(), pspec2->type().c_str());

    std::pair<state_id_t, const pulsarGatingState*>pstate =
            dm.add_state(std::make_unique<pulsarGatingState>(pspec));
    dset_id_t ds_id3 = dm.add_dataset(ds_id2, pstate.first);

    INFO("TEST: Registered dataset 0x%" PRIx64 " (%zu) with pulsar gating state.",
         ds_id3, ds_id3);

    for (auto s : dm.states())
        INFO("TEST: 0x%" PRIx64 " - %s",
             s.first, s.second->data_to_json().dump().c_str());

    for (auto s : dm.datasets())
        INFO("TEST: 0x%" PRIx64 " - 0x%" PRIx64 , s.second.state(),
                     s.second.base_dset());

    usleep(3000000);
}


BOOST_AUTO_TEST_CASE( _dataset_manager_state_known_to_broker ) {
    __log_level = 4;
    __enable_syslog = 0;

    json json_config;
    json json_config_dm;
    json_config_dm["use_dataset_broker"] = true;
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
    __log_level = 5;
    __enable_syslog = 0;

    json json_config;
    json json_config_dm;
    json_config_dm["use_dataset_broker"] = true;
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

    dm.add_dataset(input_state.first);

    // wait a bit, to make sure we see errors in any late callbacks
    usleep(1000000);
}
