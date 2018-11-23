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
// TODO: pass this here via a file instead
#define DSET_ID 10856314535876447850UL

using json = nlohmann::json;

using namespace std::string_literals;

BOOST_FIXTURE_TEST_CASE( _dataset_manager_general, CompareCTypes ) {
    __log_level = 5;
    __enable_syslog = 0;

    json json_config;
    json_config["use_dataset_broker"] = true;

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

    auto freq_state = dm.dataset_state<freqState>(DSET_ID);
    check_equal(old_freqs, freq_state->get_freqs());

    auto prod_state = dm.dataset_state<prodState>(DSET_ID);
    check_equal(old_prods, prod_state->get_prods());

    auto input_state = dm.dataset_state<inputState>(DSET_ID);
    check_equal(old_inputs, input_state->get_inputs());

    // change states:
    std::vector<input_ctype> new_inputs = {input_ctype(1, "1"),
                                           input_ctype(3, "3")};
    std::vector<std::pair<uint32_t, freq_ctype>> new_freqs = {{1, {1.1, 1}},
                                                              {3, {3, 3}}};


    // add new input state and new freq state
    std::pair<state_id_t, const inputState*> new_input_state =
            dm.add_state(std::make_unique<inputState>
                         (new_inputs, std::make_unique<freqState>(new_freqs)));

    dm.add_dataset(DSET_ID, new_input_state.first);

    std::cout << dm.summary() << std::endl;

    for (auto s : dm.states())
        std::cout << s.first << " - " << s.second->data_to_json().dump()
                  << std::endl;

    for (auto s : dm.datasets())
        std::cout << s.second.state() << " - " << s.second.base_dset() <<
                     std::endl;

    // wait a bit, to make sure we see errors in any late callbacks
    usleep(700000);
}