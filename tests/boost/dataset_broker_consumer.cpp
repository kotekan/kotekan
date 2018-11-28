#define BOOST_TEST_MODULE "test_dataset_broker_consumer"

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

// dataset id from producer2
// TODO: pass this here via a file instead
#define DSET_ID 11388855756098689401UL

using json = nlohmann::json;

using namespace std::string_literals;

BOOST_FIXTURE_TEST_CASE( _ask_broker_for_ancestors, CompareCTypes ) {
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
    std::vector<input_ctype> inputs = {input_ctype(1, "1"),
                                       input_ctype(3, "3")};
    std::vector<prod_ctype> prods = {{1, 1},
                                     {2, 2},
                                     {3, 3}};
    std::vector<std::pair<uint32_t, freq_ctype>> freqs = {{1, {1.1, 1}},
                                                          {3, {3, 3}}};

    std::cout << dm.summary() << std::endl;

    for (auto s : dm.states())
        std::cout << s.first << " - " << s.second->data_to_json().dump()
                  << std::endl;

    for (auto s : dm.datasets())
        std::cout << s.second.state() << " - " << s.second.base_dset() <<
                     std::endl;

    auto i = dm.dataset_state<inputState>(DSET_ID);
    check_equal(i->get_inputs(), inputs);

    auto f = dm.dataset_state<freqState>(DSET_ID);
    check_equal(f->get_freqs(), freqs);

    // for this ancestor it will have to ask the broker again!
    auto p = dm.dataset_state<prodState>(DSET_ID);
    std::cout << dm.summary() << std::endl;

    for (auto s : dm.states())
        std::cout << s.first << " - " << s.second->data_to_json().dump()
                  << std::endl;

    for (auto s : dm.datasets())
        std::cout << s.second.state() << " - " << s.second.base_dset() <<
                     std::endl;

    check_equal(p->get_prods(), prods);

    // wait a bit, to make sure we see errors in any late callbacks
    usleep(700000);
}