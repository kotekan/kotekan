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
#define DSET_ID 772977253621153375UL

using json = nlohmann::json;

using namespace std::string_literals;

BOOST_FIXTURE_TEST_CASE( _ask_broker_for_ancestors, CompareCTypes ) {
    __log_level = 4;
    __enable_syslog = 0;

    json json_config;
    json_config["use_dataset_broker"] = true;
    json_config["ds_broker_port"] = 12050;
    json_config["ds_broker_host"] = "127.0.0.1";
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