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
#define DSET_ID 14090398099494767442UL

using json = nlohmann::json;

using namespace std::string_literals;

BOOST_FIXTURE_TEST_CASE( _ask_broker_for_ancestors, CompareCTypes ) {
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

    // generate datasets:
    std::vector<input_ctype> inputs = {input_ctype(1, "1"),
                                       input_ctype(3, "3")};
    std::vector<prod_ctype> prods = {{1, 1},
                                     {2, 2},
                                     {3, 3}};
    std::vector<std::pair<uint32_t, freq_ctype>> freqs = {{1, {1.1, 1}},
                                                          {3, {3, 3}}};

    auto i = dm.closest_ancestor_of_type<inputState>(DSET_ID);
    check_equal(i.second->get_inputs(), inputs);

    auto f = dm.closest_ancestor_of_type<freqState>(DSET_ID);
    check_equal(f.second->get_freqs(), freqs);

    auto p = dm.closest_ancestor_of_type<prodState>(DSET_ID);
    check_equal(p.second->get_prods(), prods);

    // wait a bit, to make sure we see errors in any late callbacks
    usleep(1000000);
}