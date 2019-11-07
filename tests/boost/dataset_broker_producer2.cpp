#define BOOST_TEST_MODULE "test_dataset_broker_producer2"

#include "restClient.hpp"
#include "restServer.hpp"
#include "test_utils.hpp"
#include "visCompression.hpp"
#include "visUtil.hpp"

#include "json.hpp"

#include <boost/test/included/unit_test.hpp>
#include <iostream>
#include <string>

// the code to test:
#include "datasetManager.hpp"

using kotekan::Config;

using json = nlohmann::json;

using namespace std::string_literals;

BOOST_FIXTURE_TEST_CASE(_dataset_manager_general, CompareCTypes) {

    // The randomly chosen port for the dataset broker is passed to this test as a command line
    // argument.
    BOOST_CHECK(boost::unit_test::framework::master_test_suite().argc == 2);
    int broker_port = atoi(boost::unit_test::framework::master_test_suite().argv[1]);
    BOOST_CHECK(broker_port);

    _global_log_level = 5;
    __enable_syslog = 0;

    // We have to start the restServer here, because the datasetManager uses it (for forced-update).
    kotekan::restServer::instance().start("127.0.0.1", 0);

    json json_config;
    json json_config_dm;
    json_config_dm["use_dataset_broker"] = true;
    json_config_dm["ds_broker_port"] = broker_port;
    json_config["dataset_manager"] = json_config_dm;

    Config conf;
    conf.update_config(json_config);
    datasetManager& dm = datasetManager::instance(conf);

    // reproduce states:
    std::vector<input_ctype> old_inputs = {input_ctype(1, "1"), input_ctype(2, "2"),
                                           input_ctype(3, "3")};
    std::vector<prod_ctype> old_prods = {{1, 1}, {2, 2}, {3, 3}};
    std::vector<std::pair<uint32_t, freq_ctype>> old_freqs = {
        {1, {1.1, 1}}, {2, {2, 2.2}}, {3, {3, 3}}};

    // read ds_id from file
    std::string line;
    std::ifstream f("DS_ID.txt");
    if (f.is_open()) {
        if (!std::getline(f, line))
            std::cout << "Unable to read from file DS_ID.txt\n";
        f.close();
    } else
        std::cout << "Unable to open file DS_D.txt\n";
    dset_id_t ds_id;
    std::stringstream(line) >> ds_id;

    auto freq_state = dm.dataset_state<freqState>(ds_id);
    check_equal(old_freqs, freq_state->get_freqs());

    auto prod_state = dm.dataset_state<prodState>(ds_id);
    check_equal(old_prods, prod_state->get_prods());

    auto input_state = dm.dataset_state<inputState>(ds_id);
    check_equal(old_inputs, input_state->get_inputs());

    // change states:
    std::vector<input_ctype> new_inputs = {input_ctype(1, "1"), input_ctype(3, "3")};
    std::vector<std::pair<uint32_t, freq_ctype>> new_freqs = {{1, {1.1, 1}}, {3, {3, 3}}};

    // add new input state and new freq state
    std::pair<state_id_t, const freqState*> new_freq_state = dm.create_state<freqState>(new_freqs);
    std::pair<state_id_t, const inputState*> new_input_state =
        dm.create_state<inputState>(new_inputs);

    dset_id_t ds_id2 = dm.add_dataset({new_freq_state.first, new_input_state.first}, ds_id);

    // write ID to disk for consumer
    std::ofstream o("DS_ID2.txt");
    o << ds_id2;
    o.close();

    std::cout << dm.summary() << std::endl;

    for (auto s : dm.states())
        std::cout << s.first << " - " << s.second->data_to_json().dump() << std::endl;

    for (auto s : dm.datasets())
        std::cout << s.second.state() << " - " << s.second.base_dset() << std::endl;

    // wait a bit, to make sure we see errors in any late callbacks
    usleep(2000000);
}
