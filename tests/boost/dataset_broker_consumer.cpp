#define BOOST_TEST_MODULE "test_dataset_broker_consumer"

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

dset_id_t second_root_update;

BOOST_FIXTURE_TEST_CASE(_ask_broker_for_ancestors, CompareCTypes) {
    _global_log_level = 4;
    __enable_syslog = 0;

    // We have to start the restServer here, because the datasetManager uses it.
    kotekan::restServer::instance().start("127.0.0.1");

    json json_config;
    json json_config_dm;
    json_config_dm["use_dataset_broker"] = true;
    json_config["dataset_manager"] = json_config_dm;

    Config conf;
    conf.update_config(json_config);
    datasetManager& dm = datasetManager::instance(conf);

    // generate datasets:
    std::vector<input_ctype> inputs = {input_ctype(1, "1"), input_ctype(3, "3")};
    std::vector<prod_ctype> prods = {{1, 1}, {2, 2}, {3, 3}};
    std::vector<std::pair<uint32_t, freq_ctype>> freqs = {{1, {1.1, 1}}, {3, {3, 3}}};

    std::cout << dm.summary() << std::endl;

    for (auto s : dm.states())
        std::cout << s.first << " - " << s.second->data_to_json().dump() << std::endl;

    for (auto s : dm.datasets())
        std::cout << s.second.state() << " - " << s.second.base_dset() << std::endl;

    // read ds_id from file
    std::string line;
    std::ifstream file("DS_ID2.txt");
    if (file.is_open()) {
        if (!std::getline(file, line))
            std::cout << "Unable to read from file DS_ID2.txt\n";
        file.close();
    } else
        std::cout << "Unable to open file DS_ID2.txt\n";
    dset_id_t ds_id;
    std::stringstream(line) >> ds_id;

    auto i = dm.dataset_state<inputState>(ds_id);
    check_equal(i->get_inputs(), inputs);

    auto f = dm.dataset_state<freqState>(ds_id);
    check_equal(f->get_freqs(), freqs);

    // for this ancestor it will have to ask the broker again!
    auto p = dm.dataset_state<prodState>(ds_id);
    std::cout << dm.summary() << std::endl;

    for (auto s : dm.states())
        std::cout << s.first << " - " << s.second->data_to_json().dump() << std::endl;

    for (auto s : dm.datasets())
        std::cout << s.second.state() << " - " << s.second.base_dset() << std::endl;

    check_equal(p->get_prods(), prods);

    // wait a bit, to make sure we see errors in any late callbacks
    usleep(500000);
}

BOOST_AUTO_TEST_CASE(_dataset_manager_second_root_update) {
    _global_log_level = 5;
    __enable_syslog = 0;

    json json_config;
    json json_config_dm;
    json_config_dm["use_dataset_broker"] = true;
    json_config["dataset_manager"] = json_config_dm;

    Config conf;
    conf.update_config(json_config);
    datasetManager& dm = datasetManager::instance(conf);

    // generate datasets:
    std::vector<input_ctype> inputs = {input_ctype(1, "4")};
    std::vector<prod_ctype> prods = {{4, 1}};
    std::vector<std::pair<uint32_t, freq_ctype>> freqs = {{4, {1.1, 1}}};

    std::pair<state_id_t, const inputState*> input_state =
        dm.add_state(std::make_unique<inputState>(
            inputs, std::make_unique<prodState>(prods, std::make_unique<freqState>(freqs))));

    // read second_root from file
    std::string line;
    std::ifstream f("SECOND_ROOT.txt");
    if (f.is_open()) {
        if (!std::getline(f, line))
            std::cout << "Unable to read from file SECOND_ROOT.txt\n";
        f.close();
    } else
        std::cout << "Unable to open file SECOND_ROOT.txt\n";
    dset_id_t second_root;
    std::stringstream(line) >> second_root;

    second_root_update = dm.add_dataset(second_root, input_state.first);

    // wait a bit, to make sure we see errors in any late callbacks
    usleep(1000000);
}

BOOST_FIXTURE_TEST_CASE(_ask_broker_for_second_root, CompareCTypes) {
    _global_log_level = 4;
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

    std::cout << dm.summary() << std::endl;

    for (auto s : dm.states())
        std::cout << s.first << " - " << s.second->data_to_json().dump() << std::endl;

    for (auto s : dm.datasets())
        std::cout << s.second.state() << " - " << s.second.base_dset() << std::endl;

    // read second_root from file
    std::string line;
    std::ifstream file("SECOND_ROOT.txt");
    if (file.is_open()) {
        if (!std::getline(file, line))
            std::cout << "Unable to read from file SECOND_ROOT.txt\n";
        file.close();
    } else
        std::cout << "Unable to open file SECOND_ROOT.txt\n";
    dset_id_t second_root;
    std::stringstream(line) >> second_root;

    auto i = dm.dataset_state<inputState>(second_root);
    check_equal(i->get_inputs(), inputs);

    auto f = dm.dataset_state<freqState>(second_root);
    check_equal(f->get_freqs(), freqs);

    auto p = dm.dataset_state<prodState>(second_root);
    check_equal(p->get_prods(), prods);

    std::cout << dm.summary() << std::endl;

    for (auto s : dm.states())
        std::cout << s.first << " - " << s.second->data_to_json().dump() << std::endl;

    for (auto s : dm.datasets())
        std::cout << s.second.state() << " - " << s.second.base_dset() << std::endl;

    // wait a bit, to make sure we see errors in any late callbacks
    usleep(500000);
}

BOOST_FIXTURE_TEST_CASE(_ask_broker_for_second_root_update, CompareCTypes) {
    _global_log_level = 4;
    __enable_syslog = 0;

    json json_config;
    json json_config_dm;
    json_config_dm["use_dataset_broker"] = true;
    json_config["dataset_manager"] = json_config_dm;

    Config conf;
    conf.update_config(json_config);
    datasetManager& dm = datasetManager::instance(conf);

    // generate datasets:
    std::vector<input_ctype> inputs = {input_ctype(1, "4")};
    std::vector<prod_ctype> prods = {{4, 1}};
    std::vector<std::pair<uint32_t, freq_ctype>> freqs = {{4, {1.1, 1}}};

    std::cout << dm.summary() << std::endl;

    for (auto s : dm.states())
        std::cout << s.first << " - " << s.second->data_to_json().dump() << std::endl;

    for (auto s : dm.datasets())
        std::cout << s.second.state() << " - " << s.second.base_dset() << std::endl;

    auto i = dm.dataset_state<inputState>(second_root_update);
    check_equal(i->get_inputs(), inputs);

    auto f = dm.dataset_state<freqState>(second_root_update);
    check_equal(f->get_freqs(), freqs);

    auto p = dm.dataset_state<prodState>(second_root_update);
    check_equal(p->get_prods(), prods);

    std::cout << dm.summary() << std::endl;

    for (auto s : dm.states())
        std::cout << s.first << " - " << s.second->data_to_json().dump() << std::endl;

    for (auto s : dm.datasets())
        std::cout << s.second.state() << " - " << s.second.base_dset() << std::endl;

    // Force the dM to register everything again.
    restReply reply = restClient::instance().make_request_blocking("/dataset-manager/force-update");
    BOOST_CHECK(reply.first == true);
    BOOST_CHECK(reply.second == "");


    // wait a bit, to make sure we see errors in any late callbacks
    usleep(600000);
}
