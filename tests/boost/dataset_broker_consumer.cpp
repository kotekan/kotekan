#define BOOST_TEST_MODULE "test_dataset_broker_consumer"

#include <boost/test/included/unit_test.hpp>
#include <string>
#include <inttypes.h>
#include "json.hpp"
#include "visUtil.hpp"
#include "restClient.hpp"
#include "visCompression.hpp"
#include "test_utils.hpp"

// the code to test:
#include "datasetManager.hpp"

// dataset id from producer2
// TODO: pass this here via a file instead
#define DSET_ID 5984564855775988381UL
#define SECOND_ROOT 1355729954233464875UL
#define SECOND_ROOT_UPDATE 13121011210482281245UL
#define GATING_DS 3662088113467306579UL

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

    INFO("%s", dm.summary().c_str());

    for (auto s : dm.states())
        INFO("0x%" PRIx64 " - %s", s.first,
             s.second->data_to_json().dump().c_str());

    for (auto s : dm.datasets())
        INFO("0x%" PRIx64 " - 0x%" PRIx64 , s.second.state(),
                     s.second.base_dset());

    auto i = dm.dataset_state<inputState>(DSET_ID);
    check_equal(i->get_inputs(), inputs);

    auto f = dm.dataset_state<freqState>(DSET_ID);
    check_equal(f->get_freqs(), freqs);

    // for this ancestor it will have to ask the broker again!
    auto p = dm.dataset_state<prodState>(DSET_ID);
    INFO("%s", dm.summary().c_str());

    for (auto s : dm.states())
        INFO("0x%" PRIx64 " - %s",
             s.first, s.second->data_to_json().dump().c_str());

    for (auto s : dm.datasets())
        INFO("0x%" PRIx64 " - 0x%" PRIx64 , s.second.state(),
                     s.second.base_dset());

    check_equal(p->get_prods(), prods);

    // wait a bit, to make sure we see errors in any late callbacks
    usleep(500000);
}

BOOST_AUTO_TEST_CASE( _dataset_manager_second_root_update ) {
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
    std::vector<input_ctype> inputs = {input_ctype(1, "4")};
    std::vector<prod_ctype> prods = {{4, 1}};
    std::vector<std::pair<uint32_t, freq_ctype>> freqs = {{4, {1.1, 1}}};

    std::pair<state_id_t, const inputState*> input_state =
            dm.add_state(std::make_unique<inputState>
                         (inputs, std::make_unique<prodState>(prods,
                          std::make_unique<freqState>(freqs))));

    dm.add_dataset(SECOND_ROOT, input_state.first);

    // wait a bit, to make sure we see errors in any late callbacks
    usleep(1000000);
}

BOOST_FIXTURE_TEST_CASE( _ask_broker_for_second_root, CompareCTypes ) {
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
    std::vector<input_ctype> inputs = {input_ctype(1, "4"),
                                       input_ctype(3, "3")};
    std::vector<prod_ctype> prods = {{4, 1},
                                     {2, 2},
                                     {3, 3}};
    std::vector<std::pair<uint32_t, freq_ctype>> freqs = {{4, {1.1, 1}},
                                                          {3, {3, 3}}};

    INFO("%s", dm.summary().c_str());

    for (auto s : dm.states())
        INFO("0x%" PRIx64 " - %s",
             s.first, s.second->data_to_json().dump().c_str());

    for (auto s : dm.datasets())
        INFO("0x%" PRIx64 " - 0x%" PRIx64 , s.second.state(),
             s.second.base_dset());

    auto i = dm.dataset_state<inputState>(SECOND_ROOT);
    check_equal(i->get_inputs(), inputs);

    auto f = dm.dataset_state<freqState>(SECOND_ROOT);
    check_equal(f->get_freqs(), freqs);

    auto p = dm.dataset_state<prodState>(SECOND_ROOT);
    check_equal(p->get_prods(), prods);

    INFO("%s", dm.summary().c_str());

    for (auto s : dm.states())
        INFO("0x%" PRIx64 " - %s", s.first,
             s.second->data_to_json().dump().c_str());

    for (auto s : dm.datasets())
        INFO("0x%" PRIx64 " - 0x%" PRIx64 , s.second.state(),
             s.second.base_dset());

    // wait a bit, to make sure we see errors in any late callbacks
    usleep(500000);
}

BOOST_FIXTURE_TEST_CASE( _ask_broker_for_second_root_update, CompareCTypes ) {
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
    std::vector<input_ctype> inputs = {input_ctype(1, "4")};
    std::vector<prod_ctype> prods = {{4, 1}};
    std::vector<std::pair<uint32_t, freq_ctype>> freqs = {{4, {1.1, 1}}};

    INFO("%s", dm.summary().c_str());

    for (auto s : dm.states())
        INFO("0x%" PRIx64 " - %s", s.first,
             s.second->data_to_json().dump().c_str());

    for (auto s : dm.datasets())
        INFO("0x%" PRIx64 " - 0x%" PRIx64 , s.second.state(),
                     s.second.base_dset());

    auto i = dm.dataset_state<inputState>(SECOND_ROOT_UPDATE);
    check_equal(i->get_inputs(), inputs);

    auto f = dm.dataset_state<freqState>(SECOND_ROOT_UPDATE);
    check_equal(f->get_freqs(), freqs);

    auto p = dm.dataset_state<prodState>(SECOND_ROOT_UPDATE);
    check_equal(p->get_prods(), prods);

    INFO("%s", dm.summary().c_str());

    for (auto s : dm.states())
        INFO("0x%" PRIx64 " - %s", s.first,
             s.second->data_to_json().dump().c_str());

    for (auto s : dm.datasets())
        INFO("0x%" PRIx64 " - 0x%" PRIx64 , s.second.state(),
             s.second.base_dset());

    // wait a bit, to make sure we see errors in any late callbacks
    usleep(500000);
}

BOOST_FIXTURE_TEST_CASE( _ask_broker_for_pulsar_gating_update, CompareCTypes ) {
    __log_level = 4;
    __enable_syslog = 0;

    json json_config;
    json json_config_dm;
    json_config_dm["use_dataset_broker"] = true;
    json_config["dataset_manager"] = json_config_dm;

    Config conf;
    conf.update_config(json_config);
    datasetManager& dm = datasetManager::instance(conf);

    std::string name = "test_pulsar";

    // Reqest any gatingState of the dataset and then check of which type it is.
    auto i = dm.dataset_state<gatingState>(GATING_DS);

    // This should give us a nullptr.
    auto n = dynamic_cast<const noiseSourceGatingState*>(i);
    BOOST_CHECK_EQUAL(n, static_cast<decltype(n)>(nullptr));

    // But this should work!
    auto p = dynamic_cast<const pulsarGatingState*>(i);
    BOOST_CHECK_NE(p, static_cast<decltype(p)>(nullptr));
    BOOST_CHECK_EQUAL(p->get_name(), name);

    auto p2 = dm.dataset_state<pulsarGatingState>(GATING_DS);
    BOOST_CHECK_EQUAL(p2->get_name(), name);


    INFO("%s", dm.summary().c_str());

    for (auto s : dm.states())
        INFO("0x%" PRIx64 " - %s", s.first,
             s.second->data_to_json().dump().c_str());

    for (auto s : dm.datasets())
        INFO("0x%" PRIx64 " - 0x%" PRIx64 , s.second.state(),
                     s.second.base_dset());

    // wait a bit, to make sure we see errors in any late callbacks
    usleep(500000);
}
