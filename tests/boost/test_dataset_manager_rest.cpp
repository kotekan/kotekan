
#define BOOST_TEST_MODULE "test_datasetManager_REST"

#include <boost/test/included/unit_test.hpp>
#include <string>
#include <iostream>
#include "fmt.hpp"
#include "json.hpp"
#include "visUtil.hpp"
#include "restClient.hpp"

// the code to test:
#include "datasetManager.hpp"

using json = nlohmann::json;

using namespace std::string_literals;

struct TestContext {

    std::atomic<size_t> _dset_id_count;

    void init() {
        _dset_id_count = 0;
        restServer::instance().register_post_callback(
                    "/register-state", std::bind(&TestContext::register_state,
                                                 this, std::placeholders::_1,
                                                 std::placeholders::_2));
        restServer::instance().register_post_callback(
                    "/send-state", std::bind(&TestContext::send_state,
                                                 this, std::placeholders::_1,
                                                 std::placeholders::_2));
        restServer::instance().register_post_callback(
                    "/register-dataset", std::bind(
                        &TestContext::register_dataset, this,
                        std::placeholders::_1, std::placeholders::_2));
        restServer::instance().register_post_callback(
                    "/request-ancestors", std::bind(
                        &TestContext::request_ancestors, this,
                        std::placeholders::_1, std::placeholders::_2));
        usleep(1000);
    }

    void register_state(connectionInstance& con, json& js) {
        DEBUG("test: /register-state received: %s", js.dump().c_str());
        json reply;
        try {
            js.at("hash");
        } catch (exception& e) {
            std::string error = fmt::format(
                "Failure parsing register state message from datasetManager: " \
                "{}\n{}.", js.dump(), e.what());
            reply["result"] = error;
            con.send_json_reply(reply);
            BOOST_CHECK_MESSAGE(false, error);
        }

        BOOST_CHECK(js.at("hash").is_number());
        reply["request"] = "get_state";
        reply["hash"] = js.at("hash");
        reply["result"] = "success";
        con.send_json_reply(reply);
        DEBUG("test: /register-state: replied with %s", reply.dump().c_str());
    }

    void send_state(connectionInstance& con, json& js) {
        DEBUG("test: /send-state received: %s", js.dump().c_str());
        json reply;
        try {
            js.at("hash");
            js.at("state");
            js.at("state").at("type");
            js.at("state").at("data");
        } catch (exception& e) {
            std::string error = fmt::format(
                        "Failure parsing send-state message from " \
                        "datasetManager: {}\n{}.", js.dump(), e.what());
            reply["result"] = error;
            con.send_json_reply(reply);
            BOOST_CHECK_MESSAGE(false, error);
        }

        BOOST_CHECK(js.at("hash").is_number());

        // check the received state
        std::vector<input_ctype> inputs = {input_ctype(1, "1"),
                                           input_ctype(2, "2"),
                                           input_ctype(3, "3")};
        std::vector<prod_ctype> prods = {{1, 1},
                                         {2, 2},
                                         {3, 3}};
        std::vector<std::pair<uint32_t, freq_ctype>> freqs = {{1, {1.1, 1}},
                                                              {2, {2, 2.2}},
                                                              {3, {3, 3}}};

        state_uptr same_state = std::make_unique<inputState>(
                    inputs,
                    make_unique<prodState>(prods,
                                           make_unique<freqState>(freqs)));
        state_uptr received_state = datasetState::from_json(js.at("state"));

        BOOST_CHECK(same_state->to_json() == received_state->to_json());

        reply["result"] = "success";
        con.send_json_reply(reply);
        DEBUG("test: /send-state: replied with %s", reply.dump().c_str());
    }

    void register_dataset(connectionInstance& con, json& js) {
        DEBUG("test: /register-dataset received: %s", js.dump().c_str());
        json reply;
        try {
            js.at("state_id");
            js.at("base_ds_id");
        } catch (exception& e) {
            std::string error = fmt::format(
                        "Failure parsing register-dataset message from " \
                        "datasetManager: {}\n{}.", js.dump(), e.what());
            reply["result"] = error;
            con.send_json_reply(reply);
            BOOST_CHECK_MESSAGE(false, error);
        }

        BOOST_CHECK(js.at("state_id").is_number());
        BOOST_CHECK(js.at("base_ds_id").is_number());
        reply["result"] = "success";
        reply["base_dset_id"] = js.at("base_ds_id");
        reply["state_id"] = js.at("state_id");
        reply["new_ds_id"] = _dset_id_count++;
        con.send_json_reply(reply);
        DEBUG("test: /request-ancestors: replied with %s", reply.dump().c_str());
    }

    void request_ancestors(connectionInstance& con, json& js) {
        DEBUG("test: /request-ancestors received: %s", js.dump().c_str());
        json reply;
        try {
            js.at("ds_id");
        } catch (exception& e) {
            std::string error = fmt::format(
                        "Failure parsing request-ancestors message from " \
                        "datasetManager: {}\n{}.", js.dump(), e.what());
            reply["result"] = error;
            con.send_json_reply(reply);
            BOOST_CHECK_MESSAGE(false, error);
        }

        BOOST_CHECK(js.at("ds_id").is_number());
        reply["new_ds_id"] = _dset_id_count++;
        reply["state_id"] = js.at("state_id");
        reply["base_dset_id"] = js.at("base_ds_id");
        reply["result"] = "success";
        con.send_json_reply(reply);
        DEBUG("test: /register-dataset: replied with %s", reply.dump().c_str());
    }
};

BOOST_FIXTURE_TEST_CASE( _dataset_manager_general, TestContext ) {
    __log_level = 4;
    __enable_syslog = 0;

    json json_config;
    json_config["use_ds_broker"] = true;
    json_config["ds_broker_port"] = 12048;
    json_config["ds_broker_host"] = "localhost";
    json_config["register_state_path"] = "/register-state";
    json_config["send_state_path"] = "/send-state";
    json_config["register_dataset_path"] = "/register-dataset";
    json_config["request_ancestors_path"] = "/request-ancestors";

    TestContext::init();

    datasetManager& dm = datasetManager::instance();
    Config conf;
    conf.update_config(json_config);
    dm.apply_config(conf);

    // generate datasets:
    std::vector<input_ctype> inputs = {input_ctype(1, "1"),
                                       input_ctype(2, "2"),
                                       input_ctype(3, "3")};
    std::vector<prod_ctype> prods = {{1, 1},
                                     {2, 2},
                                     {3, 3}};
    std::vector<std::pair<uint32_t, freq_ctype>> freqs = {{1, {1.1, 1}},
                                                          {2, {2, 2.2}},
                                                          {3, {3, 3}}};

    std::pair<state_id, const inputState*> input_state =
            dm.add_state(std::make_unique<inputState>(inputs,
                                               make_unique<prodState>(prods,
                                               make_unique<freqState>(freqs))));

    dset_id init_ds_id = dm.add_dataset(input_state.first, -1);

    // register first state again
    std::pair<state_id, const inputState*>input_state3 =
            dm.add_state(std::make_unique<inputState>(inputs,
                              make_unique<prodState>(prods,
                              make_unique<freqState>(freqs))));
    // register new dataset with the twin state
    dset_id init_ds_id3 = dm.add_dataset(input_state3.first, init_ds_id);

    std::cout << dm.summary() << std::endl;

    for (auto s : dm.states())
        std::cout << s.first << " - " << s.second->data_to_json().dump()
                  << std::endl;

    for (auto s : dm.datasets())
        std::cout << s.second.first << " - " << s.second.second << std::endl;

    for (auto s : dm.ancestors(init_ds_id3))
        std::cout << s.first << " - " << s.second->data_to_json().dump()
                  << std::endl;

    usleep(3000);
}