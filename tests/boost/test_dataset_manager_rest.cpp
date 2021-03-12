
#define BOOST_TEST_MODULE "test_datasetManager_REST"

#include "Config.hpp"         // for Config
#include "Hash.hpp"           // for operator<<, hash, operator==, Hash
#include "dataset.hpp"        // for dataset
#include "datasetManager.hpp" // for state_id_t, datasetManager, dset_id_t
#include "datasetState.hpp"   // for freqState, inputState, prodState, state_uptr
#include "errors.h"           // for __enable_syslog, _global_log_level
#include "kotekanLogging.hpp" // for DEBUG_NON_OO
#include "restServer.hpp"     // for restServer, connectionInstance
#include "visUtil.hpp"        // for input_ctype, prod_ctype, freq_ctype

#include <algorithm>                         // for max
#include <atomic>                            // for atomic, __atomic_base
#include <boost/test/included/unit_test.hpp> // for BOOST_PP_IIF_1, BOOST_PP_BOOL_2, BOOST_TEST...
#include <exception>                         // for exception
#include <fmt.hpp>                           // for format, fmt
#include <functional>                        // for _Bind_helper<>::type, _Placeholder, bind, _1
#include <iostream>                          // for operator<<, endl, ostream, basic_ostream, cout
#include <json.hpp>                          // for basic_json<>::object_t, basic_json<>::value...
#include <map>                               // for map
#include <memory>                            // for allocator, make_unique, unique_ptr
#include <stddef.h>                          // for size_t
#include <stdint.h>                          // for uint32_t
#include <string>                            // for string, operator<<, string_literals
#include <sys/types.h>                       // for u_short, ushort
#include <unistd.h>                          // for usleep
#include <utility>                           // for pair
#include <vector>                            // for vector


using kotekan::connectionInstance;
using kotekan::restServer;

using json = nlohmann::json;

using namespace std::string_literals;

struct TestContext {

    std::atomic<size_t> _dset_id_count;

    void init() {
        _dset_id_count = 0;
        restServer::instance().register_post_callback(
            "/register-state", std::bind(&TestContext::register_state, this, std::placeholders::_1,
                                         std::placeholders::_2));
        restServer::instance().register_post_callback(
            "/send-state", std::bind(&TestContext::send_state, this, std::placeholders::_1,
                                     std::placeholders::_2));
        restServer::instance().register_post_callback(
            "/register-dataset", std::bind(&TestContext::register_dataset, this,
                                           std::placeholders::_1, std::placeholders::_2));
        usleep(1000);
    }

    void register_state(connectionInstance& con, json& js) {
        DEBUG_NON_OO("test: /register-state received:\n{:s}", js.dump(4));
        json reply;
        try {
            js.at("hash");
        } catch (std::exception& e) {
            std::string error =
                fmt::format("Failure parsing register state message from datasetManager: {}\n{}.",
                            js.dump(), e.what());
            reply["result"] = error;
            con.send_json_reply(reply);
            BOOST_CHECK_MESSAGE(false, error);
        }

        BOOST_CHECK(js.at("hash").is_string());
        reply["request"] = "get_state";
        reply["hash"] = js.at("hash");
        reply["result"] = "success";
        con.send_json_reply(reply);
        DEBUG_NON_OO("test: /register-state: replied with:\n{:s}", reply.dump(4));
    }

    void send_state(connectionInstance& con, json& js) {
        DEBUG_NON_OO("test: /send-state received:\n{:s}", js.dump(4));
        json reply;
        try {
            js.at("hash");
            js.at("state");
            js.at("state").at("type");
            js.at("state").at("data");
        } catch (std::exception& e) {
            std::string error = fmt::format(fmt("Failure parsing send-state message from "
                                                "datasetManager:\n{:s}\n{:s}."),
                                            js.dump(4), e.what());
            reply["result"] = error;
            con.send_json_reply(reply);
            BOOST_CHECK_MESSAGE(false, error);
        }

        BOOST_CHECK(js.at("hash").is_string());

        // check the received state
        static std::vector<input_ctype> inputs = {input_ctype(1, "1"), input_ctype(2, "2"),
                                                  input_ctype(3, "3")};
        static std::vector<prod_ctype> prods = {{1, 1}, {2, 2}, {3, 3}};
        static std::vector<std::pair<uint32_t, freq_ctype>> freqs = {
            {1, {1.1, 1}}, {2, {2, 2.2}}, {3, {3, 3}}};

        static state_uptr states[3] = {std::make_unique<inputState>(inputs),
                                       std::make_unique<prodState>(prods),
                                       std::make_unique<freqState>(freqs)};
        static bool pass[3] = {false, false, false};
        state_uptr received_state = datasetState::from_json(js.at("state"));

        for (ushort i = 0; i < 4; i++) {
            BOOST_CHECK(i < 4);
            if (states[i]->to_json() == received_state->to_json()) {
                BOOST_CHECK(pass[i] == false);
                pass[i] = true;
                break;
            }
        }

        reply["result"] = "success";
        con.send_json_reply(reply);
        DEBUG_NON_OO("test: /send-state: replied with\n{:s}", reply.dump(4));
    }

    void register_dataset(connectionInstance& con, json& js) {
        DEBUG_NON_OO("test: /register-dataset received:\n{:s}", js.dump(4));
        json reply;
        json js_ds;
        try {
            js.at("hash");
            js_ds = js.at("ds");
            js_ds.at("is_root");
            js_ds.at("state");
            if (!js_ds.at("is_root"))
                js_ds.at("base_dset");
        } catch (std::exception& e) {
            std::string error = fmt::format("Failure parsing register-dataset message from "
                                            "datasetManager: {}\n{}.",
                                            js.dump(), e.what());
            reply["result"] = error;
            con.send_json_reply(reply);
            BOOST_CHECK_MESSAGE(false, error);
        }

        BOOST_CHECK(js_ds.at("state").is_string());
        if (!js_ds.at("is_root"))
            BOOST_CHECK(js_ds.at("base_dset").is_string());
        BOOST_CHECK(js_ds.at("is_root").is_boolean());
        BOOST_CHECK(js.at("hash").is_string());

        BOOST_CHECK(hash(js_ds.dump()) == Hash::from_string(js.at("hash")));

        reply["result"] = "success";
        con.send_json_reply(reply);
        DEBUG_NON_OO("test: /register-dataset: replied with\n{:s}", reply.dump(4));
    }
};

BOOST_FIXTURE_TEST_CASE(_dataset_manager_general, TestContext) {
    _global_log_level = 4;
    __enable_syslog = 0;

    json json_config;
    json_config["use_dataset_broker"] = true;

    // kotekan restServer endpoints defined above. Start with random free port.
    restServer::instance().start("127.0.0.1", 0);
    usleep(10000);
    json_config["ds_broker_port"] = restServer::instance().port;
    std::cout << "Running RESTserver on port " << json_config["ds_broker_port"] << " for dM test."
              << std::endl;

    TestContext::init();

    kotekan::Config conf;
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

    // register first state again
    std::vector<state_id_t> states2;
    states2.push_back(dm.create_state<freqState>(freqs).first);
    states2.push_back(dm.create_state<prodState>(prods).first);
    states2.push_back(dm.create_state<inputState>(inputs).first);

    // register new dataset with the twin state
    dm.add_dataset(states2, init_ds_id);

    std::cout << dm.summary() << std::endl;

    for (auto s : dm.states())
        std::cout << s.first << " - " << s.second->data_to_json().dump() << std::endl;

    for (auto s : dm.datasets())
        std::cout << s.second.state() << " - " << s.second.base_dset() << std::endl;

    usleep(500000);
}
