#include "datasetManager.hpp"

#include "restClient.hpp"
#include "visUtil.hpp"

#include "fmt/ostream.h"

#include <cstdint>
#include <fmt.hpp>
#include <functional>
#include <inttypes.h>
#include <iostream>
#include <mutex>
#include <regex>
#include <signal.h>
#include <stdio.h>
#include <typeinfo>


dataset::dataset(json& js) {
    _state = js["state"];
    _is_root = js["is_root"];
    if (_is_root)
        _base_dset = 0;
    else
        _base_dset = js["base_dset"];
    _types = js["types"].get<std::set<std::string>>();
}

bool dataset::is_root() const {
    return _is_root;
}

state_id_t dataset::state() const {
    return _state;
}

dset_id_t dataset::base_dset() const {
    return _base_dset;
}

const std::set<std::string>& dataset::types() const {
    return _types;
}

json dataset::to_json() const {
    json j;
    j["is_root"] = _is_root;
    j["state"] = _state;
    if (!_is_root)
        j["base_dset"] = _base_dset;
    j["types"] = _types;
    return j;
}

bool dataset::equals(dataset& ds) const {
    if (_is_root != ds.is_root())
        return false;
    if (_is_root) {
        return _state == ds.state() && _types == ds.types();
    }
    return _state == ds.state() && _base_dset == ds.base_dset() && _types == ds.types();
}


std::ostream& operator<<(std::ostream& out, const datasetState& dt) {
    out << typeid(dt).name();

    return out;
}


std::vector<stack_ctype> invert_stack(uint32_t num_stack,
                                      const std::vector<rstack_ctype>& stack_map) {
    std::vector<stack_ctype> res(num_stack);
    size_t num_prod = stack_map.size();

    for (uint32_t i = 0; i < num_prod; i++) {
        uint32_t j = num_prod - i - 1;
        res[stack_map[j].stack] = {j, stack_map[j].conjugate};
    }

    return res;
}

datasetManager& datasetManager::private_instance() {
    static datasetManager dm;
    return dm;
}


datasetManager& datasetManager::instance() {
    datasetManager& dm = private_instance();

    if (!dm._config_applied) {
        ERROR("A part of kotekan that is configured to load uses the"
              " datasetManager, but no block named '%s' was found in the "
              "config.\nExiting...",
              DS_UNIQUE_NAME);
        exit(-1);
    }

    return dm;
}

datasetManager& datasetManager::instance(kotekan::Config& config) {
    datasetManager& dm = private_instance();

    dm._use_broker = config.get<bool>(DS_UNIQUE_NAME, "use_dataset_broker");
    if (dm._use_broker) {
        dm._ds_broker_port = config.get_default<uint32_t>(DS_UNIQUE_NAME, "ds_broker_port", 12050);
        dm._ds_broker_host =
            config.get_default<std::string>(DS_UNIQUE_NAME, "ds_broker_host", "127.0.0.1");
        dm._retry_wait_time_ms =
            config.get_default<uint32_t>(DS_UNIQUE_NAME, "retry_wait_time_ms", 1000);
        dm._retries_rest_client =
            config.get_default<uint32_t>(DS_UNIQUE_NAME, "retries_rest_client", 0);
        dm._timeout_rest_client_s =
            config.get_default<int32_t>(DS_UNIQUE_NAME, "timeout_rest_client", 100);

        DEBUG("datasetManager: expecting broker at %s:%d.", dm._ds_broker_host.c_str(),
              dm._ds_broker_port);
    }
    dm._config_applied = true;

    return dm;
}

datasetManager::~datasetManager() {
    _stop_request_threads = true;

    // wait for the detached threads
    std::unique_lock<std::mutex> lk(_lock_stop_request_threads);
    _cv_stop_request_threads.wait(lk, [this] { return _n_request_threads == 0; });
}

dset_id_t datasetManager::add_dataset(state_id_t state) {
    datasetState* t = nullptr;
    try {
        t = _states.at(state).get();
    } catch (std::exception& e) {
        // This must be a bug in the calling stage...
        ERROR("datasetManager: Failure registering root dataset : state "
              "0x%" PRIx64 " not found: %s",
              state, e.what());
        raise(SIGINT);
    }
    std::set<std::string> types = t->types();
    dataset ds(state, types);
    return add_dataset(ds);
}

dset_id_t datasetManager::add_dataset(dset_id_t base_dset, state_id_t state) {
    datasetState* t = nullptr;
    try {
        std::lock_guard<std::mutex> slck(_lock_states);
        t = _states.at(state).get();
    } catch (std::exception& e) {
        // This must be a bug in the calling stage...
        ERROR("datasetManager: Failure registering dataset : state "
              "0x%" PRIx64 " not found (base dataset ID: 0x%" PRIx64 "): %s",
              state, base_dset, e.what());
        raise(SIGINT);
    }
    std::set<std::string> types = t->types();
    dataset ds(state, base_dset, types);
    return add_dataset(ds);
}

// Private.
dset_id_t datasetManager::add_dataset(dataset ds) {

    dset_id_t new_dset_id = hash_dataset(ds);

    {
        // insert the new entry
        std::lock_guard<std::mutex> lck_ds(_lock_dsets);

        if (!_datasets.insert(std::pair<dset_id_t, dataset>(new_dset_id, ds)).second) {
            // There is already a dataset with the same hash.
            // Search for existing entry and return if it exists.
            auto find = _datasets.find(new_dset_id);
            if (!ds.equals(find->second)) {
                // TODO: hash collision. make the value a vector and store same
                // hash entries? This would mean the state/dset has to be sent
                // when registering.
                ERROR("datasetManager: Hash collision!\n"
                      "The following datasets have the same hash ("
                      "0x%" PRIx64 ").\n\n%s\n\n%s\n\n"
                      "datasetManager: Exiting...",
                      new_dset_id, ds.to_json().dump().c_str(),
                      find->second.to_json().dump().c_str());
                raise(SIGINT);
            }

            if (ds.is_root())
                _known_roots.insert(new_dset_id);

            // this dataset was already added
            return new_dset_id;
        }
    }

    if (_use_broker)
        register_dataset(new_dset_id, ds);

    return new_dset_id;
}

// TODO: decide if this is the best way of hashing the state and dataset.
// It has the advantage of being simple, there's a slight issue in that json
// technically doesn't guarantee order of items in an object, but in
// practice nlohmann::json ensures they are alphabetical by default. It
// might also be a little slow as it requires full serialisation.
state_id_t datasetManager::hash_state(datasetState& state) const {
    static std::hash<std::string> hash_function;

    return hash_function(state.to_json().dump());
}

state_id_t datasetManager::hash_dataset(dataset& ds) const {
    static std::hash<std::string> hash_function;

    return hash_function(ds.to_json().dump());
}

void datasetManager::register_state(state_id_t state) {
    json js_post;
    js_post["hash"] = state;
    std::string endpoint = PATH_REGISTER_STATE;
    std::function<bool(std::string&)> parser(
        std::bind(&datasetManager::register_state_parser, this, std::placeholders::_1));

    std::lock_guard<std::mutex> lk(_lock_stop_request_threads);
    std::thread t(&datasetManager::request_thread, this, std::move(js_post), std::move(endpoint),
                  std::move(parser));
    _n_request_threads++;

    // Let the request thread retry forever.
    if (t.joinable())
        t.detach();
}

void datasetManager::request_thread(const json&& request, const std::string&& endpoint,
                                    const std::function<bool(std::string&)>&& parse_reply) {

    restReply reply;

    while (true) {
        reply =
            _rest_client.make_request_blocking(endpoint, request, _ds_broker_host, _ds_broker_port,
                                               _retries_rest_client, _timeout_rest_client_s);

        // If parser succeeds, the request is done and this thread can exit.
        if (reply.first) {
            if (parse_reply(reply.second)) {
                std::unique_lock<std::mutex> lk(_lock_stop_request_threads);
                _n_request_threads--;
                return;
            }
            // Parsing errors are reported by the parsing function.
        } else {
            // Complain and retry...
            kotekan::prometheusMetrics::instance().add_stage_metric(
                "kotekan_datasetbroker_error_count", DS_UNIQUE_NAME, ++_conn_error_count);
            WARN("datasetManager: Failure in connection to broker: %s:"
                 "%d/%s. Make sure the broker is "
                 "running.",
                 _ds_broker_host.c_str(), _ds_broker_port, endpoint.c_str());
        }

        // check if datasetManager destructor was called
        if (_stop_request_threads) {
            INFO("datasetManager: Cancelling running request thread (endpoint "
                 "/%s, message %s).",
                 endpoint.c_str(), request.dump().c_str());
            std::unique_lock<std::mutex> lk(_lock_stop_request_threads);
            _n_request_threads--;
            std::notify_all_at_thread_exit(_cv_stop_request_threads, std::move(lk));
            return;
        }
    }
}

bool datasetManager::register_state_parser(std::string& reply) {
    json js_reply;

    try {
        js_reply = json::parse(reply);
    } catch (std::exception& e) {
        WARN("datasetManager: failure parsing reply received from broker "
             "after registering dataset state (reply: %s): %s",
             reply.c_str(), e.what());
        kotekan::prometheusMetrics::instance().add_stage_metric(
            "kotekan_datasetbroker_error_count", DS_UNIQUE_NAME, ++_conn_error_count);
        return false;
    }

    try {
        if (js_reply.at("result") != "success")
            throw std::runtime_error("received error from broker: " + js_reply.at("result").dump());
        // did the broker know this state already?
        if (js_reply.find("request") == js_reply.end())
            return true;
        // does the broker want the whole dataset state?
        if (js_reply.at("request") == "get_state") {
            state_id_t state = js_reply.at("hash");

            json js_post;
            js_post["hash"] = state;
            std::string endpoint = PATH_SEND_STATE;
            std::function<bool(std::string&)> parser(
                std::bind(&datasetManager::send_state_parser, this, std::placeholders::_1));

            {
                std::lock_guard<std::mutex> slck(_lock_states);
                js_post["state"] = _states.at(state)->to_json();
            }

            std::lock_guard<std::mutex> lk(_lock_stop_request_threads);
            std::thread t(&datasetManager::request_thread, this, std::move(js_post),
                          std::move(endpoint), std::move(parser));
            _n_request_threads++;

            // Let the request thread retry forever.
            if (t.joinable())
                t.detach();
        } else {
            throw std::runtime_error("datasetManager: failure parsing reply received "
                                     "from broker after registering dataset state "
                                     "(reply: "
                                     + reply + ").");
        }
    } catch (std::exception& e) {
        WARN("datasetManager: failure registering dataset state with "
             "broker: %s",
             e.what());
        kotekan::prometheusMetrics::instance().add_stage_metric(
            "kotekan_datasetbroker_error_count", DS_UNIQUE_NAME, ++_conn_error_count);
        return false;
    }
    return true;
}

bool datasetManager::send_state_parser(std::string& reply) {
    json js_reply;
    try {
        js_reply = json::parse(reply);
        if (js_reply.at("result") != "success")
            throw std::runtime_error("received error from broker: " + js_reply.at("result").dump());

        return true;
    } catch (std::exception& e) {
        WARN("datasetManager: failure parsing reply received from broker "
             "after sending dataset state (reply: %s): %s",
             reply.c_str(), e.what());
        kotekan::prometheusMetrics::instance().add_stage_metric(
            "kotekan_datasetbroker_error_count", DS_UNIQUE_NAME, ++_conn_error_count);
        return false;
    }
}

void datasetManager::register_dataset(dset_id_t hash, dataset dset) {
    json js_post;
    js_post["ds"] = dset.to_json();
    js_post["hash"] = hash;
    std::string endpoint = PATH_REGISTER_DATASET;
    std::function<bool(std::string&)> parser(
        std::bind(&datasetManager::register_dataset_parser, this, std::placeholders::_1));

    std::lock_guard<std::mutex> lk(_lock_stop_request_threads);
    std::thread t(&datasetManager::request_thread, this, std::move(js_post), std::move(endpoint),
                  std::move(parser));
    _n_request_threads++;

    // Let the request thread retry forever.
    if (t.joinable())
        t.detach();
}

bool datasetManager::register_dataset_parser(std::string& reply) {

    json js_reply;

    try {
        js_reply = json::parse(reply);
        if (js_reply.at("result") != "success")
            throw std::runtime_error("received error from broker: " + js_reply.at("result").dump());
        return true;
    } catch (std::exception& e) {
        WARN("datasetManager: failure parsing reply received from broker "
             "after registering dataset (reply: %s): %s",
             reply.c_str(), e.what());
        kotekan::prometheusMetrics::instance().add_stage_metric(
            "kotekan_datasetbroker_error_count", DS_UNIQUE_NAME, ++_conn_error_count);
        return false;
    }
}

std::string datasetManager::summary() {
    int id = 0;
    std::string out;

    // lock both of them at the same time to prevent deadlocks
    std::lock(_lock_states, _lock_dsets);
    std::lock_guard<std::mutex> slock(_lock_states, std::adopt_lock);
    std::lock_guard<std::mutex> dslock(_lock_dsets, std::adopt_lock);

    for (auto t : _datasets) {
        try {
            datasetState* dt = _states.at(t.second.state()).get();

            out += fmt::format("{:>30} : {:#x}\n", *dt, t.second.base_dset());
            id++;
        } catch (std::out_of_range& e) {
            WARN("datasetManager::summary(): This datasetManager instance "
                 "does not know state "
                 "0x%" PRIx64 ", referenced by dataset 0x%" PRIx64 ". (%s)",
                 t.second.state(), t.first, e.what());
        }
    }
    return out;
}

const std::map<state_id_t, const datasetState*> datasetManager::states() {

    std::map<state_id_t, const datasetState*> cdt;

    std::lock_guard<std::mutex> lock(_lock_states);
    for (auto& dt : _states) {
        cdt[dt.first] = dt.second.get();
    }

    return cdt;
}

const std::map<dset_id_t, dataset> datasetManager::datasets() {
    std::lock_guard<std::mutex> lock(_lock_dsets);
    return _datasets;
}

const std::vector<std::pair<dset_id_t, datasetState*>> datasetManager::ancestors(dset_id_t dset) {

    std::vector<std::pair<dset_id_t, datasetState*>> a_list;

    std::lock(_lock_states, _lock_dsets);
    std::lock_guard<std::mutex> slock(_lock_states, std::adopt_lock);
    std::lock_guard<std::mutex> dslock(_lock_dsets, std::adopt_lock);

    // make sure we know this dataset before running into trouble
    if (_datasets.find(dset) == _datasets.end()) {
        DEBUG("datasetManager: dataset 0x%" PRIx64 " was not found locally.", dset);
        return a_list;
    }

    // Walk up from the current node to the root, extracting pointers to the
    // states performed
    bool root = false;
    while (!root) {
        datasetState* t;
        try {
            t = _states.at(_datasets.at(dset).state()).get();
        } catch (...) {
            // we don't have the base dataset
            break;
        }

        // Walk over the inner states, given them all the same dataset id.
        while (t != nullptr) {
            a_list.emplace_back(dset, t);
            t = t->_inner_state.get();
        }

        // if this is the root dataset, we are done
        root = _datasets.at(dset).is_root();

        // Move on to the parent dataset...
        dset = _datasets.at(dset).base_dset();
    }

    return a_list;
}

void datasetManager::update_datasets(dset_id_t ds_id) {

    // wait for ongoing dataset updates
    std::lock(_lock_dsets, _lock_ds_update);
    std::unique_lock<std::mutex> dslock(_lock_dsets, std::adopt_lock);
    std::lock_guard<std::mutex> updatelock(_lock_ds_update, std::adopt_lock);

    // check if local dataset topology is up to date to include requested ds_id
    if (_datasets.find(ds_id) == _datasets.end()) {
        dslock.unlock();
        json js_rqst;
        js_rqst["ts"] = _timestamp_update;
        js_rqst["ds_id"] = ds_id;
        js_rqst["roots"] = _known_roots;

        restReply reply = _rest_client.make_request_blocking(
            PATH_UPDATE_DATASETS, js_rqst, _ds_broker_host, _ds_broker_port, _retries_rest_client,
            _timeout_rest_client_s);

        while (!parse_reply_dataset_update(reply)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(_retry_wait_time_ms));
            reply = _rest_client.make_request_blocking(
                PATH_UPDATE_DATASETS, js_rqst, _ds_broker_host, _ds_broker_port,
                _retries_rest_client, _timeout_rest_client_s);
        }
    }
}

bool datasetManager::parse_reply_dataset_update(restReply reply) {

    if (!reply.first) {
        WARN("datasetManager: Failure requesting update on datasets from "
             "broker: %s",
             reply.second.c_str());
        kotekan::prometheusMetrics::instance().add_stage_metric(
            "kotekan_datasetbroker_error_count", DS_UNIQUE_NAME, ++_conn_error_count);
        return false;
    }

    json js_reply;
    json timestamp;
    try {
        js_reply = json::parse(reply.second);
        if (js_reply.at("result") != "success")
            throw std::runtime_error("Broker answered with result=" + js_reply.at("result").dump());

        std::lock_guard<std::mutex> dslock(_lock_dsets);
        for (json::iterator ds = js_reply.at("datasets").begin();
             ds != js_reply.at("datasets").end(); ds++) {

            try {
                dset_id_t ds_id;
                sscanf(ds.key().c_str(), "%zu", &ds_id);
                dataset new_dset = dataset(ds.value());

                // insert the new dataset
                _datasets.insert(std::pair<dset_id_t, dataset>(ds_id, new_dset));

                if (new_dset.is_root())
                    _known_roots.insert(ds_id);

            } catch (std::exception& e) {
                WARN("datasetManager: failure parsing reply received from"
                     " broker after requesting dataset update: the following "
                     " exception was thrown when parsing dataset %s with ID "
                     "%s: %s",
                     ds.value().dump().c_str(), ds.key().c_str(), e.what());
                kotekan::prometheusMetrics::instance().add_stage_metric(
                    "kotekan_datasetbroker_error_count", DS_UNIQUE_NAME, ++_conn_error_count);
                return false;
            }
        }
        timestamp = js_reply.at("ts");
    } catch (std::exception& e) {
        WARN("datasetManager: failure parsing reply received from broker "
             "after requesting dataset update (reply: %s): %s",
             reply.second.c_str(), e.what());
        kotekan::prometheusMetrics::instance().add_stage_metric(
            "kotekan_datasetbroker_error_count", DS_UNIQUE_NAME, ++_conn_error_count);
        return false;
    }

    _timestamp_update = timestamp;
    return true;
}


REGISTER_DATASET_STATE(freqState);
REGISTER_DATASET_STATE(inputState);
REGISTER_DATASET_STATE(prodState);
REGISTER_DATASET_STATE(stackState);
REGISTER_DATASET_STATE(eigenvalueState);
REGISTER_DATASET_STATE(timeState);
REGISTER_DATASET_STATE(metadataState);
REGISTER_DATASET_STATE(gatingState);
