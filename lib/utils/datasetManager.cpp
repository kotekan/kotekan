#include <typeinfo>
#include <functional>
#include <algorithm>
#include <iostream>
#include <mutex>

#include "datasetManager.hpp"
#include "fmt.hpp"
#include "restClient.hpp"


dataset::dataset(json& js) {
    _state = js["state"];
    _base_dset = js["base_dset"];
    _is_root = js["is_root"];
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

json dataset::to_json() const {
    json j;
    j["is_root"] = _is_root;
    j["state"] = _state;
    j["base_dset"] = _base_dset;
    return j;
}

bool dataset::equals(dataset& ds) const {
    return _state == ds.state() && _base_dset == ds.base_dset()
            && _is_root == ds.is_root();
}

std::ostream& operator<<(std::ostream& out, const datasetState& dt) {
    out << typeid(dt).name();

    return out;
}


std::vector<stack_ctype> invert_stack(
    uint32_t num_stack, const std::vector<rstack_ctype>& stack_map)
{
    std::vector<stack_ctype> res(num_stack);
    size_t num_prod = stack_map.size();

    for(uint32_t i = 0; i < num_prod; i++) {
        uint32_t j = num_prod - i - 1;
        res[stack_map[j].stack] = {j, stack_map[j].conjugate};
    }

    return res;
}


datasetManager& datasetManager::instance() {
    static datasetManager dm;

    return dm;
}

datasetManager::datasetManager() {
    _conn_error_count = 0;
}

void datasetManager::apply_config(Config& config) {
    _use_broker = config.get_default<bool>(
                UNIQUE_NAME, "use_dataset_broker", false);
    if (_use_broker) {
        _ds_broker_port = config.get_default<uint32_t>(
                    UNIQUE_NAME, "ds_broker_port", 12050);
        _ds_broker_host = config.get_default<std::string>(
                    UNIQUE_NAME, "ds_broker_host", "127.0.0.1");

        DEBUG("datasetManager: expecting broker at %s:%d.",
              _ds_broker_host.c_str(), _ds_broker_port);
    }
}

dset_id_t datasetManager::add_dataset(dataset ds, bool ignore_broker) {

    dset_id_t new_dset_id = hash_dataset(ds);

    {
        // insert the new entry
        std::lock_guard<std::mutex> lck_ds(_lock_dsets);

        if (!_datasets.insert(
                std::pair<dset_id_t, dataset>(new_dset_id, ds)).second) {
            // There is already a dataset with the same hash.
            // Search for existing entry and return if it exists.
            auto find = _datasets.find(new_dset_id);
            if (!ds.equals(find->second)) {
                // TODO: hash collision. make the value a vector and store same
                // hash entries? This would mean the state/dset has to be sent
                // when registering.
                ERROR("datasetManager: Hash collision!\n"
                      "The following datasets have the same hash (%zu)." \
                      "\n\n%s\n\n%s\n\n" \
                      "datasetManager: Exiting...",
                      new_dset_id, ds.to_json().dump().c_str(),
                      find->second.to_json().dump().c_str());
                raise(SIGINT);
            }
            // this dataset was already added
            return new_dset_id;
        }
    }

    if (_use_broker && !ignore_broker)
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

    std::function<void(restReply)> callback(
                std::bind(&datasetManager::register_state_callback,
                          this, std::placeholders::_1));
    if (restClient::instance().make_request(
            PATH_REGISTER_STATE,
            callback,
            js_post, _ds_broker_host, _ds_broker_port) == false) {
        prometheusMetrics::instance().add_process_metric(
                    "kotekan_datasetbroker_error_count", UNIQUE_NAME,
                    ++_conn_error_count);
        std::string msg = fmt::format(
                    "datasetManager: Failure registering state {}\n" \
                    "datasetManager: Make sure the broker is running.", state);
        throw std::runtime_error(msg);
    }
}

void datasetManager::register_state_callback(restReply reply) {
    if (reply.first == true) {
        json js_reply;

        try {
            js_reply = json::parse(reply.second);
        } catch (std::exception& e) {
            WARN("datasetManager: failure parsing reply received from broker " \
                  "after registering dataset state (reply: %s): %s",
                  reply.second.c_str(), e.what());
            prometheusMetrics::instance().add_process_metric(
                        "kotekan_datasetbroker_error_count", UNIQUE_NAME,
                        ++_conn_error_count);
            return;
        }

        try {
            if (js_reply.at("result") != "success")
                throw std::runtime_error("received error from broker: "
                                         + js_reply.at("result").dump());
            // did the broker know this state already?
            if (js_reply.find("request") == js_reply.end())
                return;
            // does the broker want the whole dataset state?
            if (js_reply.at("request") == "get_state") {
                state_id_t state = js_reply.at("hash");

                json js_post;
                js_post["hash"] = state;

                _lock_states.lock();
                js_post["state"] = _states.at(state)->to_json();
                _lock_states.unlock();

                std::function<void(restReply)> callback(
                            std::bind(&datasetManager::send_state_callback,
                                      this, std::placeholders::_1));
                if (restClient::instance().make_request(
                        PATH_SEND_STATE,
                        callback,
                        js_post, _ds_broker_host, _ds_broker_port) == false)
                    throw std::runtime_error("datasetManager: failed sending " \
                                             "dataset state "
                                             + std::to_string(state)
                                             + " to broker.");
            } else {
                throw std::runtime_error(
                            "datasetManager: failure parsing reply received " \
                            "from broker after registering dataset state " \
                            "(reply: " + reply.second + ").");
            }
        } catch (std::exception& e) {
            WARN("datasetManager: failure registering dataset state with " \
                  "broker: %s", e.what());
            prometheusMetrics::instance().add_process_metric(
                        "kotekan_datasetbroker_error_count", UNIQUE_NAME,
                        ++_conn_error_count);
        }
    }
    else {
        WARN("datasetManager: failure registering dataset state with broker.");
        prometheusMetrics::instance().add_process_metric(
                    "kotekan_datasetbroker_error_count", UNIQUE_NAME,
                    ++_conn_error_count);
    }
}

void datasetManager::send_state_callback(restReply reply) {
    json js_reply;
    if (reply.first == true) {
        try {
            js_reply = json::parse(reply.second);
            if (js_reply.at("result") != "success")
                throw std::runtime_error("received error from broker: "
                                         + js_reply.at("result").dump());
            // success
            return;
        } catch (std::exception& e) {
            WARN("datasetManager: failure parsing reply received from broker "\
                  "after sending dataset state (reply: %s): %s",
                  reply.second.c_str(), e.what());
            prometheusMetrics::instance().add_process_metric(
                        "kotekan_datasetbroker_error_count", UNIQUE_NAME,
                        ++_conn_error_count);
        }
    }
    WARN("datasetManager: failure sending dataset state to broker.");
    prometheusMetrics::instance().add_process_metric(
                "kotekan_datasetbroker_error_count", UNIQUE_NAME,
                ++_conn_error_count);
}

void datasetManager::register_dataset(dset_id_t hash, dataset dset) {
    json js_post;
    js_post["dataset"] = dset.to_json();
    js_post["hash"] = hash;

    std::function<void(restReply)> callback(
                std::bind(&datasetManager::register_dataset_callback,
                          this, std::placeholders::_1));
    if (restClient::instance().make_request(
            PATH_REGISTER_DATASET,
            callback,
            js_post, _ds_broker_host, _ds_broker_port) == false) {
        prometheusMetrics::instance().add_process_metric(
                    "kotekan_datasetbroker_error_count", UNIQUE_NAME,
                    ++_conn_error_count);
        std::string msg = fmt::format(
                    "datasetManager: Failure registering new dataset with " \
                    "hash {} (dataset state {} applied to dataset {}}).\n" \
                    "datasetManager: Make sure the broker is running.",
                    hash, dset.state(), dset.base_dset());
        throw std::runtime_error(msg);
    }
}

void datasetManager::register_dataset_callback(restReply reply) {

    if (reply.first == false) {
        WARN("datasetManager: failure registering dataset with broker.");
        prometheusMetrics::instance().add_process_metric(
                    "kotekan_datasetbroker_error_count", UNIQUE_NAME,
                    ++_conn_error_count);
        return;
    }

    json js_reply;

    try {
        js_reply = json::parse(reply.second);
        if (js_reply.at("result") != "success")
            throw std::runtime_error("received error from broker: "
                                     + js_reply.at("result").dump());
    } catch (std::exception& e) {
        WARN("datasetManager: failure parsing reply received from broker "\
             "after registering dataset (reply: %s): %s",
              reply.second.c_str(), e.what());
        prometheusMetrics::instance().add_process_metric(
                    "kotekan_datasetbroker_error_count", UNIQUE_NAME,
                    ++_conn_error_count);
    }
}

void datasetManager::request_ancestor(dset_id_t dset_id, const char* type) {
    json js_post;
    js_post["ds_id"] = dset_id;
    js_post["type"] = type;

    std::function<void(restReply)> callback(
                std::bind(&datasetManager::request_ancestor_callback,
                          this, std::placeholders::_1));
    if (restClient::instance().make_request(
            PATH_REQUEST_ANCESTOR,
            callback,
            js_post, _ds_broker_host, _ds_broker_port) == false) {
        prometheusMetrics::instance().add_process_metric(
                    "kotekan_datasetbroker_error_count", UNIQUE_NAME,
                    ++_conn_error_count);
        throw std::runtime_error("datasetManager: failed requesting ancestor" \
                                 " of type " + std::string(type)
                                 + " of dataset " + std::to_string(dset_id)
                                 + " from broker.");
    }

    DEBUG("datasetManager: requesting ancestor of type %s of dataset %zu",
          type, dset_id);
}

void datasetManager::request_ancestor_callback(restReply reply) {

    json js_reply;

    try {
        js_reply = json::parse(reply.second);
        if (js_reply.at("result") != "success")
            throw std::runtime_error("datasetManager: Broker answered with " \
                                     "error after requesting ancestor.");
    } catch (std::exception& e) {
        WARN("datasetManager: failure parsing reply received from broker " \
              "after requesting ancestor (reply: %s): %s.",
             reply.second.c_str(), e.what());
        prometheusMetrics::instance().add_process_metric(
                    "kotekan_datasetbroker_error_count", UNIQUE_NAME,
                    ++_conn_error_count);
        return;
    }

    // in case the broker doesn't have any ancestors to the dataset
    try {
        js_reply.at("states");
        js_reply.at("datasets");
    } catch (std::exception& e) {
        DEBUG("datasetManager::request_ancestor_callback(): broker did not " \
              "reply with any ancestor: %s", js_reply.dump().c_str());
        return;
    }

    // register the received states
    {
        std::unique_lock<std::mutex> slck(_lock_states);
        for (json::iterator s = js_reply.at("states").begin();
                s != js_reply.at("states").end(); s++) {
            state_id_t s_id;
            sscanf(s.key().c_str(), "%zu", &s_id);

            state_uptr state = datasetState::from_json(s.value());
            if (state == nullptr)
                continue;

            // TODO: hash collisions should be checked for by the broker
            if (!_states.insert(std::pair<state_id_t,
                                std::unique_ptr<datasetState>>
                                (s_id, move(state))).second)
                INFO("datasetManager::request_ancestors_callback: received a " \
                     "state (with hash %zu) that is already registered " \
                     "locally.", s_id);
        }
    }

    // register the received datasets
    {
        std::unique_lock<std::mutex> dslck(_lock_dsets);
        for (json::iterator ds = js_reply.at("datasets").begin();
             ds != js_reply.at("datasets").end(); ds++) {

            try {
                dset_id_t ds_id;
                sscanf(ds.key().c_str(), "%zu", &ds_id);
                dataset new_dset = dataset(ds.value());

                if (ds_id != hash_dataset(new_dset)) {
                    WARN("datasetManager: failure parsing reply received from"\
                         " broker after requesting ancestors: the dataset (%s) has " \
                         "the hash %zu, but %zu was received from the broker. " \
                         "Ignoring this dataset...",
                          ds.value().dump().c_str(), ds_id, hash_dataset(new_dset));
                    continue;
                }

                // insert the new dataset and check if it was a known before
                // TODO: collisions should be checked for by the broker
                auto inserted = _datasets.insert(
                            std::pair<dset_id_t, dataset>(ds_id, new_dset));
                if (!inserted.second) {
                    DEBUG("datasetManager::request_ancestors_callback: received " \
                          "dataset that is already known locally: %zu : %s.",
                          ds_id, ds.value().dump().c_str());
                }
            } catch (std::exception& e) {
                WARN("datasetManager: failure parsing reply received from"\
                      " broker after requesting ancestors: the following " \
                      " exception was thrown when parsing dataset %s with ID %s:" \
                      " %s", ds.key().c_str(), ds.value().dump().c_str(), e.what());
                prometheusMetrics::instance().add_process_metric(
                            "kotekan_datasetbroker_error_count", UNIQUE_NAME,
                            ++_conn_error_count);
            }
        }
    }

    // tell closest_ancestor_of_type() that the work is done
    _cv_request_ancestor.notify_all();
}

std::string datasetManager::summary() {
    int id = 0;
    std::string out;

    // lock both of them using std::lock to prevent a deadlock
    std::lock(_lock_states, _lock_dsets);
    std::lock_guard<std::mutex> slock(_lock_states, std::adopt_lock);
    std::lock_guard<std::mutex> dslock(_lock_dsets, std::adopt_lock);

    for(auto t : _datasets) {
        try{
            datasetState* dt = _states.at(t.second.state()).get();

            out += fmt::format("{:>30} : {:2} -> {:2}\n",
                               *dt, t.second.base_dset(), id);
            id++;
        } catch (std::out_of_range& e) {
            // this is fine
            DEBUG("This datasetManager instance does not know state %zu, " \
                  "referenced by dataset %zu. (std::out_of_range: %s)",
                  t.second.state(), t.first, e.what());
        }
    }
    return out;
}

const std::map<state_id_t, const datasetState *> datasetManager::states() {

    std::map<state_id_t, const datasetState *> cdt;

    std::lock_guard<std::mutex> lock(_lock_states);
    for (auto& dt : _states) {
        cdt[dt.first] = dt.second.get();
    }

    return cdt;
}

const std::map<dset_id_t, dataset>
datasetManager::datasets() {
    std::lock_guard<std::mutex> lock(_lock_dsets);
    return _datasets;
}

const std::vector<std::pair<dset_id_t, datasetState *>>
datasetManager::ancestors(dset_id_t dset) {

    std::vector<std::pair<dset_id_t, datasetState *>> a_list;

    std::lock(_lock_states, _lock_dsets);
    std::lock_guard<std::mutex> slock(_lock_states, std::adopt_lock);
    std::lock_guard<std::mutex> dslock(_lock_dsets, std::adopt_lock);

    // make sure we know this dataset before running into trouble
    if (_datasets.find(dset) == _datasets.end()) {
        DEBUG("datasetManager: dataset %zu was not found locally.", dset);
        return a_list;
    }

    // Walk up from the current node to the root, extracting pointers to the
    // states performed
    bool root = false;
    while(!root) {
        datasetState* t;
        try {
            t = _states.at(_datasets.at(dset).state()).get();
        } catch (...) {
            // we don't have the base dataset
            break;
        }

        // Walk over the inner states, given them all the same dataset id.
        while(t != nullptr) {
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


REGISTER_DATASET_STATE(freqState);
REGISTER_DATASET_STATE(inputState);
REGISTER_DATASET_STATE(prodState);
REGISTER_DATASET_STATE(stackState);
REGISTER_DATASET_STATE(eigenvalueState);
REGISTER_DATASET_STATE(timeState);
REGISTER_DATASET_STATE(metadataState);