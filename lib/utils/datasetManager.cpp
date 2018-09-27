#include <typeinfo>
#include <functional>
#include <algorithm>
#include <iostream>
#include <mutex>

#include "datasetManager.hpp"
#include "fmt.hpp"
#include "restClient.hpp"

// static stuff
std::map<state_id, state_uptr> datasetManager::_states;
std::map<dset_id, std::pair<state_id, dset_id>> datasetManager::_datasets;
std::mutex datasetManager::_lock_states;
std::mutex datasetManager::_lock_dsets;
std::mutex datasetManager::_lock_rqst;
std::mutex datasetManager::_lock_reg;
std::condition_variable datasetManager::cv_register_dset;
std::condition_variable datasetManager::cv_request_ancestors;
std::string datasetManager::_path_register_state;
std::string datasetManager::_path_send_state;
std::string datasetManager::_path_register_dataset;
std::string datasetManager::_path_request_ancestors;
std::string datasetManager::_ds_broker_host;
unsigned short datasetManager::_ds_broker_port;

// Initialise static map of types
std::map<std::string, std::function<state_uptr(json&, state_uptr)>>&
datasetState::_registered_types()
{
    static std::map<std::string, std::function<state_uptr(json&, state_uptr)>>
        _register;

    return _register;
}

state_uptr datasetState::_create(std::string name, json & data,
                                 state_uptr inner) {

    return _registered_types()[name](data, std::move(inner));
}

state_uptr datasetState::from_json(json & data) {

    // Fetch the required properties from the json
    std::string dtype = data.at("type");
    json d = data.at("data");

    // Get the inner if it exists
    state_uptr inner = nullptr;
    if(data.count("inner")) {
        inner = datasetState::from_json(data["inner"]);
    }

    // Create and return the
    return datasetState::_create(dtype, d, std::move(inner));
}

json datasetState::to_json() const {

    json j;

    // Use RTTI to serialise the type of datasetState this is
    j["type"] = typeid(*this).name();

    // Recursively serialise any inner states
    if(_inner_state != nullptr) {
        j["inner"] = _inner_state->to_json();
    }
    j["data"] = data_to_json();

    return j;
}

std::ostream& operator<<(std::ostream& out, const datasetState& dt) {
    out << typeid(dt).name();

    return out;
}

datasetManager& datasetManager::instance() {
    static datasetManager dm;

    return dm;
}

void datasetManager::apply_config(Config& config) {
    _use_broker = config.get_bool_default(UNIQUE_NAME, "use_ds_broker", false);
    if (_use_broker) {
        _ds_broker_port = config.get_uint32(UNIQUE_NAME, "ds_broker_port");
        _ds_broker_host = config.get_string(UNIQUE_NAME, "ds_broker_host");
        _path_register_state = config.get_string(UNIQUE_NAME,
                                                 "register_state_path");
        _path_send_state = config.get_string(UNIQUE_NAME,
                                                 "send_state_path");
        _path_register_dataset = config.get_string(UNIQUE_NAME,
                                                 "register_dataset_path");
        _path_request_ancestors = config.get_string(UNIQUE_NAME,
                                                    "request_ancestors_path");

        DEBUG("datasetManager: expecting broker at %s:%d, endpoints: %s, %s, " \
              "%s, %s", _ds_broker_host.c_str(), _ds_broker_port,
              _path_register_state.c_str(), _path_send_state.c_str(),
              _path_register_dataset.c_str(), _path_request_ancestors.c_str());
    }
}

bool datasetManager::find_dataset_id(const std::pair<state_id, dset_id> dataset,
                                     dset_id& dataset_id) {
    std::map<dset_id, std::pair<state_id, dset_id>>::iterator it;

    std::lock_guard<std::mutex> lock_ds(_lock_dsets);
    for (it = _datasets.begin(); it != _datasets.end(); it++) {
        if (it->second == dataset) {
            dataset_id = it->first;
            return true;
        }
    }
    return false;
}

dset_id datasetManager::add_dataset(state_id state, dset_id input) {
    dset_id new_dset_id = 0;

    // TODO: replace datasets container with something more appropriate
    auto new_dset = std::make_pair(state, input);

    std::unique_lock<std::mutex> lck_ds(_lock_dsets);
    // Search for existing entry and return if it exists
    for (auto dset = _datasets.begin(); dset != _datasets.end(); dset++) {
        if (dset->second == new_dset)
            return dset->first;
    }
    lck_ds.unlock();

    if (!_use_broker) {
        lck_ds.lock();
        new_dset_id = _datasets.size();
        // insert a new entry and return its index.
        if (!_datasets.insert(std::pair<dset_id,
                              std::pair<state_id, dset_id>>(new_dset_id,
                                                            new_dset)).second) {
            ERROR("datasetManager: A dataset with ID %d is already known.",
                  new_dset_id);
            raise(SIGINT);
        }
    } else {

        // lock for the conditional variable cv_register_dset
        std::unique_lock<std::mutex> lck(_lock_reg);

        try {
            register_dataset(new_dset);
        } catch (std::runtime_error& e) {
            ERROR("datasetManager: Failure registering new dataset " \
                  "(dataset state %zu applied to dataset %d): %s", state, input,
                  e.what());
            ERROR("datasetManager: Make sure the broker is running. " \
                  "Exiting...");
            raise(SIGINT);
        }
        // set timeout to hear back from callback function
        std::chrono::seconds timeout(TIMEOUT_BROKER_SEC);
        auto time_point = std::chrono::system_clock::now() + timeout;

        // wait for signal from callback and check if dataset was registered
        bool found = find_dataset_id(new_dset, new_dset_id);
        while(!found) {
            if (cv_register_dset.wait_until(lck, time_point)
                  == std::cv_status::timeout) {
                ERROR("datasetManager: Timeout while registering new dataset " \
                      "(dataset state %zu applied to dataset %d).",
                      state, input);
                ERROR("datasetManager: Exiting...");
                raise(SIGINT);
            }
            found = find_dataset_id(new_dset, new_dset_id);
        }
    }

    return new_dset_id;
}

state_id datasetManager::hash_state(datasetState& state) {
    static std::hash<std::string> hash_function;

    // TODO: decide if this is the best way of hashing the state. It has the
    // advantage of being simple, there's a slight issue in that json
    // technically doesn't guarantee order of items in an object, but in
    // practice nlohmann::json ensures they are alphabetical by default. It
    // might also be a little slow as it requires full serialisation.
    return hash_function(state.to_json().dump());
}

void datasetManager::register_state(state_id state)
{
    json js_post;
    js_post["hash"] = state;

    std::function<void(restReply)> callback(
                datasetManager::register_state_callback);
    if (restClient::instance().make_request(
            _path_register_state,
            callback,
            js_post, _ds_broker_host, _ds_broker_port) == false)
        throw std::runtime_error("datasetManager: failed registering dataset " \
                                 "state " + std::to_string(state)
                                 + " with broker.");
}

void datasetManager::register_state_callback(restReply reply) {
    if (reply.first == true) {
        json js_reply;

        try {
            js_reply = json::parse(reply.second);
        } catch (exception& e) {
            ERROR("datasetManager: failure parsing reply received from broker " \
                  "after registering dataset state (reply: %s): %s",
                  reply.second.c_str(), e.what());
            ERROR("datasetManager: exiting...");
            raise(SIGINT);
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
                state_id state = js_reply.at("hash");

                json js_post;
                js_post["hash"] = state;

                _lock_states.lock();
                js_post["state"] = _states.at(state)->to_json();
                _lock_states.unlock();

                std::function<void(restReply)> callback(
                            datasetManager::send_state_callback);
                if (restClient::instance().make_request(
                        _path_send_state,
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
        } catch (exception& e) {
            ERROR("datasetManager: failure registering dataset state with " \
                  "broker: %s", e.what());
            ERROR("datasetManager: exiting...");
            raise(SIGINT);
        }
    }
    else {
        ERROR("datasetManager: failure registering dataset state with broker.");
        ERROR("datasetManager: exiting...");
        raise(SIGINT);
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
        } catch (exception& e) {
            ERROR("datasetManager: failure parsing reply received from broker "\
                  "after sending dataset state (reply: %s): %s",
                  reply.second.c_str(), e.what());
            ERROR("datasetManager: exiting...");
            raise(SIGINT);
        }
    }
    ERROR("datasetManager: failure sending dataset state to broker.");
    ERROR("datasetManager: exiting...");
    raise(SIGINT);
}

void datasetManager::register_dataset(std::pair<state_id, dset_id> dset) {
    json js_post;
    js_post["state_id"] = dset.first;
    js_post["base_ds_id"] = dset.second;
    std::function<void(restReply)> callback(
                datasetManager::register_dataset_callback);
    if (restClient::instance().make_request(
            _path_register_dataset,
            callback,
            js_post, _ds_broker_host, _ds_broker_port) == false)
        throw std::runtime_error("datasetManager: failed registering dataset " \
                                 "(" + std::to_string(dset.first) + ","
                                 + std::to_string(dset.second)
                                 + ") with broker.");
}

void datasetManager::register_dataset_callback(restReply reply) {
    dset_id new_ds_id = 0;
    std::pair<state_id, dset_id> new_dset;

    if (reply.first == false) {
        ERROR("datasetManager: failure registering dataset with broker.");
        ERROR("datasetManager: exiting...");
        raise(SIGINT);
    }

    json js_reply;

    try {
        js_reply = json::parse(reply.second);
        if (js_reply.at("result") != "success")
            throw std::runtime_error("received error from broker: "
                                     + js_reply.at("result").dump());
        new_ds_id = js_reply.at("new_ds_id");
        new_dset = std::pair<state_id, dset_id>(
                    js_reply.at("state_id"), js_reply.at("base_dset_id"));
    } catch (exception& e) {
        ERROR("datasetManager: failure parsing reply received from broker "\
             "after registering dataset (reply: %s): %s",
              reply.second.c_str(), e.what());
        ERROR("datasetManager: exiting...");
        raise(SIGINT);
        }

    // lock both the lock that protects the datasets as well as the one for the
    // conditional variable in add_dataset() at the same time, to prevent
    // a deadlock.
    std::lock(_lock_dsets, _lock_reg);
    std::lock_guard<std::mutex> dslock(_lock_dsets, std::adopt_lock);
    std::lock_guard<std::mutex> reglock(_lock_reg, std::adopt_lock);

    // insert the received new entry
    if (!_datasets.insert(std::pair<dset_id, std::pair<state_id, dset_id>>(
                new_ds_id, new_dset)).second) {
        // this can happen when two processes register the new
        INFO("datasetManager::register_dataset_callback(): A dataset with ID " \
             "%d is already known locally.", new_ds_id);
    }

    // signal add_dataset() that the work is done
    cv_register_dset.notify_all();
}

void datasetManager::request_ancestors(dset_id dset) {
    json js_post;
    js_post["ds_id"] = dset;

    std::function<void(restReply)> callback(
                datasetManager::request_ancestors_callback);
    if (restClient::instance().make_request(
            _path_request_ancestors,
            callback,
            js_post, _ds_broker_host, _ds_broker_port) == false)
        throw std::runtime_error("datasetManager: failed requesting ancestors" \
                                 " of dataset " + std::to_string(dset)
                                 + " with broker.");
}

void datasetManager::request_ancestors_callback(restReply reply) {

    json js_reply;

    try {
        js_reply = json::parse(reply.second);
        if (js_reply.at("result") != "success")
            throw std::runtime_error("datasetManager: Broker answered with " \
                                     "error after requesting ancestors.");
    } catch (std::exception& e) {
        ERROR("datasetManager: failure parsing reply received from broker " \
              "after requesting ancestors (reply: %s): %s.\ndatasetManager: " \
              "exciting...", reply.second.c_str(), e.what());
        raise(SIGINT);
    }

    // in case the broker doesn't have any ancestors to the dataset
    try {
        js_reply.at("ancestors");
    } catch (std::exception& e) {
        DEBUG("datasetManager::request_ancestors_callback(): broker did not " \
              "reply with any ancestors.");
        cv_request_ancestors.notify_all();
        return;
    }

    // acquire at the same time the lock that protects the states as well as the
    // one for cv_request_ancestors to avoid deadlocks
    std::lock(_lock_rqst, _lock_states);
    std::lock_guard<std::mutex> rqstlck(_lock_rqst, std::adopt_lock);
    std::unique_lock<std::mutex> slck(_lock_states, std::adopt_lock);

    // register the received states
    for (json::iterator s = js_reply.at("ancestors").at("states").begin();
         s != js_reply.at("ancestors").at("states").end(); s++) {
        state_id s_id;
        sscanf(s.key().c_str(), "%zu", &s_id);
        DEBUG2("datasetManager received state_id: %zu", s_id);
        DEBUG2("datasetManager received state: %s", s.value().dump().c_str());
        state_uptr state = datasetState::from_json(s.value());
        if (!_states.insert(std::pair<state_id, unique_ptr<datasetState>>(
                                s_id, move(state))).second)
            INFO("datasetManager::request_ancestors_callback: received a " \
                 "state (with hash %zu) that is already registered " \
                 "locally.", s.key());
    }

    // release the states lock and acquire the datasets lock
    // WARN: this is only acceptable as long as no other thread will ever try to
    // acquire both _lock_datasets and _lock_rqst
    slck.unlock();
    std::unique_lock<std::mutex> dslck(_lock_dsets);

    // register the received datasets
    for (json::iterator ds = js_reply.at("ancestors").at("datasets").begin();
         ds != js_reply.at("ancestors").at("datasets").end(); ds++) {

        dset_id ds_id = std::stoi(ds.key());
        if (ds_id < -1) {
            ERROR("datasetManager: failure parsing reply received from " \
                  "broker after requesting ancestors (received dset_id %d" \
                  " < -1).\ndatasetManager: exciting...", ds_id);
            raise(SIGINT);
        }

        state_id s_id;
        sscanf(ds.value().at(0).dump().c_str(), "%zu", &s_id);

        dset_id base_ds_id = std::stoi(ds.value().at(1).dump());
        if (base_ds_id < -1) {
            ERROR("datasetManager: failure parsing reply received from " \
                  "broker after requesting ancestors (received " \
                  "base_dset_id %d < -1).\ndatasetManager: exciting...",
                  base_ds_id);
            raise(SIGINT);
        }
        DEBUG2("datasetManager received dataset with id %d (state_id: %zu" \
              ", base_ds_id: %d).", ds_id, s_id, base_ds_id);
        auto new_dset = std::make_pair(s_id, base_ds_id);

        // Search for existing entry
        for (auto dset = _datasets.begin(); dset != _datasets.end(); dset++) {
            if (dset->second == new_dset) {
                if (dset->first != ds_id) {
                    ERROR("datasetManager: failure parsing reply received from"\
                          " broker after requesting ancestors: received " \
                          "dataset (ID: %d), that is known locally with a " \
                          "different ID (%d): state_id: %zu, base_ds_id: %d.\n"\
                          "datasetManager: exciting...",
                          ds_id, dset->first, s_id, base_ds_id);
                    raise(SIGINT);
                }
                DEBUG("datasetManager::request_ancestors_callback: received " \
                      "dataset that is already known locally: %d (%zu, %d).",
                      ds_id, s_id, base_ds_id);
                continue;
            } else {
                // insert the new dataset.
                if (!_datasets.insert(
                        std::pair<dset_id, std::pair<state_id, dset_id>>(
                            ds_id, new_dset)).second) {
                    // we just checked for this case, so if we end up here,
                    // something is seriously wrong
                    ERROR("datasetManager: A dataset with ID %d is already " \
                          "known (If you see this, there is a bug in " \
                          "datasetManager.).\ndatasetManager: Exiting...",
                          ds_id);
                    raise(SIGINT);
                }
            }
        }
    }

    // tell closest_ancestor_of_type() that the work is done
    cv_request_ancestors.notify_all();
}

std::string datasetManager::summary() const {
    int id = 0;
    std::string out;

    // lock both of them using std::lock to prevent a deadlock
    std::lock(_lock_states, _lock_dsets);
    std::lock_guard<std::mutex> slock(_lock_states, std::adopt_lock);
    std::lock_guard<std::mutex> dslock(_lock_dsets, std::adopt_lock);

    for(auto t : _datasets) {
        datasetState* dt = _states.at(t.second.first).get();

        out += fmt::format("{:>30} : {:2} -> {:2}\n", *dt, t.second.second, id);
        id++;
    }
    return out;
}

const std::map<state_id, const datasetState *> datasetManager::states() const {

    std::map<state_id, const datasetState *> cdt;

    std::lock_guard<std::mutex> lock(_lock_states);
    for (auto& dt : _states) {
        cdt[dt.first] = dt.second.get();
    }

    return cdt;
}

const std::map<dset_id, std::pair<state_id, dset_id>>
datasetManager::datasets() const {
    std::lock_guard<std::mutex> lock(_lock_dsets);
    return _datasets;
}

const std::vector<std::pair<dset_id, datasetState *>>
datasetManager::ancestors(dset_id dset) const {

    std::vector<std::pair<dset_id, datasetState *>> a_list;

    std::lock(_lock_states, _lock_dsets);
    std::lock_guard<std::mutex> slock(_lock_states, std::adopt_lock);
    std::lock_guard<std::mutex> dslock(_lock_dsets, std::adopt_lock);

    // make sure we know this dataset before running into trouble
    if (_datasets.find(dset) == _datasets.end())
        return a_list;

    // Walk up from the current node to the root, extracting pointers to the
    // states performed
    while(dset >= 0) {
        datasetState * t = _states.at(_datasets.at(dset).first).get();

        // Walk over the inner states, given them all the same dataset id.
        while(t != nullptr) {
            a_list.emplace_back(dset, t);
            t = t->_inner_state.get();
        }

        // Move on to the parent dataset...
        dset = _datasets.at(dset).second;

    }

    return a_list;
}


REGISTER_DATASET_STATE(freqState);
REGISTER_DATASET_STATE(inputState);
REGISTER_DATASET_STATE(prodState);
