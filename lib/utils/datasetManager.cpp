#include <typeinfo>
#include <functional>
#include <algorithm>
#include <iostream>

#include "datasetManager.hpp"
#include "fmt.hpp"

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

dset_id datasetManager::add_dataset(state_id state, dset_id input) {
    _datasets.push_back({state, input});
    return _datasets.size() - 1;
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

std::string datasetManager::summary() const {
    int id = 0;
    std::string out;
    for(auto t : _datasets) {
        datasetState* dt = _states.at(t.first).get();

        out += fmt::format("{:>30} : {:2} -> {:2}\n", *dt, t.second, id);
        id++;
    }

    return out;
}

const std::map<state_id, const datasetState *> datasetManager::states() const {

    std::map<state_id, const datasetState *> cdt;

    for (auto& dt : _states) {
        cdt[dt.first] = dt.second.get();
    }

    return cdt;
}

const std::vector<std::pair<state_id, dset_id>> datasetManager::datasets() const {
    return _datasets;
}

const std::vector<std::pair<dset_id, datasetState *>>
datasetManager::ancestors(dset_id dset) const {

    std::vector<std::pair<dset_id, datasetState *>> a_list;

    // Walk up from the current node to the root, extracting pointers to the
    // states performed
    while(dset >= 0) {
        datasetState * t = _states.at(_datasets[dset].first).get();
        // Walk over the inner states, given them all the same dataset id.
        while(t != nullptr) {
            a_list.emplace_back(dset, t);
            t = t->_inner_state.get();
        }

        // Move on to the parent dataset...
        dset = _datasets[dset].second;

    }

    return a_list;
}


REGISTER_DATASET_STATE(freqState);
REGISTER_DATASET_STATE(inputState);
REGISTER_DATASET_STATE(prodState);
