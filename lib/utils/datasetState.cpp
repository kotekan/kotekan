#include "datasetState.hpp"
#include <typeinfo>

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
    try {
        return _registered_types()[name](data, std::move(inner));
    } catch (std::bad_function_call& e) {
        WARN("datasetManager: no state of type %s is registered.", name.c_str());
        return nullptr;
    }
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

// TODO: compare without serialization
bool datasetState::equals(datasetState& s) const {
    return to_json() == s.to_json();
}

std::set<std::string> datasetState::types() const {
    std::set<std::string> types;

    types.insert(typeid(*this).name());

    const datasetState* t = _inner_state.get();

    while(t != nullptr) {
        types.insert(typeid(*t).name());
        t = t->_inner_state.get();
    }

    return types;
}