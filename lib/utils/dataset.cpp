#include "dataset.hpp" // for dataset, dset_id_t, state_id_t

#include "Hash.hpp" // for operator==

#include "json.hpp" // for json, basic_json<>::value_type, basic_json

#include <string> // for string, operator==

using nlohmann::json;


dataset::dataset(json& js) {
    _state = js["state"];
    _is_root = js["is_root"];
    if (_is_root)
        _base_dset = dset_id_t::null;
    else
        _base_dset = js["base_dset"].get<dset_id_t>();
    _type = js["type"].get<std::string>();
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

const std::string& dataset::type() const {
    return _type;
}

json dataset::to_json() const {
    json j;
    j["is_root"] = _is_root;
    j["state"] = _state;
    if (!_is_root)
        j["base_dset"] = _base_dset;
    j["type"] = _type;
    return j;
}

bool dataset::equals(dataset& ds) const {
    if (_is_root != ds.is_root())
        return false;
    if (_is_root) {
        return _state == ds.state() && _type == ds.type();
    }
    return _state == ds.state() && _base_dset == ds.base_dset() && _type == ds.type();
}
