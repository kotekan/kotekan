#include "datasetState.hpp"


using nlohmann::json;


state_uptr datasetState::from_json(const json& data) {

    // Fetch the required properties from the json
    std::string dtype = data.at("type");
    json d = data.at("data");

    // Create and return the
    return FACTORY(datasetState)::create_unique(dtype, d);
}

json datasetState::to_json() const {

    json j;

    // Use RTTI to serialise the type of datasetState this is
    j["type"] = type();
    j["data"] = data_to_json();

    return j;
}

// TODO: compare without serialization
bool datasetState::equals(datasetState& s) const {
    return to_json() == s.to_json();
}

std::string datasetState::type() const {
    return FACTORY(datasetState)::label(*this);
}

std::ostream& operator<<(std::ostream& out, const datasetState& dt) {
    out << dt.type();
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

REGISTER_DATASET_STATE(freqState, "frequencies");
REGISTER_DATASET_STATE(inputState, "inputs");
REGISTER_DATASET_STATE(prodState, "products");
REGISTER_DATASET_STATE(stackState, "stack");
REGISTER_DATASET_STATE(eigenvalueState, "eigenvalues");
REGISTER_DATASET_STATE(timeState, "time");
REGISTER_DATASET_STATE(metadataState, "metadata");
REGISTER_DATASET_STATE(gatingState, "gating");
REGISTER_DATASET_STATE(acqDatasetIdState, "acq_dataset_id");
REGISTER_DATASET_STATE(flagState, "flags");
REGISTER_DATASET_STATE(gainState, "gains");
REGISTER_DATASET_STATE(beamState, "beams");
REGISTER_DATASET_STATE(subfreqState, "sub_frequencies");
