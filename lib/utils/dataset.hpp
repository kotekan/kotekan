#ifndef DATASET_HPP
#define DATASET_HPP

#include "Hash.hpp" // for operator==, Hash

#include "json.hpp" // for json

#include <string> // for string

/// DatasetID
using dset_id_t = Hash;

/// DatasetState ID
using state_id_t = Hash;


/**
 * @brief The description of a dataset consisting of a dataset state and a base
 * dataset.
 *
 * A dataset is described by a dataset state applied to a base dataset. If the
 * flag for this dataset being a root dataset (a dataset that has no base
 * dataset), the base dataset ID value is not defined.
 */
class dataset {
public:
    /**
     * @brief Dataset constructor. Omitting the base_dset will create a root dataset.
     * @param state      The state of this dataset.
     * @param type       The name of the dataset state type.
     * @param base_dset  The ID of the base datset. Omit to create a root dataset.
     */
    dataset(state_id_t state, std::string type, dset_id_t base_dset = dset_id_t::null) :
        _state(state), _base_dset(base_dset), _is_root(base_dset == dset_id_t::null), _type(type) {}

    /**
     * @brief Dataset constructor from json object.
     * The json object must have the following fields:
     * is_root:     boolean
     * state:       integer
     * base_dset    integer
     * types        list of strings
     * @param js    Json object describing a dataset.
     */
    dataset(nlohmann::json& js);

    /**
     * @brief Access to the root dataset flag.
     * @return True if this is a root dataset (has no base dataset),
     * otherwise False.
     */
    bool is_root() const;

    /**
     * @brief Access to the dataset state ID of this dataset.
     * @return The dataset state ID.
     */
    state_id_t state() const;

    /**
     * @brief Access to the ID of the base dataset.
     * @return The base dataset ID. Undefined if this is a root dataset.
     */
    dset_id_t base_dset() const;

    /**
     * @brief Read only access to the set of states.
     * @return  The set of states that are different from the base dataset.
     */
    const std::string& type() const;

    /**
     * @brief Generates a json serialization of this dataset.
     * @return A json serialization.
     */
    nlohmann::json to_json() const;

    /**
     * @brief Compare to another dataset.
     * @param ds    Dataset to compare with.
     * @return True if datasets identical, False otherwise.
     */
    bool equals(dataset& ds) const;

private:
    /// Dataset state.
    state_id_t _state;

    /// Base dataset ID.
    dset_id_t _base_dset;

    /// Is this a root dataset?
    bool _is_root;

    /// List of the types of datasetStates
    std::string _type;
};


#endif
