#ifndef DATASET_MANAGER_HPP
#define DATASET_MANAGER_HPP

#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <mutex>

#include "json.hpp"
#include "errors.h"
#include "visUtil.hpp"

// Alias certain types to give semantic meaning to the IDs
using dset_id = int32_t;
using state_id = size_t;  // This is the output format of a std::hash (64bit so we shouldn't have collisions)

// This type is used a lot so let's use an alias
using json = nlohmann::json;
using namespace std;

// Forward declarations
class datasetState;
class datasetManager;

using state_uptr = unique_ptr<datasetState>;

/**
 * @brief A base class for representing state changes done to datasets.
 *
 * This is meant to be subclassed. All subclasses must implement a constructor
 * that calls the base class constructor to set any inner states. As a
 * convention it should pass the data and as a last argument `inner` (which
 * should be optional).
 *
 * @author Richard Shaw, Rick Nitsche
 **/
class datasetState {
public:

    /**
     * @brief Create a datasetState
     *
     * @param inner An internal state that this one wraps. Think of this
     *              like function composition.
     **/
    datasetState(state_uptr inner=nullptr) :
        _inner_state(move(inner)) {};

    virtual ~datasetState() {};

    /**
     * @brief Create a dataset from a full json serialisation.
     *
     * This will correctly instantiate the correct types and reconstruct all
     * inner states.
     *
     * @param j Full JSON serialisation.
     * @returns The created datasetState.
     **/
    static state_uptr from_json(json& j);

    /**
     * @brief Full serialisation of state into JSON.
     *
     * @returns JSON serialisation of state.
     **/
    json to_json() const;

    /**
     * @brief Save the internal data of this instance into JSON.
     *
     * This must be implement by any derived classes and should save the
     * information needed to reconstruct any subclass specific internals.
     * Information of the baseclass (e.g. inner_state) is saved
     * separately.
     *
     * @returns JSON representing the internal state.
     **/
    virtual json data_to_json() const = 0;

    /**
     * @brief Register a derived datasetState type
     *
     * @warning You shouldn't call this directly. It's only public so the macro
     * can call it.
     *
     * @returns Always returns zero.
     **/
    template<typename T>
    static inline int _register_state_type();

private:

    /**
     * @brief Create a datasetState subclass from a json serialisation.
     *
     * @param name  Name of subclass to create.
     * @param tag   Unique string label.
     * @param data  Serialisation of config.
     * @param inner Inner state to compose with.
     * @returns The created datasetState.
     **/
    static state_uptr _create(string name, json & data,
                              state_uptr inner=nullptr);

    // Reference to the internal state
    state_uptr _inner_state = nullptr;

    // List of registered subclass creating functions
    static map<string, function<state_uptr(json&, state_uptr)>>&
        _registered_types();

    // Add as friend so it can walk the inner state
    friend datasetManager;

};

#define REGISTER_DATASET_STATE(T) int _register_ ## T = \
    datasetState::_register_state_type<T>()


// Printing for datasetState
ostream& operator<<(ostream&, const datasetState&);


/**
 * @brief A dataset state that describes the frequencies in a datatset.
 *
 * @author Richard Shaw, Rick Nitsche
 */
class freqState : public datasetState {
public:
    /**
     * @brief Constructor
     * @param data  The frequency information as serialized by
     *              freqState::to_json().
     * @param inner An inner state or a nullptr.
     */
    freqState(json & data, state_uptr inner) :
        datasetState(move(inner)) {
        try {
            _freqs = data.get<vector<pair<uint32_t, freq_ctype>>>();
        } catch (exception& e) {
             throw std::runtime_error("freqState: Failure parsing json data ("
                                      + data.dump() + "): " + e.what());
        }
    };

    /**
     * @brief Constructor
     * @param freqs The frequency information as a vector of
     *              {frequency ID, frequency index map}.
     * @param inner An inner state (optional).
     */
    freqState(vector<pair<uint32_t, freq_ctype>> freqs,
              state_uptr inner=nullptr) :
        datasetState(move(inner)),
        _freqs(freqs) {};

    /**
     * @brief Get frequency information (read only).
     *
     * @return The frequency information as a vector of
     *         {frequency ID, frequency index map}
     */
    const vector<pair<uint32_t, freq_ctype>>& get_freqs() const {
        return _freqs;
    }

private:
    /// Serialize the data of this state in a json object
    json data_to_json() const override {
        json j(_freqs);
        return j;
    }

    /// IDs that describe the subset that this dataset state defines
    vector<pair<uint32_t, freq_ctype>> _freqs;
};


/**
 * @brief A dataset state that describes the inputs in a datatset.
 *
 * @author Richard Shaw, Rick Nitsche
 */
class inputState : public datasetState {
public:
    /**
     * @brief Constructor
     * @param data  The input information as serialized by
     *              inputState::to_json().
     * @param inner An inner state or a nullptr.
     */
    inputState(json & data, state_uptr inner) :
        datasetState(move(inner)) {
        try {
            _inputs = data.get<vector<input_ctype>>();
        } catch (exception& e) {
             throw std::runtime_error("inputState: Failure parsing json data ("
                                      + data.dump() + "): " + e.what());
        }
    };

    /**
     * @brief Constructor
     * @param inputs The input information as a vector of
     *               input index maps.
     * @param inner  An inner state (optional).
     */
    inputState(vector<input_ctype> inputs, state_uptr inner=nullptr) :
        datasetState(move(inner)),
        _inputs(inputs) {};

    /**
     * @brief Get input information (read only).
     *
     * @return The input information as a vector of input index maps.
     */
    const vector<input_ctype>& get_inputs() const {
        return _inputs;
    }

private:
    /// Serialize the data of this state in a json object
    json data_to_json() const override {
        json j(_inputs);
        return j;
    }

    /// The subset that this dataset state defines
    vector<input_ctype> _inputs;
};


/**
 * @brief A dataset state that describes the products in a datatset.
 *
 * @author Richard Shaw, Rick Nitsche
 */
class prodState : public datasetState {
public:
    /**
     * @brief Constructor
     * @param data  The product information as serialized by
     *              prodState::to_json().
     * @param inner An inner state or a nullptr.
     */
    prodState(json & data, state_uptr inner) :
        datasetState(move(inner)) {
        try {
            _prods = data.get<vector<prod_ctype>>();
        } catch (exception& e) {
             throw std::runtime_error("prodState: Failure parsing json data ("
                                      + data.dump() + "): " + e.what());
        }
    };

    /**
     * @brief Constructor
     * @param prods The product information as a vector of
     *              product index maps.
     * @param inner An inner state (optional).
     */
    prodState(vector<prod_ctype> prods, state_uptr inner=nullptr) :
        datasetState(move(inner)),
        _prods(prods) {};

    /**
     * @brief Get product information (read only).
     *
     * @return The prod information as a vector of product index maps.
     */
    const vector<prod_ctype>& get_prods() const {
        return _prods;
    }

private:
    /// Serialize the data of this state in a json object
    json data_to_json() const override {
        json j(_prods);
        return j;
    }

    /// IDs that describe the subset that this dataset state defines
    vector<prod_ctype> _prods;
};


/**
 * @brief Manages sets of state changes applied to datasets.
 *
 * This is a singleton class. Use `datasetManager::instance()` to get a
 * reference to it.
 *
 * This is used to manage the states of datasets that get passed through
 * kotekan processes.
 * A process in the kotekan pipeline may use the dataset ID found in an incoming
 * frame to get a set of states from the datasetManager.
 * E.g.
 * ```
 * pair<dset_id, const inputState*> input_state =
 *          dm.closest_ancestor_of_type<inputState>(ds_id_from_frame);
 * const std::vector<input_ctype>& inputs = input_state.second->get_inputs();
 * ```
 * to receive information about the inputs the datsets in the frames contain.
 *
 * A process that changes the state of the dataset in the frames it processes
 * should inform the datasetManager by adding a state. For example
 * ```
 * std::pair<state_id, const inputState*> new_state =
 *          dm.add_state(std::make_unique<inputState>(new_inputs,
 *                                         make_unique<prodState>(new_prods)));
 *  dset_id new_ds_id = dm.add_dataset(new_state.first, old_state.first);
 * ```
 * Adds an input state as well as a product dataset state to the manager. The
 * process should then write `new_ds_id` to its outgoing frames.
 *
 * If a process is altering more than one type of dataset state, it can add
 * `inner` states to the one it passes to the dataset manager.
 *
 * @author Richard Shaw, Rick Nitsche
 *
 * TODO: Centralized datasetBroker, so that states can be shared over multiple
 * kotekan instances.
 **/
class datasetManager {
public:

    /**
     * @brief Get the global datasetManager.
     *
     * @returns A reference to the global datasetManager instance.
     **/
    static datasetManager& instance();

    // Remove the implicit copy/assignments to prevent copying
    datasetManager(const datasetManager&) = delete;
    void operator=(const datasetManager&) = delete;

    /**
     * @brief Register a new dataset.
     *
     * @param trans The ID of an already registered state.
     * @param input ID of the input dataset.
     * @returns The ID assigned to the new dataset.
     **/
    dset_id add_dataset(state_id trans, dset_id input);

    /**
     * @brief Register a state with the manager.
     *
     * @param trans A pointer to the state.
     * @returns The id assigned to the state.
     **/
    template <typename T>
    inline pair<state_id, const T*> add_state(unique_ptr<T>&& state);

    /**
     * @brief Return the state table.
     *
     * @returns A string summarising the state table.
     **/
    string summary() const;

    /**
     * @brief Get a read-only vector of the states.
     *
     * @returns The set of states.
     **/
    const map<state_id, const datasetState *> states() const;

    /**
     * @brief Get a read-only vector of the datasets.
     *
     * @returns The set of datasets.
     **/
    const vector<pair<state_id, dset_id>> datasets() const;

    /**
     * @brief Get the states applied to generate the given dataset.
     *
     * @note This will flatten out inner state into the list. They are given the
     * same dataset ID as their parents.
     *
     * @returns A vector of the dataset ID and the state that was
     *          applied to previous element in the vector to generate it.
     **/
    const vector<pair<dset_id, datasetState *>> ancestors(dset_id dset) const;

    /**
     * @brief Find the closest ancestor of a given type.
     *
     * @returns The dataset ID and the state that generated it.
     * Returns `{-1, nullptr}` if not found in ancestors.
     **/
    template<typename T>
    inline pair<dset_id, const T*> closest_ancestor_of_type(dset_id) const;

private:

    datasetManager() { };

    /**
     * @brief Calculate the hash of a datasetState to use as the state_id.
     *
     * @param state State to hash.
     *
     * @returns Hash to use as ID.
     *
     * @note This deliberately isn't a method of datasetState itself to ensure
     * that only the manager can issue hashes/IDs.
     **/
    state_id hash_state(datasetState& state);

    // Store the list of all the registered states.
    map<state_id, state_uptr> _states;

    // Store a list of the datasets registered and what states they correspond to
    vector<pair<state_id, dset_id>> _datasets;

    // Lock for changing or using the states map.
    mutable std::mutex _lock_states;

    // Lock for changing or using the datasets.
    mutable std::mutex _lock_dsets;
};


//
// Implementations of templated methods
//

template<typename T>
inline int datasetState::_register_state_type() {

    // Get the unique name for the type to generate the lookup key. This is
    // the same used by RTTI which is what we use to label the serialised
    // instances.
    string key = typeid(T).name();

    DEBUG("Registering state type: %s", key.c_str());

    // Generate a lambda function that creates an instance of the type
    datasetState::_registered_types()[key] =
        [](json & data, state_uptr inner) -> state_uptr {
            return make_unique<T>(data, move(inner));
        };
    return 0;
}

template<typename T>
inline pair<dset_id, const T*>
datasetManager::closest_ancestor_of_type(dset_id dset) const {

    if (dset < 0 || _datasets.size() <= (size_t)dset)
        return {-1, nullptr};

    for(auto& t : ancestors(dset)) {
        if(typeid(*(t.second)) == typeid(T)) {
            return {t.first, dynamic_cast<T*>(t.second)};
        }
    }

    return {-1, nullptr};

}

// FIXME: add sth like
// typename enable_if_t<is_base_of<datasetState, T>::value>::state
// So that compilation for T not having datasetState as a base class fails.
template <typename T>
pair<state_id, const T*> datasetManager::add_state(unique_ptr<T>&& state) {
    state_id hash = hash_state(*state);
    std::lock_guard<std::mutex> lock(_lock_states);
    if (!_states.insert(std::pair<state_id, unique_ptr<T>>(hash,
                                                           move(state))).second)
        INFO("datasetManager a state with hash %d is already registered.",
             hash);
    return pair<state_id, const T*>(hash, (const T*)(_states.at(hash).get()));
}
#endif
