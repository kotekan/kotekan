#ifndef DATASET_MANAGER_HPP
#define DATASET_MANAGER_HPP

#include <string>
#include <vector>
#include <iostream>
#include <memory>

#include "json.hpp"
#include "errors.h"

// Alias certain types to give semantic meaning to the IDs
using dset_id = int32_t;
using state_id = size_t;  // This is the output format of a std::hash (64bit so we shouldn't have collisions)

// This type is used a lot so let's use an alias
using json = nlohmann::json;

// Forward declarations
class datasetState;
class datasetManager;

using state_uptr = std::unique_ptr<datasetState>;

/**
 * @brief A base class for representing state changes done to datasets.
 *
 * This is meant to be subclassed. All subclasses must implement a constructor
 * that calls the base class constructor to set any inner states. As a
 * convention tag should be the first argument, and inner the last (and should
 * be optional).
 *
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
        _inner_state(std::move(inner)) {};

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
     * Information the baseclass (e.g. tag, inner_state) is saved
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
    static state_uptr _create(std::string name, json & data,
                              state_uptr inner=nullptr);

    // Reference to the internal state
    state_uptr _inner_state = nullptr;

    // List of registerd subclass creating functions
    static std::map<std::string,
                    std::function<state_uptr(json&, state_uptr)>> _type_create_funcs;

    // Add as friend so it can walk the inner state
    friend datasetManager;

};

#define REGISTER_DATASET_STATE(T) int _register_ ## T = datasetState::_register_state_type<T>()


// Printing for datasetState
std::ostream& operator<<(std::ostream&, const datasetState&);


class freqState : virtual public datasetState {
public:
    freqState(std::string t, state_uptr inner=nullptr) :
        datasetState(std::move(inner))
    {
        std::cout << t << std::endl;
    }

    freqState(json & data, state_uptr inner) :
        freqState("Hello", std::move(inner)) {};

    json data_to_json() const override { json j; return j; }
};


class inputState : virtual public datasetState {
public:
    inputState(state_uptr inner=nullptr) :
        datasetState(std::move(inner)) {};

    inputState(json & data, state_uptr inner) :
        inputState(std::move(inner)) {};

    json data_to_json() const override { json j; return j; }
};


/**
 * @brief Manages sets of state changes applied to datasets.
 *
 * This is a singleton class. Use datasetManager::get to return a reference to
 * it.
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
     * @brief Register a state with the manager.
     *
     * @param trans A pointer to the state.
     * @returns The id assigned to the state.
     **/
    state_id add_state(state_uptr&& trans);

    /**
     * @brief Register a new dataset.
     *
     * @param trans The ID of an already registered state.
     * @param input ID of the input dataset.
     * @returns The ID assigned to the new dataset.
     **/
    dset_id add_dataset(state_id trans, dset_id input);

    /**
     * @brief Return the state table.
     *
     * @returns A string summarising the state table.
     **/
    std::string summary() const;

    /**
     * @brief Get a read-only vector of the states.
     *
     * @returns The set of states.
     **/
    const std::map<state_id, const datasetState *> states() const;

    /**
     * @brief Get a read-only vector of the datasets.
     *
     * @returns The set of datasets.
     **/
    const std::vector<std::pair<state_id, dset_id>> datasets() const;

    /**
     * @brief Get the states applied to generate the given dataset.
     *
     * @note This will flatten out inner state into the list. They are given the
     * same dataset ID as their parents.
     *
     * @returns A vector of the dataset ID and the state that was
     *          applied to previous element in the vector to generate it.
     **/
    const std::vector<std::pair<dset_id, datasetState *>> ancestors(
        dset_id dset
    ) const;

    /**
     * @brief Find the closest ancestor of a given type.
     *
     * @returns The dataset ID and the state that generated it.
     * Returns `{-1, nullptr}` if not found in ancestors.
     **/
    template<typename T>
    inline std::pair<dset_id, const T*> closest_ancestor_of_type(dset_id) const;

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
    std::map<state_id, state_uptr> _states;

    // Store a list of the datasets registered and what states they correspond to
    std::vector<std::pair<state_id, dset_id>> _datasets;
};


//
// Implementations of templated methods
//

template<typename T>
inline int datasetState::_register_state_type() {

    // Get the unique name for the type to generate the lookup key. This is
    // the same used by RTTI which is what we use to label the serialised
    // instances.
    std::string key = typeid(T).name();

    DEBUG("Registering state type: %s", key);

    // Generate a lambda function that creates an instance of the type
    datasetState::_type_create_funcs[key] =
        [](json & data, state_uptr inner) {
            return std::make_unique<T>(data, std::move(inner));
        };
    return 0;
}

template<typename T>
inline std::pair<dset_id, const T*>
datasetManager::closest_ancestor_of_type(dset_id dset) const {

    for(auto& t : ancestors(dset)) {
        if(typeid(*(t.second)) == typeid(T)) {
            return {t.first, dynamic_cast<T*>(t.second)};
        }
    }

    return {-1, nullptr};

}
#endif