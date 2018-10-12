#ifndef DATASET_MANAGER_HPP
#define DATASET_MANAGER_HPP

#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <mutex>
#include <condition_variable>

#include "json.hpp"
#include "errors.h"
#include "visUtil.hpp"
#include "restClient.hpp"

#define UNIQUE_NAME "/dataset_manager"
#define TIMEOUT_BROKER_SEC 30

// Alias certain types to give semantic meaning to the IDs
// This is the output format of a std::hash
// (64bit so we shouldn't have collisions)
using dset_id_t = size_t;
using state_id_t = size_t;

// This type is used a lot so let's use an alias
using json = nlohmann::json;
using namespace std;

// Forward declarations
class datasetState;
class datasetManager;

using state_uptr = unique_ptr<datasetState>;

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
    * @brief Dataset constructor.
    * @param state      The state of this dataset.
    * @param base_dset  The ID of the base datset.
    * @param is_root    True if this is a root dataset (has no base dataset).
    */
    dataset(state_id_t state, dset_id_t base_dset, bool is_root = false)
        : _state(state), _base_dset(base_dset), _is_root(is_root)
    { }

    /**
     * @brief Dataset constructor from json object.
     * The json object must have the following fields:
     * is_root:     boolean
     * state:       integer
     * base_dset    integer
     * @param js    Json object describing a dataset.
     */
    dataset(json& js);

    /**
     * @brief Access to the root dataset flag.
     * @return True if this is a root dataset (has no base dataset),
     * otherwise False.
     */
    const bool is_root() const;

    /**
     * @brief Access to the dataset state ID of this dataset.
     * @return The dataset state ID.
     */
    const state_id_t state() const;

    /**
     * @brief Access to the ID of the base dataset.
     * @return The base dataset ID. Undefined if this is a root dataset.
     */
    const dset_id_t base_dset() const;

    /**
     * @brief Generates a json serialization of this dataset.
     * @return A json serialization.
     */
    json to_json() const;

    /**
     * @brief Compare to another dataset.
     * @param ds    Dataset to compare with.
     * @return True if datasets identical, False otherwise.
     */
    const bool equals(dataset& ds) const;

private:
    /// Dataset state.
    state_id_t _state;

    /// Base dataset ID.
    dset_id_t _base_dset;

    /// Is this a root dataset?
    bool _is_root;
};

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
     * @brief Create a dataset state from a full json serialisation.
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
 * The dataset broker is a centralized part of the dataset management system.
 * Using it allows the synchronization of datasets and states between multiple
 * kotekan instances.
 *
 * @config use_ds_broker    If true, states and datasets will be registered with
 *                          the dataset broker. If an ancestor can not be found
 *                          locally, `closest_ancestor_of_type` will ask the
 *                          broker.
 * @config ds_broker_port   The port of the dataset broker (if `use_ds_broker`
 *                          is `True`).
 * @config ds_broker_host   Address to the dataset broker (if 'use_ds_broke` is
 *                          `True`. Prefer numerical address).
 * @config _path register_state     Path to the `register-state` endpoint (if
 *                                  `use_ds_broker` is `True`).
 * @config _path send_state         Path to the `send-state` endpoint (if
 *                                  `use_ds_broker` is `True`).
 * @config _path register_dataset   Path to the `register-dataset` endpoint (if
 *                                  `use_ds_broker` is `True`).
 * @config _path request_ancestors  Path to the `request_ancestors` endpoint (if
 *                                  `use_ds_broker` is `True`).
 *
 * @author Richard Shaw, Rick Nitsche
 **/
class datasetManager {
public:

    /**
     * @brief Get the global datasetManager.
     *
     * @returns A reference to the global datasetManager instance.
     **/
    static datasetManager& instance();

    /**
     * @brief Set and apply the static config to datasetManager
     * @param config         The config.
     */
    void apply_config(Config& config);

    // Remove the implicit copy/assignments to prevent copying
    datasetManager(const datasetManager&) = delete;
    void operator=(const datasetManager&) = delete;

    /**
     * @brief Register a new dataset.
     *
     * If `use_ds_broker` is set, this function will ask the dataset broker to
     * assign an ID to the new dataset and is blocking. If you want to
     * do something while waiting for the return value, use an std::future.
     *
     * @param trans The ID of an already registered state.
     * @param input ID of the input dataset.
     * @returns The ID assigned to the new dataset.
     **/
    dset_id_t add_dataset(const dataset ds);

    /**
     * @brief Register a state with the manager.
     *
     * If `use_ds_broker` is set, this function will also register the new state
     * with the broker (not blocking).
     *
     * The second argument of this function can be ignored. Its purpose is to
     * prevent compilation of this function with `T` not having the base class
     * `datasetState`.
     *
     * @param trans A pointer to the state.
     * @returns The id assigned to the state and a read-only pointer to the
     * state.
     **/
    template <typename T>
    inline pair<state_id_t, const T*> add_state(
            unique_ptr<T>&& state,
            typename std::enable_if<is_base_of<datasetState, T>::value>::type*
            = 0);

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
    const map<state_id_t, const datasetState *> states() const;

    /**
     * @brief Get a read-only vector of the datasets.
     *
     * @returns The set of datasets.
     **/
    const std::map<dset_id_t, dataset> datasets() const;

    /**
     * @brief Get the states applied to generate the given dataset.
     *
     * @note This will flatten out inner state into the list. They are given the
     * same dataset ID as their parents.
     *
     * @returns A vector of the dataset ID and the state that was
     *          applied to previous element in the vector to generate it.
     **/
    const vector<pair<dset_id_t, datasetState *>> ancestors(dset_id_t dset) const;

    /**
     * @brief Find the closest ancestor of a given type.
     *
     * If `use_ds_broker` is set and no ancestor of the given type is found,
     * this will ask the broker for a complete list of ancestors for the given
     * dataset. In that case, this function is blocking, until the broker
     * answeres. If you want to do something else, while waiting for the return
     * value of this function, use std::future.
     *
     * @returns The dataset ID and the state that generated it.
     * Returns `{<undefined>, nullptr}` if not found in ancestors.
     **/
    template<typename T>
    inline pair<dset_id_t, const T*> closest_ancestor_of_type(dset_id_t) const;

private:

    datasetManager() = default;

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
    static const state_id_t hash_state(datasetState& state);

    /**
     * @brief Calculate the hash of a dataset to use as the dset_id.
     *
     * @param state Dataset to hash.
     *
     * @returns Hash to use as ID.
     *
     * @note This deliberately isn't a method of dataset itself to ensure
     * that only the manager can issue hashes/IDs.
     **/
    static const dset_id_t hash_dataset(dataset& ds);

    /// register the given state with the dataset broker
    static void register_state(state_id_t state);

    /// register the given dataset with the dataset broker
    static void register_dataset(const dset_id_t hash, const dataset ds);

    /// callback function for register_state()
    static void register_state_callback(restReply reply);

    /// callback function for sending a state to the dataset broker
    /// from register_state_callback()
    static void send_state_callback(restReply reply);

    /// callback function for register_dataset()
    static void register_dataset_callback(restReply reply);

    /// request closest ancestor of type
    static void request_ancestor(dset_id_t dset_id, const char* type);

    /// callback function for request_ancestor()
    static void request_ancestor_callback(restReply reply);

    /// Store the list of all the registered states.
    static std::map<state_id_t, state_uptr> _states;

    /// Store a list of the datasets registered and what states
    /// and input datasets they correspond to
    static std::map<dset_id_t, dataset> _datasets;

    /// Lock for changing or using the states map.
    static std::mutex _lock_states;

    /// Lock for changing or using the datasets.
    static std::mutex _lock_dsets;

    /// Lock for the ancestors request cv.
    static std::mutex _lock_rqst;

    /// Lock for the register dataset cv.
    static std::mutex _lock_reg;

    /// conditional variable for registering dataset
    static std::condition_variable cv_register_dset;

    /// conditional variable for requesting ancestors
    static std::condition_variable cv_request_ancestor;

    // config params
    bool _use_broker = false;
    static std::string _path_register_state;
    static std::string _path_send_state;
    static std::string _path_register_dataset;
    static std::string _path_request_ancestor;
    static std::string _ds_broker_host;
    static unsigned short _ds_broker_port;
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
inline pair<dset_id_t, const T*>
datasetManager::closest_ancestor_of_type(dset_id_t dset) const {

    if (!_use_broker) {
        std::unique_lock<std::mutex> dslck(_lock_dsets);
        if (_datasets.find(dset) == _datasets.end())
            return {0, nullptr};
        dslck.unlock();
    }

    for(auto& t : ancestors(dset)) {
        if(typeid(*(t.second)) == typeid(T)) {
            return {t.first, dynamic_cast<T*>(t.second)};
        }
    }

    // no ancestor found locally -> ask broker
    if (_use_broker) {

        // lock for conditional variable
        std::unique_lock<std::mutex> lck(_lock_rqst);

        try {
            request_ancestor(dset, typeid(T).name());
        } catch (std::runtime_error& e) {
            ERROR("datasetManager: Failure requesting ancestors: %s", e.what());
            ERROR("datasetManager: Make sure the broker is running.\n" \
                  "Exiting...");
            raise(SIGINT);
        }
        // set timeout to hear back from callback function
        std::chrono::seconds timeout(TIMEOUT_BROKER_SEC);
        auto time_point = std::chrono::system_clock::now() + timeout;

        for(auto& t : ancestors(dset)) {
            if(typeid(*(t.second)) == typeid(T)) {
                return {t.first, dynamic_cast<T*>(t.second)};
            }
        }
        while(true) {
            if (cv_request_ancestor.wait_until(lck, time_point)
                  == std::cv_status::timeout) {
                ERROR("datasetManager: Timeout while requesting ancestors of " \
                      "type %s of dataset %zu.", typeid(T).name(), dset);
                ERROR("datasetManager: Exiting...");
                raise(SIGINT);
            }
            for(auto& t : ancestors(dset)) {
                if(typeid(*(t.second)) == typeid(T)) {
                    return {t.first, dynamic_cast<T*>(t.second)};
                }
            }
        }
    }

    return {0, nullptr};
}

template <typename T>
pair<state_id_t, const T*> datasetManager::add_state(
        unique_ptr<T>&& state,
        typename std::enable_if<is_base_of<datasetState, T>::value>::type*)
{

    state_id_t hash = hash_state(*state);

    // insert the new state
    // FIXME: check for and handle hash collicion
    std::lock_guard<std::mutex> slock(_lock_states);
    if (!_states.insert(std::pair<state_id_t, unique_ptr<T>>(hash,
                                                           move(state))).second)
        INFO("datasetManager: a state with hash %zu is already registered " \
             "locally.", hash);

    if (_use_broker) {
        try {
            register_state(hash);
        } catch (std::runtime_error& e) {
            ERROR("datasetManager: Failure registering state: %s", e.what());
            ERROR("datasetManager: Make sure the broker is running.\n" \
                  "Exiting...");
            raise(SIGINT);
        }
    }

    return pair<state_id_t, const T*>(hash, (const T*)(_states.at(hash).get()));
}
#endif
