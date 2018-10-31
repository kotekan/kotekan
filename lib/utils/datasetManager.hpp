#ifndef DATASET_MANAGER_HPP
#define DATASET_MANAGER_HPP

#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <fmt.hpp>

#include "json.hpp"
#include "errors.h"
#include "visUtil.hpp"
#include "restClient.hpp"
#include "prometheusMetrics.hpp"

#define UNIQUE_NAME "/dataset_manager"
#define TIMEOUT_BROKER_SEC 10

// Alias certain types to give semantic meaning to the IDs
// This is the output format of a std::hash
// (64bit so we shouldn't have collisions)
using dset_id_t = size_t;
using state_id_t = size_t;

// This type is used a lot so let's use an alias
using json = nlohmann::json;

// Forward declarations
class datasetState;
class datasetManager;

using state_uptr = std::unique_ptr<datasetState>;

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
     * @param data  Serialisation of config.
     * @param inner Inner state to compose with.
     * @returns The created datasetState.
     **/
    static state_uptr _create(string name, json & data,
                              state_uptr inner=nullptr);

    // Reference to the internal state
    state_uptr _inner_state = nullptr;

    // List of registered subclass creating functions
    static map<string, std::function<state_uptr(json&, state_uptr)>>&
        _registered_types();

    // Add as friend so it can walk the inner state
    friend datasetManager;

};

#define REGISTER_DATASET_STATE(T) int _register_ ## T = \
    datasetState::_register_state_type<T>()


// Printing for datasetState
std::ostream& operator<<(std::ostream&, const datasetState&);


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
            _freqs = data.get<std::vector<std::pair<uint32_t, freq_ctype>>>();
        } catch (std::exception& e) {
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
    freqState(std::vector<std::pair<uint32_t, freq_ctype>> freqs,
              state_uptr inner=nullptr) :
        datasetState(move(inner)),
        _freqs(freqs) {};

    /**
     * @brief Get frequency information (read only).
     *
     * @return The frequency information as a vector of
     *         {frequency ID, frequency index map}
     */
    const std::vector<std::pair<uint32_t, freq_ctype>>& get_freqs() const {
        return _freqs;
    }

private:
    /// Serialize the data of this state in a json object
    json data_to_json() const override {
        json j(_freqs);
        return j;
    }

    /// IDs that describe the subset that this dataset state defines
    std::vector<std::pair<uint32_t, freq_ctype>> _freqs;
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
            _inputs = data.get<std::vector<input_ctype>>();
        } catch (std::exception& e) {
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
    inputState(std::vector<input_ctype> inputs, state_uptr inner=nullptr) :
        datasetState(move(inner)),
        _inputs(inputs) {};

    /**
     * @brief Get input information (read only).
     *
     * @return The input information as a vector of input index maps.
     */
    const std::vector<input_ctype>& get_inputs() const {
        return _inputs;
    }

private:
    /// Serialize the data of this state in a json object
    json data_to_json() const override {
        json j(_inputs);
        return j;
    }

    /// The subset that this dataset state defines
    std::vector<input_ctype> _inputs;
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
            _prods = data.get<std::vector<prod_ctype>>();
        } catch (std::exception& e) {
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
    prodState(std::vector<prod_ctype> prods, state_uptr inner=nullptr) :
        datasetState(move(inner)),
        _prods(prods) {};

    /**
     * @brief Get product information (read only).
     *
     * @return The prod information as a vector of product index maps.
     */
    const std::vector<prod_ctype>& get_prods() const {
        return _prods;
    }

private:
    /// Serialize the data of this state in a json object
    json data_to_json() const override {
        json j(_prods);
        return j;
    }

    /// IDs that describe the subset that this dataset state defines
    std::vector<prod_ctype> _prods;
};


std::vector<stack_ctype> invert_stack(
    uint32_t num_stack, const std::vector<rstack_ctype>& stack_map);


/**
 * @brief A dataset state that describes a redundant baseline stacking.
 *
 * @author Richard Shaw
 */
class stackState : public datasetState {
public:
    /**
     * @brief Constructor
     * @param data  The stack information as serialized by
     *              stackState::to_json().
     * @param inner An inner state or a nullptr.
     */
    stackState(json& data, state_uptr inner) :
        datasetState(move(inner))
    {
        try {
            _rstack_map = data["rstack"].get<std::vector<rstack_ctype>>();
            _num_stack = data["num_stack"].get<uint32_t>();
            _stacked = data["stacked"].get<bool>();
        } catch (std::exception& e) {
             throw std::runtime_error("stackState: Failure parsing json data: "
                                      + std::string(e.what()));
        }
    };

    /**
     * @brief Constructor
     * @param rstack_map Definition of how the products were stacked.
     * @param num_stack Number of stacked visibilites.
     * @param inner  An inner state (optional).
     */
    stackState(uint32_t num_stack, std::vector<rstack_ctype>&& rstack_map,
               state_uptr inner=nullptr) :
        datasetState(std::move(inner)),
        _num_stack(num_stack),
        _rstack_map(rstack_map),
        _stacked(true) { }


    /**
     * @brief Constructor for an empty stack state
     *
     * This constructs a stackState describing a dataset that is not stacked.
     */
    stackState(state_uptr inner=nullptr) :
        datasetState(std::move(inner)),
        _stacked(false) {}


    /**
     * @brief Get stack map information (read only).
     *
     * For every product this says which stack to add the product into and
     * whether it needs conjugating before doing so.
     *
     * @return The stack map.
     */
    const std::vector<rstack_ctype>& get_rstack_map() const
    {
        return _rstack_map;
    }

    /**
     * @brief Get the number of stacks (read only).
     *
     * @return The number of stacks.
     */
    const uint32_t get_num_stack() const
    {
        return _num_stack;
    }

    /**
     * @brief Tells if the data is stacked (read only).
     *
     * @return True for stacked data, otherwise False.
     */
    const bool is_stacked() const
    {
        return _stacked;
    }

    /**
     * @brief Calculate and return the stack->prod mapping.
     *
     * This is calculated on demand and so a full fledged vector is returned.
     *
     * @returns The stack map.
     **/
    std::vector<stack_ctype> get_stack_map() const
    {
        return invert_stack(_num_stack, _rstack_map);
    }

    /// Serialize the data of this state in a json object
    json data_to_json() const override
    {
        return {{"rstack", _rstack_map }, {"num_stack", _num_stack},
                {"stacked", _stacked}};
    }

private:

    /// Total number of stacks
    uint32_t _num_stack;

    /// The stack definition
    std::vector<rstack_ctype> _rstack_map;

    /// Is the data stacked at all?
    bool _stacked;
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
 * std::pair<dset_id, const inputState*> input_state =
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
 * @conf use_ds_broker Bool. If true, states and datasets will be
 *                          registered with
 *                          the dataset broker. If an ancestor can not be found
 *                          locally, `closest_ancestor_of_type` will ask the
 *                          broker.
 * @conf ds_broker_port   Int. The port of the dataset broker
 *                          (if `use_ds_broker` is `True`).
 * @conf ds_broker_host   String. Address to the dataset broker
 *                          (if 'use_ds_broke` is `True`. Prefer numerical
 *                          address, because the DNS lookup is blocking).
 * @conf _path register_state     String. Path to the `register-state`
 *                                  endpoint (if `use_ds_broker` is `True`).
 * @conf _path send_state         String. Path to the `send-state` endpoint
 *                                  (if `use_ds_broker` is `True`).
 * @conf _path register_dataset   String. Path to the `register-dataset`
 *                                  endpoint (if `use_ds_broker` is `True`).
 * @conf _path request_ancestors  String. Path to the `request_ancestors`
 *                                  endpoint (if `use_ds_broker` is `True`).
 *
 *
 * @par metrics
 * @metric kotekan_datasetbroker_error_count Number of errors encountered in
 *                                           communication with the broker.
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
     * assign an ID to the new dataset.
     *
     * @param ds The dataset to be added.
     * @param ignore_broker If true, the dataset is not sent to the broker.
     * @returns The ID assigned to the new dataset.
     **/
    dset_id_t add_dataset(const dataset ds, bool ignore_broker = false);

    /**
     * @brief Register a state with the manager.
     *
     * If `use_ds_broker` is set, this function will also register the new state
     * with the broker.
     *
     * The third argument of this function is to
     * prevent compilation of this function with `T` not having the base class
     * `datasetState`.
     *
     * @param state The state to be added.
     * @param ignore_broker If true, the state is not sent to the broker.
     * @returns The id assigned to the state and a read-only pointer to the
     * state.
     **/
    template <typename T>
    inline std::pair<state_id_t, const T*> add_state(
            std::unique_ptr<T>&& state,
            bool ignore_broker = false,
            typename std::enable_if<std::is_base_of<datasetState,
                                    T>::value>::type* = 0);

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
    const std::vector<std::pair<dset_id_t, datasetState *>>
    ancestors(dset_id_t dset) const;

    /**
     * @brief Find the closest ancestor of a given type.
     *
     * If `use_ds_broker` is set and no ancestor of the given type is found,
     * this will ask the broker for a complete list of ancestors for the given
     * dataset. In that case, this function is blocking, until the broker
     * answeres. If you want to do something else, while waiting for the return
     * value of this function, use std::future.
     *
     * @returns A read-only pointer to the ancestor state.
     * Returns a `nullptr` if not found in ancestors or in a
     * failure case.
     **/
    template<typename T>
    inline const T* dataset_state(dset_id_t) const;

private:
    /// Constructor
    datasetManager();

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
     * @param ds Dataset to hash.
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
    static void request_ancestor(dset_id_t dset_id, const char *type);

    /// callback function for request_ancestor()
    static void request_ancestor_callback(restReply reply);

    template<typename T>
    static inline const T* get_closest_ancestor(dset_id_t dset);

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

    /// conditional variable for requesting ancestors
    static std::condition_variable _cv_request_ancestor;

    /// counter for connection and parsing errors
    static std::atomic<uint32_t> _conn_error_count;

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
            return std::make_unique<T>(data, move(inner));
        };
    return 0;
}

/* TODO:
 * atm this receives a list from the broker of all it is missing to know the
 * ancestor itself. Instead, receive only the requested ancexstor state and
 * cache it in a hash map. The callback function then receives a string
 * describing the type and has to find the type_index to use as a key for the
 * hash map and it has to dynamically cast to that type. */
template<typename T>
inline const T* datasetManager::dataset_state(dset_id_t dset) const {
    if (!_use_broker) {
        // check if we know that dataset at all
        {
            std::unique_lock<std::mutex> dslck(_lock_dsets);
            if (_datasets.find(dset) == _datasets.end())
                return nullptr;
        }
    }

    // is the ancestor known locally?
    const T* ancestor = get_closest_ancestor<T>(dset);
    if (ancestor)
        return ancestor;

    // no ancestor found locally -> ask broker
    if (_use_broker) {

        // lock for conditional variable
        std::unique_lock<std::mutex> lck(_lock_rqst);

        try {
            request_ancestor(dset, typeid(T).name());
        } catch (std::runtime_error& e) {
            prometheusMetrics::instance().add_process_metric(
                        "kotekan_datasetbroker_error_count", UNIQUE_NAME,
                        ++_conn_error_count);
            std::string msg = fmt::format(
                        "datasetManager: Failure requesting ancestors, make " \
                        "sure the broker is running: {}", e.what());
            throw std::runtime_error(msg);
        }
        // set timeout to hear back from callback function
        std::chrono::seconds timeout(TIMEOUT_BROKER_SEC);
        auto time_point = std::chrono::system_clock::now() + timeout;

        while(true) {
            if (!_cv_request_ancestor.wait_until(lck, time_point,
                               std::bind(get_closest_ancestor<T>, dset))) {

                std::string msg = fmt::format(
                            "datasetManager: Timeout while requesting " \
                            "ancestors of type {} of dataset {}.",
                            typeid(T).name(), dset);
                prometheusMetrics::instance().add_process_metric(
                            "kotekan_datasetbroker_error_count", UNIQUE_NAME,
                            ++_conn_error_count);
                throw std::runtime_error(msg);
            }
            // If the request was successful, the ancestor will be here:
            ancestor = get_closest_ancestor<T>(dset);
            if (ancestor)
                return ancestor;
        }
    }

    // not found
    return nullptr;
}

template <typename T>
std::pair<state_id_t, const T*> datasetManager::add_state(
        std::unique_ptr<T>&& state,
        bool ignore_broker,
        typename std::enable_if<std::is_base_of<datasetState, T>::value>::type*)
{

    state_id_t hash = hash_state(*state);

    // insert the new state
    // FIXME: check for and handle hash collicion
    std::lock_guard<std::mutex> slock(_lock_states);
    if (!_states.insert(
            std::pair<state_id_t, std::unique_ptr<T>>(hash, move(state))).second)
        INFO("datasetManager: a state with hash %zu is already registered " \
             "locally.", hash);

    if (_use_broker && !ignore_broker) {
        try {
            register_state(hash);
        } catch (std::runtime_error& e) {
            prometheusMetrics::instance().add_process_metric(
                        "kotekan_datasetbroker_error_count", UNIQUE_NAME,
                        ++_conn_error_count);
            std::string msg = fmt::format(
                        "datasetManager: Failure registering state: {}\n" \
                        "datasetManager: Make sure the broker is running."
                        , e.what());
            throw std::runtime_error(msg);
        }
    }

    return std::pair<state_id_t, const T*>(hash,
                                           (const T*)(_states.at(hash).get()));
}

template<typename T>
inline const T*
datasetManager::get_closest_ancestor(dset_id_t dset) {

    std::lock(_lock_dsets, _lock_states);
    std::lock_guard<std::mutex> dslock(_lock_dsets, std::adopt_lock);
    std::lock_guard<std::mutex> slock(_lock_states, std::adopt_lock);

    // Walk up from the current node to the root, extracting pointers to the
    // states
    bool root = false;
    while(!root) {
        datasetState* t;
        try {
            t = _states.at(_datasets.at(dset).state()).get();
        } catch (std::out_of_range& e) {
            // we don't have the base dataset
            DEBUG2("datasetManager: found a dead reference when looking for " \
                   "locally known ancestor: %s", e.what());
            break;
        }

        // Walk over the inner states.
        while(t != nullptr) {
            if(typeid(*t) == typeid(T))
                return dynamic_cast<T*>(t);
            t = t->_inner_state.get();
        }

        // if this is the root dataset, we are done
        root = _datasets.at(dset).is_root();

        // Move on to the parent dataset...
        dset = _datasets.at(dset).base_dset();
    }

    // not found
    return nullptr;
}

#endif
