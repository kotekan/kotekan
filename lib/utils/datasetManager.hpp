#ifndef DATASET_MANAGER_HPP
#define DATASET_MANAGER_HPP

#include <stdint.h>
#include <time.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>
#include <inttypes.h>

#include "json.hpp"

#include "Config.hpp"
#include "datasetState.hpp"
#include "errors.h"
#include "prometheusMetrics.hpp"
#include "restClient.hpp"
#include "signal.h"


#define DS_UNIQUE_NAME "/dataset_manager"

// names of broker endpoints
const std::string PATH_REGISTER_STATE = "/register-state";
const std::string PATH_SEND_STATE = "/send-state";
const std::string PATH_REGISTER_DATASET = "/register-dataset";
const std::string PATH_UPDATE_DATASETS = "/update-datasets";
const std::string PATH_REQUEST_STATE = "/request-state";

// Alias certain types to give semantic meaning to the IDs
// This is the output format of a std::hash
// (64bit so we shouldn't have collisions)
using dset_id_t = size_t;
using state_id_t = size_t;


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
    * @brief Dataset constructor for a root dataset.
    * @param state      The state of this dataset.
    * @param types      The set of state types that are different from the base
    *                   dataset.
    */
    dataset(state_id_t state, std::set<std::string> types)
        : _state(state), _base_dset(0), _is_root(true),
          _types(types) { }

    /**
    * @brief Dataset constructor for a non-root dataset.
    * @param state      The state of this dataset.
    * @param base_dset  The ID of the base datset.
    * @param types      The set of state types that are different from the base
    *                   dataset.
    */
    dataset(state_id_t state, dset_id_t base_dset,
            std::set<std::string> types)
        : _state(state), _base_dset(base_dset), _is_root(false),
          _types(types) { }

    /**
     * @brief Dataset constructor from json object.
     * The json object must have the following fields:
     * is_root:     boolean
     * state:       integer
     * base_dset    integer
     * types        list of strings
     * @param js    Json object describing a dataset.
     */
    dataset(json& js);

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
    const std::set<std::string>& types() const;

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
    bool equals(dataset& ds) const;

private:
    /// Dataset state.
    state_id_t _state;

    /// Base dataset ID.
    dset_id_t _base_dset;

    /// Is this a root dataset?
    bool _is_root;

    /// List of the types of datasetStates
    std::set<std::string> _types;
};


/**
 * @brief Manages sets of state changes applied to datasets.
 *
 * This is a singleton class. Use `datasetManager::instance()` to get a
 * reference to it.
 *
 * The datasetManager is used to manage the states of datasets that get passed
 * through kotekan processes.
 * A process in the kotekan pipeline may use the dataset ID found in an incoming
 * frame to get a set of states from the datasetManager.
 *
 * To receive information about the inputs the datsets in the frames contain, it
 * could do thew following:
 * ```
 * auto input_state = dm.dataset_state<inputState>(ds_id_from_frame);
 * const std::vector<input_ctype>& inputs = input_state->get_inputs();
 * ```
 *
 * A process that changes the state of the dataset in the frames it processes
 * should inform the datasetManager by adding a new state and dataset.
 *  If a process is altering more than one type of dataset state, it can add
 * `inner` states to the one it passes to the dataset manager.
 * The following adds an input state as well as a product state. The
 * process should then write `new_ds_id` to its outgoing frames.
 * ```
 * auto new_state = dm.add_state(std::make_unique<inputState>(
 *                              new_inputs, make_unique<prodState>(new_prods)));
 *  dset_id_t new_ds_id = dm.add_dataset(old_dataset_id, new_state);
 * ```
 *
 *
 * The dataset broker is a centralized part of the dataset management system.
 * Using it allows the synchronization of datasets and states between multiple
 * kotekan instances.
 *
 * @conf use_dataset_broker     Bool. If true, states and datasets will be
 *                              registered with the dataset broker. If an
 *                              ancestor can not be found locally,
 *                              `dataset_state` will ask the broker.
 * @conf ds_broker_port         Int. The port of the dataset broker (if
 *                              `use_dataset_broker` is `True`). Default 12050.
 * @conf ds_broker_host         String. Address to the dataset broker (if
 *                              'use_ds_broke` is `True`. Prefer numerical
 *                              address, because the DNS lookup is blocking).
 *                              Default "127.0.0.1".
 * @conf retry_wait_time_ms     Int. Time to wait after failed request to broker
 *                              before retrying in ms. Default 1000.
 * @conf retries_rest_client    Int. Retry value passed to libevent. Caution:
 *                              Infinite retries are performed by the
 *                              datasetManager. Default 0.
 * @conf timeout_rest_client_s  Int. Timeout value passed to libevent. -1 will
 *                              use libevent default value (50s). Default 100.
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
     *
     * @returns A reference to the global datasetManager instance.
     */
    static datasetManager& instance(Config& config);

    // Remove the implicit copy/assignments to prevent copying
    datasetManager(const datasetManager&) = delete;
    void operator=(const datasetManager&) = delete;

    /**
     * @brief Register a new root dataset.
     *
     * If `use_dataset_broker` is set, this function will ask the dataset broker
     * to assign an ID to the new dataset.
     *
     * @param state         The ID of the dataset state that describes the
     *                      difference to the base dataset.
     * @returns The ID assigned to the new dataset.
     **/
    dset_id_t add_dataset(state_id_t state);

    /**
     * @brief Register a new non-root dataset.
     *
     * If `use_dataset_broker` is set, this function will ask the dataset broker
     * to assign an ID to the new dataset.
     *
     * @param base_dset     The ID of the dataset this dataset is based on.
     * @param state         The ID of the dataset state that describes the
     *                      difference to the base dataset.
     * @returns The ID assigned to the new dataset.
     **/
    dset_id_t add_dataset(dset_id_t base_dset, state_id_t state);

    /**
     * @brief Register a state with the manager.
     *
     * If `use_dataset_broker` is set, this function will also register the new
     * state with the broker.
     *
     * The third argument of this function is to
     * prevent compilation of this function with `T` not having the base class
     * `datasetState`.
     *
     * @param state The state to be added.
     * @returns The id assigned to the state and a read-only pointer to the
     * state.
     **/
    template <typename T>
    inline std::pair<state_id_t, const T*> add_state(
            std::unique_ptr<T>&& state,
            typename std::enable_if<std::is_base_of<datasetState,
                                    T>::value>::type* = 0);

    /**
     * @brief Return the state table.
     *
     * @returns A string summarising the state table.
     **/
    std::string summary();

    /**
     * @brief Get a read-only vector of the states.
     *
     * @returns The set of states.
     **/
    const map<state_id_t, const datasetState *> states();

    /**
     * @brief Get a read-only vector of the datasets.
     *
     * @returns The set of datasets.
     **/
    const std::map<dset_id_t, dataset> datasets();

    /**
     * @brief Find the closest ancestor of a given type.
     *
     * If `use_dataset_broker` is set and no ancestor of the given type is found,
     * this will ask the broker for a complete list of ancestors for the given
     * dataset. In that case, this function is blocking, until the broker
     * answeres. If you want to do something else, while waiting for the return
     * value of this function, use std::future.
     *
     * @returns A read-only pointer to the ancestor state.
     * Returns a `nullptr` if not found in ancestors or in a
     * failure case.
     **/
    template<typename T> inline const T* dataset_state(dset_id_t dset);

private:
    /// Constructor
    datasetManager() :
        _conn_error_count(0),
        _timestamp_update(json(0)),
        _stop_request_threads(false),
        _n_request_threads(0),
        _config_applied(false),
        _rest_client(restClient::instance()) {}

    /// Generate a private static instance so that the overloaded instance()
    /// members can use the same static variable
    static datasetManager& private_instance();

    /// Destructor. Joins all request threads.
    ~datasetManager();

    /**
     * @brief Register a new dataset.
     *
     * If `use_dataset_broker` is set, this function will ask the dataset broker
     * to assign an ID to the new dataset.
     *
     * @param ds     The dataset to get registered.
     * @returns The ID assigned to the new dataset.
     **/
    dset_id_t add_dataset(dataset ds);

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
    ancestors(dset_id_t dset);

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
    state_id_t hash_state(datasetState& state) const;

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
    dset_id_t hash_dataset(dataset& ds) const;

    /// register the given state with the dataset broker
    void register_state(state_id_t state);

    /// register the given dataset with the dataset broker
    void register_dataset(const dset_id_t hash, const dataset ds);

    /// parser function for register_state()
    bool register_state_parser(std::string& reply);

    /// parser function for sending a state to the dataset broker
    /// from register_state_parser()
    bool send_state_parser(std::string& reply);

    /// parser function for register_dataset()
    bool register_dataset_parser(std::string& reply);

    /// request an update on the topology of datasets (blocking)
    void update_datasets(dset_id_t ds_id);

    /// Helper function to parse the reply for update_datasets()
    bool parse_reply_dataset_update(restReply reply);

    /// To be left in a detached thread: Infinitely retries request parse.
    /// Stopped by the destructor if still unsuccessfully retrying.
    void request_thread(const json&& request, const std::string&& endpoint,
                        const std::function<bool(std::string&)>&& parse_reply);

    /// Gets the closest ancestor of the given dataset of the given dataset
    /// state type. If it is not known locally, it will be sent from the broker.
    template<typename T>
    inline const T* get_closest_ancestor(dset_id_t dset);

    /// Wait for any ongoing requests of the same state OR request state.
    template<typename T>
    inline const T* request_state(state_id_t state_id) ;

    /// Store the list of all the registered states.
    std::map<state_id_t, state_uptr> _states;

    /// Store a list of the datasets registered and what states
    /// and input datasets they correspond to
    std::map<dset_id_t, dataset> _datasets;

    /// Set of known root datasets (Protected by _lock_datasets).
    std::set<dset_id_t> _known_roots;

    /// Lock for changing or using the states map.
    std::mutex _lock_states;

    /// Lock for changing or using the datasets.
    std::mutex _lock_dsets;

    /// Lock for the ancestors request cv.
    std::mutex _lock_rqst;

    /// Lock for the register dataset cv.
    std::mutex _lock_reg;

    /// Lock for the receive state cv.
    std::mutex _lock_recv_state;

    /// Lock for the stop request threads cv
    std::mutex _lock_stop_request_threads;

    /// Lock to only allow one dataset update at a time.
    std::mutex _lock_ds_update;

    /// conditional variable to signal a received state.
    std::condition_variable _cv_received_state;

    /// Condition Variable to signal request threads to stop on exit.
    std::condition_variable _cv_stop_request_threads;

    /// counter for connection and parsing errors
    std::atomic<uint32_t> _conn_error_count;

    /// Timestamp of last topology update (generated by broker).
    /// It is protected by _lock_dsets.
    json _timestamp_update;

    /// set of the states currently requested from the broker.
    /// Protected by _lock_recv_state.
    std::set<state_id_t> _requested_states;

    /// Set to true by the destructor.
    std::atomic<bool> _stop_request_threads;

    /// Number of running request threads (for destructor to wait).
    uint64_t _n_request_threads;

    /// Check if config loaded for this singleton before handing out instances
    std::atomic<bool> _config_applied;

    /// config params
    bool _use_broker = false;
    std::string _ds_broker_host;
    unsigned short _ds_broker_port;
    uint32_t _retry_wait_time_ms;
    uint32_t _retries_rest_client;
    int32_t _timeout_rest_client_s;

    /// a reference to the restClient instance
    restClient& _rest_client;
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

    DEBUG("Registering state type: %s", key.c_str());

    // Generate a lambda function that creates an instance of the type
    datasetState::_registered_types()[key] =
        [](json & data, state_uptr inner) -> state_uptr {
            return std::make_unique<T>(data, move(inner));
        };
    return 0;
}

template<typename T>
inline const T* datasetManager::dataset_state(dset_id_t dset) {

    if (!_use_broker)
        return get_closest_ancestor<T>(dset);

    // get an update on the dataset topology (blocking)
    update_datasets(dset);
    
    // get the state or ask broker for it
    const T* state = get_closest_ancestor<T>(dset);

    return state;
}

template <typename T>
std::pair<state_id_t, const T*> datasetManager::add_state(
        std::unique_ptr<T>&& state,
        typename std::enable_if<std::is_base_of<datasetState, T>::value>::type*)
{

    state_id_t hash = hash_state(*state);

    // check if there is a hash collision
    if (_states.find(hash) != _states.end()) {
        auto find = _states.find(hash);
        if (!state->equals(*(find->second))) {
            // FIXME: hash collision. make the value a vector and store same
            // hash entries? This would mean the state/dset has to be sent
            // when registering.
            ERROR("datasetManager: Hash collision!\n"
                  "The following states have the same hash (0x%" PRIx64 ")." \
                  "\n\n%s\n\n%s\n\n" \
                  "datasetManager: Exiting...",
                  hash, state->to_json().dump().c_str(),
                  find->second->to_json().dump().c_str());
            raise(SIGINT);
        }
    } else {
        // insert the new state
        std::lock_guard<std::mutex> slock(_lock_states);
        if (!_states.insert(std::pair<state_id_t, std::unique_ptr<T>>
                                                  (hash, move(state))).second) {
            DEBUG("datasetManager: a state with hash 0x%" PRIx64 " is already "\
                 "registered locally.", hash);
        }

        // tell the broker about it
        if (_use_broker)
            register_state(hash);
    }

    return std::pair<state_id_t, const T*>(hash,
                                           (const T*)(_states.at(hash).get()));
}

template<typename T>
inline const T*
datasetManager::get_closest_ancestor(dset_id_t dset) {
    {
        std::lock_guard<std::mutex> dslock(_lock_dsets);
        state_id_t ancestor;

        // Check if we can find requested state in dataset topology.
        // Walk up from the current node to the root.
        while(true) {
            // Search for the requested type in each dataset (includes inner
            // states).
            try {
                if (_datasets.at(dset).types().count(typeid(T).name())) {
                    ancestor = _datasets.at(dset).state();
                    break;
                }

                // if this is the root dataset, we don't have that ancestor
                if (_datasets.at(dset).is_root())
                    return nullptr;

                // Move on to the parent dataset...
                dset = _datasets.at(dset).base_dset();

            } catch (std::out_of_range& e) {
                // we don't have the base dataset
                DEBUG2("datasetManager: found a dead reference when looking for " \
                       "locally known ancestor: %s", e.what());
                return nullptr;
            }
        }

        // Check if we have that state already
        const datasetState* state = nullptr;
        try {
            state = _states.at(ancestor).get();

            // walk through the inner states until we find the right type
            while (state != nullptr) {
                if (typeid(*state) == typeid(T))
                    return (const T*)state;
                state = state->_inner_state.get();
            }
        } catch (std::out_of_range& e) {
            DEBUG("datasetManager: requested state 0x%" PRIx64 " not known " \
                  "locally.", ancestor);
        }
        if (_use_broker) {
            // Request the state from the broker.
            state = request_state<T>(ancestor);
            while (!state) {
                WARN("datasetManager: Failure requesting state " \
                     "0x%" PRIx64 " from broker.\nRetrying...");
                std::this_thread::sleep_for(
                            std::chrono::milliseconds(_retry_wait_time_ms));
                state = request_state<T>(ancestor);
            }
            return (const T*)state;
        } else
            return nullptr;
    }
}

template<typename T>
inline const T* datasetManager::request_state(state_id_t state_id) {

    // If this state is requested already, wait for it.
    if (_requested_states.count(state_id)) {
        std::unique_lock<std::mutex> lck_rcvd(_lock_recv_state);
        _cv_received_state.wait(lck_rcvd, [this, state_id]() {
            return !_requested_states.count(state_id);
        });
    }

    // If an ongoing request returned just when this function was
    // called, we are done.
    {
        std::lock_guard<std::mutex> lck_states(_lock_states);
        if(_states.count(state_id))
            return (const T*)_states.at(state_id).get();
    }

    // Request state from broker
    _requested_states.insert(state_id);
    json js_request;
    js_request["id"] = state_id;
    restReply reply = _rest_client.make_request_blocking(
                PATH_REQUEST_STATE, js_request,
                _ds_broker_host, _ds_broker_port);
    if (!reply.first) {
        WARN("datasetManager: Failure requesting state from " \
             "broker: %s", reply.second.c_str());
        prometheusMetrics::instance().add_process_metric(
                    "kotekan_datasetbroker_error_count", DS_UNIQUE_NAME,
                    ++_conn_error_count);
        return nullptr;
    }

    json js_reply;
    try {
        js_reply = json::parse(reply.second);
        if (js_reply.at("result") != "success")
            throw std::runtime_error("Broker answered with result="
                                     + js_reply.at("result").dump());

        state_id_t s_id = js_reply.at("id");

        state_uptr state =
                datasetState::from_json(js_reply.at("state"));
        if (state == nullptr) {
            throw(std::runtime_error("Failed to parse state received from " \
                                     "broker: " + js_reply.at("state").dump()));
        }

        // register the received state
        std::unique_lock<std::mutex> slck(_lock_states);
        auto new_state = _states.insert(std::pair<state_id_t,
                                   std::unique_ptr<datasetState>>
                                   (s_id, move(state)));
        slck.unlock();

        // signal other waiting state requests, that we received this state
        {
            std::unique_lock<std::mutex> _lck_rcvd(_lock_recv_state);
            _requested_states.erase(state_id);
        }
        _cv_received_state.notify_all();

        // hash collisions are checked for by the broker
        if (!new_state.second)
            INFO("datasetManager::request_state: received a " \
                 "state (with hash 0x%" PRIx64 ") that is already registered " \
                 "locally.", s_id);

        // get a pointer out of that iterator
        const datasetState* s =
                (const datasetState*) new_state.first->second.get();

        // find the inner state matching the type
        while (true) {
            if (typeid(T) == typeid(*s))
                return (const T*)s;
            if (s->_inner_state == nullptr)
                throw std::runtime_error("Broker sent state that didn't match "\
                                         "requested type (" +
                                         std::string(typeid(T).name()) +
                                         "): " + js_reply.at("state").dump());
            s = s->_inner_state.get();
        }
    } catch (std::exception& e) {
        WARN("datasetManager: failure parsing reply received from broker " \
              "after requesting state (reply: %s): %s",
              reply.second.c_str(), e.what());
        prometheusMetrics::instance().add_process_metric(
                    "kotekan_datasetbroker_error_count", DS_UNIQUE_NAME,
                    ++_conn_error_count);
        return nullptr;
    }
}

#endif
