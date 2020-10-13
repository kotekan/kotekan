#ifndef DATASET_MANAGER_HPP
#define DATASET_MANAGER_HPP

#include "Config.hpp"            // for Config
#include "Hash.hpp"              // for Hash, operator<
#include "dataset.hpp"           // for dataset
#include "datasetState.hpp"      // for datasetState, state_uptr, _factory_aliasdatasetState
#include "factory.hpp"           // for FACTORY
#include "kotekanLogging.hpp"    // for WARN_NON_OO, DEBUG_NON_OO, DEBUG2_NON_OO, FATAL_ERROR_N...
#include "prometheusMetrics.hpp" // for Gauge
#include "restClient.hpp"        // for restClient::restReply, restClient
#include "restServer.hpp"        // for connectionInstance

#include "fmt.hpp"  // for fmt
#include "json.hpp" // for json, basic_json<>::object_t, basic_json, operator!=

#include <atomic>             // for atomic, __atomic_base
#include <chrono>             // for milliseconds
#include <condition_variable> // for condition_variable
#include <exception>          // for exception
#include <functional>         // for function
#include <map>                // for map, _Rb_tree_iterator
#include <memory>             // for unique_ptr, operator==, make_unique
#include <mutex>              // for mutex, unique_lock, lock_guard
#include <optional>           // for optional
#include <set>                // for set
#include <stdexcept>          // for runtime_error, out_of_range
#include <stdint.h>           // for uint32_t, int32_t, uint64_t
#include <string>             // for string, basic_string
#include <thread>             // for sleep_for
#include <type_traits>        // for is_base_of, enable_if, enable_if_t
#include <typeinfo>           // for type_info
#include <utility>            // for pair, move, forward
#include <vector>             // for vector

/// Alias certain types to give semantic meaning to the IDs
/// These use a 128 bit hash type so there shouldn't be any collisions.
using state_id_t = Hash;
using dset_id_t = Hash;
using fingerprint_t = Hash;

#define DS_UNIQUE_NAME "/dataset_manager"
#define DS_FORCE_UPDATE_ENDPOINT_NAME "/dataset-manager/force-update"

// names of broker endpoints
const std::string PATH_REGISTER_STATE = "/register-state";
const std::string PATH_SEND_STATE = "/send-state";
const std::string PATH_REGISTER_DATASET = "/register-dataset";
const std::string PATH_UPDATE_DATASETS = "/update-datasets";
const std::string PATH_REQUEST_STATE = "/request-state";

/**
 * @class datasetManager
 * @brief Manages sets of state changes applied to datasets.
 *
 * This is a singleton class. Use `datasetManager::instance()` to get a
 * reference to it.
 *
 * The datasetManager is used to manage the states of datasets that get passed
 * through kotekan stages.
 * A stage in the kotekan pipeline may use the dataset ID found in an incoming
 * frame to get a set of states from the datasetManager.
 *
 * To receive information about the inputs the datsets in the frames contain, it
 * could do thew following:
 * ```
 * auto input_state = dm.dataset_state<inputState>(ds_id_from_frame);
 * const std::vector<input_ctype>& inputs = input_state->get_inputs();
 * ```
 *
 * A stage that changes the state of the dataset in the frames it processes should inform the
 * datasetManager by adding a new state and dataset. If multiple states are being applied at the
 * same time a vector of states can be passed to `add_dataset`. This causes datasets linking them to
 * be generated, but only final one is returned.
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
 * @par REST Endpoints
 * @endpoint    /force-update ``GET`` Forces the datasetManager to register
 *                                    all datasets and states with the
 *                                    dataset_broker.
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
    static datasetManager& instance(kotekan::Config& config);

    // Remove the implicit copy/assignments to prevent copying
    datasetManager(const datasetManager&) = delete;
    void operator=(const datasetManager&) = delete;

    /**
     * @brief Signal to stop request threads.
     **/
    void stop();

    // TODO: 0 is not a good sentinel value. Move to std::optional typing when we use C++17
    /**
     * @brief Register a new dataset. Omitting base_dset adds a root dataset.
     *
     * If `use_dataset_broker` is set, this function will ask the dataset broker
     * to assign an ID to the new dataset.
     *
     * @param state         The ID of the dataset state that describes the
     *                      difference to the base dataset.
     * @param base_dset     The ID of the dataset this dataset is based on.
     *                      Omit to create a root dataset.
     * @returns The ID assigned to the new dataset.
     **/
    dset_id_t add_dataset(state_id_t state, dset_id_t base_dset = dset_id_t::null);

    /**
     * @brief Register a new dataset with multiple states.
     *
     * If `use_dataset_broker` is set, this function will ask the dataset broker
     * to assign an ID to the new dataset.
     *
     * @param states        The IDs of the dataset states that describes the
     *                      difference to the base dataset.
     * @param base_dset     The ID of the dataset this dataset is based on.
     *                      Omit to create a root dataset.
     * @returns The ID assigned to the new dataset.
     **/
    dset_id_t add_dataset(const std::vector<state_id_t>& states,
                          dset_id_t base_dset = dset_id_t::null);

    /**
     * @brief Create *and* register a state with the manager.
     *
     * This is the recommended way to create a datasetState as it will directly
     * create the datasetState instance under the ownership of the
     * datasetManager. The calling function is returned the ID and a const
     * pointer to the created state.
     *
     * If `use_dataset_broker` is set, this function will also register the new
     * state with the broker.
     *
     * @param  args  Arguments forwarded through to the constructor of the sub-type.
     * @returns      The id assigned to the state and a read-only pointer to the
     *               state.
     **/
// Sphinx can't correctly parse the template definition here, so we need to make sure Doxygen passes
// on a sanitized version
#ifdef _DOXYGEN_
    template<typename T, typename... Args>
#else
    template<typename T, typename... Args,
             typename std::enable_if_t<std::is_base_of<datasetState, T>::value>* = nullptr>
#endif
    inline std::pair<state_id_t, const T*> create_state(Args&&... args);

    /**
     * @brief Register a state with the manager.
     *
     * If `use_dataset_broker` is set, this function will also register the new
     * state with the broker.
     *
     * The second argument of this function is to
     * prevent compilation of this function with `T` not having the base class
     * `datasetState`.
     *
     * @param state The state to be added.
     * @returns The id assigned to the state and a read-only pointer to the
     * state.
     **/
    template<typename T>
    inline std::pair<state_id_t, const T*>
    add_state(std::unique_ptr<T>&& state,
              typename std::enable_if<std::is_base_of<datasetState, T>::value>::type* = nullptr);

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
    const std::map<state_id_t, const datasetState*> states();

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
     * answers. If you want to do something else, while waiting for the return
     * value of this function, use std::future.
     *
     * @param  dset  The ID of the dataset to start from.
     * @param  type  The type name of the state change we are searching for.
     *
     * @returns      The dataset entry matching the type. Unset if no state of given
     *               type exists.
     **/
    std::optional<std::pair<dset_id_t, dataset>> closest_dataset_of_type(dset_id_t dset,
                                                                         const std::string& type);

    /**
     * @brief Find the closest ancestor of a given type.
     *
     * If `use_dataset_broker` is set and no ancestor of the given type is found,
     * this will ask the broker for a complete list of ancestors for the given
     * dataset. In that case, this function is blocking, until the broker
     * answers. If you want to do something else, while waiting for the return
     * value of this function, use std::future.
     *
     * @param  dset  The ID of the dataset to start from.
     *
     * @returns      A read-only pointer to the ancestor state.
     *               Returns a `nullptr` if not found in ancestors or in a
     *               failure case.
     **/
    template<typename T>
    inline const T* dataset_state(dset_id_t dset);


    /**
     * @brief Fingerprint a dataset for specified states.
     *
     * Generate a summary of the specified states present in the requested
     * dataset. This will be unique for datasets where one or more of the
     * requested states differ. Datasets that share all these states will give
     * the same fingerprint regardless of differences in any other states.
     *
     * The fingerprint does not depend on the order of state_types. It is also
     * specific to the types, even when states are missing. This means that for
     * a dataset which contains a state of `type_A`, but neither of `type_B` or
     * `type_C`, the fingerprints with respect to `{type_A, type_B}` and
     * `{type_A, type_C}` will be different.
     *
     * @param  ds_id        Dataset ID of the incoming frame.
     * @param  state_types  Names of the state types to fingerprint.
     *
     * @return              Finger print of the dataset.
     **/
    fingerprint_t fingerprint(dset_id_t ds_id, const std::set<std::string>& state_types);

    /**
     * @brief Callback for endpoint `force-update` called by the restServer.
     * @param conn The HTTP connection object.
     */
    void force_update_callback(kotekan::connectionInstance& conn);

private:
    /// Constructor
    datasetManager();

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
     * @returns A vector of the dataset ID and the state that was
     *          applied to previous element in the vector to generate it.
     **/
    const std::vector<std::pair<dset_id_t, datasetState*>> ancestors(dset_id_t dset);

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
    void register_dataset(const dset_id_t hash, const dataset& ds);

    /// parser function for register_state()
    bool register_state_parser(std::string& reply);

    /// parser function for sending a state to the dataset broker
    /// from register_state_parser()
    bool send_state_parser(std::string& reply);

    /// parser function for register_dataset()
    bool register_dataset_parser(std::string& reply);

    /// request an update on the topology of datasets (blocking)
    /// this will check to see if any ancestors of ds_id are not known, and try
    /// to fetch any that are missing
    void update_datasets(dset_id_t ds_id);

    /// Helper function to parse the reply for update_datasets()
    bool parse_reply_dataset_update(restClient::restReply reply);

    /// To be left in a detached thread: Infinitely retries request parse.
    /// Stopped by the destructor if still unsuccessfully retrying.
    void request_thread(const nlohmann::json&& request, const std::string&& endpoint,
                        const std::function<bool(std::string&)>&& parse_reply);

    /// Wait for any ongoing requests of the same state OR request state.
    template<typename T>
    inline const T* request_state(state_id_t state_id);

    /// Store the list of all the registered states.
    std::map<state_id_t, state_uptr> _states;

    /// Store a list of the datasets registered and what states
    /// and input datasets they correspond to
    std::map<dset_id_t, dataset> _datasets;

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

    // TODO: this should be a counter, but we don't have it using atomic Ints
    kotekan::prometheus::Gauge& error_counter;
};


//
// Implementations of templated methods
//

template<typename T>
inline const T* datasetManager::dataset_state(dset_id_t dset) {

    // Try to find a matching dataset
    std::string type = FACTORY(datasetState)::label<T>();
    auto ret = closest_dataset_of_type(dset, type);

    DEBUG2_NON_OO("Finding state type {} from dset={}", type, dset.to_string());

    // If not found, return null
    if (!ret)
        return nullptr;

    state_id_t state_id = ret.value().second.state();

    // NOTE: we may want to reconsider if we should have released the lock
    // between here and the `closest_dataset_of_type` call above. There's a
    // possibility we may end up doing multiple requests to comet
    {
        std::lock_guard<std::mutex> dslock(_lock_dsets);

        // Check if we have that state already
        const datasetState* state = nullptr;
        try {
            state = _states.at(state_id).get();
            return (const T*)state;
        } catch (std::out_of_range& e) {
            DEBUG_NON_OO("datasetManager: requested state {} not known locally.", state_id);
        }

        if (_use_broker) {
            // Request the state from the broker.
            state = request_state<T>(state_id);
            while (!state && !_stop_request_threads) {
                WARN_NON_OO("datasetManager: Failure requesting state {} from broker.\nRetrying...",
                            state_id);
                std::this_thread::sleep_for(std::chrono::milliseconds(_retry_wait_time_ms));
                state = request_state<T>(state_id);
            }
            return (const T*)state;
        } else
            return nullptr;
    }
}


template<typename T, typename... Args,
         typename std::enable_if_t<std::is_base_of<datasetState, T>::value>*>
std::pair<state_id_t, const T*> datasetManager::create_state(Args&&... args) {
    // Create the instance and register the state
    auto t = std::make_unique<T>(std::forward<Args>(args)...);
    return add_state(std::move(t));
}


template<typename T>
std::pair<state_id_t, const T*>
datasetManager::add_state(std::unique_ptr<T>&& state,
                          typename std::enable_if<std::is_base_of<datasetState, T>::value>::type*) {

    state_id_t hash = hash_state(*state);

    // check if there is a hash collision
    if (_states.find(hash) != _states.end()) {
        auto find = _states.find(hash);
        if (!state->equals(*(find->second))) {
            // FIXME: hash collision. make the value a vector and store same
            // hash entries? This would mean the state/dset has to be sent
            // when registering.
            FATAL_ERROR_NON_OO("datasetManager: Hash collision!\nThe following states have the "
                               "same hash {}.\n\n{:s}\n\n{:s}\n\ndatasetManager: Exiting...",
                               hash, state->to_json().dump(4), find->second->to_json().dump(4));
        }
    } else {
        // insert the new state
        std::lock_guard<std::mutex> slock(_lock_states);
        if (!_states.insert(std::pair<state_id_t, std::unique_ptr<T>>(hash, move(state))).second) {
            DEBUG_NON_OO("datasetManager: a state with hash {} is already registered locally.",
                         hash);
        }

        // tell the broker about it
        if (_use_broker)
            register_state(hash);
    }

    return std::pair<state_id_t, const T*>(hash, (const T*)(_states.at(hash).get()));
}


template<typename T>
inline const T* datasetManager::request_state(state_id_t state_id) {

    // If this state is requested already, wait for it.
    if (_requested_states.count(state_id)) {
        std::unique_lock<std::mutex> lck_rcvd(_lock_recv_state);
        _cv_received_state.wait(lck_rcvd,
                                [this, state_id]() { return !_requested_states.count(state_id); });
    }

    // If an ongoing request returned just when this function was
    // called, we are done.
    {
        std::lock_guard<std::mutex> lck_states(_lock_states);
        if (_states.count(state_id))
            return (const T*)_states.at(state_id).get();
    }

    // Request state from broker
    _requested_states.insert(state_id);
    nlohmann::json js_request;
    js_request["id"] = state_id;
    restClient::restReply reply = _rest_client.make_request_blocking(
        PATH_REQUEST_STATE, js_request, _ds_broker_host, _ds_broker_port);
    if (!reply.first) {
        WARN_NON_OO("datasetManager: Failure requesting state from broker: {:s}", reply.second);
        error_counter.set(++_conn_error_count);
        return nullptr;
    }

    nlohmann::json js_reply;
    try {
        js_reply = nlohmann::json::parse(reply.second);
        if (js_reply.at("result") != "success")
            throw std::runtime_error(fmt::format(fmt("Broker answered with result={:s}"),
                                                 js_reply.at("result").dump(4)));

        state_id_t s_id = js_reply.at("id");

        state_uptr state = datasetState::from_json(js_reply.at("state"));
        if (state == nullptr) {
            throw(std::runtime_error(fmt::format(fmt("Failed to parse state received from "
                                                     "broker: {:s}"),
                                                 js_reply.at("state").dump(4))));
        }

        // register the received state
        std::unique_lock<std::mutex> slck(_lock_states);
        auto new_state =
            _states.insert(std::pair<state_id_t, std::unique_ptr<datasetState>>(s_id, move(state)));
        slck.unlock();

        // signal other waiting state requests, that we received this state
        {
            std::unique_lock<std::mutex> _lck_rcvd(_lock_recv_state);
            _requested_states.erase(state_id);
        }
        _cv_received_state.notify_all();

        // hash collisions are checked for by the broker
        if (!new_state.second)
            INFO_NON_OO("datasetManager::request_state: received a state (with hash {}) that "
                        "is already registered locally.",
                        s_id);

        // get a pointer out of that iterator
        const datasetState* s = (const datasetState*)new_state.first->second.get();

        // Check the state matches the type
        if (typeid(T).hash_code() == typeid(*s).hash_code())
            return (const T*)s;
        else {
            throw std::runtime_error(
                fmt::format(fmt("Broker sent state that didn't match requested type ({:s}): {:s}"),
                            FACTORY(datasetState)::label<T>(), js_reply.at("state").dump(4)));
        }
    } catch (std::exception& e) {
        WARN_NON_OO("datasetManager: failure parsing reply received from broker after requesting "
                    "state (reply: {:s}): {:s}",
                    reply.second, e.what());
        error_counter.set(++_conn_error_count);
        return nullptr;
    }
}

#endif
