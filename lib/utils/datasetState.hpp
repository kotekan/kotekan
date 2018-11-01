#ifndef DATASETSTATE_HPP
#define DATASETSTATE_HPP

#include <memory>

#include "json.hpp"
#include "visUtil.hpp"

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

    /**
     * @brief Compare to another dataset state.
     * @param s    State to compare with.
     * @return True if states identical, False otherwise.
     */
    const bool equals(datasetState& s) const;

private:

    /**
     * @brief Create a datasetState subclass from a json serialisation.
     *
     * @param name  Name of subclass to create.
     * @param data  Serialisation of config.
     * @param inner Inner state to compose with.
     * @returns The created datasetState.
     **/
    static state_uptr _create(std::string name, json & data,
                              state_uptr inner=nullptr);

    // Reference to the internal state
    state_uptr _inner_state = nullptr;

    // List of registered subclass creating functions
    static std::map<string, std::function<state_uptr(json&, state_uptr)>>&
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

#endif // DATASETSTATE_HPP

