#ifndef DATASETSTATE_HPP
#define DATASETSTATE_HPP

#include "Hash.hpp"     // for Hash
#include "factory.hpp"  // for REGISTER_NAMED_TYPE_WITH_FACTORY, CREATE_FACTORY, FACTORY, Factory
#include "gateSpec.hpp" // for gateSpec, _factory_aliasgateSpec
#include "visUtil.hpp"  // for prod_ctype, rstack_ctype, time_ctype, input_ctype, freq_ctype

#include "fmt.hpp"  // for format, fmt
#include "json.hpp" // for json, basic_json<>::object_t, json_ref, basic_json, basic_json<>...

#include <algorithm> // for copy
#include <cstdint>   // for uint32_t
#include <exception> // for exception
#include <iosfwd>    // for ostream
#include <memory>    // for allocator, unique_ptr
#include <numeric>   // for iota
#include <stddef.h>  // for size_t
#include <stdexcept> // for runtime_error, out_of_range
#include <string>    // for string
#include <utility>   // for pair
#include <vector>    // for vector, vector<>::iterator

class datasetManager;
class datasetState; // IWYU pragma: keep


/// Unique pointer to a datasetState
using state_uptr = std::unique_ptr<datasetState>;

/// DatasetState ID
using state_id_t = Hash;

/// DatasetID
using dset_id_t = Hash;

/**
 * @brief A base class for representing state changes done to datasets.
 *
 * This is meant to be subclassed. All subclasses must implement a constructor
 * that can build the type from a `json` argument, and a `data_to_json` method
 * that can serialise the type into a `json` object.
 *
 * @author Richard Shaw, Rick Nitsche
 **/
class datasetState {
public:
    virtual ~datasetState(){};

    /**
     * @brief Create a dataset state from a full json serialisation.
     *
     * This will correctly instantiate the correct type from the json.
     *
     * @param j Full JSON serialisation.
     * @returns The created datasetState or a nullptr in a failure case.
     **/
    static state_uptr from_json(const nlohmann::json& j);

    /**
     * @brief Full serialisation of state into JSON.
     *
     * @returns JSON serialisation of state.
     **/
    nlohmann::json to_json() const;

    /**
     * @brief Save the internal data of this instance into JSON.
     *
     * This must be implement by any derived classes and should save the
     * information needed to reconstruct any subclass specific internals.
     *
     * @returns JSON representing the internal state.
     **/
    virtual nlohmann::json data_to_json() const = 0;

    /**
     * @brief Compare to another dataset state.
     * @param s    State to compare with.
     * @return True if states identical, False otherwise.
     */
    bool equals(datasetState& s) const;

    /**
     * @brief Get the name of this state.
     * @return The state name.
     */
    std::string type() const;

private:
    // Add as friend so it can walk the inner state
    friend datasetManager;
};


CREATE_FACTORY(datasetState, const nlohmann::json&);


#define REGISTER_DATASET_STATE(T, s) REGISTER_NAMED_TYPE_WITH_FACTORY(datasetState, T, s);


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
     */
    freqState(const nlohmann::json& data) {
        try {
            _freqs = data.get<std::vector<std::pair<uint32_t, freq_ctype>>>();
        } catch (std::exception& e) {
            throw std::runtime_error(fmt::format(
                fmt("freqState: Failure parsing json data ({:s}): {:s}"), data.dump(4), e.what()));
        }
    };

    /**
     * @brief Constructor
     * @param freqs The frequency information as a vector of
     *              {frequency ID, frequency index map}.
     */
    freqState(std::vector<std::pair<uint32_t, freq_ctype>> freqs) : _freqs(freqs){};

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
    nlohmann::json data_to_json() const override {
        nlohmann::json j(_freqs);
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
     */
    inputState(const nlohmann::json& data) {
        try {
            _inputs = data.get<std::vector<input_ctype>>();
        } catch (std::exception& e) {
            throw std::runtime_error(fmt::format(
                fmt("inputState: Failure parsing json data ({:s}): {:s}"), data.dump(4), e.what()));
        }
    };

    /**
     * @brief Constructor
     * @param inputs The input information as a vector of
     *               input index maps.
     */
    inputState(std::vector<input_ctype> inputs) : _inputs(inputs){};

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
    nlohmann::json data_to_json() const override {
        nlohmann::json j(_inputs);
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
     */
    prodState(const nlohmann::json& data) {
        try {
            _prods = data.get<std::vector<prod_ctype>>();
        } catch (std::exception& e) {
            throw std::runtime_error(fmt::format(
                fmt("prodState: Failure parsing json data ({:s}): {:s}"), data.dump(4), e.what()));
        }
    };

    /**
     * @brief Constructor
     * @param prods The product information as a vector of
     *              product index maps.
     */
    prodState(std::vector<prod_ctype> prods) : _prods(prods){};

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
    nlohmann::json data_to_json() const override {
        nlohmann::json j(_prods);
        return j;
    }

    /// IDs that describe the subset that this dataset state defines
    std::vector<prod_ctype> _prods;
};


/**
 * @brief A dataset state that keeps the time information of a datatset.
 *
 * @author Rick Nitsche
 */
class timeState : public datasetState {
public:
    /**
     * @brief Constructor
     * @param data  The time information as serialized by
     *              timeState::to_json().
     */
    timeState(const nlohmann::json& data) {
        try {
            _times = data.get<std::vector<time_ctype>>();
        } catch (std::exception& e) {
            throw std::runtime_error(fmt::format(
                fmt("timeState: Failure parsing json data ({:s}): {:s}"), data.dump(4), e.what()));
        }
    };

    /**
     * @brief Constructor
     * @param times The time information as a vector of
     *              time index maps.

     */
    timeState(std::vector<time_ctype> times) : _times(times){};

    /**
     * @brief Get time information (read only).
     *
     * @return The time information as a vector of time index maps.
     */
    const std::vector<time_ctype>& get_times() const {
        return _times;
    }

private:
    /// Serialize the data of this state in a json object
    nlohmann::json data_to_json() const override {
        nlohmann::json j(_times);
        return j;
    }

    /// Time index map of the dataset state.
    std::vector<time_ctype> _times;
};

/**
 * @brief A dataset state that keeps the eigenvalues of a datatset.
 *
 * @author Rick Nitsche
 */
class eigenvalueState : public datasetState {
public:
    /**
     * @brief Constructor
     * @param data  The eigenvalues as serialized by
     *              eigenvalueState::to_json().
     */
    eigenvalueState(const nlohmann::json& data) {
        try {
            _ev = data.get<std::vector<uint32_t>>();
        } catch (std::exception& e) {
            throw std::runtime_error(fmt::format(fmt("eigenvectorState: Failure parsing json "
                                                     "data ({:s}): {:s}"),
                                                 data.dump(4), e.what()));
        }
    };

    /**
     * @brief Constructor
     * @param ev The eigenvalues.
     */
    eigenvalueState(std::vector<uint32_t> ev) : _ev(ev){};

    /**
     * @brief Constructor
     * @param num_ev The number of eigenvalues. The indices will end up
     *               running from 0 to num_ev - 1
     */
    eigenvalueState(size_t num_ev) : _ev(num_ev) {
        std::iota(_ev.begin(), _ev.end(), 0);
    }

    /**
     * @brief Get eigenvalues (read only).
     *
     * @return The eigenvalues.
     */
    const std::vector<uint32_t>& get_ev() const {
        return _ev;
    }

    /**
     * @brief Get the number of eigenvalues
     *
     * @return The number of eigenvalues.
     */
    size_t get_num_ev() const {
        return _ev.size();
    }

private:
    /// Serialize the data of this state in a json object
    nlohmann::json data_to_json() const override {
        nlohmann::json j(_ev);
        return j;
    }

    /// Eigenvalues of the dataset state.
    std::vector<uint32_t> _ev;
};


/**
 * @brief Take an rstack map and generate a stack->prod mapping.
 *
 * @param num_stack Total number of stacks.
 * @param stack_map The prod->stack mapping.
 *
 * @returns The stack->prod mapping.
 **/
std::vector<stack_ctype> invert_stack(uint32_t num_stack,
                                      const std::vector<rstack_ctype>& stack_map);


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
     */
    stackState(const nlohmann::json& data) {
        try {
            _rstack_map = data["rstack"].get<std::vector<rstack_ctype>>();
            _num_stack = data["num_stack"].get<uint32_t>();
        } catch (std::exception& e) {
            throw std::runtime_error(
                fmt::format(fmt("stackState: Failure parsing json data: {:s}"), e.what()));
        }
    };

    /**
     * @brief Constructor
     * @param rstack_map Definition of how the products were stacked.
     * @param num_stack Number of stacked visibilities.
     */
    stackState(uint32_t num_stack, std::vector<rstack_ctype>&& rstack_map) :
        _num_stack(num_stack),
        _rstack_map(rstack_map) {}

    /**
     * @brief Get stack map information (read only).
     *
     * For every product this says which stack to add the product into and
     * whether it needs conjugating before doing so.
     *
     * @return The stack map.
     */
    const std::vector<rstack_ctype>& get_rstack_map() const {
        return _rstack_map;
    }

    /**
     * @brief Get the number of stacks.
     *
     * @return The number of stacks.
     */
    uint32_t get_num_stack() const {
        return _num_stack;
    }

    /**
     * @brief Calculate and return the stack->prod mapping.
     *
     * This is calculated on demand and so a full fledged vector is returned.
     *
     * @returns The stack map.
     **/
    std::vector<stack_ctype> get_stack_map() const {
        return invert_stack(_num_stack, _rstack_map);
    }

    /// Serialize the data of this state in a json object
    nlohmann::json data_to_json() const override {
        return {{"rstack", _rstack_map}, {"num_stack", _num_stack}};
    }

private:
    /// Total number of stacks
    uint32_t _num_stack;

    /// The stack definition
    std::vector<rstack_ctype> _rstack_map;
};


/**
 * @brief A dataset state that describes all the metadata that is written to
 * file as "attributes", but not defined by other states yet.
 *
 * @author Rick Nitsche
 */
class metadataState : public datasetState {
public:
    /**
     * @brief Constructor
     * @param data  The metadata as serialized by
     *              metadataState::to_json():
     * weight_type: string
     * instrument_name: string
     * git_version_number: string
     *
     */
    metadataState(const nlohmann::json& data) {
        try {
            _weight_type = data.at("weight_type").get<std::string>();
            _instrument_name = data.at("instrument_name").get<std::string>();
            _git_version_tag = data.at("git_version_tag").get<std::string>();
        } catch (std::exception& e) {
            throw std::runtime_error(fmt::format(fmt("metadataState: Failure parsing json "
                                                     "data ({:s}): {:s}"),
                                                 data.dump(4), e.what()));
        }
    }

    /**
     * @brief Constructor
     * @param weight_type       The weight type attribute.
     * @param instrument_name   The instrument name attribute.
     * @param git_version_tag   The git version tag attribute.
     */
    metadataState(std::string weight_type, std::string instrument_name,
                  std::string git_version_tag) :
        _weight_type(weight_type),
        _instrument_name(instrument_name),
        _git_version_tag(git_version_tag) {}

    /**
     * @brief Get the weight type (read only).
     *
     * @return The weigh type.
     */
    const std::string& get_weight_type() const {
        return _weight_type;
    }

    /**
     * @brief Get the instrument name (read only).
     *
     * @return The instrument name.
     */
    const std::string& get_instrument_name() const {
        return _instrument_name;
    }

    /**
     * @brief Get the git version tag (read only).
     *
     * @return The git version tag.
     */
    const std::string& get_git_version_tag() const {
        return _git_version_tag;
    }

private:
    /// Serialize the data of this state in a json object
    nlohmann::json data_to_json() const override {
        nlohmann::json j;
        j["weight_type"] = _weight_type;
        j["instrument_name"] = _instrument_name;
        j["git_version_tag"] = _git_version_tag;
        return j;
    }

    // the actual metadata
    std::string _weight_type, _instrument_name, _git_version_tag;
};


/**
 * @brief A state to describe any applied gating.
 *
 * @author Richard Shaw
 **/
class gatingState : public datasetState {
public:
    /**
     * @brief Construct a gating state
     *
     * @param  spec  gateSpec to describe what's happening.
     **/
    gatingState(const gateSpec& spec) :
        gating_type(FACTORY(gateSpec)::label(spec)),
        gating_data(spec.to_dm_json()) {}

    /**
     * @brief Construct a gating state
     *
     * @param  data   Full serialised data.
     **/
    gatingState(const nlohmann::json& data) :
        gating_type(data["type"].get<std::string>()),
        gating_data(data["data"]) {}


    /**
     * @brief Serialise the gatingState data.
     *
     * @return  JSON serialisation.
     **/
    nlohmann::json data_to_json() const override {
        return {{"type", gating_type}, {"data", gating_data}};
    }

    /// Type of gating
    const std::string gating_type;

    /// Type specific data
    const nlohmann::json gating_data;
};


/**
 * @brief A dataset state that describes the gains applied to the data.
 *
 * @author Richard Shaw
 */
class gainState : public datasetState {
public:
    /**
     * @brief Constructor
     * @param data  The product information as serialized by
     *              gainState::to_json().
     */
    gainState(const nlohmann::json& data) {
        try {
            _update_id = data["update_id"].get<std::string>();
            _transition_interval = data["transition_interval"].get<double>();
        } catch (std::exception& e) {
            throw std::runtime_error(fmt::format(
                fmt("gainState: Failure parsing json data ({:s}): {:s}"), data.dump(4), e.what()));
        }
    };

    /**
     * @brief Constructor
     * @param  update_id  The string update_id labelling the applied gains.
     * @param  transition_interval  The length of time to blend updates over.
     */
    gainState(std::string update_id, double transition_interval) :
        _update_id(update_id),
        _transition_interval(transition_interval){};

    /**
     * @brief Get the update_id
     **/
    const std::string& get_update_id() const {
        return _update_id;
    }

    /**
     * @brief Get the length of time to blend this new update with the previous one.
     **/
    double get_transition_interval() const {
        return _transition_interval;
    }

private:
    /// Serialize the data of this state in a json object
    nlohmann::json data_to_json() const override {
        nlohmann::json j;
        j["update_id"] = _update_id;
        j["transition_interval"] = _transition_interval;
        return j;
    }

    // The label for the gains
    std::string _update_id;

    // The length of time (in seconds) the previous gain update is blended with this one.
    double _transition_interval;
};


/**
 * @brief A dataset state that describes the input flags being applied.
 *
 * @author Richard Shaw
 */
class flagState : public datasetState {
public:
    /**
     * @brief Constructor
     *
     * @param data  The product information as serialized by
     *              flagState::to_json().
     */
    flagState(const nlohmann::json& data) {
        try {
            _update_id = data.get<std::string>();
        } catch (std::exception& e) {
            throw std::runtime_error(fmt::format(
                fmt("flagState: Failure parsing json data ({:s}): {:s}"), data.dump(4), e.what()));
        }
    };

    /**
     * @brief Constructor
     *
     * @param  update_id  The string update_id labelling the applied flags.
     */
    flagState(std::string update_id) : _update_id(update_id){};

    const std::string& get_update_id() const {
        return _update_id;
    }

private:
    /// Serialize the data of this state in a json object
    nlohmann::json data_to_json() const override {
        nlohmann::json j(_update_id);
        return j;
    }

    // The label for the flags
    std::string _update_id;
};

/**
 * @brief A dataset state that keeps the beam information of a datatset.
 *
 * @author James Willis
 */
class beamState : public datasetState {
public:
    /**
     * @brief Constructor
     * @param data  The beam information as serialized by
     *              beamState::to_json().
     */
    beamState(const nlohmann::json& data) {
        try {
            _beams = data.get<std::vector<uint32_t>>();
        } catch (std::exception& e) {
            throw std::runtime_error(fmt::format(
                fmt("beamState: Failure parsing json data ({:s}): {:s}"), data.dump(4), e.what()));
        }
    };

    /**
     * @brief Constructor
     * @param beams The beam information as a vector of
     *              beam index maps.

     */
    beamState(std::vector<uint32_t> beams) : _beams(beams){};

    /**
     * @brief Constructor
     * @param num_beams The number of beams. The indices will end up
     *                  running from 0 to num_beams - 1
     */
    beamState(size_t num_beams) : _beams(num_beams) {
        std::iota(_beams.begin(), _beams.end(), 0);
    }

    /**
     * @brief Get beam information (read only).
     *
     * @return The beam information as a vector of beam index maps.
     */
    const std::vector<uint32_t>& get_beams() const {
        return _beams;
    }

private:
    /// Serialize the data of this state in a json object
    nlohmann::json data_to_json() const override {
        nlohmann::json j(_beams);
        return j;
    }

    /// Time index map of the dataset state.
    std::vector<uint32_t> _beams;
};

/**
 * @brief A dataset state that keeps the sub-frequency information of a datatset.
 *
 * @author James Willis
 */
class subfreqState : public datasetState {
public:
    /**
     * @brief Constructor
     * @param data  The sub-frequency information as serialized by
     *              subfreqState::to_json().
     */
    subfreqState(const nlohmann::json& data) {
        try {
            _subfreqs = data.get<std::vector<uint32_t>>();
        } catch (std::exception& e) {
            throw std::runtime_error(
                fmt::format(fmt("subfreqState: Failure parsing json data ({:s}): {:s}"),
                            data.dump(4), e.what()));
        }
    };

    /**
     * @brief Constructor
     * @param subfreqs The sub-frequency information as a vector of
     *              subfreq index maps.
     */
    subfreqState(std::vector<uint32_t> subfreqs) : _subfreqs(subfreqs){};

    /**
     * @brief Constructor
     * @param num_subfreqs The number of sub-frequencies. The indices will end up
     *                  running from 0 to num_subfreqs - 1
     */
    subfreqState(size_t num_subfreqs) : _subfreqs(num_subfreqs) {
        std::iota(_subfreqs.begin(), _subfreqs.end(), 0);
    }


    /**
     * @brief Get sub-frequency information (read only).
     *
     * @return The sub-frequency information as a vector of subfreq index maps.
     */
    const std::vector<uint32_t>& get_subfreqs() const {
        return _subfreqs;
    }

private:
    /// Serialize the data of this state in a json object
    nlohmann::json data_to_json() const override {
        nlohmann::json j(_subfreqs);
        return j;
    }

    /// Time index map of the dataset state.
    std::vector<uint32_t> _subfreqs;
};

/**
 * @brief A dataset state that keeps the RFI frame-dropping information of a datatset.
 *
 * @author Rick Nitsche
 */
class RFIFrameDropState : public datasetState {
public:
    /**
     * @brief Constructor
     * @param data  The RFI frame-dropping information as serialized by
     *              RFIFrameDropState::to_json().
     */
    RFIFrameDropState(const nlohmann::json& data) {
        try {
            enabled = data["enabled"].get<bool>();
            thresholds = data["thresholds"].get<std::vector<std::pair<float, float>>>();
        } catch (std::exception& e) {
            throw std::runtime_error(
                fmt::format(fmt("RFIFrameDropState: Failure parsing json data ({:s}): {:s}"),
                            data.dump(4), e.what()));
        }
    }

    /**
     * @brief Constructor
     * @param enabled       True, if RFI frame-dropping enabled
     * @param thresholds    Vector of pairs: thresholds and fractions
     */
    RFIFrameDropState(bool enabled, std::vector<std::pair<float, float>> thresholds) :
        enabled(enabled),
        thresholds(thresholds) {}

    /**
     * @brief Get RFI frame-dropping enabled information.
     *
     * @return True if RFI frame-dropping enabled.
     */
    bool get_enabled() const {
        return enabled;
    }

    /**
     * @brief Get RFI frame-dropping thresholds (read only).
     *
     * @return Vector of pairs containing <threshold, fraction>, each, in this order.
     */
    const std::vector<std::pair<float, float>>& get_thresholds() const {
        return thresholds;
    }

private:
    /// Serialize the data of this state in a json object
    nlohmann::json data_to_json() const override {
        nlohmann::json j;
        j["enabled"] = enabled;
        j["thresholds"] = thresholds;
        return j;
    }

    /// Tells if frame dropping is enabled in the RFIFrameDrop stage.
    bool enabled;

    std::vector<std::pair<float, float>> thresholds;
};

#endif // DATASETSTATE_HPP
