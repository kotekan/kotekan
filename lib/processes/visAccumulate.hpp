
/*****************************************
@file
@brief Accumulation and gating of visibility data.
- visAccumulate : public KotekanProcess
*****************************************/
#ifndef VIS_ACCUMULATE_HPP
#define VIS_ACCUMULATE_HPP

#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <time.h>

#include "json.hpp"

#include "factory.hpp"
#include "buffer.h"
#include "KotekanProcess.hpp"
#include "visUtil.hpp"
#include "pulsarTiming.hpp"


/**
 * @brief Base class for specifying a gated data accumulation.
 *
 **/
class gateSpec {
public:

    /**
     * @brief Create a new gateSpec
     *
     * @param  name  Name of the gated dataset.
     **/
    gateSpec(const std::string& name);
    virtual ~gateSpec() = 0;

    /**
     * @brief Create a subtype by name.
     *
     * @param type Name of gateSpec subtype.
     * @param name Name of the gated dataset.
     *
     * @returns A pointer to the gateSpec instance.
     **/
    static std::unique_ptr<gateSpec> create(const std::string& type,
                                            const::std::string& name);


    /**
     * @brief A callback to update the gating specification.
     **/
    virtual bool update_spec(nlohmann::json &json) = 0;

    /**
     * @brief Get a function/closure to calculate the weights for a subsample.
     *
     * @note This must return a closure that captures by value such that its
     *       lifetime can be longer than the gateSpec object that generated it.
     *
     * @returns A function to calculate the weights.
     **/
    virtual std::function<float(timespec, timespec, float)> weight_function() const = 0;

    /**
     * @brief Is this enabled at the moment?
     **/
    const bool& enabled() const { return _enabled; }

    /**
     * @brief Get the name of the gated dataset.
     **/
    const std::string& name() const { return _name; }

protected:

    // Name of the gated dataset in the config
    const std::string _name;

    // Is the dataset enabled?
    bool _enabled = false;
};

// Create a factory for gateSpecs
CREATE_FACTORY(gateSpec, const std::string&);
#define REGISTER_GATESPEC(specType, name) REGISTER_NAMED_TYPE_WITH_FACTORY(gateSpec, specType, name)


/**
 * @brief Pulsar gating.
 *
 * Config message must contain:
 * @conf  enabled      Bool. Is the gating enabled or not.
 * @conf  pulsar_name  String. Name of the pulsar.
 * @conf  dm           Float. Dispersion measure in pc/cm^3.
 * @conf  t_ref        Float. Reference time for solution. Should be close to
 *                     the observing time.
 * @conf  phase_ref    Float. Phase of pulsar at t_ref.
 * @conf  rot_freq     Float. Rotational frequency in Hz.
 * @conf  pulse_width  Float. Width of pulse in s.
 * @conf  coeff        Array of floats. Polyco coefficients for timing solution.
 **/
class pulsarSpec : public gateSpec {

public:
    pulsarSpec(const std::string& name) : gateSpec(name) {};

    bool update_spec(nlohmann::json &json) override;
    std::function<float(timespec, timespec, float)> weight_function() const override;

private:
    // Config parameters for pulsar gating
    std::string pulsar_name;
    float dm;           // in pc / cm^3
    double tmid;        // in days since MJD
    double phase_ref;   // in number of rotations
    double rot_freq;    // in Hz
    float pulse_width;  // in s
    Polyco polyco;
};


/**
 * @brief Uniformly weight all data.
 *
 * @note This is used to implement the nominal visibility dataset. It does not
 *       actually gate.
 **/
class uniformSpec : public gateSpec {
public:
    uniformSpec(const std::string& name);
    bool update_spec(nlohmann::json &json) override;
    std::function<float(timespec, timespec, float)> weight_function() const override;
};


/**
 * @class visAccumulate
 * @brief Accumulate the high rate GPU output into integrated visBuffers.
 *
 * This process will accumulate the GPU output and calculate the within sample
 * variance for weights.
 *
 * @par Buffers
 * @buffer in_buf
 *         @buffer_format GPU packed upper triangle
 *         @buffer_metadata chimeMetadata
 * @buffer out_buf
 *         @buffer_format visBuffer
 *         @buffer_metadata visMetadata
 *
 * @conf  samples_per_data_set  Int. The number of samples each GPU buffer has
 *                              been integrated for.
 * @conf  num_gpu_frames        Int. The number of GPU frames to accumulate over.
 * @conf  integration_time      Float. Requested integration time in seconds.
 *                              This can be used as an alterative to
 *                              `num_gpu_frames` (which it overrides).
 *                              Internally it picks the nearest acceptable value
 *                              of `num_gpu_frames`.
 * @conf  num_elements          Int. The number of elements (i.e. inputs) in the
 *                              correlator data.
 * @conf  num_freq_in_frame     Int. Number of frequencies in each GPU frame.
 * @conf  block_size            Int. The block size of the packed data.
 * @conf  num_ev                Int. The number of eigenvectors to be stored
 * @conf  input_reorder         Array of [int, int, string]. The reordering mapping.
 *                              Only the first element of each sub-array is used and it is the the index of
 *                              the input to move into this new location. The remaining elements of the
 *                              subarray are for correctly labelling the input in ``visWriter``.
 *
 * @author Richard Shaw, Tristan Pinsonneault-Marotte
 */
class visAccumulate : public KotekanProcess {
public:
    visAccumulate(Config& config,
                  const string& unique_name,
                  bufferContainer &buffer_container);
    ~visAccumulate();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread() override;

private:

    // NOTE: Annoyingly this can't be forward declared, and defined fully externally
    // as the std::deque needs the complete type
    /**
     * @class internalState
     * @brief Hold the internal state of a gated accumulation.
     **/
    struct internalState {

        /**
         * @brief Initialise the required fields.
         *
         * Everything else will be set by the reset_state call during
         * initialisation.
         *
         * @param out_buf   Buffer we will output into.
         * @param gate_spec Specification of how any gating is done.
         **/
        internalState(Buffer* out_buf, std::unique_ptr<gateSpec> gate_spec, size_t nprod);

        /// The buffer we are outputting too
        Buffer* buf;

        // Current frame ID of the buffer we are using
        frameID frame_id;

        /// Specification of how we are gating
        //gateSpec* spec;
        std::unique_ptr<gateSpec> spec;

        /// The weighted number of total samples accumulated. Must be reset every
        /// integration period.
        float sample_weight_total;

        /// Function for applying the weighting. While this can essentially be
        /// derived from the gateSpec we need to cache it so the gating can be
        /// updated externally within an accumulation.
        std::function<float(timespec, timespec, float)> calculate_weight;

        /// Mutex to control update of gateSpec
        std::mutex state_mtx;

        /// Accumulation vectors
        std::vector<cfloat> vis1;
        std::vector<float> vis2;

        friend visAccumulate;
    };

    // Buffers to read/write
    Buffer* in_buf;
    Buffer* out_buf;  // Output for the main vis dataset only

    // Parameters saved from the config files
    size_t num_elements;
    size_t num_freq_in_frame;
    size_t num_eigenvectors;
    size_t block_size;
    size_t samples_per_data_set;
    size_t num_gpu_frames;

    // Derived from config
    size_t num_prod_gpu;

    // The mapping from buffer element order to output file element ordering
    std::vector<uint32_t> input_remap;

    // Helper methods to make code clearer

    // Set initial values of visBuffer
    void initialise_output(internalState& state,
                           int in_frame_id, int freq_ind);

    // Fill in data sections of visBuffer
    void finalise_output(internalState& state, int freq_ind,
                         uint32_t total_samples);

    // List of gating specifications
    std::map<std::string, gateSpec*> gating_specs;

    /**
     * @brief Reset the state when we restart an integration.
     *
     * @returns Return if this accumulation was enabled.
     **/
    bool reset_state(internalState& state);


    // Hold the state for any gated data
    std::deque<internalState> gated_datasets;
};

#endif