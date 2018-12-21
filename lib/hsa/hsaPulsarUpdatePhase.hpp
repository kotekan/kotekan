/**
 * @file
 * @brief Pulsar update phases for brute-force beamform
 *  - hsaPulsarUpdatePhase : public hsaCommand
 */

#ifndef HSA_PULSAR_UPDATE_PHASE_H
#define HSA_PULSAR_UPDATE_PHASE_H

#include "hsaCommand.hpp"
#include "restServer.hpp"

#include <mutex>
#include <thread>

/**
 * @class hsaPulsarUpdatePhase
 * @brief hsaCommand to get phase delay for brute-force beamform
 *
 * This is an hsaCommand that calculateas phase delays for brute-form beamforming.
 * It forms 10 tracking beams with phases updated every second.
 * Ten pairs of nominal coordinates (RA & Dec) is provided by the config file.
 * This script calculates the phase delay of these 10 sources,
 * for each feed position. The gain of each input is multiplied into the same array,
 * to give an array of phase delay of size 2048*10*2 to be provided as input for
 * hsaBeamformPulsar.cpp and the kernel pulsar_beamformer.hsaco.
 * There are two banks of phases to alternate between to avoid read/write conflict.
 * Each of the 10 source position (and an associated scaling factor) can be
 * changed/re-pointed on a per beam basis via endpoint.
 *
 * The gain path is registered as a subscriber to an updatable config block.
 *
 * @par REST Endpoints
 * @endpoint  /update_pulsar/<gpu id>   ``POST`` Trigger re-pointing of a
 *            specific beam at RA+Dec with a scaling factor.
 *            requires json values      "beam", "ra", "dec", "scaling"
 *            update config             source_ra[beam], source_dec[beam], psr_scaling[beam]
 *
 * @par GPU Memory
 * @gpu_mem  beamform_phase     Array of phase delays size 2048x10x2
 *     @gpu_mem_type            staging
 *     @gpu_mem_format          Array of @c float
 *     @gpu_mem_metadata        none
 *
 * @conf   num_elements         Int (default 2048). Number of elements
 * @conf   num_beams            Int (default 10). Number of pulsars
 * @conf   feed_sep_NS          Float (default 0.3048). N-S feed separation in m.
 * @conf   feed_sep_EW          Float (default 22.0). E-W feed separation in m.
 * @conf   default_gains        Float array (default 1+1j). Default gain value if gain file is
 * missing
 * @conf   source_ra            Float array - 10 initial RA (in deg) to form beams on.
 * @conf   source_dec           Float array - 10 initial Dec (in deg) to form beams on.
 * @conf   psr_scaling          Int array - 10 nominal scaling for all beams (can be changed on per
 * beam basis via endpoint)
 *
 * @author Cherry Ng
 *
 */


class hsaPulsarUpdatePhase : public hsaCommand {
public:
    /// Constructor, also initializes internal variables from config, allocates host_gain,
    /// host_phase_0, host_pahse_1, and set up 2 endpoints
    hsaPulsarUpdatePhase(Config& config, const string& unique_name, bufferContainer& host_buffers,
                         hsaDeviceInterface& device);

    /// Destructor, cleans up local allocs.
    virtual ~hsaPulsarUpdatePhase();

    /// Wait for full metadata frame and keep track of precondition_id
    int wait_on_precondition(int gpu_frame_id) override;

    /// Endpoint for providing new directory path for gain updates
    bool update_gains_callback(nlohmann::json& json);

    /// Figure our LST at this frame and the Alt-Az of the 10 sources, then calculate phase delays
    /// at each input
    void calculate_phase(struct psrCoord psr_coord, timespec time_now, float freq_now, float* gain,
                         float* output);

    /// Load gain, update phases every second by alternating the use of 2 banks.
    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

    void finalize_frame(int frame_id);

    /// Endpoint for providing new pulsar target (RA, Dec, sacling factor, beam_id)
    void pulsar_grab_callback(connectionInstance& conn, json& json_request);

private:
    /// Length of arrray of phases, should be 2048 x 10 x 2 for complex
    int32_t phase_frame_len;
    /// One of two alternating array of host phase
    float* host_phase_0;
    /// Two of two alternating array of host phase
    float* host_phase_1;
    /// 2048 elements x 2 for complex
    int32_t gain_len;
    /// Directory path where gain files are
    vector<string> _gain_dir;
    /// Default gain values if gain file is missing for this freq
    vector<float> default_gains;
    /// Array of gains, float size of 2048*2
    float* host_gain;

    /// Number of elements, should be 2048
    uint32_t _num_elements;
    /// Number of pulsar beams, should be 10
    int16_t _num_beams;

    /// Metadata buffer ID
    int32_t metadata_buffer_id;
    /// Metadata buffer precondition ID
    int32_t metadata_buffer_precondition_id;
    /// Buffer for accessing metadata
    Buffer* metadata_buf;

    /// 10 pulsar RA, DEC and scaling factor
    struct psrCoord psr_coord;               // active coordinates to be passed to metatdata
    struct psrCoord psr_coord_latest_update; // Last updated coordinates
    vector<float> _source_ra;
    vector<float> _source_dec;
    vector<int> _source_scl;
    /// Time now in second
    struct timespec time_now_gps;

    /// Freq bin index, where the 0th is at 800MHz
    int32_t freq_idx;
    /// Freq in MHz
    float freq_MHz;
    /// N-S feed separation in m
    float _feed_sep_NS;
    /// E-W feed separation in m
    int32_t _feed_sep_EW;

    /// Which phase bank (0 or 1) is used by with gpu_frame_id
    uint* bankID;
    /// Keep track of outstanding async copies involving phase bank 0
    uint bank_use_0;
    /// Keep track of outstanding async copies involving phase bank 1
    uint bank_use_1;
    /// The ID of the active bank of phase to be used (0 or 1)
    uint bank_active;

    /// Keep track of the passing of time, in order to trigger update phase every second
    uint second_now;
    uint second_last;

    /// mutex lock prevent psr_coord to be read while it is being updated.
    std::mutex _pulsar_lock;

    /// Flag to control gains to be only loaded on request.
    bool update_gains;
    /// Flag to avoid re-calculating freq-specific params except at first pass
    bool first_pass;
    /// Flag to trigger phase update, either because of endpt message or every second
    bool update_phase;

    /// Endpoint for updating psr coordinates
    std::string endpoint_psrcoord;

    /// Config base (@TODO this is a huge hack replace with updatable config)
    string config_base;
};

#endif
