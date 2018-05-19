/**
 * @file
 * @brief Pulsar update phases for brute-force beamform
 *  - hsaPulsarUpdatePhase : public hsaCommand
 */

#ifndef HSA_PULSAR_UPDATE_PHASE_H
#define HSA_PULSAR_UPDATE_PHASE_H

#include "hsaCommand.hpp"
#include <mutex>
#include <thread>
#include "restServer.hpp"

/**
 * @class hsaPulsarUpdatePhase
 * @brief hsaCommand to get phase delay for brute-force beamform
 *
 * This is an hsaCommand that calculates phase delays for brute-form beamforming.
 * It forms 10 tracking beams with phases updated every second.
 * One nominal coordinates (RA & Dec) is provided by the config file.
 * This script calculates the phase delay of that source and 9 positions around it,
 * for each feed position. The gain of each input is multiplied into the same array,
 * to give an array of phase delay of size 2048*10*2 to be provided as input for
 * hsaBeamformPulsar.cpp and the kernel pulsar_beamformer.hsaco.
 * There are two banks of phases to alternate between to avoid read/write conflict.
 * Each of the 10 source position (and an associated scaling factor) can be changed/re-pointed
 * on a per beam basis via endpoint.
 *
 * @par GPU Memory
 * @gpu_mem  beamform_phase     Array of phase delays size 2048x10x2
 *     @gpu_mem_type            staging
 *     @gpu_mem_format          Array of @c float
 *     @gpu_mem_metadata        none
 *
 * @conf   num_elements         Int (default 2048). Number of elements
 * @conf   num_pulsar           Int (default 10). Number of pulsars
 * @conf   feed_sep_NS          Float (default 0.3048). N-S feed separation in m.
 * @conf   feed_sep_EW          Float (default 22.0). E-W feed separation in m.
 * @conf   gain_dir             String - directory path where gain files are
 * @conf   default_gains        Float array (default 1+1j). Default gain value if gain file is missing
 * @conf   source_ra            Float - one initial RA (in hr) to form beams on.
 * @conf   source_dec           Float - one initial Dec (in deg) to form beams on.
 * @conf   psr_scaling          Float - nominal scaling for all beams (can be changed on per beam basis via endpoint)
 *
 * @todo change time_now from system time to gps time
 *
 * @author Cherry Ng
 *
 */


class hsaPulsarUpdatePhase: public hsaCommand
{
public:
    ///Constructor, also initializes internal variables from config, allocates host_gain, host_phase_0, host_pahse_1, and set up 2 endpoints
    hsaPulsarUpdatePhase( Config &config,const string &unique_name,
                        bufferContainer &host_buffers, hsaDeviceInterface &device);

    /// Destructor, cleans up local allocs.
    virtual ~hsaPulsarUpdatePhase();

    /// Wait for full metadata frame and keep track of precondition_id
    int wait_on_precondition(int gpu_frame_id) override;

    /// Endpoint for providing new directory path for gain updates
    void update_gains_callback(connectionInstance& conn, json& json_request);

    /// Figure our LST at this frame and the Alt-Az of the 10 sources, then calculate phase delays at each input
    void calculate_phase(struct psrCoord psr_coord, timeval time_now, float freq_now, float *gain, float *output);

    /// Load gain, update phases every second by alternating the use of 2 banks.
    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq,
                         hsa_signal_t precede_signal) override;

    void finalize_frame(int frame_id);

    /// Endpoint for providing new pulsar target (RA, Dec, sacling factor, beam_id)
    void pulsar_grab_callback(connectionInstance& conn, json& json_request);

private:

    /// Length of arrray of phases, should be 2048 x 10 x 2 for complex
    int32_t phase_frame_len;
    /// One of two alternating array of host phase
    float * host_phase_0;
    /// Two of two alternating array of host phase
    float * host_phase_1;
    /// 2048 elements x 2 for complex
    int32_t gain_len;
    /// Directory path where gain files are
    string _gain_dir;
    /// Default gain values if gain file is missing for this freq
    vector<float> default_gains;
    /// Array of gains, float size of 2048*2
    float * host_gain;

    /// One initial RA from config
    float _source_ra;
    /// One initial Dec from config
    float _source_dec;
    /// One initial scaling factor from config
    uint32_t _psr_scaling;

    /// Number of elements, should be 2048
    int32_t _num_elements;
    /// Number of pulsar beams, should be 10
    int16_t _num_pulsar;

    /// Metadata buffer ID
    int32_t metadata_buffer_id;
    /// Metadata buffer precondition ID
    int32_t metadata_buffer_precondition_id;
    /// Buffer for accessing metadata
    Buffer * metadata_buf;

    /// 10 pulsar RA, DEC and scaling factor
    struct psrCoord psr_coord;
    /// Time now in second
    struct timeval time_now;

    /// Freq bin index, where the 0th is at 800MHz
    int32_t freq_idx;
    /// Freq in MHz
    float freq_MHz;
    /// N-S feed separation in m
    float _feed_sep_NS;
    /// E-W feed separation in m
    int32_t _feed_sep_EW;

    /// Determine which bank of phases to read from, normally its either 0 or 1
    uint16_t bank_read_id;
    /// Determine which bank of phases to write to, when updating phases, should be 0 or 1
    uint16_t bank_write;
    /// mutex lock to prevent a bank from being read while it is being written to.
    std::mutex mtx_read;
    /// mutex lock prevent psr_coord to be read while it is being updated.
    std::mutex _pulsar_lock;

    ///Flag to control gains to be only loaded on request.
    bool update_gains;
    /// Flag to avoid re-calculating freq-specific params except at first pass
    bool first_pass;

};

#endif
