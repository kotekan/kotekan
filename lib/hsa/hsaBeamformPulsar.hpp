/*****************************************
File Contents:
- hsaBeamformPulsar : public hsaCommand
*****************************************/

#ifndef HSA_BEAMFORM_PULSAR_H
#define HSA_BEAMFORM_PULSAR_H

#include "hsaCommand.hpp"

/**
 * @class hsaBeamformPulsar
 * @brief hsaCommand to brute-force beamform for pulsar obs
 *
 *
 * This is an hsaCommand that launches the kernel (pulsar_beamformer) for 
 * brute-force coherent beamforming and is most applicable to pulsar observations. 
 * An array of phases (shape @c n_psr x @c n_elem) is calculated by hsaPulsarUpdatePhase.cpp. 
 * The default number of pulsar beams to be formed is 10. The phases are matrix 
 * multiplied with the input data (shape @c n_samp x @c n_elem) and the output is of dimension 
 * (@c n_samp x @c n_psr).
 *
 * @requires_kernel    pulsar_beamformer.hasco
 *
 * @gpu_mem  input           Input data of size input_frame_len
 *     @gpu_mem_type         staging
 *     @gpu_mem_format       Array of @c uchar
 *     @gpu_mem_metadata     chimeMetadata
 * @gpu_mem  bf_output       Output data of size output_frame_len
       @gpu_mem_type         staging
 *     @gpu_mem_format       Array of @c uint8_t
 *     @gpu_mem_metadata     chimeMetadata
 * @gpu_mem  beamform_phase  Array of phases of size phase_len
       @gpu_mem_type         static
 *     @gpu_mem_format       Array of @c float
 *     @gpu_mem_metadata     none
 *
 *
 * @conf   num_elements         Int (default 2048). Number of elements
 * @conf   num_pulsar           Int (default 10). Number of pulsar beams to be formed
 * @conf   samples_per_data_set Int (default 49152). Number of time samples in a data set
 * @conf   num_pol              Int (default 2). Number of polarizations
 *
 * 
 * @todo   Currently the phases are intialized to some dummy values (beam_id/10) in 
 *         the constructor. This shouldn't be necessary because the phases 
 *         should have been calculated in another piece of code (hsaPulsarUpdatePhase.cpp), 
 *         and will be overwriten in execute to sensible values. I haven't checked in 
 *         hsaPulsarUpdatePhase.cpp to master yet that is why I inserted those dummy values
 *         to be on the safe side. This will be tidied up in a few days. 
 *
 * @author Cherry Ng
 *
 */


class hsaBeamformPulsar: public hsaCommand
{
public:
    /// Constructor, also initializes internal variables from config and initializes the array of phases.
    hsaBeamformPulsar(const string &kernel_name, const string &kernel_file_name,
                        hsaDeviceInterface &device, Config &config,
                        bufferContainer &host_buffers,
                        const string &unique_name);

    /// Destructor, cleans up local allocs.
    virtual ~hsaBeamformPulsar();

    /// Parse config
    void apply_config(const uint64_t& fpga_seq) override;

    /// Allocate kernel argument buffer, set kernel dimensions, enqueue kernel
    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq,
                         hsa_signal_t precede_signal) override;

private:
    /// Input length, should be nsamp x n_elem x 2 for complex / 2 since we pack two 4-bit in one
    int32_t input_frame_len;
    /// Output length, should be 10psr x nsamp x 2 pol x 2 for complex / 2 since we pack two 4-bit in one
    int32_t output_frame_len;


    ///Length of the array of phases for beamforming, should be 10 psr * 2048 elem * 2 for complex
    int32_t phase_len;
    /// pointer to the phase array
    float * host_phase;


    ///numbler of elements, should be 2048
    int32_t _num_elements;
    /// number of pulsar beams to be formed, should be 10
    int32_t _num_pulsar;
    /// number of polarizations in the data, should be 2
    int32_t _num_pol;
    /// number of samples
    int32_t _samples_per_data_set;
};

#endif
