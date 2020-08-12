/**
 * @file
 * @brief Brute-force beamform for kotekan pulsar obs
 *  - hsaTrackingBeamform : public hsaCommand
 */

#ifndef HSA_BEAMFORM_PULSAR_H
#define HSA_BEAMFORM_PULSAR_H

#include "Config.hpp"             // for Config
#include "bufferContainer.hpp"    // for bufferContainer
#include "hsa/hsa.h"              // for hsa_signal_t
#include "hsaCommand.hpp"         // for hsaCommand
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface

#include <stdint.h> // for int32_t
#include <string>   // for string
/**
 * @class hsaTrackingBeamform
 * @brief hsaCommand to brute-force beamform for pulsar obs
 *
 *
 * This is an hsaCommand that launches the kernel (pulsar_beamformer) for
 * brute-force coherent beamforming and is most applicable to pulsar observations.
 * An array of phases (shape @c n_trk x @c n_elem x 2) is calculated by hsaTrackingUpdatePhase.cpp.
 * The default number of pulsar beams to be formed is 10. The phases are matrix
 * multiplied with the input data (shape @c n_samp x @c n_elem) and the output is of dimension
 * (@c n_samp x @c n_trk x @c n_pol x 2). Output data type is 4-4b int packed as char. Currently
 * it is float, as we are pending on decision of data truncation scheme.
 *
 * @requires_kernel    pulsar_beamformer.hasco
 *
 * @par GPU Memory
 * @gpu_mem  input_reordered Input data of size input_frame_len
 *     @gpu_mem_type         static
 *     @gpu_mem_format       Array of @c uchar
 *     @gpu_mem_metadata     chimeMetadata
 * @gpu_mem  bf_output       Output data of size output_frame_len
       @gpu_mem_type         staging
 *     @gpu_mem_format       Array of @c uint8_t
 *     @gpu_mem_metadata     chimeMetadata
 * @gpu_mem  beamform_phase  Array of phases of size phase_len
       @gpu_mem_type         staging
 *     @gpu_mem_format       Array of @c float
 *     @gpu_mem_metadata     none
 *
 * @conf   num_elements         Int (default 2048). Number of elements
 * @conf   num_pulsar           Int (default 10). Number of pulsar beams to be formed
 * @conf   samples_per_data_set Int (default 49152). Number of time samples in a data set
 * @conf   num_pol              Int (default 2). Number of polarizations
 * @conf   command              String (defualt: "pulsarbf"). Kernel command.
 * @conf   kernel               String (default: "pulsar_beamformer.hsaco"). Kernel filename.
 *
 * @todo   finalize output truncation scheme
 *
 * @author Cherry Ng
 *
 */


class hsaTrackingBeamform : public hsaCommand {
public:
    /// Constructor, also initializes internal variables from config and initializes the array of
    /// phases.
    hsaTrackingBeamform(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);

    /// Destructor, cleans up local allocs.
    virtual ~hsaTrackingBeamform();

    /// Allocate kernel argument buffer, set kernel dimensions, enqueue kernel
    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

private:
    /// Input length, should be nsamp x n_elem x 2 for complex / 2 since we pack two 4-bit in one
    int32_t input_frame_len;
    /// Output length, should be 10trk x nsamp x 2 pol x 2 for complex / 2 since we pack two 4-bit
    /// in one
    int32_t output_frame_len;

    /// Length of the array of phases for beamforming, should be 10 trk * 2048 elem * 2 for complex
    int32_t phase_len;

    /// numbler of elements, should be 2048
    int32_t _num_elements;
    /// number of pulsar beams to be formed, should be 10
    int32_t _num_beams;
    /// number of polarizations in the data, should be 2
    int32_t _num_pol;
    /// number of samples
    int32_t _samples_per_data_set;
};

#endif
