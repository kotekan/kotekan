/*
 * @file rfiAVXVDIF.hpp
 * @brief Contains RFI spectral kurtosis estimator using AVX2 intrinsics
 *  - rfiAVXVDIF : public kotekan::Stage
 */
#ifndef RFI_AVX_VDIF_HPP
#define RFI_AVX_VDIF_HPP

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "bufferContainer.hpp"

#include <stdint.h> // for uint32_t, uint8_t
#include <string>   // for string

/*
 * @class rfiAVXVDIF
 * @brief Consumer ``kotekan::Stage`` which consumer a buffer filled with VDIF data and
 * produces a buffer filled with spectral kurtosis estimates.
 *
 * This stage read input VDIF data and computes spectral kurtosis estimates at a variable time
 * cadence. The estimates are computed through AVX2 intrinsics. The data is read in sections of 64
 * 4-bit numbers before the power and square power are computed and stored. After a certain amount
 * of time, the integrated values are turned into a single spectral kurtosis estimate and added to
 * the output array.
 *
 * @par Buffers
 * @buffer vdif_in      The kotekan buffer containing VDIF input data.
 *      @buffer_format  Array of @c uint8_t
 *      @buffer_metadata chimeMetadata
 * @buffer rfi_out      The kotekan buffer to be filled with spectral kurtosis estimates.
 *      @buffer_format  Array of @c float
 *      @buffer_metadata chimeMetadata
 *
 * @conf   num_elements         Int (default 2048). Number of elements.
 * @conf   num_local_freq       Int (default 1). Number of local freq.
 * @conf   samples_per_data_set Int (default 32768). Number of time samples in a data set.
 * @conf   sk_step              Int Length of time integration in SK estimate.
 * @conf   frames_per_packet    Int The Number of frames to average over before sending each UDP
 * packet.
 * @conf   rfi_combined         Bool Whether or not the kurtosis measurements include an input sum.
 *
 * @author Jacob Taylor
 */
class rfiAVXVDIF : public kotekan::Stage {
public:
    // Constructor, resgister producer/consumer and apply config
    rfiAVXVDIF(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);
    // Destructor, do nothing
    ~rfiAVXVDIF();
    // Gets frames, creates thread to perform SK estimates, marks frame empty
    void main_thread() override;

private:
    // Performs integration and computes spectral kurtosis estimates
    inline void fastSKVDIF(uint8_t* data, uint32_t* temp_buf, uint32_t* sq_temp_buf, float* output);
    // Declraes arrays and instanciates correct number of fastSKVDIF instances
    void parallelSpectralKurtosis(uint32_t loop_idx, uint32_t loop_length);
    // Input buffer (VDIF)
    Buffer* buf_in;
    // Output buffer (SK estimates)
    Buffer* buf_out;
    // General Config Parameters
    /// Number of elements (2048 for CHIME or 256 for Pathfinder)
    uint32_t _num_elements;
    /// Number of frequencies per GPU (1 for CHIME or 8 for Pathfinder)
    uint32_t _num_local_freq;
    /// Number of time samples per frame (Usually 32768 or 49152)
    uint32_t _samples_per_data_set;
    // RFI config parameters
    /// The kurtosis step (How many timesteps per kurtosis estimate)
    uint32_t _sk_step;
    /// Flag for element summation in kurtosis estimation process
    bool _rfi_combined;
    // Arrays to hold current input/output frames
    uint8_t* in_local;
    uint8_t* out_local;
};

#endif
