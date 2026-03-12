#ifndef TEST_XPOSE_HPP
#define TEST_XPOSE_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for int32_t, uint32_t, uint8_t
#include <string>   // for string
#include <vector>   // for vector

/**
 * @brief Produces test data for xpose2048.
 *
 * @par Buffers
 * @buffer cpu_xposed_voltage_buf Kotekan buffer for xposed voltages
 *     @buffer_metadata chordMetadata
 * @buffer host_scatter_indices_buf Kotekan buffer with scattering indices
 *     @buffer_metadata chordMetadata
 * @buffer out_bufs Kotekan buffers with unxposed voltages
 *     @buffer_metadata chordMetadata
 *
 * @author Roland Haas
 */
class testXpose : public kotekan::Stage {
public:
    /// Standard constructor
    testXpose(kotekan::Config& config, const std::string& unique_name,
                            kotekan::bufferContainer& buffer_container);

    /// Destructor
    ~testXpose();

    /// Main thead which zeros the data from the lost_samples_buf
    void main_thread() override;

private:
    /// The buffer with the expected output
    Buffer* out_xposed_buf;

    /// The buffer with the scattering indices
    Buffer* scatter_indices_buf;

    /// The buffer with the input buffers for xpose2048
    std::vector<Buffer*> out_bufs;

    /// number of polarizations
    const int num_polarizations;

    /// number of dishes
    const int num_dishes;

    /// number of time samples per frame
    const int num_times;

    /// number of frequencies in xpose2048 input buffer
    const int num_frequencies;

    /// number of frequencies in expected output  buffer
    const int num_xposed_frequencies;
};

#endif /* TEST_XPOSE_HPP */
