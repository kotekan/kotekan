#ifndef KOTEKAN_DPDKSHUFFLESIMULATE_HPP
#define KOTEKAN_DPDKSHUFFLESIMULATE_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for int64_t, uint32_t
#include <string>   // for string

/**
 * @class DPDKShuffleSimulate
 * @brief Simulates the 4-way DPDK shuffle for ICEBoard data
 *
 * Note this currently only simulates the data rate, not the actual data the ICEBoards generate
 *
 * @par Buffers
 * @buffer voltage_data_buf_N The buffers to store the voltage data from the FPGAs
 *     @buffer_format 4+4-bit complex number
 *     @buffer_metadata chimeMetadata
 *
 * @buffer lost_samples_buf Buffer of flags set to 1 if the corresponding sample in the voltage
 *                          buffer was lost.
 *     @buffer_format uint8_t flags
 *     @buffer_metadata chimeMetadata
 *
 * @conf    samples_per_data_set  The number of samples in each frame.
 *
 * @todo Add an option to generate actual simulated voltage data
 *
 * @author Andre Renard
 */
class DPDKShuffleSimulate : public kotekan::Stage {
public:
    DPDKShuffleSimulate(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& buffer_container);
    ~DPDKShuffleSimulate();
    void main_thread() override;

private:
    static const uint32_t shuffle_size = 4;
    Buffer* voltage_data_buf[shuffle_size];
    Buffer* lost_samples_buf;
    int64_t _num_samples_per_dataset;
};


#endif // KOTEKAN_DPDKSHUFFLESIMULATE_HPP
