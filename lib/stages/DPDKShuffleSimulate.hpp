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
 * and fills voltage/lost-sample buffers with zeros. Four voltage buffers are produced
 * (`voltage_data_buf_0`..`3`) along with a lost-samples flag buffer; metadata (FPGA sequence and
 * first packet time) are populated to look like live data. Frame timing is throttled to match
 * the configured samples-per-frame and telescope sequence length.
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
 * @conf    samples_per_data_set  Int. The number of samples in each frame.
 *
 * @par Example
 * @code
 * dpdk_shuffle_simulate:
 *   kotekan_stage: DPDKShuffleSimulate
 *   samples_per_data_set: 49152
 *   voltage_data_buf_0: vol0
 *   voltage_data_buf_1: vol1
 *   voltage_data_buf_2: vol2
 *   voltage_data_buf_3: vol3
 *   lost_samples_buf: lost_samples
 * @endcode
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
