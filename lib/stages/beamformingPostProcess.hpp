/**
 * @file
 * @brief Format GPU beamformer output into VDIF frames.
 * - beamformingPostProcess : public kotekan::Stage
 */
#ifndef BEAMFORMING_POST_PROCESS
#define BEAMFORMING_POST_PROCESS

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for uint32_t, int32_t
#include <string>   // for string
#include <vector>   // for vector


/**
 * @class beamformingPostProcess
 * @brief Packages multiple GPU beamformer links into dual-pol VDIF frames.
 *
 * Consumes one buffer per GPU link, verifies all inputs share the same FPGA sequence number, and
 * writes interleaved VDIF frames (5032 bytes) into `vdif_out_buf`. Each VDIF payload carries both
 * polarizations for each coarse frequency in `_num_local_freq`, using the coarse frequency as the
 * VDIF thread ID. Headers are filled once per output buffer, then payload samples are copied from
 * the raw 4+4-bit complex beam data. A background cadence is not used: the stage runs in the main
 * thread until inputs drain or stop is requested.
 *
 * @par Buffers
 * @buffer beam_in_buf_N  Beamformer output per GPU (N = 0..num_gpus-1).
 *     @buffer_format 4+4-bit complex voltages laid out [time][freq][pol] (pol packed as bytes)
 *     @buffer_metadata chordMetadata
 * @buffer vdif_out_buf   Output VDIF buffer.
 *     @buffer_format Raw VDIF frames (no kotekan metadata attached)
 *     @buffer_metadata none
 *
 * @conf num_links            UInt. Number of FPGA links to combine into each VDIF frame.
 * @conf num_gpus             UInt. Number of GPU input buffers.
 * @conf samples_per_data_set UInt. Samples per FPGA frame.
 * @conf num_data_sets        UInt. Number of FPGA frames per GPU frame.
 * @conf link_map             Array<Int>. Maps link index -> GPU buffer index.
 * @conf num_local_freq       UInt. Number of coarse frequencies per link (used to size payload).
 *
 * @par Example
 * @code
 * beamforming_post_process:
 *   kotekan_stage: beamformingPostProcess
 *   num_links: 4
 *   num_gpus: 2
 *   samples_per_data_set: 49152
 *   num_data_sets: 8
 *   num_local_freq: 4
 *   link_map: [0, 1, 0, 1]
 *   vdif_out_buf: vdif_out
 *   beam_in_buf_0: beam_gpu0
 *   beam_in_buf_1: beam_gpu1
 * @endcode
 */
class beamformingPostProcess : public kotekan::Stage {
public:
    beamformingPostProcess(kotekan::Config& config, const std::string& unique_name,
                           kotekan::bufferContainer& buffer_container);
    virtual ~beamformingPostProcess();
    void main_thread() override;

private:
    void fill_headers(unsigned char* out_buf, struct VDIFHeader* vdif_header, const uint32_t second,
                      const uint32_t fpga_seq_num, const uint32_t num_links, uint32_t* thread_id);

    Buffer** in_buf;
    Buffer* vdif_buf;

    // Config variables
    uint32_t _num_fpga_links;
    uint32_t _samples_per_data_set;
    uint32_t _num_data_sets;
    std::vector<int32_t> _link_map;
    uint32_t _num_local_freq;
    uint32_t _num_gpus;
};


#endif
