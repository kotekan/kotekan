#ifndef TEST_N2K_GEN_H
#define TEST_N2K_GEN_H

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata
#include "dataset.hpp"         // for dset_id_t
#include "visUtil.hpp"         // for frameID

#include <array>    // for array
#include <memory>   // for shared_ptr
#include <optional> // for optional
#include <stdint.h> // for int32_t, uint32_t, uint64_t
#include <string>   // for string, basic_string
#include <vector>   // for vector

/**
 * @class testN2kGen
 * @brief Generate test N2k data as a standin for N2k.
 *
 * @par Buffers
 * @buffer out_buf Buffer to fill with unnormalized visibilities
 *         @buffer_format complex int32+32
 *         @buffer_shape [samples_per_data_set/sub_integration_ntime,
 *              num_local_freq, num_corr_blocks, 16, 16]
 *         @buffer_metadata chordMetadata time_downsample_fpga[] = sub_integration_ntime
 * @buffer out_counts_buf Buffer to fill with counts
 *         @buffer_format int32
 *         @buffer_shape [samples_per_data_set/sub_integration_ntime,
 *              num_local_freq, num_count_blocks, 8, 8]
 *         @buffer_metadata chordMetadata time_downsample_fpga[] = sub_integration_ntime
 *
 * @conf  correlation_name      String. quantity name for correlation in chordMetadata
 * @conf  counts_name           String. quantity name for counts in chordMetadata
 * @conf  correlation_type      String. "const", "random".
 * @conf  counts_type           String. "const", "random".
 * @conf  correlation_value     Pair of ints. Used when `correlation_type` is "const".
 * @conf  correlation_values    Vector of int pairs. Optional cycle for "const" correlation frames.
 * @conf  counts_value          Int. Used when `counts_type` is "const" or "const_scalar".
 * @conf  counts_values         Vector of ints. Optional cycle for "const" count frames.
 * @conf  seed                  Int. Default 0. Seeds the deterministic RNG for "random" correlation
 *                              and counts variants.
 * @conf  dataset_id            Hash string. Optional dataset id to set for CHIME * pipelines.
 * @conf  first_frame_index     Int. Default 0. Starting FPGA frame number, fro
 *                              frames of size samples_per_data_set.
 * @conf  samples_per_data_set  Int. How often to produce data.
 * @conf  num_frames            Int. How many frames to produce. Default inf.
 * @conf  num_freq_in_frame     Int. Number of frequencies in each GPU frame.
 *
 * @author Geoffrey Ryan
 */
class testN2kGen : public kotekan::Stage {
public:
    testN2kGen(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);
    ~testN2kGen(){};
    void main_thread() override;

private:
    Buffer* corr_buf;
    Buffer* count_buf;
    std::string corr_name;
    std::string corr_type;
    std::string count_name;
    std::string count_type;
    std::array<int32_t, 2> corr_value;
    std::vector<std::array<int32_t, 2>> corr_value_array;
    std::array<int32_t, 2> corr_min;
    std::array<int32_t, 2> corr_max;
    int32_t count_value;
    std::vector<int32_t> count_value_array;
    int32_t count_min;
    int32_t count_max;
    bool mul_correlation_by_counts;
    int32_t samples_per_data_set;
    int32_t sub_integration_ntime;
    int32_t num_frames;
    int32_t num_local_freq;
    int32_t num_elements;
    std::vector<uint32_t> freq_ids;
    uint32_t seed;
    std::optional<dset_id_t> dset_id;
    int64_t first_frame_index;
    int repeat_count;

    static constexpr int corr_blocksize = 16; // ALWAYS 16
    static constexpr int count_blocksize = 8; // ALWAYS 8
    int corr_lin_blocks;
    int count_lin_blocks;
    int corr_num_blocks;
    int count_num_blocks;
    int num_integrations;

    int corr_num_entries;
    int count_num_entries;

    std::shared_ptr<chordMetadata> get_new_metadata(Buffer* buf, frameID frame_id);
    void set_shared_metadata(const std::shared_ptr<chordMetadata>& meta, uint64_t seq_num);
    void get_blocked_indices(int i, int j, int blocksize, int& ihi, int& jhi, int& ilo, int& jlo0,
                             int& block_idx);
};

#endif
