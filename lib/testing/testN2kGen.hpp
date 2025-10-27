#ifndef TEST_N2K_GEN_H
#define TEST_N2K_GEN_H

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata
#include "visUtil.hpp"         // for frameID

#include "json.hpp" // for json

#include <memory>   // for shared_ptr
#include <stddef.h> // for size_t
#include <stdint.h> // for uint32_t
#include <string>   // for string
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
 * @conf  value                 Int. Required for type "const" and "ramp".
 * @conf  values                Vector of ints. Only used for type "onehot" - sets the array element
 * to a different value for each frame; loops through the values.
 * @conf  seed                  Int. For type "random", "random_signed", and "onehot".  If non-zero,
 * seeds the random number generator on startup for reproducible results.
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
    Buffer* rfi_buf;
    std::string corr_name;
    std::string corr_type;
    std::string count_name;
    std::string count_type;
    std::string rfi_name;
    std::string rfi_type;
    std::array<int32_t, 2> corr_value;
    std::vector<std::array<int32_t, 2>> corr_value_array;
    std::array<int32_t, 2> corr_min;
    std::array<int32_t, 2> corr_max;
    int32_t count_value;
    std::vector<int32_t> count_value_array;
    int32_t count_min;
    int32_t count_max;
    int32_t rfi_value;
    std::vector<int32_t> rfi_value_array;
    int32_t rfi_min;
    int32_t rfi_max;
    bool mul_correlation_by_counts;
    int32_t samples_per_data_set;
    int32_t sub_integration_ntime;
    int32_t num_frames;
    int32_t num_local_freq;
    int32_t num_elements;
    std::vector<uint32_t> freq_ids;
    uint32_t seed;

    int corr_blocksize;  // ALWAYS 16
    int count_blocksize; // ALWAYS 8
    int corr_lin_blocks;
    int count_lin_blocks;
    int corr_num_blocks;
    int count_num_blocks;
    int num_integrations;

    const std::shared_ptr<chordMetadata> get_new_metadata(Buffer* buf, frameID frame_id);
    void set_correlation_metadata(std::shared_ptr<chordMetadata> meta, uint64_t seq_num);
    void set_counts_metadata(std::shared_ptr<chordMetadata> meta, uint64_t seq_num);
    void get_blocked_indices(int i, int j, int blocksize, int& ihi, int& jhi, int& ilo, int& jlo0,
                             int& block_idx);
};

#endif
