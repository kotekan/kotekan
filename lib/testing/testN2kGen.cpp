#include "testN2kGen.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"       // for Telescope, stream_t
#include "buffer.hpp"          // for Buffer, allocate_new_metadata_object, mark_frame_full
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"
#include "kotekanLogging.hpp" // for INFO, DEBUG
#include "visUtil.hpp"        // for frameID

#include <algorithm>  // for copy, max
#include <assert.h>   // for assert
#include <atomic>     // for atomic_bool
#include <cmath>      // for fmod
#include <cstdint>    // for uint64_t
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, _Placeholder, bind, _1, _2, function
#include <random>
#include <regex>       // for match_results<>::_Base_type
#include <stdexcept>   // for runtime_error, invalid_argument
#include <stdint.h>    // for uint64_t, uint32_t, uint8_t, int32_t
#include <stdlib.h>    // for rand, srand
#include <strings.h>   // for bzero
#include <sys/time.h>  // for gettimeofday, timeval
#include <sys/types.h> // for uint
#include <unistd.h>    // for usleep
#include <vector>      // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(testN2kGen);

testN2kGen::testN2kGen(Config& config, const std::string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&testN2kGen::main_thread, this)) {

    // Get buffers
    corr_buf = get_buffer("out_buf");
    corr_buf->register_producer(unique_name);
    count_buf = get_buffer("out_counts_buf");
    count_buf->register_producer(unique_name);

    corr_name = config.get_default<std::string>(unique_name, "correlation_name", "n2k_correlation");
    count_name = config.get_default<std::string>(unique_name, "counts_name", "n2k_counts");

    corr_type = config.get<std::string>(unique_name, "correlation_type");
    count_type = config.get<std::string>(unique_name, "counts_type");
    assert(corr_type == "const" || corr_type == "random");
    assert(count_type == "const" || count_type == "random");
    
    corr_value = config.get_default<std::array<int32_t, 2>>(unique_name, "correlation_value", std::array<int32_t, 2>({1234, 5678}));
    count_value = config.get_default<int32_t>(unique_name, "counts_value", 4444);
    
    corr_value_array = config.get_default<std::vector<std::array<int32_t, 2>>>(unique_name, "correlation_values", std::vector<std::array<int32_t, 2>>());
    count_value_array = config.get_default<std::vector<int32_t>>(unique_name, "count_values", std::vector<int32_t>());

    corr_min = config.get_default<std::array<int32_t, 2>>(unique_name, "correlation_min", {-524288, -524288});
    corr_max = config.get_default<std::array<int32_t, 2>>(unique_name, "correlation_max", {524288, 524288});
    count_min = config.get_default<int32_t>(unique_name, "count_min", 0);
    count_max = config.get_default<int32_t>(unique_name, "count_max", 8192);


    samples_per_data_set = config.get_default<size_t>(unique_name, "samples_per_data_set", 8192);
    sub_integration_ntime = config.get_default<size_t>(unique_name, "sub_integration_ntime", 8192);

    num_frames = config.get_default<int>(unique_name, "num_frames", -1);
    num_local_freq = config.get_default<size_t>(unique_name, "num_local_freq", 1);
    num_elements = config.get_default<size_t>(unique_name, "num_elements", 16);
    freq_ids = config.get_default<std::vector<uint32_t>>(unique_name, "freq_ids", std::vector<uint32_t>({4096}));
    seed = config.get_default<uint32_t>(unique_name, "seed", 0);

    // now thing we calculate
    corr_blocksize = 16;    // ALWAYS 16
    count_blocksize = 8;    // ALWAYS 8
    
    assert(samples_per_data_set % sub_integration_ntime == 0);
    assert(num_elements % corr_blocksize == 0);
    assert(num_elements % (8 * count_blocksize) == 0);
    assert(freq_ids.size() > 0);

    num_integrations = samples_per_data_set / sub_integration_ntime;

    corr_lin_blocks = num_elements / corr_blocksize;
    count_lin_blocks = (num_elements / 8) / count_blocksize;

    corr_num_blocks = (corr_lin_blocks * (corr_lin_blocks + 1)) / 2;
    count_num_blocks = (count_lin_blocks * (count_lin_blocks + 1)) / 2;
}

const std::shared_ptr<chordMetadata> testN2kGen::get_new_metadata(Buffer *buf, frameID frame_id) {
    buf->allocate_new_metadata_object(frame_id);
    
    const std::shared_ptr<metadataObject> mc = buf->get_metadata(frame_id);
    if (!mc) {
        FATAL_ERROR("Buffer {:s} frame {:d} cannot allocate metadata", buf->buffer_name,
                    frame_id);
    }
    assert(mc);
    if (!metadata_is_chord(mc)) {
        FATAL_ERROR("Buffer {:s} frame {:d} does not have CHORD metadata", buf->buffer_name,
                    frame_id);
    }
    assert(metadata_is_chord(mc));
    const std::shared_ptr<chordMetadata> meta = get_chord_metadata(mc);
    assert(meta);

    return meta;
}

void testN2kGen::set_correlation_metadata(std::shared_ptr<chordMetadata> meta,
                                      uint64_t seq_num) {
    meta->set_name(corr_name);
    meta->fpga_seq_num = seq_num;
    meta->type = kotekan::int32;
    meta->dims = 6;
    assert(meta->dims <= CHORD_META_MAX_DIM);
    meta->set_array_dimension(0, num_integrations, "Tc");
    meta->set_array_dimension(1, num_local_freq, "F");
    meta->set_array_dimension(2, corr_num_blocks, "DPhi");
    meta->set_array_dimension(3, corr_blocksize, "DPlo1");
    meta->set_array_dimension(4, corr_blocksize, "DPlo2");
    meta->set_array_dimension(5, 2, "C");
    meta->set_strides_simple();
    
    meta->fpga_seq_num = seq_num;
    meta->sample0_offset = seq_num;
    meta->offset_downsampling = 1;

    meta->nfreq = num_local_freq;
    assert(meta->nfreq <= CHORD_META_MAX_FREQ);
    for (int f = 0; f < meta->nfreq; f++) {
        meta->coarse_freq[f] = freq_ids[f % freq_ids.size()];
        meta->freq_upchan_factor[f] = 1;
        meta->half_fpga_sample0[f] = 0;
        meta->time_downsampling_fpga[f] = sub_integration_ntime;
    }
}

void testN2kGen::set_counts_metadata(std::shared_ptr<chordMetadata> meta,
                                      uint64_t seq_num) {
    meta->set_name(count_name);
    meta->fpga_seq_num = seq_num;
    meta->type = kotekan::int32;
    meta->dims = 5;
    assert(meta->dims <= CHORD_META_MAX_DIM);
    meta->set_array_dimension(0, num_integrations, "Tc");
    meta->set_array_dimension(1, num_local_freq, "F");
    meta->set_array_dimension(2, count_num_blocks, "D8Phi");
    meta->set_array_dimension(3, count_blocksize, "D8Plo1");
    meta->set_array_dimension(4, count_blocksize, "D8Plo2");
    meta->set_strides_simple();
    
    meta->fpga_seq_num = seq_num;
    meta->sample0_offset = seq_num;
    meta->offset_downsampling = 1;

    meta->nfreq = num_local_freq;
    assert(meta->nfreq <= CHORD_META_MAX_FREQ);
    for (int f = 0; f < meta->nfreq; f++) {
        meta->coarse_freq[f] = freq_ids[f % freq_ids.size()];
        meta->freq_upchan_factor[f] = 1;
        meta->half_fpga_sample0[f] = 0;
        meta->time_downsampling_fpga[f] = sub_integration_ntime;
    }
}


void testN2kGen::main_thread() {

    frameID corr_frame_id(corr_buf);
    frameID count_frame_id(count_buf);
    int num_frames_generated = 0;
    uint64_t seq_num = 0;

    std::random_device rd;
    if (seed == 0)
        seed = rd();
    std::mt19937 gen(seed);

    while (!stop_thread) {

        // grab frames
        int32_t *corr = (int32_t*)corr_buf->wait_for_empty_frame(unique_name, corr_frame_id);
        if (corr == nullptr)
            break;
        int32_t *count = (int32_t*)count_buf->wait_for_empty_frame(unique_name, count_frame_id);
        if (count == nullptr)
            break;

        // create metadata
        const std::shared_ptr<chordMetadata> corr_meta = get_new_metadata(corr_buf, corr_frame_id);
        const std::shared_ptr<chordMetadata> count_meta = get_new_metadata(count_buf, count_frame_id);

        // fill metadata
        set_correlation_metadata(corr_meta, seq_num);
        set_counts_metadata(count_meta, seq_num);

        // block, freq, and time strides for access into the 
        // correlation and counts buffers
        int db_corr = 2 * corr_blocksize * corr_blocksize;
        int df_corr = db_corr * corr_num_blocks;
        int dt_corr = df_corr * num_local_freq;
        int db_count = count_blocksize * count_blocksize;
        int df_count = db_count * count_num_blocks;
        int dt_count = df_count * num_local_freq;

        for(int t = 0; t < num_integrations; t++) {
            for(int f = 0; f < num_local_freq; f++) {
                // Fill the correlation array
                int corr_block_idx = 0;

                for(int ihi = 0; ihi < corr_lin_blocks; ihi++) {
                    // Lower triangular only
                    for(int jhi = 0; jhi <= ihi; jhi++) {
                        for(int ilo = 0; ilo < corr_blocksize; ilo++) {
                            for(int jlo = 0; jlo < corr_blocksize; jlo++) {
                                int idx = 2 * (jlo + ilo * corr_blocksize) + corr_block_idx * db_corr + f * df_corr + t * dt_corr;

                                corr[idx + 0] = 0;  // Real
                                corr[idx + 1] = 0;  // Imag
                            }
                        }

                        corr_block_idx++;
                    }
                }

                // Fill the count array
                int count_block_idx = 0;

                for(int ihi = 0; ihi < count_lin_blocks; ihi++) {
                    // Lower triangular only
                    for(int jhi = 0; jhi <= ihi; jhi++) {
                        for(int ilo = 0; ilo < count_blocksize; ilo++) {
                            for(int jlo = 0; jlo < count_blocksize; jlo++) {
                                int idx = jlo + ilo * count_blocksize + count_block_idx * db_count + f * df_count + t * dt_count;

                                count[idx] = 0;
                            }
                        }

                        count_block_idx++;
                    }
                }
            } // f
        } // t

        /*
        if (value_array.size() && (type == "const"))
            // Cycle through "values" array, if given
            value = value_array[frame_id % value_array.size()];

        for (uint f = 0; f < num_local_freq; f++)
            for (uint t = 0; t < nt_output; t++) {
                size_t idx = f * nt_output + t;
                if (type == "const")
                    frame[idx] = value;
                else if (type == "all_true")
                    frame[idx] = 0xFFFF'FFFF;
                else if (type == "all_false")
                    frame[idx] = 0;
                else if (type == "random")
                    frame[idx] = gen();
            }
            */

        DEBUG("Generated a {:s} test correlation data set in {:s}[{:d}]", corr_type, corr_buf->buffer_name, corr_frame_id);
        DEBUG("Generated a {:s} test counts data set in {:s}[{:d}]", count_type, count_buf->buffer_name, count_frame_id);
        DEBUG("Corr sample size is: {:d}", corr_meta->sample_bytes());
        DEBUG("Counts sample size is: {:d}", count_meta->sample_bytes());

        corr_buf->mark_frame_full(unique_name, corr_frame_id++);
        count_buf->mark_frame_full(unique_name, count_frame_id++);

        num_frames_generated++;
        seq_num += samples_per_data_set;

        if (num_frames >= 0 && num_frames_generated >= num_frames) {
            INFO("Generated the requested number of frames ({:d}) - exiting", num_frames);
            break;
        }
    }
}
