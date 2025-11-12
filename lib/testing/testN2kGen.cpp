#include "testN2kGen.hpp"

#include "Config.hpp"          // for Config
#include "DataType.hpp"        // for DataType
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata, metadata_is_chord, CHORD_META_MAX_DIM, CHO...
#include "kotekanLogging.hpp"  // for FATAL_ERROR, DEBUG, INFO
#include "metadata.hpp"        // for metadataObject
#include "visUtil.hpp"         // for frameID, modulo

#include "fmt.hpp" // for compile_string_to_view

#include <assert.h>   // for assert
#include <cstdlib>    // for abort, size_t
#include <functional> // for bind, function
#include <random>     // for uniform_int_distribution, mt19937
#include <stdint.h>   // for int32_t, uint32_t, uint64_t, int64_t
#include <utility>    // for swap
#include <vector>     // for vector


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
    rfi_buf = get_buffer("in_rfimask_buf");
    rfi_buf->register_consumer(unique_name);

    // Get generation parameters
    corr_name = config.get_default<std::string>(unique_name, "correlation_name", "n2k_correlation");
    count_name = config.get_default<std::string>(unique_name, "counts_name", "n2k_counts");

    corr_type = config.get<std::string>(unique_name, "correlation_type");
    count_type = config.get<std::string>(unique_name, "counts_type");
    assert(corr_type == "const" || corr_type == "random" || corr_type == "f_times_ee"
           || corr_type == "f_times_ee_plus_t");
    assert(count_type == "const" || count_type == "random" || count_type == "random_scalar"
           || count_type == "const_scalar");

    corr_value = config.get_default<std::array<int32_t, 2>>(unique_name, "correlation_value",
                                                            std::array<int32_t, 2>({1234, 5678}));
    count_value = config.get_default<int32_t>(unique_name, "counts_value", 4444);

    corr_value_array = config.get_default<std::vector<std::array<int32_t, 2>>>(
        unique_name, "correlation_values", std::vector<std::array<int32_t, 2>>());
    count_value_array = config.get_default<std::vector<int32_t>>(unique_name, "counts_values",
                                                                 std::vector<int32_t>());

    corr_min = config.get_default<std::array<int32_t, 2>>(unique_name, "correlation_min",
                                                          {-524288, -524288});
    corr_max = config.get_default<std::array<int32_t, 2>>(unique_name, "correlation_max",
                                                          {524288, 524288});
    count_min = config.get_default<int32_t>(unique_name, "count_min", 0);
    count_max = config.get_default<int32_t>(unique_name, "count_max", 8192);

    mul_correlation_by_counts =
        config.get_default<bool>(unique_name, "mul_correlation_by_counts", false);

    // Get array size parameters
    samples_per_data_set = config.get_default<size_t>(unique_name, "samples_per_data_set", 8192);
    sub_integration_ntime = config.get_default<size_t>(unique_name, "sub_integration_ntime", 8192);

    num_frames = config.get_default<int>(unique_name, "num_frames", -1);
    num_local_freq = config.get_default<size_t>(unique_name, "num_local_freq", 1);
    num_elements = config.get_default<size_t>(unique_name, "num_elements", 16);
    freq_ids = config.get_default<std::vector<uint32_t>>(unique_name, "freq_ids",
                                                         std::vector<uint32_t>({4096}));
    seed = config.get_default<uint32_t>(unique_name, "seed", 0);

    // Check parameter compatibility
    if (num_elements <= 0) {
        FATAL_ERROR("num_elements ({:d}) is not positive.", num_elements);
        std::abort();
    }
    if (num_local_freq <= 0) {
        FATAL_ERROR("num_local_freq ({:d}) is not positive.", num_local_freq);
        std::abort();
    }
    if (samples_per_data_set <= 0) {
        FATAL_ERROR("samples_per_data_set ({:d}) is not positive.", samples_per_data_set);
        std::abort();
    }
    if (sub_integration_ntime <= 0) {
        FATAL_ERROR("sub_integration_ntime ({:d}) is not positive.", sub_integration_ntime);
        std::abort();
    }
    if (samples_per_data_set % sub_integration_ntime != 0) {
        FATAL_ERROR(
            "samples_per_data_set ({:d}) is not a multiple of sub_integration_ntime ({:d}).",
            samples_per_data_set, sub_integration_ntime);
        std::abort();
    }
    if (samples_per_data_set % 1024 != 0) {
        FATAL_ERROR("samples_per_data_set ({:d}) is not a multiple of 1024.", samples_per_data_set);
        std::abort();
    }
    if (num_elements % corr_blocksize != 0) {
        FATAL_ERROR("num_elements ({:d}) is not a multiple of corr_blocksize ({:d}).", num_elements,
                    corr_blocksize);
        std::abort();
    }
    if (num_elements % (8 * count_blocksize) != 0) {
        FATAL_ERROR("num_elements ({:d}) / 8 is not a multiple of count_blocksize ({:d}).",
                    num_elements, count_blocksize);
        std::abort();
    }
    if (freq_ids.empty()) {
        FATAL_ERROR("freq_ids must have at least one element.");
        std::abort();
    }

    num_integrations = samples_per_data_set / sub_integration_ntime;

    corr_lin_blocks = num_elements / corr_blocksize;
    count_lin_blocks = (num_elements / 8) / count_blocksize;

    corr_num_blocks = (corr_lin_blocks * (corr_lin_blocks + 1)) / 2;
    count_num_blocks = (count_lin_blocks * (count_lin_blocks + 1)) / 2;

    // Check frame sizes
    size_t corr_frame_size = 2 * sizeof(int32_t) * corr_blocksize * corr_blocksize * corr_num_blocks
                             * num_local_freq * num_integrations;
    size_t count_frame_size = sizeof(int32_t) * count_blocksize * count_blocksize * count_num_blocks
                              * num_local_freq * num_integrations;
    size_t rfi_frame_size = num_local_freq * samples_per_data_set / 8;

    if (corr_buf->frame_size != corr_frame_size) {
        FATAL_ERROR("out_buf ({:s}) has frame size {:d}, expected {:d}.", corr_buf->buffer_name,
                    corr_buf->frame_size, corr_frame_size);
        std::abort();
    }
    if (count_buf->frame_size != count_frame_size) {
        FATAL_ERROR("out_count_buf ({:s}) has frame size {:d}, expected {:d}.",
                    count_buf->buffer_name, count_buf->frame_size, count_frame_size);
        std::abort();
    }
    if (rfi_buf->frame_size != rfi_frame_size) {
        FATAL_ERROR("out_buf ({:s}) has frame size {:d}, expected {:d}.", rfi_buf->buffer_name,
                    rfi_buf->frame_size, rfi_frame_size);
        std::abort();
    }
}

std::shared_ptr<chordMetadata> testN2kGen::get_new_metadata(Buffer* buf, frameID frame_id) {
    buf->allocate_new_metadata_object(frame_id);

    const std::shared_ptr<metadataObject> mc = buf->get_metadata(frame_id);
    if (!mc) {
        FATAL_ERROR("Buffer {:s} frame {:d} cannot allocate metadata", buf->buffer_name, frame_id);
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

void testN2kGen::allocate_correlation_frame_desc(Buffer* buf, frameID frame_id) {
    buf->allocate_new_frame_desc(
        frame_id, kotekan::int32, "n2k_correlation",
        {num_integrations, num_local_freq, corr_num_blocks, corr_blocksize, corr_blocksize, 2},
        {"Tc", "F", "DPhi", "DPlo1", "DPlo2", "C"});
}

void testN2kGen::allocate_counts_frame_desc(Buffer* buf, frameID frame_id) {
    buf->allocate_new_frame_desc(
        frame_id, kotekan::int32, "n2k_counts",
        {num_integrations, num_local_freq, count_num_blocks, count_blocksize, count_blocksize},
        {"Tc", "F", "D8Phi", "D8Plo1", "D8Plo2"});
}

void testN2kGen::set_correlation_metadata(const std::shared_ptr<chordMetadata>& meta,
                                          uint64_t seq_num) {
    meta->set_name(corr_name);
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

    meta->set_fpga_seq_num(seq_num);
    meta->set_sample0_offset(seq_num / sub_integration_ntime);
    meta->set_offset_downsampling(1);

    std::vector<int> coarse_freq(num_local_freq);
    std::vector<int> freq_upchan_factor(num_local_freq);
    std::vector<int> time_downsampling_fpga(num_local_freq);
    std::vector<int64_t> half_fpga_sample0(num_local_freq);

    for (int f = 0; f < num_local_freq; f++) {
        coarse_freq[f] = freq_ids[f % freq_ids.size()];
        freq_upchan_factor[f] = 1;
        time_downsampling_fpga[f] = sub_integration_ntime;
        half_fpga_sample0[f] = sub_integration_ntime - 1;
    }

    meta->set_coarse_freq(coarse_freq);
    meta->set_freq_upchan_factor(freq_upchan_factor);
    meta->set_time_downsampling_fpga(time_downsampling_fpga);
    meta->set_half_fpga_sample0(half_fpga_sample0);
    assert(meta->get_nfreq() <= CHORD_META_MAX_FREQ);
}

void testN2kGen::set_counts_metadata(const std::shared_ptr<chordMetadata>& meta, uint64_t seq_num) {
    meta->set_name(count_name);
    meta->type = kotekan::int32;
    meta->dims = 5;
    assert(meta->dims <= CHORD_META_MAX_DIM);
    meta->set_array_dimension(0, num_integrations, "Tc");
    meta->set_array_dimension(1, num_local_freq, "F");
    meta->set_array_dimension(2, count_num_blocks, "D8Phi");
    meta->set_array_dimension(3, count_blocksize, "D8Plo1");
    meta->set_array_dimension(4, count_blocksize, "D8Plo2");
    meta->set_strides_simple();

    meta->set_fpga_seq_num(seq_num);
    meta->set_sample0_offset(seq_num / sub_integration_ntime);
    meta->set_offset_downsampling(1);

    std::vector<int> coarse_freq(num_local_freq);
    std::vector<int> freq_upchan_factor(num_local_freq);
    std::vector<int> time_downsampling_fpga(num_local_freq);
    std::vector<int64_t> half_fpga_sample0(num_local_freq);

    for (int f = 0; f < num_local_freq; f++) {
        coarse_freq[f] = freq_ids[f % freq_ids.size()];
        freq_upchan_factor[f] = 1;
        time_downsampling_fpga[f] = sub_integration_ntime;
        half_fpga_sample0[f] = sub_integration_ntime - 1;
    }

    meta->set_coarse_freq(coarse_freq);
    meta->set_freq_upchan_factor(freq_upchan_factor);
    meta->set_time_downsampling_fpga(time_downsampling_fpga);
    meta->set_half_fpga_sample0(half_fpga_sample0);
    assert(meta->get_nfreq() <= CHORD_META_MAX_FREQ);
}

void testN2kGen::get_blocked_indices(int i, int j, int blocksize, int& ihi, int& jhi, int& ilo,
                                     int& jlo, int& block_idx) {

    ihi = i / blocksize;
    jhi = j / blocksize;
    ilo = i % blocksize;
    jlo = j % blocksize;

    // Only reference lower triangular blocks (jhi <= ihi), so swap if jhi > ihi
    if (jhi > ihi) {
        std::swap(ihi, jhi);
        std::swap(ilo, jlo);
    }

    // Block idx:
    //       j
    //       0  1  2  3  4
    //     ---------------
    // i 0|  0
    //   1|  1  2
    //   2|  3  4  5
    //   3|  6  7  8  9
    //   4| 10 11 12 13 14
    //
    // block_idx is index of row start (j=0) + col_idx (j)
    block_idx = (ihi * (ihi + 1)) / 2 + jhi;
}

void testN2kGen::main_thread() {

    frameID corr_frame_id(corr_buf);
    frameID count_frame_id(count_buf);
    frameID rfi_frame_id(rfi_buf);
    int num_frames_generated = 0;
    uint64_t seq_num = 0;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> count_dist(count_min, count_max);
    std::uniform_int_distribution<int32_t> corr_real_dist(corr_min[0], corr_max[0]);
    std::uniform_int_distribution<int32_t> corr_imag_dist(corr_min[1], corr_max[1]);

    int corr_val_idx = 0;
    int count_val_idx = 0;

    while (!stop_thread) {

        // grab frames
        int32_t* corr = (int32_t*)corr_buf->wait_for_empty_frame(unique_name, corr_frame_id);
        if (corr == nullptr)
            break;
        int32_t* count = (int32_t*)count_buf->wait_for_empty_frame(unique_name, count_frame_id);
        if (count == nullptr)
            break;
        int32_t* rfi = (int32_t*)rfi_buf->wait_for_full_frame(unique_name, rfi_frame_id);
        if (rfi == nullptr)
            break;

        // create metadata
        const std::shared_ptr<chordMetadata> corr_meta = get_new_metadata(corr_buf, corr_frame_id);
        const std::shared_ptr<chordMetadata> count_meta =
            get_new_metadata(count_buf, count_frame_id);

        // fill metadata
        set_correlation_metadata(corr_meta, seq_num);
        set_counts_metadata(count_meta, seq_num);

        // allocate frame descriptors
        allocate_correlation_frame_desc(corr_buf, corr_frame_id);
        allocate_counts_frame_desc(count_buf, count_frame_id);

        // check frame descriptors match metadata
        corr_meta->check_frame_desc(corr_buf->get_frame_desc(corr_frame_id));
        count_meta->check_frame_desc(count_buf->get_frame_desc(count_frame_id));

        // block, freq, and time strides for access into the
        // correlation and counts buffers
        int db_corr = 2 * corr_blocksize * corr_blocksize;
        int df_corr = db_corr * corr_num_blocks;
        int dt_corr = df_corr * num_local_freq;
        int db_count = count_blocksize * count_blocksize;
        int df_count = db_count * count_num_blocks;
        int dt_count = df_count * num_local_freq;

        for (int t = 0; t < num_integrations; t++) {
            int t_glob = num_frames_generated * num_integrations + t;

            for (int f = 0; f < num_local_freq; f++) {

                // Fill the count array
                int count_block_idx = 0;

                int32_t count_scalar_val = -1;

                if (count_type == "random_scalar")
                    count_scalar_val = count_dist(rng);
                if (count_type == "const_scalar") {
                    if (count_value_array.size() > 0) {
                        count_scalar_val =
                            count_value_array[count_val_idx % count_value_array.size()];
                        count_val_idx++;
                    } else
                        count_scalar_val = count_value;
                }

                for (int ihi = 0; ihi < count_lin_blocks; ihi++) {
                    // Lower triangular only
                    for (int jhi = 0; jhi <= ihi; jhi++) {

                        assert(count_block_idx == (ihi * (ihi + 1)) / 2 + jhi);

                        for (int ilo = 0; ilo < count_blocksize; ilo++) {
                            for (int jlo = 0; jlo < count_blocksize; jlo++) {
                                int idx = jlo + ilo * count_blocksize + count_block_idx * db_count
                                          + f * df_count + t * dt_count;

                                if (count_type == "const") {
                                    if (count_value_array.size() > 0) {
                                        count[idx] = count_value_array[count_val_idx
                                                                       % count_value_array.size()];
                                        count_val_idx++;
                                    } else {
                                        count[idx] = count_value;
                                    }
                                } else if (count_type == "random") {
                                    count[idx] = count_dist(rng);
                                } else if (count_type == "const_scalar") {
                                    count[idx] = count_scalar_val;
                                } else if (count_type == "random_scalar") {
                                    count[idx] = count_scalar_val;
                                } else {
                                    count[idx] = 0;
                                }
                            } // count jlo
                        } // count ilo

                        count_block_idx++;
                    } // count jhi
                } // count ihi


                // Fill the correlation array
                int corr_block_idx = 0;

                for (int ihi = 0; ihi < corr_lin_blocks; ihi++) {
                    // Lower triangular only
                    for (int jhi = 0; jhi <= ihi; jhi++) {

                        assert(corr_block_idx == (ihi * (ihi + 1)) / 2 + jhi);

                        for (int ilo = 0; ilo < corr_blocksize; ilo++) {
                            for (int jlo = 0; jlo < corr_blocksize; jlo++) {
                                int idx = 2 * (jlo + ilo * corr_blocksize)
                                          + corr_block_idx * db_corr + f * df_corr + t * dt_corr;

                                if (corr_type == "const") {
                                    if (corr_value_array.size() > 0) {
                                        corr[idx + 0] =
                                            corr_value_array[corr_val_idx % corr_value_array.size()]
                                                            [0];
                                        corr[idx + 1] =
                                            corr_value_array[corr_val_idx % corr_value_array.size()]
                                                            [1];
                                        corr_val_idx++;
                                    } else {
                                        corr[idx + 0] = corr_value[0];
                                        corr[idx + 1] = corr_value[1];
                                    }
                                } else if (corr_type == "random") {
                                    corr[idx + 0] = corr_real_dist(rng);
                                    corr[idx + 1] = corr_imag_dist(rng);
                                } else if (corr_type == "f_times_ee") {
                                    int i = ilo + corr_blocksize * ihi;
                                    int j = jlo + corr_blocksize * jhi;
                                    corr[idx + 0] = f * i;
                                    corr[idx + 1] = f * j;
                                } else if (corr_type == "f_times_ee_plus_t") {
                                    int i = ilo + corr_blocksize * ihi;
                                    int j = jlo + corr_blocksize * jhi;
                                    corr[idx + 0] = f * i + t_glob;
                                    corr[idx + 1] = f * j + t_glob;
                                } else {
                                    corr[idx + 0] = 0; // Real
                                    corr[idx + 1] = 0; // Imag
                                }

                                if (mul_correlation_by_counts) {
                                    int cihi, cjhi, cilo, cjlo, cb_idx;
                                    int ci = (ihi * corr_blocksize + ilo) / 8;
                                    int cj = (jhi * corr_blocksize + jlo) / 8;
                                    get_blocked_indices(ci, cj, count_blocksize, cihi, cjhi, cilo,
                                                        cjlo, cb_idx);

                                    int count_idx = cjlo + cilo * count_blocksize
                                                    + cb_idx * db_count + f * df_count
                                                    + t * dt_count;

                                    corr[idx + 0] *= count[count_idx];
                                    corr[idx + 1] *= count[count_idx];
                                }
                            } // corr jlo
                        } // corr ilo

                        corr_block_idx++;
                    } // corr jhi
                } // corr ihi
            } // f
        } // t

        DEBUG("Generated a {:s} test correlation data set in {:s}[{:d}]", corr_type,
              corr_buf->buffer_name, corr_frame_id);
        DEBUG("Generated a {:s} test counts data set in {:s}[{:d}]", count_type,
              count_buf->buffer_name, count_frame_id);
        DEBUG("Corr sample size is: {:d}", corr_meta->sample_bytes());
        DEBUG("Counts sample size is: {:d}", count_meta->sample_bytes());

        corr_buf->mark_frame_full(unique_name, corr_frame_id++);
        count_buf->mark_frame_full(unique_name, count_frame_id++);
        rfi_buf->mark_frame_empty(unique_name, rfi_frame_id++);

        num_frames_generated++;
        seq_num += samples_per_data_set;

        if (num_frames >= 0 && num_frames_generated >= num_frames) {
            INFO("Generated the requested number of frames ({:d}) - exiting", num_frames);
            break;
        }
    }
}
