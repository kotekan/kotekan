#include "Config.hpp"          // for Config
#include "DataType.hpp"        // for DataType
#include "NDArray.hpp"         // for GenericNDArray
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
#ifdef WITH_OMP
#include <omp.h> // for omp_get_wtime
#endif
#include <random>   // for uniform_int_distribution, mt19937
#include <stdint.h> // for int32_t, uint32_t, uint64_t, int64_t
#include <utility>  // for swap
#include <vector>   // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

/**
 * @class testRFImaskGen
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
 * @conf  first_frame_index     Int. Default 0. Starting FPGA frame number, for
 *                              frames of size samples_per_data_set.
 * @conf  samples_per_data_set  Int. How often to produce data.
 * @conf  num_frames            Int. How many frames to produce. Default inf.
 * @conf  num_freq_in_frame     Int. Number of frequencies in each GPU frame.
 *
 * @author Geoffrey Ryan
 */
class testRFImaskGen : public Stage {
public:
    testRFImaskGen(Config& config, const std::string& unique_name,
                   bufferContainer& buffer_container);
    ~testRFImaskGen() {};
    void main_thread() override;

private:
    Buffer* out_buf;
    const std::string name;
    const std::string type;
    const uint8_t value;
    const std::vector<uint8_t> value_array;
    const int64_t samples_per_data_set;
    const int64_t rfi_downsampling_factor;
    const int64_t num_frames;
    const int64_t num_local_freq;
    const std::vector<uint32_t> freq_ids;
    const uint64_t seed;
    const int64_t first_frame_index;
    const int64_t repeat_count;
    const int64_t num_entries;

    std::shared_ptr<chordMetadata> get_new_metadata(Buffer* buf, frameID frame_id);
    void set_metadata(const std::shared_ptr<chordMetadata>& meta, uint64_t seq_num);
};

REGISTER_KOTEKAN_STAGE(testRFImaskGen);

testRFImaskGen::testRFImaskGen(Config& config, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&testRFImaskGen::main_thread, this)),
    name(config.get_default<std::string>(unique_name, "name", "RFImask")),
    type(config.get_default<std::string>(unique_name, "type", "const")),
    value(config.get_default<uint8_t>(unique_name, "value", 0)),
    value_array(config.get_default<std::vector<uint8_t>>(unique_name, "value_array", {})),
    samples_per_data_set(config.get<int64_t>(unique_name, "samples_per_data_set")),
    rfi_downsampling_factor(config.get<int64_t>(unique_name, "rfi_downsampling_factor")),
    num_frames(config.get<int64_t>(unique_name, "num_frames")),
    num_local_freq(config.get<int64_t>(unique_name, "num_local_freq")),
    freq_ids(config.get_default<std::vector<uint32_t>>(unique_name, "freq_ids", {4096})),
    seed(config.get_default<uint64_t>(unique_name, "seed", 123245)),
    first_frame_index(config.get_default<int64_t>(unique_name, "first_frame_index", 0)),
    repeat_count(config.get_default<int64_t>(unique_name, "repeat_count", 0)),
    num_entries((samples_per_data_set * num_local_freq) / 8) {

    // Get buffers
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    assert(type == "const" || type == "random");

    if (num_local_freq <= 0) {
        FATAL_ERROR("num_local_freq ({:d}) is not positive.", num_local_freq);
    }
    if (samples_per_data_set <= 0) {
        FATAL_ERROR("samples_per_data_set ({:d}) is not positive.", samples_per_data_set);
    }
    if (samples_per_data_set % 1024 != 0) {
        FATAL_ERROR("samples_per_data_set ({:d}) is not a multiple of 1024.", samples_per_data_set);
    }
    if (freq_ids.empty()) {
        FATAL_ERROR("freq_ids must have at least one element.");
    }
    if (repeat_count > 0 && num_frames <= 0) {
        FATAL_ERROR("If repeat_count > 0, num_frames must also be > 0");
    }

    // allocate frame descriptors
    out_buf->require_frame_desc(kotekan::GenericNDArray::describe(
        kotekan::uint1x8, name, {samples_per_data_set / 1024, num_local_freq, 128},
        {"T8hi128", "F", "T8lo128"}, {1024, 1, 8}));

    if (out_buf->frame_size != out_buf->frames_desc->get_byte_size())
        FATAL_ERROR("out_but {:s} has frame size {:d}, expected {:d}.", out_buf->buffer_name,
                    out_buf->frame_size, out_buf->frames_desc->get_byte_size());
}

std::shared_ptr<chordMetadata> testRFImaskGen::get_new_metadata(Buffer* buf, frameID frame_id) {
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

void testRFImaskGen::set_metadata(const std::shared_ptr<chordMetadata>& meta, uint64_t seq_num) {
    meta->set_from_frame_desc(out_buf->get_frame_desc<kotekan::GenericNDArray>());

    meta->set_fpga_seq_num(seq_num);
    meta->set_time_downsampling_fpga(1024);

    std::vector<int> coarse_freq(num_local_freq);
    std::vector<int> freq_upchan_factor(num_local_freq);
    std::vector<int> freq_upchan_index(num_local_freq);

    for (int f = 0; f < num_local_freq; f++) {
        coarse_freq[f] = freq_ids[f % freq_ids.size()];
        freq_upchan_factor[f] = 1;
        freq_upchan_index[f] = 0;
    }

    meta->set_coarse_freq(coarse_freq);
    meta->set_freq_upchan_factor(freq_upchan_factor);
    meta->set_freq_upchan_index(freq_upchan_index);
    assert(meta->get_nfreq() <= CHORD_META_MAX_FREQ);
}


void testRFImaskGen::main_thread() {

    frameID frame_id(out_buf);
    int num_frames_generated = 0;
    int64_t seq_num = first_frame_index * samples_per_data_set;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint8_t> dist;

    int total_frames = num_frames;
    if (repeat_count > 0) {
        total_frames *= repeat_count;
    }

    int val_idx = num_entries * first_frame_index;

    // If repeating, buffers to store the constructed frames.
    std::vector<uint8_t> store;

    if (repeat_count > 0) {
        store.resize(num_entries * num_frames);
    }

#ifdef WITH_OMP
    [[maybe_unused]] double last_time = omp_get_wtime();
#endif

    while (!stop_thread) {

        // grab frames
        uint8_t* rfimask = (uint8_t*)out_buf->wait_for_empty_frame(unique_name, frame_id);
        if (rfimask == nullptr)
            break;

#ifdef WITH_OMP
        [[maybe_unused]] double start_time = omp_get_wtime();
#endif

        // create metadata
        const std::shared_ptr<chordMetadata> meta = get_new_metadata(out_buf, frame_id);

        // fill metadata
        set_metadata(meta, seq_num);

        // check frame descriptors match metadata
        meta->check_frame_desc(out_buf->get_frame_desc<kotekan::GenericNDArray>());

        // If we're not repeating, or we're in the first num_frames, generate data
        if (repeat_count <= 0 || (num_frames > 0 && num_frames_generated < num_frames)) {

            int64_t num_thi = samples_per_data_set / 1024;
            int64_t num_tlo = 1024 / 8; // Writing rfimask as u8's

            int64_t df = num_tlo;
            int64_t dthi = num_tlo * num_local_freq;

            for (int64_t thi = 0; thi < num_thi; thi++) {
                for (int f = 0; f < num_local_freq; f++) {
                    for (int64_t tlo = 0; tlo < num_tlo; tlo++) {
                        int64_t idx = tlo + f * df + thi * dthi;
                        if (type == "const") {
                            if (value_array.size() > 0) {
                                rfimask[idx] = value_array[val_idx % value_array.size()];
                                val_idx++;
                            } else {
                                rfimask[idx] = value;
                            }
                        } else if (type == "random") {
                            rfimask[idx] = dist(rng);
                        } else {
                            FATAL_ERROR("unknown generation type: {:s}", type);
                        }
                    } // tlo
                } // f
            } // thi

            // If we're repeating, copy the frames into storage before moving on.
            if (repeat_count > 0) {
                std::copy(rfimask, rfimask + num_entries,
                          store.begin() + num_frames_generated * num_entries);
            }

            DEBUG("Generated a {:s} test correlation data set in {:s}[{:d}] at seq {:d}", type,
                  out_buf->buffer_name, frame_id, seq_num);
        }

#ifdef WITH_OMP
        [[maybe_unused]] double curr_time = omp_get_wtime();
        DEBUG("Frame generation took {:f} ms + {:f} ms idle", (curr_time - start_time) * 1000,
              (start_time - last_time) * 1000);
        last_time = curr_time;
#endif

        out_buf->mark_frame_full(unique_name, frame_id++);

        num_frames_generated++;
        seq_num += samples_per_data_set;

        if (num_frames >= 0 && num_frames_generated >= total_frames) {
            INFO("Generated the requested number of frames ({:d}) - exiting", total_frames);
            break;
        }
    }
}
