#include "Config.hpp"          // for Config
#include "DataType.hpp"        // for DataType
#include "N2Util.hpp"          // for frameID
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata, metadata_is_chord, CHORD_META_MAX_DIM, CHO...
#include "div.hpp"             // for div_ceil, num_triangle_blocks
#include "kotekanLogging.hpp"  // for FATAL_ERROR, DEBUG, INFO
#include "metadata.hpp"        // for metadataObject

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
using N2::frameID;

/**
 * @class testLostCountsGen
 * @brief Generate test lost counts buffer as a standin for CountLostPLSamplesScalar or RfiMaskSum.
 *
 * @par Buffers
 * @buffer in_buf Buffer of n2k counts, OPTIONAL: controlled by @conf use_n2k_counts
 *         @buffer_format int32
 *         @buffer_shape n2k_counts shape, see cudaPL1bitCorrelator.
 *         @buffer_metadata chordMetadata
 * @buffer out_buf Buffer to fill with counts
 *         @buffer_format int32
 *         @buffer_shape [samples_per_data_set/sub_integration_ntime,
 *              num_local_freq]
 *         @buffer_metadata chordMetadata
 *
 * @conf  name                  String. quantity name.
 * @conf  type                  String. "const", "random".
 * @conf  value                 Int. Used when `type` is "const".
 * @conf  values                Vector of ints. Optional cycle for "const".
 * @conf  seed                  Int. Default 123245. Seeds the deterministic RNG for "random".
 * @conf  first_frame_index     Int. Default 0. Starting FPGA frame number, for
 *                              frames of size samples_per_data_set.
 * @conf  samples_per_data_set  Int. FPGA samples encompassed by one frame.
 * @conf  sub_integration_ntime Int. FPGA samples per integration. Multiple integrations may be in a
 * frame.
 * @conf  num_frames            Int. How many frames to produce. Default inf (-1)
 * @conf  num_local_freq        Int. Number of frequencies in each GPU frame.
 * @conf  num_elements          Int. Number of elements (dish-polarization pairs).
 * @conf  freq_ids              Vector of Ints. freq_ids to insert into metadata.
 * @conf  repeat_count          Int. If positive, the 'num_frames' frames will be repeated this many
 * times, producing num_frames * repeat_count total frames.
 * @conf  use_n2k_counts        Bool. If true, read n2k_counts off @buffer in_buf and ensure output
 * PL Lost counts are consistent.  Default, false.
 *
 * @author Geoffrey Ryan
 */
class testLostCountsGen : public Stage {
public:
    testLostCountsGen(Config& config, const std::string& unique_name,
                      bufferContainer& buffer_container);
    ~testLostCountsGen(){};
    void main_thread() override;

private:
    Buffer* in_buf;
    Buffer* out_buf;
    const std::string name;
    const std::string type;
    const int32_t value;
    const std::vector<int32_t> value_array;
    const int64_t samples_per_data_set;
    const int64_t sub_integration_ntime;
    const int64_t num_frames;
    const int64_t num_local_freq;
    const bool use_n2k_counts;
    const int64_t num_elements;
    const std::vector<uint32_t> freq_ids;
    const uint64_t seed;
    const int64_t first_frame_index;
    const int64_t repeat_count;
    const int64_t num_integrations;
    const int64_t num_entries;
    const int64_t n2k_counts_lin_blocks;
    const int64_t n2k_counts_num_blocks;
    const int64_t n2k_counts_num_prod;

    static constexpr int64_t n2k_counts_blocksize = 8; // THIS IS ALWAYS 8

    std::shared_ptr<chordMetadata> get_new_metadata(Buffer* buf, frameID frame_id);
    void set_metadata(const std::shared_ptr<chordMetadata>& meta, uint64_t seq_num);
};

REGISTER_KOTEKAN_STAGE(testLostCountsGen);

testLostCountsGen::testLostCountsGen(Config& config, const std::string& unique_name,
                                     bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&testLostCountsGen::main_thread, this)),
    name(config.get<std::string>(unique_name, "name")),
    type(config.get_default<std::string>(unique_name, "type", "const")),
    value(config.get_default<int32_t>(unique_name, "value", 0)),
    value_array(config.get_default<std::vector<int32_t>>(unique_name, "value_array", {})),
    samples_per_data_set(config.get<int64_t>(unique_name, "samples_per_data_set")),
    sub_integration_ntime(config.get<int64_t>(unique_name, "sub_integration_ntime")),
    num_frames(config.get<int64_t>(unique_name, "num_frames")),
    num_local_freq(config.get<int64_t>(unique_name, "num_local_freq")),
    use_n2k_counts(config.get_default<bool>(unique_name, "use_n2k_counts", false)),
    num_elements(config.get_default<int64_t>(unique_name, "num_elements", 64)),
    freq_ids(config.get_default<std::vector<uint32_t>>(unique_name, "freq_ids", {4096})),
    seed(config.get_default<uint64_t>(unique_name, "seed", 123245)),
    first_frame_index(config.get_default<int64_t>(unique_name, "first_frame_index", 0)),
    repeat_count(config.get_default<int64_t>(unique_name, "repeat_count", 0)),
    num_integrations(samples_per_data_set / sub_integration_ntime),
    num_entries(num_integrations * num_local_freq),
    n2k_counts_lin_blocks(kotekan::div_ceil(num_elements / 8, n2k_counts_blocksize)),
    n2k_counts_num_blocks(kotekan::num_triangle_blocks(num_elements / 8, n2k_counts_blocksize)),
    n2k_counts_num_prod(n2k_counts_num_blocks * n2k_counts_blocksize * n2k_counts_blocksize) {

    // Get buffers
    if (use_n2k_counts) {
        in_buf = get_buffer("in_buf");
        in_buf->register_consumer(unique_name);
    }
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    assert(type == "const" || type == "random");

    if (num_local_freq <= 0) {
        FATAL_ERROR("num_local_freq ({:d}) is not positive.", num_local_freq);
    }
    if (samples_per_data_set <= 0) {
        FATAL_ERROR("samples_per_data_set ({:d}) is not positive.", samples_per_data_set);
    }
    if (sub_integration_ntime <= 0) {
        FATAL_ERROR("sub_integration_ntime ({:d}) is not positive.", sub_integration_ntime);
    }
    if (samples_per_data_set % sub_integration_ntime != 0) {
        FATAL_ERROR(
            "samples_per_data_set ({:d}) is not a multiple of sub_integration_ntime ({:d}).",
            samples_per_data_set, sub_integration_ntime);
    }
    if (freq_ids.empty()) {
        FATAL_ERROR("freq_ids must have at least one element.");
    }
    if (repeat_count > 0 && num_frames <= 0) {
        FATAL_ERROR("If repeat_count > 0, num_frames must also be > 0");
    }

    // allocate frame descriptors
    if (use_n2k_counts) {
        if (num_elements % (8 * n2k_counts_blocksize) != 0)
            FATAL_ERROR("num_elements {:d} is not a multiple of {:d}", num_elements,
                        8 * n2k_counts_blocksize);

        in_buf->allocate_ndarray_frame_desc<int32_t, 5>(
            "n2k_counts",
            {num_integrations, num_local_freq, n2k_counts_num_blocks, n2k_counts_blocksize,
             n2k_counts_blocksize},
            {"Tc", "F", "D8Phi", "D8Plo1", "D8Plo2"});
    }
    out_buf->allocate_ndarray_frame_desc<int32_t, 2>(name, {num_integrations, num_local_freq},
                                                     {"Tc", "F"});
}

std::shared_ptr<chordMetadata> testLostCountsGen::get_new_metadata(Buffer* buf, frameID frame_id) {
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

void testLostCountsGen::set_metadata(const std::shared_ptr<chordMetadata>& meta, uint64_t seq_num) {
    meta->set_from_frame_desc(out_buf->get_ndarray_frame_desc());

    meta->set_fpga_seq_num(seq_num);
    meta->set_time_downsampling_fpga(sub_integration_ntime);

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


void testLostCountsGen::main_thread() {

    int in_frame_id = 0;
    frameID frame_id(out_buf);
    int num_frames_generated = 0;
    int64_t seq_num = first_frame_index * samples_per_data_set;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> dist(0, sub_integration_ntime);

    int total_frames = num_frames;
    if (repeat_count > 0) {
        total_frames *= repeat_count;
    }

    int val_idx = num_entries * first_frame_index;

    // If repeating, buffers to store the constructed frames.
    std::vector<int32_t> store;

    if (repeat_count > 0) {
        store.resize(num_entries * num_frames);
    }

    while (!stop_thread) {

        // grab frames
        int32_t* counts = (int32_t*)out_buf->wait_for_empty_frame(unique_name, frame_id);
        if (counts == nullptr)
            break;

        int32_t* n2k_counts = nullptr;
        if (use_n2k_counts) {
            n2k_counts = (int32_t*)in_buf->wait_for_full_frame(unique_name, in_frame_id);
            if (n2k_counts == nullptr)
                break;
        }

        // create metadata
        const std::shared_ptr<chordMetadata> meta = get_new_metadata(out_buf, frame_id);

        // fill metadata
        set_metadata(meta, seq_num);

        // check frame descriptors match metadata
        meta->check_frame_desc(out_buf->get_ndarray_frame_desc());

        // If we're not repeating, or we're in the first num_frames, generate data
        if (repeat_count <= 0 || (num_frames > 0 && num_frames_generated < num_frames)) {

            for (int64_t tc = 0; tc < num_integrations; tc++) {
                for (int f = 0; f < num_local_freq; f++) {

                    int64_t idx = f + tc * num_local_freq;
                    if (type == "const") {
                        if (value_array.size() > 0) {
                            counts[idx] = value_array[val_idx % value_array.size()];
                            val_idx++;
                        } else {
                            counts[idx] = value;
                        }
                    } else if (type == "random") {
                        counts[idx] = dist(rng);
                    } else {
                        FATAL_ERROR("unknown generation type: {:s}", type);
                    }

                    if (use_n2k_counts) {
                        int32_t n2k_count = n2k_counts[idx * n2k_counts_num_prod];

                        int32_t max_lost_count = sub_integration_ntime - n2k_count;

                        if (counts[idx] > max_lost_count)
                            counts[idx] = max_lost_count;
                    }
                } // f
            } // tc

            // If we're repeating, copy the frames into storage before moving on.
            if (repeat_count > 0) {
                std::copy(counts, counts + num_entries,
                          store.begin() + num_frames_generated * num_entries);
            }

            DEBUG("Generated a {:s} test PL count data set in {:s}[{:d}] at seq {:d}", type,
                  out_buf->buffer_name, frame_id, seq_num);
        }

        out_buf->mark_frame_full(unique_name, frame_id++);

        if (use_n2k_counts) {
            in_buf->mark_frame_empty(unique_name, in_frame_id);
            in_frame_id = (in_frame_id + 1) % in_buf->num_frames;
        }

        num_frames_generated++;
        seq_num += samples_per_data_set;

        if (num_frames >= 0 && num_frames_generated >= total_frames) {
            INFO("Generated the requested number of frames ({:d}) - exiting", total_frames);
            break;
        }
    }
}
