#include "Config.hpp"          // for Config
#include "DataType.hpp"        // for DataType
#include "N2Util.hpp"          // for frameID
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata, metadata_is_chord, CHORD_META_MAX_DIM, CHO...
#include "kotekanLogging.hpp"  // for FATAL_ERROR, DEBUG, INFO
#include "metadata.hpp"        // for metadataObject

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
using N2::frameID;

/**
 * @class testRFIFrameMaskGen
 * @brief Generate test RFI Frame Mask as a standin for RfiFrameMask.
 *
 * @par Buffers
 * @buffer out_buf Buffer to fill with 0's and 1's
 *         @buffer_format uint8
 *         @buffer_shape [samples_per_data_set/sub_integration_ntime,
 *              num_local_freq]
 *         @buffer_metadata chordMetadata
 *
 * @conf  name                  String. quantity name for RFI Frame Mask
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
 * @conf  freq_ids              Vector of Ints. freq_ids to insert into metadata.
 * @conf  repeat_count          Int. If positive, the 'num_frames' frames will be repeated this many
 * times, producing num_frames * repeat_count total frames.
 *
 * @author Geoffrey Ryan
 */
class testRFIFrameMaskGen : public Stage {
public:
    testRFIFrameMaskGen(Config& config, const std::string& unique_name,
                        bufferContainer& buffer_container);
    ~testRFIFrameMaskGen(){};
    void main_thread() override;

private:
    Buffer* out_buf;
    const std::string name;
    const std::string type;
    const uint8_t value;
    const std::vector<uint8_t> value_array;
    const int64_t samples_per_data_set;
    const int64_t sub_integration_ntime;
    const int64_t num_frames;
    const int64_t num_local_freq;
    const std::vector<uint32_t> freq_ids;
    const bool enabled;
    const std::vector<float> threshold;
    const std::vector<float> fraction;
    const uint64_t seed;
    const int64_t first_frame_index;
    const int64_t repeat_count;
    const int64_t num_integrations;
    const int64_t num_entries;

    std::shared_ptr<chordMetadata> get_new_metadata(Buffer* buf, frameID frame_id);
    void set_metadata(const std::shared_ptr<chordMetadata>& meta, uint64_t seq_num);
};

REGISTER_KOTEKAN_STAGE(testRFIFrameMaskGen);

testRFIFrameMaskGen::testRFIFrameMaskGen(Config& config, const std::string& unique_name,
                                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&testRFIFrameMaskGen::main_thread, this)),
    name(config.get_default<std::string>(unique_name, "name", "RFIFrameMask")),
    type(config.get_default<std::string>(unique_name, "type", "const")),
    value(config.get_default<uint8_t>(unique_name, "value", 0)),
    value_array(config.get_default<std::vector<uint8_t>>(unique_name, "value_array", {})),
    samples_per_data_set(config.get<int64_t>(unique_name, "samples_per_data_set")),
    sub_integration_ntime(config.get<int64_t>(unique_name, "sub_integration_ntime")),
    num_frames(config.get<int64_t>(unique_name, "num_frames")),
    num_local_freq(config.get<int64_t>(unique_name, "num_local_freq")),
    freq_ids(config.get_default<std::vector<uint32_t>>(unique_name, "freq_ids", {4096})),
    enabled(config.get_default<bool>(unique_name, "enabled", true)),
    threshold(config.get_default<std::vector<float>>(unique_name, "threshold", {2.0f})),
    fraction(config.get_default<std::vector<float>>(unique_name, "fraction", {0.8f})),
    seed(config.get_default<uint64_t>(unique_name, "seed", 123245)),
    first_frame_index(config.get_default<int64_t>(unique_name, "first_frame_index", 0)),
    repeat_count(config.get_default<int64_t>(unique_name, "repeat_count", 0)),
    num_integrations(samples_per_data_set / sub_integration_ntime),
    num_entries(num_integrations * num_local_freq) {

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
    out_buf->allocate_ndarray_frame_desc(kotekan::uint8, name, {num_integrations, num_local_freq},
                                         {"Tc", "F"});
}

std::shared_ptr<chordMetadata> testRFIFrameMaskGen::get_new_metadata(Buffer* buf,
                                                                     frameID frame_id) {
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

void testRFIFrameMaskGen::set_metadata(const std::shared_ptr<chordMetadata>& meta,
                                       uint64_t seq_num) {
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

    std::vector<std::array<float, 2>> thresholds;

    for (size_t i = 0; i < std::max(threshold.size(), fraction.size()); i++) {
        thresholds.push_back({threshold.at(i % threshold.size()), fraction.at(i % fraction.size())});
    }

    meta->set_rfi_frame_excision_enabled(enabled);
    meta->set_rfi_frame_excision_thresholds(thresholds);

    assert(meta->get_nfreq() <= CHORD_META_MAX_FREQ);
    }


    void testRFIFrameMaskGen::main_thread() {

        frameID frame_id(out_buf);
        int num_frames_generated = 0;
        int64_t seq_num = first_frame_index * samples_per_data_set;

        std::mt19937 rng(seed);
        std::uniform_int_distribution<uint8_t> dist(0, 1);

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
            uint8_t* framemask = (uint8_t*)out_buf->wait_for_empty_frame(unique_name, frame_id);
            if (framemask == nullptr)
                break;

#ifdef WITH_OMP
            [[maybe_unused]] double start_time = omp_get_wtime();
#endif

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
                                framemask[idx] = value_array[val_idx % value_array.size()];
                                val_idx++;
                            } else {
                                framemask[idx] = value;
                            }
                        } else if (type == "random") {
                            framemask[idx] = dist(rng);
                        } else {
                            FATAL_ERROR("unknown generation type: {:s}", type);
                        }
                    } // f
                } // tc

                // If we're repeating, copy the frames into storage before moving on.
                if (repeat_count > 0) {
                    std::copy(framemask, framemask + num_entries,
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
