#include "Config.hpp"          // for Config
#include "DataType.hpp"        // for DataType
#include "NDArray.hpp"         // for NDArray, GenericNDArray, Config
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


/**
 * @class testRFIS012Gen
 * @brief Generate test RFI S012 data as a standin for the cudaRFIS012*.
 *
 * Given a packet loss (PL) mask and voltage (E, complex) time series, S0/1/2 are defined as:
 *
 * S0 = \sum_{t \in t_rfi} PL[t]
 * S1 = \sum_{t \in t_rfi} PL[t] x |E[t]|^2
 * S2 = \sum_{t \in t_rfi} PL[t] x |E[t]|^4
 *
 * `t_rfi` is a downsampled time variable, covering `downsampling_factor` fast time samples `t`.
 *
 * Can emulate both cudaRFIS012 and cudaRFIS012bar outputs.
 *
 * @par Buffers
 * @buffer out_buf Buffer to fill with S012 values
 *         @buffer_format uint64
 *         @buffer_shape [samples_per_data_set/downsampling_factor, num_local_freq, 3,
 *              num_polarizations, num_dishes]
 *         @buffer_metadata chordMetadata time_downsample_fpga[] = downsampling_factor
 *
 * @conf  type                  String. "const", "random".
 * @conf  samples_per_data_set  Int. Number of fast time samples per buffer.
 * @conf  num_local_freq        Int. Number of frequencies.
 * @conf  num_polarizations     Int. Number of polarizations.
 * @conf  num_dishes            Int. Number of dishes.
 * @conf  num_frames            Int. Number of frames to produce.
 * @conf  downsampling_factor   Int. Number of fast times `t` in a `t_rfi`. Must divide
 *                              `samples_per_data_set`. Output buffer will contain
 *                              `samples_per_data_set` / `downsampling_factor` time samples.
 * @conf  bar_mode              Bool. Whether to output an S012bar buffer or S012 buffer.
 * @conf  S0_value              Int. Used when `type` is "const". Fixed value for S0.
 * @conf  S1_value              Int. Used when `type` is "const". Fixed value for S1.
 * @conf  S2_value              Int. Used when `type` is "const". Fixed value for S2.
 * @conf  S0_values             Array<Int>. Used when `type` is "const". Set of S0 values to
 *                              cycle through. If present, overrides single S0_value.
 * @conf  S1_values             Array<Int>. Used when `type` is "const". Set of S1 values to
 *                              cycle through. If present, overrides single S1_value.
 * @conf  S2_values             Array<Int>. Used when `type` is "const". Set of S2 values to
 *                              cycle through. If present, overrides single S2_value.
 * @conf  freq_ids              Array<Int>. frequency IDs to put in metadata coarse_freq field.
 * @conf  seed                  Int. Default 0. Seeds the deterministic RNG for "random" type
 *                              and counts variants.
 * @conf  first_frame_index     Int. Default 0. Starting frame number
 *
 * @author Geoffrey Ryan
 */
class testRFIS012Gen : public kotekan::Stage {
public:
    testRFIS012Gen(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);
    ~testRFIS012Gen() {};
    void main_thread() override;

private:
    const std::string type;
    const int64_t samples_per_data_set;
    const int64_t num_local_freq;
    const int64_t num_polarizations;
    const int64_t num_dishes;
    const int64_t num_frames;
    const int64_t downsampling_factor;
    /// number of t_rfi samples. = samples_per_data_set / downsampling_factor
    const int64_t num_rfi_samples;
    /// = num_dishes * num_polarizations
    const int64_t num_elements;
    const bool bar_mode;
    const uint64_t S0_value;
    const uint64_t S1_value;
    const uint64_t S2_value;
    const std::vector<uint64_t> S0_value_array;
    const std::vector<uint64_t> S1_value_array;
    const std::vector<uint64_t> S2_value_array;
    const std::vector<uint32_t> freq_ids;
    const uint64_t seed;
    const uint64_t first_frame_index;

    Buffer* out_buf;

    /// Acquires a new metadata object on out_buf for given frame_id.
    std::shared_ptr<chordMetadata> get_new_metadata(frameID frame_id);

    /// Sets the S012/S012bar metadata at the given FPGA sequence number.
    void set_metadata(const std::shared_ptr<chordMetadata>& meta, uint64_t seq_num);
};

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(testRFIS012Gen);

testRFIS012Gen::testRFIS012Gen(Config& config, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&testRFIS012Gen::main_thread, this)),
    type(config.get<std::string>(unique_name, "type")),
    samples_per_data_set(config.get<int64_t>(unique_name, "samples_per_data_set")),
    num_local_freq(config.get<int64_t>(unique_name, "num_local_freq")),
    num_polarizations(config.get<int64_t>(unique_name, "num_polarizations")),
    num_dishes(config.get<int64_t>(unique_name, "num_dishes")),
    num_frames(config.get<int64_t>(unique_name, "num_frames")),
    downsampling_factor(config.get<int64_t>(unique_name, "downsampling_factor")),
    num_rfi_samples(samples_per_data_set / downsampling_factor),
    num_elements(num_dishes * num_polarizations),
    bar_mode(config.get<bool>(unique_name, "bar_mode")),
    S0_value(config.get_default<uint64_t>(unique_name, "S0_value", downsampling_factor)),
    S1_value(config.get_default<uint64_t>(unique_name, "S1_value", downsampling_factor)),
    S2_value(config.get_default<uint64_t>(unique_name, "S2_value", downsampling_factor)),
    S0_value_array(config.get_default<std::vector<uint64_t>>(unique_name, "S0_values",
                                                             std::vector<uint64_t>())),
    S1_value_array(config.get_default<std::vector<uint64_t>>(unique_name, "S1_values",
                                                             std::vector<uint64_t>())),
    S2_value_array(config.get_default<std::vector<uint64_t>>(unique_name, "S2_values",
                                                             std::vector<uint64_t>())),
    freq_ids(config.get_default<std::vector<uint32_t>>(unique_name, "freq_ids",
                                                       std::vector<uint32_t>({4096}))),
    seed(config.get_default<uint64_t>(unique_name, "seed", 12345)),
    first_frame_index(config.get_default<uint64_t>(unique_name, "first_frame_index", 0)) {

    assert(type == "const" || type == "random");

    // Get buffers
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    // Check parameter compatibility
    if (num_elements <= 0) {
        FATAL_ERROR("num_elements ({:d}) is not positive.", num_elements);
    }
    if (num_local_freq <= 0) {
        FATAL_ERROR("num_local_freq ({:d}) is not positive.", num_local_freq);
    }
    if (samples_per_data_set <= 0) {
        FATAL_ERROR("samples_per_data_set ({:d}) is not positive.", samples_per_data_set);
    }
    if (samples_per_data_set % downsampling_factor != 0) {
        FATAL_ERROR("samples_per_data_set ({:d}) is not a multiple of downsampling_factor ({:d}).",
                    samples_per_data_set, downsampling_factor);
    }
    if (freq_ids.empty()) {
        FATAL_ERROR("freq_ids must have at least one element.");
    }

    // allocate frame descriptor
    if (bar_mode) {
        out_buf->require_frame_desc(kotekan::NDArray<uint64_t, 5>::describe(
            "S012bar", {num_rfi_samples, num_local_freq, 3, num_polarizations, num_dishes},
            {"Trfibar", "F", "S", "P", "D"}, {downsampling_factor, 1, 1, 1, 1}));
    } else {
        out_buf->require_frame_desc(kotekan::NDArray<uint64_t, 5>::describe(
            "S012", {num_rfi_samples, num_local_freq, 3, num_polarizations, num_dishes},
            {"Trfi", "F", "S", "P", "D"}, {downsampling_factor, 1, 1, 1, 1}));
    }
}

std::shared_ptr<chordMetadata> testRFIS012Gen::get_new_metadata(frameID frame_id) {
    out_buf->allocate_new_metadata_object(frame_id);

    const std::shared_ptr<metadataObject> mc = out_buf->get_metadata(frame_id);
    if (!mc) {
        FATAL_ERROR("Buffer {:s} frame {:d} cannot allocate metadata", out_buf->buffer_name,
                    frame_id);
    }
    assert(mc);
    if (!metadata_is_chord(mc)) {
        FATAL_ERROR("Buffer {:s} frame {:d} does not have CHORD metadata", out_buf->buffer_name,
                    frame_id);
    }
    assert(metadata_is_chord(mc));
    const std::shared_ptr<chordMetadata> meta = get_chord_metadata(mc);
    assert(meta);

    return meta;
}

void testRFIS012Gen::set_metadata(const std::shared_ptr<chordMetadata>& meta, uint64_t seq_num) {
    meta->set_from_frame_desc(out_buf->get_frame_desc<kotekan::GenericNDArray>());

    meta->set_fpga_seq_num(seq_num);
    meta->set_time_downsampling_fpga(downsampling_factor);

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

void testRFIS012Gen::main_thread() {

    frameID frame_id(out_buf);
    int num_frames_generated = 0;
    uint64_t seq_num = first_frame_index * samples_per_data_set;

    std::mt19937 rng(seed);

    uint64_t s0_val_idx = 0;
    uint64_t s1_val_idx = 0;
    uint64_t s2_val_idx = 0;

    while (!stop_thread) {

        // grab frame
        uint64_t* s012 = (uint64_t*)out_buf->wait_for_empty_frame(unique_name, frame_id);
        if (s012 == nullptr)
            break;

        // create metadata
        const std::shared_ptr<chordMetadata> meta = get_new_metadata(frame_id);

        // fill metadata
        set_metadata(meta, seq_num);

        // check frame descriptors match metadata
        meta->check_frame_desc(out_buf->get_frame_desc<kotekan::GenericNDArray>());

        // Access Strides into s012
        int64_t ds = num_elements;
        int64_t df = 3 * num_elements;
        int64_t dt = num_local_freq * 3 * num_elements;

        // Maximum voltage (E-field) value in 4 + 4 bit quantizing.
        uint64_t max_E = 7;
        // Maximum possible value of |E|^2  (E is complex)
        uint64_t max_E2 = 2 * max_E * max_E;
        // Maximum possible value of |E|^4
        uint64_t max_E4 = max_E2 * max_E2;

        for (int64_t t = 0; t < num_rfi_samples; t++) {
            for (int64_t f = 0; f < num_local_freq; f++) {
                int64_t idx = t * dt + f * df;
                for (int64_t e = 0; e < num_elements; e++) {
                    uint64_t s0 = 0;
                    uint64_t s1 = 0;
                    uint64_t s2 = 0;

                    if (type == "const") {
                        if (S0_value_array.size() > 0) {
                            s0 = S0_value_array[s0_val_idx % S0_value_array.size()];
                            s0_val_idx++;
                        } else
                            s0 = S0_value;

                        if (S1_value_array.size() > 0) {
                            s1 = S1_value_array[s1_val_idx % S1_value_array.size()];
                            s1_val_idx++;
                        } else
                            s1 = S1_value;

                        if (S2_value_array.size() > 0) {
                            s2 = S2_value_array[s2_val_idx % S2_value_array.size()];
                            s2_val_idx++;
                        } else
                            s2 = S2_value;
                    } else if (type == "random") {
                        // If generating random data, try to generate mathematically possible
                        // data.

                        // s0 is counts, can only be as many as seq_nums
                        s0 = rng() % downsampling_factor;
                        if (s0 > 0) {
                            // Restrict s1 and s2 to be mathematically possible values
                            uint64_t min_s1 = 0;
                            uint64_t max_s1 = s0 * max_E2;
                            s1 = min_s1 + rng() % (max_s1 - min_s1 + 1);

                            // s2 in [min_s2, max_s2]
                            uint64_t max_s2 = s0 * max_E4;
                            uint64_t min_s2 = s1 * s1 / s0; // triangle inequality
                            if (max_s2 < min_s2)
                                FATAL_ERROR(
                                    "For s0 = {:d}, s1 = {:d}, min_s2 = {:d} > max_s2 = {:d}", s0,
                                    s1, min_s2, max_s2);

                            s2 = min_s2 + rng() % (max_s2 - min_s2 + 1);
                        } else {
                            // s0 is 0, so the others must be too.
                            s1 = 0;
                            s2 = 0;
                        }
                    } else {
                        FATAL_ERROR("Unknown 'type': {:s}", type);
                    }

                    s012[idx + 0 * ds + e] = s0;
                    s012[idx + 1 * ds + e] = s1;
                    s012[idx + 2 * ds + e] = s2;

                    if (e == 0 && f == 0)
                        DEBUG("Wrote S012[{:d}, {:d}, :, {:d}] = ({:d}, {:d}, {:d})", t, f, e, s0,
                              s1, s2);
                } // e
            } // f
        } // t

        DEBUG("Generated a {:s} test RFI S012 data set in {:s}[{:d}]", type, out_buf->buffer_name,
              frame_id);

        out_buf->mark_frame_full(unique_name, frame_id++);

        num_frames_generated++;
        seq_num += samples_per_data_set;

        if (num_frames >= 0 && num_frames_generated >= num_frames) {
            INFO("Generated the requested number of frames ({:d}) - exiting", num_frames);
            break;
        }
    }
}
