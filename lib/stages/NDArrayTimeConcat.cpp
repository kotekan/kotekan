#include "Config.hpp"
#include "NDArray.hpp"
#include "Stage.hpp"
#include "StageFactory.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "chordMetadata.hpp"
#include "kotekanLogging.hpp"
#include "visUtil.hpp"

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

/**
 * @class NDArrayTimeConcat
 * @brief Concatenate single-sample NDArray frames along time into longer frames.
 *
 * Each input frame holds one time sample: dimension 0 is a time axis of length 1 whose
 * scaling is the frame's time_downsampling_fpga. An output frame holds N samples, where N
 * is the ratio of the output and input frame sizes, and covers a fixed window of N sample
 * periods. Windows are aligned to multiples of N periods in fpga_seq_num, offset by the
 * phase of the first input frame, so streams with the same timing share window boundaries.
 * A frame is sent when its last row is filled, or when an input frame falls in a later
 * window. Rows with no input frame keep the byte @c fill_value.
 *
 * The output metadata is copied from the first input frame of the window, with fpga_seq_num
 * set to the window start and the array description of out_buf. Input frames must arrive in
 * time order with a fixed time_downsampling_fpga and coarse_freq.
 *
 * @par Buffers
 * @buffer in_buf   Single-sample frames.
 *     @buffer_format   NDArray [1, ...]
 *     @buffer_metadata chordMetadata
 * @buffer out_buf  Concatenated frames.
 *     @buffer_format   NDArray [N, ...], otherwise the same as in_buf.
 *     @buffer_metadata chordMetadata
 *
 * @conf fill_value  Int. Byte written to rows with no input frame. Default 255, which reads
 *                   as -1 for signed integers and as the largest value for unsigned ones.
 */
class NDArrayTimeConcat : public kotekan::Stage {
public:
    NDArrayTimeConcat(kotekan::Config& config, const std::string& unique_name,
                      kotekan::bufferContainer& buffers) :
        Stage(config, unique_name, buffers, std::bind(&NDArrayTimeConcat::main_thread, this)),
        in_buf(get_buffer("in_buf")), out_buf(get_buffer("out_buf")),
        fill_value(config.get_default<uint8_t>(unique_name, "fill_value", 0xFF)) {
        in_buf->register_consumer(unique_name);
        out_buf->register_producer(unique_name);

        in_desc = in_buf->get_frame_desc<kotekan::GenericNDArray>();
        if (!in_desc)
            FATAL_ERROR("NDArrayTimeConcat: in_buf {:s} needs an NDArray frame description",
                        in_buf->buffer_name);
        std::vector<std::ptrdiff_t> extents = in_desc->get_extents();
        if (extents.empty() || extents[0] != 1)
            FATAL_ERROR("NDArrayTimeConcat: dimension 0 of in_buf {:s} must have length 1",
                        in_buf->buffer_name);
        if (out_buf->frame_size == 0 || out_buf->frame_size % in_buf->frame_size != 0)
            FATAL_ERROR("NDArrayTimeConcat: out_buf frame size {:d} is not a multiple of the "
                        "in_buf frame size {:d}",
                        out_buf->frame_size, in_buf->frame_size);
        num_rows = out_buf->frame_size / in_buf->frame_size;
        extents[0] = num_rows;
        out_buf->ensure_frame_desc(kotekan::GenericNDArray::describe(
            in_desc->get_value_datatype(), in_desc->get_quantity_name(), extents,
            in_desc->get_dimnames(), in_desc->get_dimscalings()));
        out_desc = out_buf->get_frame_desc<kotekan::GenericNDArray>();
    }

    void main_thread() override {
        frameID in_id(in_buf), out_id(out_buf);
        uint8_t* out_frame = nullptr;
        bool started = false;
        int64_t period = 0, phase = 0, window_start = 0, last_seq = 0;
        std::vector<int> freq;

        while (!stop_thread) {
            const uint8_t* in_frame = in_buf->wait_for_full_frame(unique_name, in_id);
            if (in_frame == nullptr)
                break;
            const auto meta = get_chord_metadata(in_buf, in_id);
            if (!meta)
                FATAL_ERROR("NDArrayTimeConcat: {:s}[{:d}] has no chordMetadata",
                            in_buf->buffer_name, (int)in_id);
            meta->check_frame_desc(in_desc);
            if (!meta->has_fpga_seq_num() || !meta->has_time_downsampling_fpga()
                || !meta->has_coarse_freq())
                FATAL_ERROR("NDArrayTimeConcat: {:s} frames need fpga_seq_num, "
                            "time_downsampling_fpga and coarse_freq",
                            in_buf->buffer_name);
            const int64_t seq = meta->get_fpga_seq_num();
            if (!started) {
                period = meta->get_time_downsampling_fpga();
                if (period <= 0 || period != out_desc->get_dimscaling(0))
                    FATAL_ERROR("NDArrayTimeConcat: time_downsampling_fpga {:d} does not match "
                                "the dimension 0 scaling {:d} of {:s}",
                                period, out_desc->get_dimscaling(0), out_buf->buffer_name);
                phase = positive_mod(seq, period);
                freq = meta->get_coarse_freq();
                started = true;
            } else if (meta->get_time_downsampling_fpga() != period
                       || meta->get_coarse_freq() != freq) {
                FATAL_ERROR("NDArrayTimeConcat: {:s} changed time_downsampling_fpga or coarse_freq",
                            in_buf->buffer_name);
            } else if (seq <= last_seq) {
                FATAL_ERROR("NDArrayTimeConcat: {:s} fpga_seq_num {:d} does not follow {:d}",
                            in_buf->buffer_name, seq, last_seq);
            }
            if (positive_mod(seq - phase, period) != 0)
                FATAL_ERROR("NDArrayTimeConcat: {:s} fpga_seq_num {:d} is off the sample grid",
                            in_buf->buffer_name, seq);

            const int64_t start = seq - positive_mod(seq - phase, num_rows * period);
            if (out_frame && start != window_start) {
                out_buf->mark_frame_full(unique_name, out_id++);
                out_frame = nullptr;
            }
            if (!out_frame) {
                out_frame = out_buf->wait_for_empty_frame(unique_name, out_id);
                if (out_frame == nullptr)
                    break;
                std::memset(out_frame, fill_value, out_buf->frame_size);
                out_buf->allocate_new_metadata_object(out_id);
                const auto out_meta = get_chord_metadata(out_buf, out_id);
                out_meta->deepCopy(meta);
                out_meta->set_from_frame_desc(out_desc);
                out_meta->set_fpga_seq_num(start);
                window_start = start;
            }
            const int64_t row = (seq - start) / period;
            std::memcpy(out_frame + row * in_buf->frame_size, in_frame, in_buf->frame_size);
            last_seq = seq;
            in_buf->mark_frame_empty(unique_name, in_id++);

            if (row == num_rows - 1) {
                out_buf->mark_frame_full(unique_name, out_id++);
                out_frame = nullptr;
            }
        }
    }

private:
    static int64_t positive_mod(int64_t a, int64_t b) {
        return ((a % b) + b) % b;
    }

    Buffer* const in_buf;
    Buffer* const out_buf;
    const uint8_t fill_value;
    std::shared_ptr<const kotekan::GenericNDArray> in_desc;
    std::shared_ptr<const kotekan::GenericNDArray> out_desc;
    int64_t num_rows = 0;
};

REGISTER_KOTEKAN_STAGE(NDArrayTimeConcat);
