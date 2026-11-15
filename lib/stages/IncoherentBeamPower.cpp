#include "Config.hpp"
#include "NDArray.hpp"
#include "Stage.hpp"
#include "StageFactory.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "chordMetadata.hpp"
#include "kotekanLogging.hpp"
#include "visUtil.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

/**
 * @class IncoherentBeamPower
 * @brief Reduce S012tilde frames to the 16-bit power of the incoherent beam.
 *
 * S012tilde holds, for each time and frequency, the number of voltage samples summed over the
 * good feeds (S0) and the sums of their power (S1) and squared power (S2). The output is the
 * mean power of one voltage sample in units of 1/256: round(256 * S1 / S0), and 0 where S0 is
 * 0. A 4-bit voltage has a power of at most 128, so the largest value is 32768; anything larger
 * saturates at 65535.
 *
 * The output metadata is copied from the input frame, with the array description of out_buf.
 *
 * @par Buffers
 * @buffer in_buf   S012tilde frames.
 *     @buffer_format   NDArray uint64 [Trfi, F, 3]
 *     @buffer_metadata chordMetadata
 * @buffer out_buf  Power frames.
 *     @buffer_format   NDArray uint16 [Trfi, F]
 *     @buffer_metadata chordMetadata
 */
class IncoherentBeamPower : public kotekan::Stage {
public:
    IncoherentBeamPower(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& buffers) :
        Stage(config, unique_name, buffers, std::bind(&IncoherentBeamPower::main_thread, this)),
        in_buf(get_buffer("in_buf")), out_buf(get_buffer("out_buf")) {
        in_buf->register_consumer(unique_name);
        out_buf->register_producer(unique_name);

        in_desc = in_buf->get_frame_desc<kotekan::GenericNDArray>();
        if (!in_desc)
            FATAL_ERROR("IncoherentBeamPower: in_buf {:s} needs an NDArray frame description",
                        in_buf->buffer_name);
        const std::vector<std::ptrdiff_t> extents = in_desc->get_extents();
        if (in_desc->get_value_datatype() != kotekan::uint64 || extents.size() != 3
            || extents[2] != 3)
            FATAL_ERROR("IncoherentBeamPower: in_buf {:s} must be uint64 [Trfi, F, 3]",
                        in_buf->buffer_name);
        num_values = extents[0] * extents[1];
        const std::vector<kotekan::Symbol> dimnames = in_desc->get_dimnames();
        const std::vector<std::ptrdiff_t> dimscalings = in_desc->get_dimscalings();
        out_buf->ensure_frame_desc(kotekan::GenericNDArray::describe(
            kotekan::uint16, "incoherent_beam_power", {extents[0], extents[1]},
            {dimnames[0], dimnames[1]}, {dimscalings[0], dimscalings[1]}));
        out_desc = out_buf->get_frame_desc<kotekan::GenericNDArray>();
    }

    void main_thread() override {
        frameID in_id(in_buf), out_id(out_buf);

        while (!stop_thread) {
            const uint8_t* in_frame = in_buf->wait_for_full_frame(unique_name, in_id);
            if (in_frame == nullptr)
                break;
            const auto meta = get_chord_metadata(in_buf, in_id);
            if (!meta)
                FATAL_ERROR("IncoherentBeamPower: {:s}[{:d}] has no chordMetadata",
                            in_buf->buffer_name, (int)in_id);
            meta->check_frame_desc(in_desc);

            uint8_t* out_frame = out_buf->wait_for_empty_frame(unique_name, out_id);
            if (out_frame == nullptr)
                break;
            const uint64_t* const s012 = (const uint64_t*)in_frame;
            uint16_t* const power = (uint16_t*)out_frame;
            for (std::ptrdiff_t i = 0; i < num_values; ++i) {
                const uint64_t s0 = s012[3 * i];
                const uint64_t s1 = s012[3 * i + 1];
                power[i] = s0 ? (uint16_t)std::min<uint64_t>((256 * s1 + s0 / 2) / s0, 65535) : 0;
            }
            out_buf->allocate_new_metadata_object(out_id);
            const auto out_meta = get_chord_metadata(out_buf, out_id);
            out_meta->deepCopy(meta);
            out_meta->set_from_frame_desc(out_desc);

            in_buf->mark_frame_empty(unique_name, in_id++);
            out_buf->mark_frame_full(unique_name, out_id++);
        }
    }

private:
    Buffer* const in_buf;
    Buffer* const out_buf;
    std::shared_ptr<const kotekan::GenericNDArray> in_desc;
    std::shared_ptr<const kotekan::GenericNDArray> out_desc;
    std::ptrdiff_t num_values = 0;
};

REGISTER_KOTEKAN_STAGE(IncoherentBeamPower);
