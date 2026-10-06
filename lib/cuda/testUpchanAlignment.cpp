#include "Config.hpp"                // for Config
#include "DataType.hpp"              // for float16_t, int4x2_swapped_withoffset_t
#include "Stage.hpp"                 // for Stage
#include "StageFactory.hpp"          // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"                // for Buffer
#include "bufferContainer.hpp"       // for bufferContainer
#include "chordMetadata.hpp"         // for chordMetadata, get_chord_metadata
#include "errors.h"                  // for TEST_PASSED
#include "kotekanLogging.hpp"        // for FATAL_ERROR, INFO
#include "upchannelizeReference.hpp" // for upchan_default_num_taps

#include "fmt.hpp" // for format, join

#include <algorithm>  // for max
#include <cmath>      // for fabs
#include <cstddef>    // for ptrdiff_t
#include <cstdint>    // for int64_t, uint64_t
#include <cstring>    // for memcpy
#include <functional> // for function
#include <map>        // for map
#include <memory>     // for shared_ptr
#include <set>        // for set
#include <string>     // for string
#include <vector>     // for vector

/**
 * @class testUpchanAlignment
 * @brief Check that the upchannelizers, the FRB1 beamformers and the PL mask upchannelizers
 *        place a signal at the FPGA sequence numbers their metadata claim.
 *
 * The voltages hold a constant signal in the FPGA samples [box_begin, box_end) and are zero
 * elsewhere, and packets are lost in the same samples (see `inventVoltage` and `inventPLMask`).
 * The box must cover whole FRB1 output samples. This stage computes on the CPU which output
 * samples the box reaches, labels every output sample with the `fpga_seq_num` from its frame's
 * metadata, and checks:
 * - FRB1 beams, for each upchannelization factor U in the shared output buffer: For U=1, the
 *   output samples with nonzero power are exactly those overlapping the box. For U>1, the
 *   output samples with at least half the maximum power are exactly those.
 * - Upchannelized voltages: every output sample with nonzero power has its window of M*U input
 *   samples overlap the box, and the power-weighted centre equals the box centre.
 * - Upchannelized PL masks: the masked output samples are exactly those whose window overlaps
 *   the box.
 *
 * @par Buffers
 * @buffer frb1_beams      The FRB1 beams I, [Ttilde][Fbar][beamQ][beamP], float16
 * @buffer upchan_voltage  List of upchannelized voltages, [Tbar][Fbar][P][D], int4+4
 * @buffer upchan_pl_mask  List of upchannelized PL masks, [Thi64][F][P][D8][Tlo64], uint1
 *
 * @conf box_begin               Int. First FPGA sample of the signal.
 * @conf box_end                 Int. One past the last FPGA sample of the signal.
 * @conf upchan_factors          List of int. The upchannelization factor of each buffer in
 *                               `upchan_voltage` and `upchan_pl_mask`.
 * @conf max_centre_offset       Float. How far the power-weighted centre of the upchannelized
 *                               voltages may lie from the box centre, in upchannelized samples
 *                               (U FPGA samples). The two edges of the box are rounded to int4
 *                               with different phases, so the centre is not exact.
 */
class testUpchanAlignment : public kotekan::Stage {
    const std::int64_t box_begin = config.get<std::int64_t>(unique_name, "box_begin");
    const std::int64_t box_end = config.get<std::int64_t>(unique_name, "box_end");
    const std::vector<int> upchan_factors =
        config.get<std::vector<int>>(unique_name, "upchan_factors");
    const double max_centre_offset = config.get<double>(unique_name, "max_centre_offset");

    static constexpr int M = kotekan::upchan_default_num_taps;
    // Look at the output samples within this many FPGA samples of the box. This must cover the
    // smearing of the upchannelizer window and of the FRB1 downsampling.
    static constexpr std::int64_t margin = 1024;

    enum class kind_t { frb1_beams, upchan_voltage, upchan_pl_mask };
    struct stream_t {
        Buffer* buffer;
        kind_t kind;
        int upchan_factor;   // 0 for frb1_beams
        int frame_index = 0; // next frame to read
        std::int64_t end_seq_num = -1;
        bool done = false;
        // Output sample label (fpga_seq_num) -> power (or 1 for masked PL samples). For the
        // FRB1 beams, keyed first by upchannelization factor.
        std::map<int, std::map<std::int64_t, double>> power;
        // FPGA samples per output sample
        int tds = 0;

        stream_t(Buffer* const buffer, const kind_t kind, const int upchan_factor) :
            buffer(buffer), kind(kind), upchan_factor(upchan_factor) {}
    };
    std::vector<stream_t> streams;

public:
    testUpchanAlignment(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container, [](const kotekan::Stage& stage) {
            return const_cast<kotekan::Stage&>(stage).main_thread();
        }) {
        if (!(box_begin < box_end))
            FATAL_ERROR("box_begin={:d} must be smaller than box_end={:d}", box_begin, box_end);
        const std::vector<Buffer*> voltages = get_buffer_or_array("upchan_voltage");
        const std::vector<Buffer*> pl_masks = get_buffer_or_array("upchan_pl_mask");
        if (!(voltages.size() == upchan_factors.size() && pl_masks.size() == upchan_factors.size()))
            FATAL_ERROR("upchan_voltage ({:d} buffers) and upchan_pl_mask ({:d} buffers) need one "
                        "buffer per entry of upchan_factors ({:d} entries)",
                        voltages.size(), pl_masks.size(), upchan_factors.size());
        streams.emplace_back(get_buffer("frb1_beams"), kind_t::frb1_beams, 0);
        for (std::size_t n = 0; n < upchan_factors.size(); ++n) {
            streams.emplace_back(voltages.at(n), kind_t::upchan_voltage, upchan_factors.at(n));
            streams.emplace_back(pl_masks.at(n), kind_t::upchan_pl_mask, upchan_factors.at(n));
        }
        for (const stream_t& stream : streams)
            stream.buffer->register_consumer(unique_name);
    }

    virtual ~testUpchanAlignment() {}

    void main_thread() override {
        // Read the streams in step, always the one that lags most, so that no producer blocks
        // on a full buffer that we are not reading.
        while (!stop_thread) {
            stream_t* next = nullptr;
            for (stream_t& stream : streams)
                if (!stream.done && (!next || stream.end_seq_num < next->end_seq_num))
                    next = &stream;
            if (!next)
                break;
            if (!read_frame(*next))
                return;
        }
        if (stop_thread)
            return;

        for (const stream_t& stream : streams) {
            if (stream.kind == kind_t::frb1_beams)
                check_frb1_beams(stream);
            else if (stream.kind == kind_t::upchan_voltage)
                check_upchan_voltage(stream);
            else
                check_upchan_pl_mask(stream);
        }
        INFO("All outputs are aligned with the signal at FPGA samples [{:d}, {:d})", box_begin,
             box_end);
        TEST_PASSED();
    }

private:
    // Whether the output sample [seq, seq+tds) overlaps the box
    bool overlaps_box(const std::int64_t seq, const std::int64_t tds) const {
        return seq < box_end && seq + tds > box_begin;
    }

    // Whether the window of the upchannelizer output sample labelled `seq` overlaps the box.
    // The sample is centred on its window of M*U input samples, which thus covers
    // [seq - (M-1)*U/2, seq + (M+1)*U/2).
    bool window_overlaps_box(const std::int64_t seq, const int U) const {
        return seq - (M - 1) * U / 2 < box_end && seq + (M + 1) * U / 2 > box_begin;
    }

    bool in_range(const std::int64_t seq) const {
        return seq >= box_begin - margin && seq < box_end + margin;
    }

    // Read one frame of a stream; returns false if we are shutting down
    bool read_frame(stream_t& stream) {
        Buffer* const buffer = stream.buffer;
        const int frame_id = stream.frame_index % buffer->num_frames;
        const std::uint8_t* const frame = buffer->wait_for_full_frame(unique_name, frame_id);
        if (!frame)
            return false;
        const std::shared_ptr<const chordMetadata> meta = get_chord_metadata(buffer, frame_id);
        const std::int64_t seq0 = meta->get_fpga_seq_num();
        const int ntimes = meta->dim[0];
        const std::ptrdiff_t time_stride = buffer->frame_size / ntimes;
        if (stream.frame_index == 0 && !(seq0 <= box_begin - margin))
            FATAL_ERROR("Buffer {:s} starts at fpga_seq_num={:d}, after the region to check "
                        "[{:d}, {:d})",
                        buffer->buffer_name, seq0, box_begin - margin, box_end + margin);

        if (stream.kind == kind_t::frb1_beams) {
            // [Ttilde][Fbar][beamQ][beamP]
            stream.tds = meta->get_time_downsampling_fpga();
            const int nfreqs = meta->dim[1];
            const std::ptrdiff_t freq_stride = time_stride / nfreqs;
            const std::vector<int> freq_upchan_factor = meta->get_freq_upchan_factor();
            for (int t = 0; t < ntimes; ++t) {
                const std::int64_t seq = seq0 + std::int64_t(t) * stream.tds;
                if (!in_range(seq))
                    continue;
                for (int f = 0; f < int(freq_upchan_factor.size()) && f < nfreqs; ++f) {
                    const int U = freq_upchan_factor.at(f);
                    if (U <= 0)
                        continue; // not written
                    const float16_t* const beams = reinterpret_cast<const float16_t*>(
                        frame + t * time_stride + f * freq_stride);
                    double power = 0;
                    for (std::ptrdiff_t b = 0; b < freq_stride / std::ptrdiff_t(sizeof *beams); ++b)
                        power += double(beams[b]);
                    stream.power[U][seq] += power;
                }
            }
        } else if (stream.kind == kind_t::upchan_voltage) {
            // [Tbar][Fbar][P][D]; only the first `get_nfreq()` frequencies are written
            stream.tds = meta->get_time_downsampling_fpga();
            const std::ptrdiff_t nbytes =
                std::ptrdiff_t(meta->get_nfreq()) * meta->dim[2] * meta->dim[3];
            for (int t = 0; t < ntimes; ++t) {
                const std::int64_t seq = seq0 + std::int64_t(t) * stream.tds;
                if (!in_range(seq))
                    continue;
                const kotekan::int4x2_swapped_withoffset_t* const values =
                    reinterpret_cast<const kotekan::int4x2_swapped_withoffset_t*>(
                        frame + t * time_stride);
                double power = 0;
                for (std::ptrdiff_t n = 0; n < nbytes; ++n)
                    power += values[n][0] * values[n][0] + values[n][1] * values[n][1];
                stream.power[0][seq] = power;
            }
        } else {
            // [Thi64][F][P][D8][Tlo64]. Only frequency 0 is checked: the frequency metadata
            // describe the input, not which output frequencies were written.
            const int U = stream.upchan_factor;
            stream.tds = U;
            if (meta->get_time_downsampling_fpga() != 64 * U)
                FATAL_ERROR("Buffer {:s} has time_downsampling_fpga={:d}, expected {:d}",
                            buffer->buffer_name, meta->get_time_downsampling_fpga(), 64 * U);
            const int nwords = meta->dim[2] * meta->dim[3]; // P * D8
            for (int t = 0; t < ntimes; ++t) {
                for (int bit = 0; bit < 64; ++bit) {
                    const std::int64_t seq = seq0 + (std::int64_t(t) * 64 + bit) * U;
                    if (!in_range(seq))
                        continue;
                    int num_masked = 0;
                    for (int w = 0; w < nwords; ++w) {
                        std::uint64_t word;
                        std::memcpy(&word, frame + t * time_stride + 8 * w, 8);
                        num_masked += !((word >> bit) & 1);
                    }
                    if (num_masked != 0 && num_masked != nwords)
                        FATAL_ERROR("Buffer {:s}: at fpga_seq_num={:d}, {:d} of {:d} inputs are "
                                    "masked; expected none or all",
                                    buffer->buffer_name, seq, num_masked, nwords);
                    stream.power[0][seq] = num_masked != 0;
                }
            }
        }

        stream.end_seq_num = seq0 + std::int64_t(ntimes) * meta->get_time_downsampling_fpga();
        buffer->mark_frame_empty(unique_name, frame_id);
        ++stream.frame_index;
        if (stream.end_seq_num >= box_end + margin) {
            stream.done = true;
            buffer->unregister_consumer(unique_name);
        }
        return true;
    }

    // Labels of the output samples where `select` holds
    static std::set<std::int64_t>
    select_seqs(const std::map<std::int64_t, double>& power,
                const std::function<bool(std::int64_t, double)>& select) {
        std::set<std::int64_t> seqs;
        for (const auto& [seq, p] : power)
            if (select(seq, p))
                seqs.insert(seq);
        return seqs;
    }

    static std::string to_string(const std::set<std::int64_t>& seqs) {
        return fmt::format("{}", fmt::join(seqs, ", "));
    }

    void check_frb1_beams(const stream_t& stream) const {
        const auto& power_by_U = stream.power;
        if (power_by_U.count(1) == 0 || power_by_U.size() < 2)
            FATAL_ERROR("The FRB1 beams hold {:d} upchannelization factors; need U=1 and at least "
                        "one other",
                        power_by_U.size());
        const std::int64_t tds = stream.tds;
        for (const auto& [U, power] : power_by_U) {
            const std::set<std::int64_t> expected = select_seqs(
                power, [&](std::int64_t seq, double) { return overlaps_box(seq, tds); });
            double max_power = 0;
            for (const auto& [seq, p] : power)
                max_power = std::max(max_power, p);
            // Without upchannelization the signal stays exactly within the box. Upchannelization
            // smears it by about M*U/2 FPGA samples on either side.
            const double threshold = U == 1 ? 0 : max_power / 2;
            const std::set<std::int64_t> found =
                select_seqs(power, [&](std::int64_t, double p) { return p > threshold && p > 0; });
            INFO("FRB1 beams, U={:d}: signal in samples [{:s}], expected [{:s}]", U,
                 to_string(found), to_string(expected));
            if (found != expected)
                FATAL_ERROR("FRB1 beams, U={:d}: the signal is in the samples starting at "
                            "fpga_seq_num [{:s}], expected [{:s}]",
                            U, to_string(found), to_string(expected));
        }
    }

    void check_upchan_voltage(const stream_t& stream) const {
        const int U = stream.upchan_factor;
        const auto& power = stream.power.at(0);
        const std::set<std::int64_t> found =
            select_seqs(power, [](std::int64_t, double p) { return p > 0; });
        const std::set<std::int64_t> allowed = select_seqs(
            power, [&](std::int64_t seq, double) { return window_overlaps_box(seq, U); });
        // Power-weighted centre, using the centre of each output sample
        double sum_power = 0, sum_seq_power = 0;
        for (const auto& [seq, p] : power) {
            sum_power += p;
            sum_seq_power += (seq + U / 2.0) * p;
        }
        const double centre = sum_seq_power / sum_power;
        const double box_centre = (box_begin + box_end) / 2.0;
        INFO("Upchannelized voltages, U={:d}: signal in samples [{:s}], centre {:.3f}, expected "
             "within [{:s}], centre {:.3f}",
             U, to_string(found), centre, to_string(allowed), box_centre);
        if (found.empty())
            FATAL_ERROR("Upchannelized voltages, U={:d}: no signal found", U);
        for (const std::int64_t seq : found)
            if (allowed.count(seq) == 0)
                FATAL_ERROR("Upchannelized voltages, U={:d}: signal at fpga_seq_num={:d}, outside "
                            "the samples whose window overlaps the box [{:s}]",
                            U, seq, to_string(allowed));
        if (!(std::fabs(centre - box_centre) <= max_centre_offset * U))
            FATAL_ERROR("Upchannelized voltages, U={:d}: the signal is centred at {:.3f}, but the "
                        "box at {:.3f}",
                        U, centre, box_centre);
    }

    void check_upchan_pl_mask(const stream_t& stream) const {
        const int U = stream.upchan_factor;
        const auto& masked = stream.power.at(0);
        const std::set<std::int64_t> found =
            select_seqs(masked, [](std::int64_t, double m) { return m != 0; });
        const std::set<std::int64_t> expected = select_seqs(
            masked, [&](std::int64_t seq, double) { return window_overlaps_box(seq, U); });
        INFO("Upchannelized PL mask, U={:d}: masked samples [{:s}], expected [{:s}]", U,
             to_string(found), to_string(expected));
        if (found != expected)
            FATAL_ERROR("Upchannelized PL mask, U={:d}: masked samples start at fpga_seq_num "
                        "[{:s}], expected [{:s}]",
                        U, to_string(found), to_string(expected));
    }
};

REGISTER_KOTEKAN_STAGE(testUpchanAlignment);
