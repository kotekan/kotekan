#include "cudaPLMaskFRB1.hpp" // for launch_frb1_pl_mask

#include "Config.hpp"                // for Config
#include "DataType.hpp"              // for uint1x8_t
#include "NDArray.hpp"               // for NDArray
#include "NDArrayRingBuffer.hpp"     // for NDArrayRingBuffer, extent_t, read_descriptor_t
#include "bufferContainer.hpp"       // for bufferContainer
#include "chordMetadata.hpp"         // for chordMetadata
#include "cudaCommand.hpp"           // for cudaCommand, cudaCommandState, REGISTER_CUDA_COMMAND...
#include "cudaDeviceInterface.hpp"   // for cudaDeviceInterface
#include "div.hpp"                   // for div_ceil, div_noremainder
#include "gpuCommand.hpp"            // for gpuCommandType
#include "kotekanLogging.hpp"        // for DEBUG, FATAL_ERROR
#include "upchannelizeReference.hpp" // for upchan_default_num_taps

#include "fmt.hpp" // for compile_string_to_view

#include <algorithm>      // for min
#include <array>          // for array
#include <cassert>        // for assert
#include <cstddef>        // for ptrdiff_t
#include <cstdint>        // for uint64_t, uint8_t, int64_t
#include <driver_types.h> // for cudaEvent_t
#include <memory>         // for shared_ptr, static_pointer_cast
#include <mutex>          // for mutex, lock_guard
#include <string>         // for string
#include <vector>         // for vector

using kotekan::div_ceil;
using kotekan::div_noremainder;

// State shared by all instances of `cudaPLMaskFRB1`: the number of output samples produced so far.
// Our position in the input stream is derived from it, since a frame can end in the middle of an
// input row.
class cudaPLMaskFRB1State : public cudaCommandState {
public:
    cudaPLMaskFRB1State(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device) :
        cudaCommandState(config, unique_name, host_buffers, device), num_produced(0) {}
    // `wait_on_precondition` runs in frame order on the gpuProcess thread, so this needs no lock
    std::ptrdiff_t num_produced;
};

// clang-format off
/**
 * @class cudaPLMaskFRB1
 * @brief cudaCommand for calculating the packet loss (PL) mask of the FRB1 output.
 *
 * @author Erik Schnetter
 *
 * This is the PL mask counterpart of the FRB1 beamformers. An FRB1 output sample `Ttilde` sums the
 * `Td` input samples `Tbar` [Td * Ttilde, Td * (Ttilde + 1)) over all dishes and polarizations, so
 * it is valid only if all of them are valid -- a logical AND. The input is the expanded PL mask
 * (for U=1) or the PL mask upchannelized by `cudaPLMaskUpchannelizer` (for U>1), which have the
 * same layout. For U=1 the first `(M-1) * max_upchannelization_factor / 2` input samples are
 * skipped, as the U=1 FRB1 beamformer does. As for the FRB1 output `I`, several of these commands
 * (one per upchannelization factor) write disjoint frequency ranges of a shared output ring buffer.
 *
 * The output holds one byte per output sample and output frequency: 1 if the sample is valid and
 * 0 if it depends on a lost packet. Input frequency `f` is written to the `U` output frequencies
 * starting at `Fbar_out_min + U * (f - Fmin)`, matching the frequency order of the upchannelized
 * voltages.
 *
 * @par GPU Memory
 * @gpu_mem Input (upchannelized) expanded PL mask
 *   @gpu_mem_buffer       @c ring
 *   @gpu_mem_quantity     @c pl_mask_exp
 *   @gpu_mem_type         @c uint1x8
 *   @gpu_mem_dim_name     [@c Thi64][@c F][@c P][@c D8][@c Tlo64]
 *   @gpu_mem_shape        [@c buffer_depth * num_times / (64 * upchannelization_factor)]
 *                         [@c num_frequencies][@c num_polarizations][@c num_dishes / 8][@c 64 / 8]
 *   @gpu_mem_metadata     @c chordMetadata
 * @gpu_mem Output FRB1 PL mask
 *   @gpu_mem_buffer       @c ring
 *   @gpu_mem_quantity     @c pl_mask_I
 *   @gpu_mem_type         @c uint8
 *   @gpu_mem_dim_name     [@c Ttilde][@c Fbar]
 *   @gpu_mem_shape        [@c num_output_times][@c num_output_frequencies]
 *   @gpu_mem_metadata     @c chordMetadata
 * @conf  buffer_depth                 Int.  The number of GPU frames used for pipelining.
 * @conf  num_times                    Int.  Number of FPGA samples per frame (input cadence).
 * @conf  num_frequencies              Int.  Number of input frequencies.
 * @conf  num_polarizations            Int.  Number of polarizations (2).
 * @conf  num_dishes                   Int.  Number of dishes.
 * @conf  upchannelization_factor      Int.  Upchannelization factor U of the input.
 * @conf  downsampling_factor          Int.  Time downsampling factor Td of the FRB1 beamformer,
 *                                           counted in input samples.
 * @conf  max_upchannelization_factor  Int.  Only for U=1: the largest upchannelization factor
 *                                           used in this run, which determines the skipped input
 *                                           samples.
 * @conf  Fmin                         Int.  First input frequency to process.
 * @conf  Fmax                         Int.  One past the last input frequency to process.
 * @conf  Fbar_out_min                 Int.  First output frequency to write.
 * @conf  Fbar_out_max                 Int.  One past the last output frequency to write; must be
 *                                           `Fbar_out_min + U * (Fmax - Fmin)`.
 * @conf  num_output_frequencies       Int.  Number of output frequencies (of all producers).
 * @conf  num_output_times             Int.  Size of the output ring buffer, in output samples.
 * @conf  pl_mask_name                 String.  Base name for the input PL mask buffers.
 * @conf  frb_pl_mask_name             String.  Base name for the output PL mask buffers.
 */
// clang-format on
class cudaPLMaskFRB1 : public cudaCommand {
public:
    cudaPLMaskFRB1(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                   const int instance_num, const std::shared_ptr<cudaCommandState>& state);
    virtual ~cudaPLMaskFRB1();

    int wait_on_precondition() override;
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame() override;

private:
    // Number of PFB taps
    static constexpr int cuda_number_of_taps = kotekan::upchan_default_num_taps;

    // Parameters
    const int buffer_depth;
    const int num_times;
    const int num_frequencies;
    const int num_polarizations;
    const int num_dishes;
    const int upchannelization_factor;
    const int downsampling_factor;
    const int max_upchannelization_factor;
    // Number of input samples we skip at startup (see the U=1 FRB1 beamformer)
    const std::ptrdiff_t T_skip;
    const int Fmin;
    const int Fmax;
    const int Fbar_out_min;
    const int Fbar_out_max;
    const int num_output_frequencies;
    const int num_output_times;
    const bool poison_buffers;

    // Kotekan buffer names
    const std::string pl_mask_name;
    const std::string frb_pl_mask_name;

    // Buffers
    NDArrayRingBuffer<kotekan::uint1x8_t, 5> pl_mask;
    NDArrayRingBuffer<std::uint8_t, 2> frb_pl_mask;

    // State shared by all instances
    const std::shared_ptr<cudaPLMaskFRB1State> state;

    // Where this instance's input begins in its first input row, and how many output samples
    // it produces
    int bit_offset;
    std::ptrdiff_t num_times_out;

    // Set once, on the first frame; see `NDArrayRingBuffer::set_metadata`
    bool did_set_metadata;
};

// Serializes the metadata updates of all producers of the shared output buffer
static std::mutex frb_pl_mask_lock;

REGISTER_CUDA_COMMAND_WITH_STATE(cudaPLMaskFRB1, cudaPLMaskFRB1State);

cudaPLMaskFRB1::cudaPLMaskFRB1(kotekan::Config& config, const std::string& unique_name,
                               kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                               const int instance_num,
                               const std::shared_ptr<cudaCommandState>& state) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, state, "cudaPLMaskFRB1"),
    // Parameters
    buffer_depth(config.get<int>(unique_name, "buffer_depth")),
    num_times(config.get<int>(unique_name, "num_times")),
    num_frequencies(config.get<int>(unique_name, "num_frequencies")),
    num_polarizations(config.get<int>(unique_name, "num_polarizations")),
    num_dishes(config.get<int>(unique_name, "num_dishes")),
    upchannelization_factor(config.get<int>(unique_name, "upchannelization_factor")),
    downsampling_factor(config.get<int>(unique_name, "downsampling_factor")),
    max_upchannelization_factor(upchannelization_factor == 1
                                    ? config.get<int>(unique_name, "max_upchannelization_factor")
                                    : 0),
    T_skip(upchannelization_factor == 1
               ? (cuda_number_of_taps - 1) * std::ptrdiff_t(max_upchannelization_factor) / 2
               : 0),
    Fmin(config.get<int>(unique_name, "Fmin")), Fmax(config.get<int>(unique_name, "Fmax")),
    Fbar_out_min(config.get<int>(unique_name, "Fbar_out_min")),
    Fbar_out_max(config.get<int>(unique_name, "Fbar_out_max")),
    num_output_frequencies(config.get<int>(unique_name, "num_output_frequencies")),
    num_output_times(config.get<int>(unique_name, "num_output_times")),
    poison_buffers(config.get_default<bool>(unique_name, "poison_buffers", false)),
    // Buffer names
    pl_mask_name(config.get<std::string>(unique_name, "pl_mask_name")),
    frb_pl_mask_name(config.get<std::string>(unique_name, "frb_pl_mask_name")),
    // Buffers
    pl_mask(pl_mask_name, "pl_mask_exp",
            std::array<std::ptrdiff_t, 5>{
                buffer_depth * div_noremainder(num_times, 64 * upchannelization_factor),
                num_frequencies, num_polarizations, div_noremainder(num_dishes, 8), 64 / 8},
            std::array<std::string, 5>{"Thi64", "F", "P", "D8", "Tlo64"},
            std::array<std::ptrdiff_t, 5>{64 * upchannelization_factor, 1, 1, 8,
                                          8 * upchannelization_factor},
            *this),
    frb_pl_mask(frb_pl_mask_name, "pl_mask_I",
                std::array<std::ptrdiff_t, 2>{num_output_times, num_output_frequencies},
                std::array<std::string, 2>{"Ttilde", "Fbar"},
                std::array<std::ptrdiff_t, 2>{
                    std::ptrdiff_t(upchannelization_factor) * downsampling_factor, 1},
                *this),
    // The factory creates the state object of the type registered for this command
    state(std::static_pointer_cast<cudaPLMaskFRB1State>(state)), bit_offset(-1), num_times_out(-1),
    did_set_metadata(false)
//
{
    if (!this->state)
        FATAL_ERROR("cudaPLMaskFRB1 {:s}: missing command state", unique_name);
    if (!(upchannelization_factor >= 1 && downsampling_factor >= 1))
        FATAL_ERROR("upchannelization_factor ({:d}) and downsampling_factor ({:d}) must be "
                    "positive",
                    upchannelization_factor, downsampling_factor);
    if (upchannelization_factor == 1
        && !(max_upchannelization_factor >= 1
             && (cuda_number_of_taps - 1) * max_upchannelization_factor % 2 == 0))
        FATAL_ERROR("max_upchannelization_factor={:d} must be positive, and "
                    "(taps-1)*max_upchannelization_factor must be even (taps={:d})",
                    max_upchannelization_factor, cuda_number_of_taps);
    if (!(0 <= Fmin && Fmin <= Fmax && Fmax <= num_frequencies))
        FATAL_ERROR("Invalid channel range: require 0 <= Fmin ({:d}) <= Fmax ({:d}) <= "
                    "num_frequencies ({:d})",
                    Fmin, Fmax, num_frequencies);
    if (!(0 <= Fbar_out_min && Fbar_out_min <= Fbar_out_max
          && Fbar_out_max <= num_output_frequencies))
        FATAL_ERROR("Invalid output channel range: require 0 <= Fbar_out_min ({:d}) <= "
                    "Fbar_out_max ({:d}) <= num_output_frequencies ({:d})",
                    Fbar_out_min, Fbar_out_max, num_output_frequencies);
    if (Fbar_out_max - Fbar_out_min != upchannelization_factor * (Fmax - Fmin))
        FATAL_ERROR("The output channel range [{:d},{:d}) must hold U={:d} times the {:d} input "
                    "channels [{:d},{:d})",
                    Fbar_out_min, Fbar_out_max, upchannelization_factor, Fmax - Fmin, Fmin, Fmax);

    // We read up to a quarter of the input ring buffer at a time. That must suffice for one output
    // sample starting anywhere in an input row.
    const std::ptrdiff_t in_ringbuf = pl_mask.get_ndarray().extent(0);
    if (!(div_ceil(63 + std::ptrdiff_t(downsampling_factor), 64) <= in_ringbuf / 4))
        FATAL_ERROR("The input ring buffer holds {:d} rows of 64 samples; a quarter of that "
                    "is too small for one output sample of {:d} samples",
                    in_ringbuf, downsampling_factor);
    if (!(T_skip / 64 <= in_ringbuf / 4))
        FATAL_ERROR("The input ring buffer holds {:d} rows of 64 samples; a quarter of that "
                    "is too small to skip {:d} samples",
                    in_ringbuf, T_skip);

    pl_mask.register_consumer();
    frb_pl_mask.register_producer();

    set_command_type(gpuCommandType::KERNEL);
}

cudaPLMaskFRB1::~cudaPLMaskFRB1() {}

int cudaPLMaskFRB1::wait_on_precondition() {
    // Skip the whole rows of `T_skip` once, at startup; the kernel skips the remaining bits
    {
        const int errcode = pl_mask.skip_at_start(T_skip / 64);
        if (errcode < 0)
            return errcode;
    }

    // Where does our input begin? Output sample `k` covers the input samples
    // [T_skip + Td * k, T_skip + Td * (k + 1)).
    const std::ptrdiff_t T_begin = T_skip + downsampling_factor * state->num_produced;
    const std::ptrdiff_t read_head = pl_mask.peek_read_head();
    if (read_head < 0)
        return -1; // shutting down
    if (read_head != T_begin / 64)
        FATAL_ERROR("Input ring buffer {:s} is at row {:d}, but the next output sample begins "
                    "at input sample {:d}",
                    pl_mask_name, read_head, T_begin);
    bit_offset = int(T_begin % 64);

    const std::ptrdiff_t in_ringbuf = pl_mask.get_ndarray().extent(0);
    const std::ptrdiff_t out_ringbuf = frb_pl_mask.get_ndarray().extent(0);

    DEBUG("Waiting for pl_mask input ringbuffer data for frame {:d}...", gpu_frame_id);
    const int in_errcode = pl_mask.wait_and_claim_readable([&](const std::ptrdiff_t available) {
        using std::min;
        // Produce as many whole output samples as the readable rows hold
        const std::ptrdiff_t rows = min(available, in_ringbuf / 4);
        num_times_out = min((64 * rows - bit_offset) / downsampling_factor, out_ringbuf / 4);
        if (num_times_out <= 0)
            return read_descriptor_t{.claimed = 0, .read = 0};
        // Claim the rows we finished; a partially used last row is read again next time
        const std::ptrdiff_t T_end = bit_offset + downsampling_factor * num_times_out;
        return read_descriptor_t{.claimed = T_end / 64, .read = div_ceil(T_end, 64)};
    });
    if (in_errcode < 0)
        return in_errcode;
    state->num_produced += num_times_out;

    DEBUG("Done waiting for pl_mask for frame {:d}; will read {:d} input rows starting at bit "
          "{:d} and produce {:d} output samples",
          gpu_frame_id, pl_mask.get_read_valid().size(), bit_offset, num_times_out);

    const int out_errcode = frb_pl_mask.wait_for_writable(num_times_out);
    if (out_errcode < 0)
        return out_errcode;

    return 0;
}

cudaEvent_t cudaPLMaskFRB1::execute(cudaPipelineState& /*pipestate*/,
                                    const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();
    record_start_event();

    pl_mask.check_metadata();
    // Set the ring buffer metadata once; see `NDArrayRingBuffer::set_metadata`
    if (instance_num == 0 && !did_set_metadata) {
        did_set_metadata = true;

        // Our output buffer has multiple producers. They must serialize setting their part of
        // the metadata.
        std::lock_guard<std::mutex> lock(frb_pl_mask_lock);

        const auto in_meta = pl_mask.get_metadata();
        const bool out_has_metadata = frb_pl_mask.has_metadata();
        if (!out_has_metadata)
            frb_pl_mask.set_metadata(in_meta);
        const auto out_meta = frb_pl_mask.get_metadata();

        const std::vector<int> in_coarse_freq = in_meta->get_coarse_freq();
        const std::vector<int> in_freq_upchan_factor = in_meta->get_freq_upchan_factor();
        const std::vector<int> in_freq_upchan_index = in_meta->get_freq_upchan_index();
        if (!(std::ptrdiff_t(in_coarse_freq.size()) >= Fmax
              && in_freq_upchan_factor.size() == in_coarse_freq.size()
              && in_freq_upchan_index.size() == in_coarse_freq.size()))
            FATAL_ERROR("Input buffer {:s} must describe at least {:d} frequencies, but its "
                        "metadata list {:d} coarse frequencies, {:d} upchannelization factors and "
                        "{:d} upchannelization indices",
                        pl_mask_name, Fmax, in_coarse_freq.size(), in_freq_upchan_factor.size(),
                        in_freq_upchan_index.size());

        // Output frequency `Fbar_out_min + U * (f - Fmin) + u` is the `u`th upchannelized
        // channel of input frequency `f`, as for the upchannelized voltages
        std::vector<int> out_coarse_freq, out_freq_upchan_factor, out_freq_upchan_index;
        if (out_has_metadata) {
            out_coarse_freq = out_meta->get_coarse_freq();
            out_freq_upchan_factor = out_meta->get_freq_upchan_factor();
            out_freq_upchan_index = out_meta->get_freq_upchan_index();
        }
        out_coarse_freq.resize(num_output_frequencies, -1);
        out_freq_upchan_factor.resize(num_output_frequencies, -1);
        out_freq_upchan_index.resize(num_output_frequencies, -1);
        for (int f = Fmin; f < Fmax; ++f) {
            for (int u = 0; u < upchannelization_factor; ++u) {
                const int fbar = Fbar_out_min + upchannelization_factor * (f - Fmin) + u;
                out_coarse_freq.at(fbar) = in_coarse_freq.at(f);
                out_freq_upchan_factor.at(fbar) =
                    in_freq_upchan_factor.at(f) * upchannelization_factor;
                out_freq_upchan_index.at(fbar) =
                    in_freq_upchan_index.at(f) * upchannelization_factor + u;
            }
        }
        out_meta->set_coarse_freq(out_coarse_freq);
        out_meta->set_freq_upchan_factor(out_freq_upchan_factor);
        out_meta->set_freq_upchan_index(out_freq_upchan_index);

        // `time_downsampling_fpga` counts the input's rows of 64 samples
        const std::int64_t out_time_downsampling_fpga =
            div_noremainder(in_meta->get_time_downsampling_fpga(), 64) * downsampling_factor;
        if (!out_has_metadata)
            out_meta->set_time_downsampling_fpga(out_time_downsampling_fpga);
        else if (out_meta->get_time_downsampling_fpga() != out_time_downsampling_fpga)
            FATAL_ERROR("Another producer of buffer {:s} set time_downsampling_fpga={:d}, but "
                        "{:s} produces {:d}",
                        frb_pl_mask_name, out_meta->get_time_downsampling_fpga(), get_unique_name(),
                        out_time_downsampling_fpga);

        // All producers must agree on the start time, as for the FRB1 output `I`
        if (in_meta->has_fpga_seq_num()) {
            const std::int64_t out_fpga_seq_num =
                in_meta->get_fpga_seq_num()
                + T_skip * div_noremainder(in_meta->get_time_downsampling_fpga(), 64);
            if (!out_has_metadata)
                out_meta->set_fpga_seq_num(out_fpga_seq_num);
            else if (!out_meta->has_fpga_seq_num()
                     || out_meta->get_fpga_seq_num() != out_fpga_seq_num)
                FATAL_ERROR("Another producer of buffer {:s} starts at fpga_seq_num={:d}, but "
                            "{:s} starts at {:d}",
                            frb_pl_mask_name,
                            out_meta->has_fpga_seq_num() ? out_meta->get_fpga_seq_num() : -1,
                            get_unique_name(), out_fpga_seq_num);
        } else if (out_has_metadata && out_meta->has_fpga_seq_num()) {
            FATAL_ERROR("Another producer of buffer {:s} starts at fpga_seq_num={:d}, but the "
                        "input of {:s} has no fpga_seq_num",
                        frb_pl_mask_name, out_meta->get_fpga_seq_num(), get_unique_name());
        }
    }

    const std::uint64_t* const in_memory =
        reinterpret_cast<const std::uint64_t*>(pl_mask.get_ndarray().data());
    std::uint8_t* const out_memory = frb_pl_mask.get_ndarray().data();

    const std::ptrdiff_t size_in = pl_mask.get_ndarray().extent(0);
    const std::ptrdiff_t pos_in = pl_mask.get_read_valid().begin() % size_in;
    const std::ptrdiff_t size_out = frb_pl_mask.get_ndarray().extent(0);
    const std::ptrdiff_t pos_out = frb_pl_mask.get_write_valid().begin() % size_out;
    assert(frb_pl_mask.get_write_valid().size() == num_times_out);

    // Inner (fast) input dimension as `uint64` elements: P * (D/8)
    const std::ptrdiff_t num_elements =
        std::ptrdiff_t(num_polarizations) * div_noremainder(num_dishes, 8);

    launch_frb1_pl_mask(out_memory, in_memory, num_elements, num_frequencies,
                        num_output_frequencies, Fmin, Fmax, Fbar_out_min, upchannelization_factor,
                        downsampling_factor, num_times_out, size_in, pos_in, bit_offset, size_out,
                        pos_out, device.getStream(cuda_stream_id));

    return record_end_event();
}

void cudaPLMaskFRB1::finalize_frame() {
    // Advance the ring buffers
    pl_mask.finish_read();
    frb_pl_mask.finish_write();

    cudaCommand::finalize_frame();
}
