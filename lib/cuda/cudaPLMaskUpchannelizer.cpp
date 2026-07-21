#include <driver_types.h>          // for cudaEvent_t
#include <algorithm>               // for min
#include <array>                   // for array
#include <cstddef>                 // for ptrdiff_t
#include <cstdint>                 // for uint64_t
#include <functional>              // for function
#include <memory>                  // for shared_ptr
#include <string>                  // for string
#include <vector>                  // for vector

#include "Config.hpp"              // for Config
#include "DataType.hpp"            // for uint1x8_t
#include "NDArray.hpp"             // for NDArray
#include "NDArrayRingBuffer.hpp"   // for NDArrayRingBuffer, extent_t, read_descriptor_t
#include "bufferContainer.hpp"     // for bufferContainer
#include "chordMetadata.hpp"       // for chordMetadata
#include "cudaCommand.hpp"         // for cudaCommand, cudaPipelineState, REGISTER_CUDA_COMMAND
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "cudaPLMaskUpchannelizer.hpp" // for launch_upchannelize_pl_mask
#include "div.hpp"                 // for div_ceil, div_noremainder, round_down
#include "gpuCommand.hpp"          // for gpuCommandType
#include "kotekanLogging.hpp"      // for DEBUG, FATAL_ERROR
#include "fmt.hpp"                 // for compile_string_to_view

using kotekan::div_ceil;
using kotekan::div_noremainder;
using kotekan::round_down;

/**
 * @class cudaPLMaskUpchannelizer
 * @brief cudaCommand for upchannelizing the packet loss (PL) mask.
 *
 * @author Erik Schnetter
 *
 * Propagates the (already expanded) PL mask onto the upchannelized time grid. An upchannelized
 * sample depends on `M*U` consecutive input samples (M = PFB taps = 4, U = upchannelization
 * factor), so an output sample is valid only if all of them are valid -- a logical AND that also
 * downsamples the time axis by `U`. Only input frequency channels [Fmin, Fmax) are processed; the
 * result is written to the output's channels [0, Fmax-Fmin).
 *
 * @par GPU Memory
 * @gpu_mem Input expanded PL mask
 *   @gpu_mem_buffer       @c ring
 *   @gpu_mem_quantity     @c pl_mask
 *   @gpu_mem_type         @c uint1x8
 *   @gpu_mem_dim_name     [@c Thi64][@c F][@c P][@c D8][@c Tlo64]
 *   @gpu_mem_dim_scaling  [@c Thi64][@c F][@c P][@c D8][@c Tlo64]
 *   @gpu_mem_shape        [@c buffer_depth * num_times / 64][@c num_frequencies][@c num_polarizations]
 *                         [@c num_dishes / 8][@c 64 / 8]
 *   @gpu_mem_metadata     @c chordMetadata
 * @gpu_mem Output upchannelized expanded PL mask
 *   @gpu_mem_buffer       @c ring
 *   @gpu_mem_quantity     @c pl_mask
 *   @gpu_mem_type         @c uint1x8
 *   @gpu_mem_dim_name     [@c Thi64][@c F][@c P][@c D8][@c Tlo64]
 *   @gpu_mem_dim_scaling  [@c Thi64][@c F][@c P][@c D8][@c Tlo64]
 *   @gpu_mem_shape        [@c buffer_depth * num_times / (64 * upchannelization_factor)]
 *                         [@c num_frequencies_out][@c num_polarizations][@c num_dishes / 8][@c 64 / 8]
 *   @gpu_mem_metadata     @c chordMetadata
 * @conf  buffer_depth                         Int.  The number of GPU frames used for pipelining.
 * @conf  num_times                            Int.  Number of time samples per frame (input cadence).
 * @conf  num_frequencies                      Int.  Number of (input) frequencies on this node.
 * @conf  num_frequencies_out                  Int.  Output frequency count (total available; the
 *                                                   buffer may over-allocate, only Fmax-Fmin used).
 * @conf  num_polarizations                    Int.  Number of polarizations (2).
 * @conf  num_dishes                           Int.  Number of dishes.
 * @conf  upchannelization_factor              Int.  Time downsampling factor U (power of two).
 * @conf  Fmin                                 Int.  First input coarse channel to process.
 * @conf  Fmax                                 Int.  One past the last input coarse channel.
 * @conf  expanded_pl_mask_name                String.  Base name for the input pl_mask buffers.
 * @conf  upchannelized_expanded_pl_mask_name  String.  Base name for the output pl_mask buffers.
 */
class cudaPLMaskUpchannelizer : public cudaCommand {
public:
    cudaPLMaskUpchannelizer(kotekan::Config& config, const std::string& unique_name,
                            kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                            const int instance_num);
    virtual ~cudaPLMaskUpchannelizer();

    int wait_on_precondition() override;
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame() override;

private:
    // Number of PFB taps (must match the kernel).
    static constexpr int cuda_number_of_taps = 4;

    // Parameters
    const int buffer_depth;
    const int num_times;
    const int num_frequencies;     // input frequency count
    const int num_frequencies_out; // output frequency count (total available; over-allocated)
    const int num_polarizations;
    const int num_dishes;
    const int upchannelization_factor;
    // Input coarse-channel range [Fmin, Fmax) to process; the output is written densely to channels
    // [0, Fmax-Fmin).
    const int Fmin;
    const int Fmax;
    const bool poison_buffers;

    // Kotekan buffer names
    const std::string expanded_pl_mask_name;
    const std::string upchannelized_expanded_pl_mask_name;

    // Buffers
    NDArrayRingBuffer<kotekan::uint1x8_t, 5> pl_expanded_mask;
    NDArrayRingBuffer<kotekan::uint1x8_t, 5> pl_upchannelized_expanded_mask;
};

REGISTER_CUDA_COMMAND(cudaPLMaskUpchannelizer);

cudaPLMaskUpchannelizer::cudaPLMaskUpchannelizer(kotekan::Config& config,
                                                 const std::string& unique_name,
                                                 kotekan::bufferContainer& host_buffers,
                                                 cudaDeviceInterface& device,
                                                 const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "cudaPLMaskUpchannelizer"),
    // Parameters
    buffer_depth(config.get<int>(unique_name, "buffer_depth")),
    num_times(config.get<int>(unique_name, "num_times")),
    num_frequencies(config.get<int>(unique_name, "num_frequencies")),
    num_frequencies_out(config.get<int>(unique_name, "num_frequencies_out")),
    num_polarizations(config.get<int>(unique_name, "num_polarizations")),
    num_dishes(config.get<int>(unique_name, "num_dishes")),
    upchannelization_factor(config.get<int>(unique_name, "upchannelization_factor")),
    Fmin(config.get<int>(unique_name, "Fmin")),
    Fmax(config.get<int>(unique_name, "Fmax")),
    poison_buffers(config.get_default<bool>(unique_name, "poison_buffers", false)),
    // Buffer names
    expanded_pl_mask_name(config.get<std::string>(unique_name, "expanded_pl_mask_name")),
    upchannelized_expanded_pl_mask_name(
        config.get<std::string>(unique_name, "upchannelized_expanded_pl_mask_name")),
    // Buffers. The output keeps the same elements; its frequency dimension is the (possibly
    // over-allocated) num_frequencies_out, of which only [0, Fmax-Fmin) is written.
    pl_expanded_mask(expanded_pl_mask_name, "pl_mask_exp",
                     std::array<std::ptrdiff_t, 5>{buffer_depth * div_noremainder(num_times, 64),
                                                   num_frequencies, num_polarizations,
                                                   div_noremainder(num_dishes, 8), 64 / 8},
                     std::array<std::string, 5>{"Thi64", "F", "P", "D8", "Tlo64"},
                     std::array<std::ptrdiff_t, 5>{64, 1, 1, 8, 8}, *this),
    pl_upchannelized_expanded_mask(
        upchannelized_expanded_pl_mask_name, "pl_mask_exp",
        std::array<std::ptrdiff_t, 5>{
            buffer_depth * div_noremainder(num_times, 64 * upchannelization_factor),
            num_frequencies_out, num_polarizations, div_noremainder(num_dishes, 8), 64 / 8},
        std::array<std::string, 5>{"Thi64", "F", "P", "D8", "Tlo64"},
        std::array<std::ptrdiff_t, 5>{64 * upchannelization_factor, 1, 1, 8,
                                      8 * upchannelization_factor},
        *this)
//
{
    if (!(0 <= Fmin && Fmin <= Fmax && Fmax <= num_frequencies))
        FATAL_ERROR("Invalid channel range: require 0 <= Fmin ({:d}) <= Fmax ({:d}) <= "
                    "num_frequencies ({:d})",
                    Fmin, Fmax, num_frequencies);
    if (Fmax - Fmin > num_frequencies_out)
        FATAL_ERROR("Output buffer too small: Fmax-Fmin ({:d}) > num_frequencies_out ({:d})",
                    Fmax - Fmin, num_frequencies_out);

    // The kernel wraps ring indices with a power-of-two bitmask, so both ring sizes (the slowest,
    // "Thi64" dimension) must be powers of two.
    const std::ptrdiff_t in_size = pl_expanded_mask.get_ndarray().extent(0);
    const std::ptrdiff_t out_size = pl_upchannelized_expanded_mask.get_ndarray().extent(0);
    if (in_size <= 0 || (in_size & (in_size - 1)) != 0)
        FATAL_ERROR("pl_expanded_mask ring size {:d} is not a power of two", in_size);
    if (out_size <= 0 || (out_size & (out_size - 1)) != 0)
        FATAL_ERROR("pl_upchannelized_expanded_mask ring size {:d} is not a power of two", out_size);

    pl_expanded_mask.register_consumer();
    pl_upchannelized_expanded_mask.register_producer();

    set_command_type(gpuCommandType::KERNEL);
}

cudaPLMaskUpchannelizer::~cudaPLMaskUpchannelizer() {}

int cudaPLMaskUpchannelizer::wait_on_precondition() {
    // The kernel reads (M-1)*U input bits past the data it consumes (the PFB forward stencil). In
    // time_64 row units that is `overlap` extra rows that must be readable but are not advanced.
    const std::ptrdiff_t overlap =
        div_ceil((cuda_number_of_taps - 1) * std::ptrdiff_t(upchannelization_factor), 64);
    const std::ptrdiff_t in_ringbuf = pl_expanded_mask.get_ndarray().extent(0);
    const std::ptrdiff_t read_max = in_ringbuf / 4;

    DEBUG("Waiting for pl_expanded_mask input ringbuffer data for frame {:d}...", gpu_frame_id);
    std::ptrdiff_t consumed = 0;
    const int in_errcode =
        pl_expanded_mask.wait_and_claim_readable([&](const std::ptrdiff_t available) {
            using std::min;
            // Leave room for the look-ahead, then round the consumed amount down to a multiple of U
            // (so the output is a whole number of `time_64` rows).
            const std::ptrdiff_t usable = min(available, read_max) - overlap;
            consumed = usable <= 0 ? 0 : round_down(usable, upchannelization_factor);
            const std::ptrdiff_t read = consumed == 0 ? 0 : consumed + overlap;
            return read_descriptor_t{.claimed = consumed, .read = read};
        });
    if (in_errcode < 0)
        return in_errcode;

    const std::ptrdiff_t written = consumed / upchannelization_factor;
    DEBUG("Done waiting for pl_expanded_mask for frame {:d}; will consume {:d} input rows and "
          "produce {:d} output rows",
          gpu_frame_id, consumed, written);

    const int out_errcode = pl_upchannelized_expanded_mask.wait_for_writable(written);
    if (out_errcode < 0)
        return out_errcode;

    return 0;
}

cudaEvent_t cudaPLMaskUpchannelizer::execute(cudaPipelineState& /*pipestate*/,
                                             const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();
    record_start_event();

    pl_expanded_mask.check_metadata();
    pl_upchannelized_expanded_mask.set_metadata(pl_expanded_mask.get_metadata());

    // The upchannelizer downsamples the time axis by `upchannelization_factor`; the frequency
    // layout is unchanged (all other metadata is copied by set_metadata above).
    // TODO: Set this metadata only once.
    const auto& in_meta = pl_expanded_mask.get_metadata();
    const auto& out_meta = pl_upchannelized_expanded_mask.get_metadata();
    out_meta->set_time_downsampling_fpga(in_meta->get_time_downsampling_fpga()
                                         * upchannelization_factor);

    kotekan::uint1x8_t* const in_memory = pl_expanded_mask.get_ndarray().data();
    kotekan::uint1x8_t* const out_memory = pl_upchannelized_expanded_mask.get_ndarray().data();

    // Ring buffer geometry, in time_64 rows. The positions wrap into the actual array index to
    // avoid overflow inside the kernel.
    const std::ptrdiff_t size_in = pl_expanded_mask.get_ndarray().extent(0);
    const std::ptrdiff_t pos_in = pl_expanded_mask.get_read_valid().begin() % size_in;
    const std::ptrdiff_t size_out = pl_upchannelized_expanded_mask.get_ndarray().extent(0);
    const std::ptrdiff_t pos_out =
        pl_upchannelized_expanded_mask.get_write_valid().begin() % size_out;

    // Number of input rows to consume (excludes the look-ahead); the kernel produces
    // num_times_64 / U output rows.
    const std::ptrdiff_t num_times_64 = pl_expanded_mask.get_read_claimed().size();
    // Inner (fast) dimension as `uint64` elements: P * (D/8). Each such element packs the 64 "Tlo64"
    // time bits; the frequency axis (read range [Fmin, Fmax)) is handled separately.
    const std::ptrdiff_t num_elements =
        std::ptrdiff_t(num_polarizations) * div_noremainder(num_dishes, 8);

    launch_upchannelize_pl_mask(
        reinterpret_cast<std::uint64_t*>(out_memory),
        reinterpret_cast<const std::uint64_t*>(in_memory), num_elements, num_frequencies,
        num_frequencies_out, Fmin, Fmax, num_times_64, size_in, pos_in, size_out, pos_out,
        upchannelization_factor, device.getStream(cuda_stream_id));

    return record_end_event();
}

void cudaPLMaskUpchannelizer::finalize_frame() {
    // Advance the ring buffers
    pl_expanded_mask.finish_read();
    pl_upchannelized_expanded_mask.finish_write();

    cudaCommand::finalize_frame();
}
