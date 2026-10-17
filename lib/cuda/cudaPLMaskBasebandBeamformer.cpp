#include "cudaPLMaskBasebandBeamformer.hpp" // for launch_bb_pl_mask

#include "Config.hpp"              // for Config
#include "DataType.hpp"            // for uint1x8_t
#include "NDArray.hpp"             // for NDArray
#include "NDArrayRingBuffer.hpp"   // for NDArrayRingBuffer, extent_t, read_descriptor_t
#include "bufferContainer.hpp"     // for bufferContainer
#include "chordMetadata.hpp"       // for chordMetadata
#include "cudaCommand.hpp"         // for cudaCommand, cudaPipelineState, REGISTER_CUDA_COMMAND
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "div.hpp"                 // for div_noremainder
#include "gpuCommand.hpp"          // for gpuCommandType
#include "kotekanLogging.hpp"      // for DEBUG

#include "fmt.hpp" // for compile_string_to_view

#include <algorithm>      // for min
#include <array>          // for array
#include <cstddef>        // for ptrdiff_t
#include <cstdint>        // for uint64_t
#include <driver_types.h> // for cudaEvent_t
#include <functional>     // for function
#include <memory>         // for shared_ptr
#include <string>         // for string
#include <vector>         // for vector

using kotekan::div_noremainder;

/**
 * @class cudaPLMaskBasebandBeamformer
 * @brief cudaCommand for the packet loss (PL) mask of the baseband beamformer outputs.
 *
 * @author Erik Schnetter
 *
 * Every dish with a nonzero phase contributes to every baseband beam. This mask treats a beam
 * sample as valid only if all dishes are valid at its time, frequency and polarization: the logical
 * AND of the expanded PL mask over its dish axis. The result is the same for all beams. It is
 * conservative: a lost packet from a dish with zero phase (e.g. a bad feed) still marks the sample
 * as lost. The baseband beamformer does not change the time axis, so this mask has the time axis
 * of the expanded PL mask.
 *
 * The polarization axis is that of the expanded PL mask. For CHIME, the voltage transpose may move
 * inputs between polarizations, so this is exact only if packet loss is the same for both
 * polarizations. All current PL mask producers (`lostSamplesToPLMask`, `ProcessPacketMask`,
 * `inventPLMask`) mask all inputs at a given time and frequency.
 *
 * @par GPU Memory
 * @gpu_mem Input expanded PL mask
 *   @gpu_mem_buffer    @c ring
 *   @gpu_mem_quantity  @c pl_mask_exp
 *   @gpu_mem_type      @c uint1x8
 *   @gpu_mem_dim_name  [@c Thi64][@c F][@c P][@c D8][@c Tlo64]
 *   @gpu_mem_shape     [@c buffer_depth * num_times / 64][@c num_frequencies][@c num_polarizations]
 *                      [@c num_dishes / 8][@c 64 / 8]
 *   @gpu_mem_metadata  @c chordMetadata
 * @gpu_mem Output baseband beamformer PL mask
 *   @gpu_mem_buffer    @c ring
 *   @gpu_mem_quantity  @c bb_pl_mask
 *   @gpu_mem_type      @c uint1x8
 *   @gpu_mem_dim_name  [@c Thi64][@c F][@c P][@c Tlo64]
 *   @gpu_mem_shape     [@c buffer_depth * num_times / 64][@c num_frequencies][@c num_polarizations]
 *                      [@c 64 / 8]
 *   @gpu_mem_metadata  @c chordMetadata
 * @conf  buffer_depth           Int.     The number of GPU frames used for pipelining commands.
 * @conf  num_times              Int.     Number of time samples per frame.
 * @conf  num_frequencies        Int.     Number of frequencies handled by this node.
 * @conf  num_polarizations      Int.     Number of polarizations (2)
 * @conf  num_dishes             Int.     Number of dishes
 * @conf  pl_expanded_mask_name  String.  Base name for the input pl_expanded_mask buffers.
 * @conf  bb_pl_mask_name        String.  Base name for the output bb_pl_mask buffers.
 */
class cudaPLMaskBasebandBeamformer : public cudaCommand {
public:
    cudaPLMaskBasebandBeamformer(kotekan::Config& config, const std::string& unique_name,
                                 kotekan::bufferContainer& host_buffers,
                                 cudaDeviceInterface& device, const int instance_num);
    virtual ~cudaPLMaskBasebandBeamformer();

    int wait_on_precondition() override;
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame() override;

private:
    // Parameters
    const int buffer_depth;
    const int num_times;
    const int num_frequencies;
    const int num_polarizations;
    const int num_dishes;

    // Kotekan buffer names
    const std::string pl_expanded_mask_name;
    const std::string bb_pl_mask_name;

    // Buffers
    NDArrayRingBuffer<kotekan::uint1x8_t, 5> pl_expanded_mask;
    NDArrayRingBuffer<kotekan::uint1x8_t, 4> bb_pl_mask;
    // Set once, on the first frame; see `NDArrayRingBuffer::set_metadata`
    bool did_set_metadata;
};

REGISTER_CUDA_COMMAND(cudaPLMaskBasebandBeamformer);

cudaPLMaskBasebandBeamformer::cudaPLMaskBasebandBeamformer(kotekan::Config& config,
                                                           const std::string& unique_name,
                                                           kotekan::bufferContainer& host_buffers,
                                                           cudaDeviceInterface& device,
                                                           const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "cudaPLMaskBasebandBeamformer"),
    // Parameters
    buffer_depth(config.get<int>(unique_name, "buffer_depth")),
    num_times(config.get<int>(unique_name, "num_times")),
    num_frequencies(config.get<int>(unique_name, "num_frequencies")),
    num_polarizations(config.get<int>(unique_name, "num_polarizations")),
    num_dishes(config.get<int>(unique_name, "num_dishes")),
    // Buffer names
    pl_expanded_mask_name(config.get<std::string>(unique_name, "pl_expanded_mask_name")),
    bb_pl_mask_name(config.get<std::string>(unique_name, "bb_pl_mask_name")),
    // Buffers
    pl_expanded_mask(pl_expanded_mask_name, "pl_mask_exp",
                     std::array<std::ptrdiff_t, 5>{buffer_depth * div_noremainder(num_times, 64),
                                                   num_frequencies, num_polarizations,
                                                   div_noremainder(num_dishes, 8), 64 / 8},
                     std::array<std::string, 5>{"Thi64", "F", "P", "D8", "Tlo64"},
                     std::array<std::ptrdiff_t, 5>{64, 1, 1, 8, 8}, *this),
    bb_pl_mask(bb_pl_mask_name, "bb_pl_mask",
               std::array<std::ptrdiff_t, 4>{buffer_depth * div_noremainder(num_times, 64),
                                             num_frequencies, num_polarizations, 64 / 8},
               std::array<std::string, 4>{"Thi64", "F", "P", "Tlo64"},
               std::array<std::ptrdiff_t, 4>{64, 1, 1, 8}, *this),
    did_set_metadata(false)
//
{
    pl_expanded_mask.register_consumer();
    bb_pl_mask.register_producer();

    set_command_type(gpuCommandType::KERNEL);
}

cudaPLMaskBasebandBeamformer::~cudaPLMaskBasebandBeamformer() {}

int cudaPLMaskBasebandBeamformer::wait_on_precondition() {
    DEBUG("Waiting for pl_expanded_mask input ringbuffer data for frame {:d}...", gpu_frame_id);
    // Read at most one frame
    const std::ptrdiff_t read_max = pl_expanded_mask.get_ndarray().extent(0) / buffer_depth;
    std::ptrdiff_t read = -1;
    const int in_errcode =
        pl_expanded_mask.wait_and_claim_readable([&](const std::ptrdiff_t available) {
            using std::min;
            read = min(available, read_max);
            return read_descriptor_t{.claimed = read, .read = read};
        });
    if (in_errcode < 0)
        return in_errcode;
    DEBUG("Done waiting for pl_expanded_mask for frame {:d}; will read {:d} rows", gpu_frame_id,
          read);

    // One output row per input row
    const int out_errcode = bb_pl_mask.wait_for_writable(read);
    if (out_errcode < 0)
        return out_errcode;

    return 0;
}

cudaEvent_t cudaPLMaskBasebandBeamformer::execute(cudaPipelineState& /*pipestate*/,
                                                  const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();
    record_start_event();

    pl_expanded_mask.check_metadata();
    // Set the ring buffer metadata once; see `NDArrayRingBuffer::set_metadata`. Time axis and
    // frequencies are those of the input.
    if (instance_num == 0 && !did_set_metadata) {
        did_set_metadata = true;
        bb_pl_mask.set_metadata(pl_expanded_mask.get_metadata());
    }

    const kotekan::uint1x8_t* const in_memory = pl_expanded_mask.get_ndarray().data();
    kotekan::uint1x8_t* const out_memory = bb_pl_mask.get_ndarray().data();

    // Ring buffer geometry, in time_64 rows. The positions wrap into the actual array index to
    // avoid overflow inside the kernel.
    const std::ptrdiff_t size_in = pl_expanded_mask.get_ndarray().extent(0);
    const std::ptrdiff_t pos_in = pl_expanded_mask.get_read_valid().begin() % size_in;
    const std::ptrdiff_t size_out = bb_pl_mask.get_ndarray().extent(0);
    const std::ptrdiff_t pos_out = bb_pl_mask.get_write_valid().begin() % size_out;
    const std::ptrdiff_t num_rows = bb_pl_mask.get_write_valid().size();

    launch_bb_pl_mask(reinterpret_cast<std::uint64_t*>(out_memory),
                      reinterpret_cast<const std::uint64_t*>(in_memory),
                      std::ptrdiff_t(num_frequencies) * num_polarizations,
                      div_noremainder(num_dishes, 8), num_rows, size_in, pos_in, size_out, pos_out,
                      device.getStream(cuda_stream_id));

    return record_end_event();
}

void cudaPLMaskBasebandBeamformer::finalize_frame() {
    // Advance the ring buffers
    pl_expanded_mask.finish_read();
    bb_pl_mask.finish_write();

    cudaCommand::finalize_frame();
}
