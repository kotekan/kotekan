#include "Config.hpp"              // for Config
#include "DataType.hpp"            // for int4x2_swapped_withoffset_t, uint1x8_t
#include "NDArray.hpp"             // for NDArray
#include "NDArrayBuffer.hpp"       // for NDArrayBuffer
#include "NDArrayRingBuffer.hpp"   // for NDArrayRingBuffer, read_descriptor_t, extent_t
#include "bufferContainer.hpp"     // for bufferContainer
#include "chordMetadata.hpp"       // for chordMetadata
#include "cudaCommand.hpp"         // for cudaCommand, cudaPipelineState, REGISTER_CUDA_COMMAND
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "cudaUtils.hpp"           // for CHECK_CUDA_ERROR
#include "div.hpp"                 // for div_noremainder, round_down
#include "gpuCommand.hpp"          // for gpuCommandType
#include "kotekanLogging.hpp"      // for DEBUG, FATAL_ERROR
#include "n2k/rfi_kernels.hpp"     // for launch_s0_kernel, launch_s12_kernel

#include <algorithm>          // for min
#include <array>              // for array
#include <cassert>            // for assert
#include <cstddef>            // for ptrdiff_t, size_t
#include <cstdint>            // for uint64_t, uint8_t
#include <cuda_runtime_api.h> // for cudaStreamSynchronize
#include <driver_types.h>     // for CUstream_st, cudaEvent_t, CUevent_st
#include <fmt.hpp>            // for compile_string_to_view
#include <functional>         // for function
#include <memory>             // for allocator, shared_ptr, __shared_ptr_access
#include <string>             // for basic_string, string
#include <sys/types.h>        // for ulong
#include <vector>             // for vector

using kotekan::div_noremainder;
using kotekan::round_down;

/**
 * @class cudaPLMaskAccumulator
 * @brief cudaCommand for downsampling the Packet Loss (PL) mask in time, with CUDA code from
 * Kendrick.
 *
 * @author Geoffrey Ryan
 *
 * An example of this stage being used can be found in `config/tests/verify_cuda_pl_accum.yaml`.
 *
 * A CPU implementation is in `lib/testing/gpuSimulatePLMaskAccumulator.hpp`.
 *
 * @par GPU Memory
 * @gpu_mem Input PL mask
 *   @gpu_mem_buffer    @c ring
 *   @gpu_mem_quantity  @c pl_mask
 *   @gpu_mem_type      @c uint1x8
 *   @gpu_mem_dim_name  [@c T2hi64][@c F4][@c P][@c D8][@c T2lo64]
 *   @gpu_mem_shape     [@c buffer_depth * num_times/128][@c num_freq/4][@c num_pol][@c
 * num_dishes/8][@c 8]
 *   @gpu_mem_metadata  @c chordMetadata
 * @gpu_mem Output PL counts
 *   num_subintegrations := samples_per_data_set / sub_integration_ntime
 *   @gpu_mem_buffer    @c standard
 *   @gpu_mem_quantity  @c pl_counts
 *   @gpu_mem_type      @c uint64
 *   @gpu_mem_dim_name  [@c Tc][@c F][@c P][@c D]
 *   @gpu_mem_shape     [@c num_subintegrations][@c num_freq][@c num_pol][@c num_dishes]
 *   @gpu_mem_metadata  @c chordMetadata
 * @conf  buffer_depth           Int.     The number of GPU frames used for pipelining commands.
 * @conf  num_times              Int.     Number of time samples per frame.
 * @conf  num_frequencies        Int.     Number of time samples per frame.
 * @conf  num_polarizations      Int.     Number of polarizations.
 * @conf  num_dishes             Int.     Number of dishes.
 * @conf  sub_integration_ntime  Int.     Number of time samples that will be summed into the
 *                                        correlation matrix.
 * @conf  pl_mask_name           String.  Base name for the PL mask buffers.
 * @conf  pl_counts_name         String.  Base name for the PL counts buffers.
 */
class cudaPLMaskAccumulator : public cudaCommand {
public:
    cudaPLMaskAccumulator(kotekan::Config& config, const std::string& unique_name,
                          kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                          const int instance_num);
    virtual ~cudaPLMaskAccumulator();

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
    const int sub_integration_ntime;
    const int num_subintegrations;

    // Kotekan buffer names
    const std::string pl_mask_name;
    const std::string pl_counts_name;

    // Buffers
    NDArrayRingBuffer<kotekan::uint1x8_t, 5> pl_mask;
    NDArrayBuffer<std::uint64_t, 4> pl_counts;
};

REGISTER_CUDA_COMMAND(cudaPLMaskAccumulator);

cudaPLMaskAccumulator::cudaPLMaskAccumulator(kotekan::Config& config,
                                             const std::string& unique_name,
                                             kotekan::bufferContainer& host_buffers,
                                             cudaDeviceInterface& device, const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "cudaPLMaskAccumulator"),
    // Parameters
    buffer_depth(config.get<int>(unique_name, "buffer_depth")),
    num_times(config.get<int>(unique_name, "num_times")),
    num_frequencies(config.get<int>(unique_name, "num_frequencies")),
    num_polarizations(config.get<int>(unique_name, "num_polarizations")),
    num_dishes(config.get<int>(unique_name, "num_dishes")),
    sub_integration_ntime(config.get<int>(unique_name, "sub_integration_ntime")),
    num_subintegrations(div_noremainder(num_times, sub_integration_ntime)),
    // Buffer names
    pl_mask_name(config.get<std::string>(unique_name, "pl_mask_name")),
    pl_counts_name(config.get<std::string>(unique_name, "pl_counts_name")),
    // Buffers
    pl_mask(pl_mask_name, "pl_mask",
            std::array<std::ptrdiff_t, 5>{buffer_depth * div_noremainder(num_times, 2 * 64),
                                          div_noremainder(num_frequencies, 4), num_polarizations,
                                          div_noremainder(num_dishes, 8), 64 / 8},
            std::array<std::string, 5>{"T2hi64", "F4", "P", "D8", "T2lo64"}, *this),
    pl_counts(pl_counts_name, "pl_counts",
              std::array<std::ptrdiff_t, 4>{num_subintegrations, num_frequencies, num_polarizations,
                                            num_dishes},
              std::array<std::string, 4>{"Tc", "F", "P", "D"}, *this)
//
{
    // For pl_mask_T128_sample_bytes
    if (sub_integration_ntime % 2 != 0)
        FATAL_ERROR("sub_integration_ntime % 2 != 0");

    if (num_times % 128 != 0)
        FATAL_ERROR("num_times % 128 != 0");

    if (num_times % sub_integration_ntime != 0)
        FATAL_ERROR("num_times % sub_integration_ntime != 0");
    
    if ((num_dishes * num_polarizations) % 128 != 0)
        FATAL_ERROR("num_elements (dish x pol) % 128 != 0");

    pl_mask.register_consumer();
    pl_counts.register_producer();

    set_command_type(gpuCommandType::KERNEL);
    set_name("cudaPLMaskAccumulator");
}

cudaPLMaskAccumulator::~cudaPLMaskAccumulator() {}

int cudaPLMaskAccumulator::wait_on_precondition() {
    // Wait for data to be available in input ringbuffers
    DEBUG("Waiting for pl_mask input ringbuffer data for frame {:d}...", gpu_frame_id);
    const std::ptrdiff_t pl_samples_per_frame = num_times / 128;
    const int pl_mask_errcode =
        pl_mask.wait_and_claim_readable([&](const std::ptrdiff_t available_elements) {
            if (available_elements < pl_samples_per_frame)
                return read_descriptor_t{.claimed = 0, .read = 0};
            else
                return read_descriptor_t{.claimed = pl_samples_per_frame,
                                         .read = pl_samples_per_frame};
        });
    if (pl_mask_errcode < 0)
        return pl_mask_errcode;

    DEBUG("Done waiting for pl_mask input ringbuffer data for frame {:d}; will read {:d} elements",
          gpu_frame_id, pl_samples_per_frame);

    return 0;
}

cudaEvent_t cudaPLMaskAccumulator::execute(cudaPipelineState& /*pipestate*/,
                                           const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();
    record_start_event();

    pl_mask.check_metadata();
    pl_counts.set_metadata(pl_mask.get_metadata());

    const auto& pl_counts_meta = pl_counts.get_metadata();
    pl_counts_meta->set_fpga_seq_num(pl_mask.get_read_valid().begin() * 128);
    pl_counts_meta->set_time_downsampling_fpga(
        div_noremainder(pl_counts_meta->get_time_downsampling_fpga(), 128) * sub_integration_ntime);

    const kotekan::uint1x8_t* const pl_mask_memory = pl_mask.get_ndarray().data();
    std::uint64_t* const pl_counts_memory = pl_counts.get_ndarray().data();

    const std::ptrdiff_t Tplsize = pl_mask.get_ndarray().extent(0);
    // Tmin wraps around into actual array index to avoid overflows
    const std::ptrdiff_t Tplmin = pl_mask.get_read_valid().begin() % Tplsize;
    const std::ptrdiff_t Tpl = pl_mask.get_read_valid().size();
    if (Tplmin + Tpl > Tplsize) {
        FATAL_ERROR(
            "Chunk starting at Tplmin={:d} of size Tpl={:d} runs past end of ringbuffer {:s} "
            "of size Tplsize={:d}",
            Tplmin, Tpl, pl_mask.get_buffer_name(), Tplsize);
    }
    assert(Tplmin + Tpl <= Tplsize);

    const std::ptrdiff_t T = Tpl * 128;
    const std::ptrdiff_t Tmin = Tplmin * 128;
    const std::ptrdiff_t Tsize = Tplsize * 128;

    const std::ptrdiff_t F_stride = pl_counts.get_ndarray().stride(1);

    const auto& pl_mask_meta = pl_mask.get_metadata();
    const std::ptrdiff_t Tpl_stride = pl_mask.get_ndarray().stride(0);
    const std::ptrdiff_t Tpl_offset = Tplmin * Tpl_stride;

    DEBUG(
        "Executing s0 kernel on frame {:d}, for T={:d} from Tmin={:d}, reading PL from offset {:d}",
        gpu_frame_id, T, Tmin, Tpl_offset, 0);

    n2k::launch_s0_kernel((ulong*)pl_counts_memory, (const ulong*)(pl_mask_memory + Tpl_offset), T,
                          0, Tsize, num_frequencies, num_dishes * num_polarizations,
                          sub_integration_ntime, F_stride, device.getStream(cuda_stream_id));
#ifdef DEBUGGING
    CHECK_CUDA_ERROR(cudaStreamSynchronize(device.getStream(cuda_stream_id)));
#endif

    return record_end_event();
}

void cudaPLMaskAccumulator::finalize_frame() {
    // Advance the ring buffers
    pl_mask.finish_read();

    cudaCommand::finalize_frame();
}
