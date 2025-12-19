#include "Config.hpp"              // for Config
#include "NDArray.hpp"             // for NDArray
#include "bufferContainer.hpp"     // for bufferContainer
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "driver_types.h"          // for cudaEvent_t, CUevent_st, CUstream_st
#include "gpuCommand.hpp"          // for gpuCommandType
#include "kotekanLogging.hpp"      // for DEBUG, FATAL_ERROR
#include "n2k/pl_kernels.hpp"      // for launch_pl_mask_expander

#include "fmt/format.h" // for compile_string_to_view

#include <DataType.hpp>          // for uint1x8_t
#include <NDArrayRingBuffer.hpp> // for NDArrayRingBuffer, extent_t, read_descriptor_t
#include <algorithm>             // for fill_n, min
#include <array>                 // for array
#include <cassert>               // for assert
#include <chordMetadata.hpp>     // for chordMetadata
#include <cstddef>               // for ptrdiff_t
#include <cudaCommand.hpp>       // for cudaCommand, cudaPipelineState, REGISTER_CUDA_COMMAND
#include <div.hpp>               // for div_noremainder, round_down
#include <functional>            // for function
#include <memory>                // for allocator, shared_ptr, __shared_ptr_access
#include <stdint.h>              // for int64_t
#include <string>                // for basic_string, string
#include <sys/types.h>           // for ulong
#include <vector>                // for vector

using kotekan::div_noremainder;
using kotekan::round_down;

/**
 * @class cudaPLMaskExpander
 * @brief cudaCommand for expanding the packet loss (PL) mask.
 *
 * @author Erik Schnetter
 *
 * @par GPU Memory
 * @gpu_mem Input PL mask
 *   @gpu_mem_buffer    @c ring
 *   @gpu_mem_quantity  @c pl_mask
 *   @gpu_mem_type      @c uint1x8
 *   @gpu_mem_dim_name  [@c T2hi64][@c F4][@c P][@c D8][@c T2lo64]
 *   @gpu_mem_shape     [@c buffer_depth * num_times / 2 / 64][@c num_frequencies / 4]
 *                      [@c num_polarizations][@c num_dishes / 8][@c 64 / 8]
 *   @gpu_mem_metadata  @c chordMetadata
 * @gpu_mem Output expanded PL mask
 *   @gpu_mem_buffer    @c ring
 *   @gpu_mem_quantity  @c pl_mask
 *   @gpu_mem_type      @c uint1x8
 *   @gpu_mem_dim_name  [@c Thi64][@c F][@c P][@c D8][@c Tlo64]
 *   @gpu_mem_shape     [@c buffer_depth * num_times / 64][@c num_frequencies][@c num_polarizations]
 *                      [@c num_dishes / 8][@c 64 / 8]
 *   @gpu_mem_metadata  @c chordMetadata
 * @conf  buffer_depth           Int.     The number of GPU frames used for pipelining commands.
 * @conf  num_times              Int.     Number of time samples per frame.
 * @conf  num_frequenciees       Int.     Number of frequencies handled by this X-Engine node.
 * @conf  num_polarizations      Int.     Number of polarizations (2)
 * @conf  num_dishes             Int.     Number of dishes
 * @conf  pl_mask_name           String.  Base name for the pl_mask buffers.
 * @conf  pl_expanded_mask_name  String.  Base name for the pl_expanded_mask buffers.
 */
class cudaPLMaskExpander : public cudaCommand {
public:
    cudaPLMaskExpander(kotekan::Config& config, const std::string& unique_name,
                       kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                       const int instance_num);
    virtual ~cudaPLMaskExpander();

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
    const std::string pl_mask_name;
    const std::string pl_expanded_mask_name;

    // Buffers
    NDArrayRingBuffer<kotekan::uint1x8_t, 5> pl_mask;
    NDArrayRingBuffer<kotekan::uint1x8_t, 5> pl_expanded_mask;
};

REGISTER_CUDA_COMMAND(cudaPLMaskExpander);

cudaPLMaskExpander::cudaPLMaskExpander(kotekan::Config& config, const std::string& unique_name,
                                       kotekan::bufferContainer& host_buffers,
                                       cudaDeviceInterface& device, const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "cudaPLMaskExpander"),
    // Parameters
    buffer_depth(config.get<int>(unique_name, "buffer_depth")),
    num_times(config.get<int>(unique_name, "num_times")),
    num_frequencies(config.get<int>(unique_name, "num_frequencies")),
    num_polarizations(config.get<int>(unique_name, "num_polarizations")),
    num_dishes(config.get<int>(unique_name, "num_dishes")),
    // Buffer names
    pl_mask_name(config.get<std::string>(unique_name, "pl_mask_name")),
    pl_expanded_mask_name(config.get<std::string>(unique_name, "pl_expanded_mask_name")),
    // Buffers
    pl_mask(pl_mask_name, "pl_mask",
            std::array<std::ptrdiff_t, 5>{buffer_depth * div_noremainder(num_times, 2 * 64),
                                          div_noremainder(num_frequencies, 4), num_polarizations,
                                          div_noremainder(num_dishes, 8), 64 / 8},
            std::array<std::string, 5>{"T2hi64", "F4", "P", "D8", "T2lo64"}, *this),
    pl_expanded_mask(pl_expanded_mask_name, "pl_mask",
                     std::array<std::ptrdiff_t, 5>{buffer_depth * div_noremainder(num_times, 64),
                                                   num_frequencies, num_polarizations,
                                                   div_noremainder(num_dishes, 8), 64 / 8},
                     std::array<std::string, 5>{"Thi64", "F", "P", "D8", "Tlo64"}, *this)
//
{
    // For pl_mask_T128_sample_bytes
    if (num_frequencies % 4 != 0)
        FATAL_ERROR("num_frequencies % 4 != 0");

    pl_mask.register_consumer();
    pl_expanded_mask.register_producer();

    set_command_type(gpuCommandType::KERNEL);
}

cudaPLMaskExpander::~cudaPLMaskExpander() {}

int cudaPLMaskExpander::wait_on_precondition() {
    // Wait for data to be available in input ringbuffers
    DEBUG("Waiting for pl_mask input ringbuffer data for frame {:d}...", gpu_frame_id);
    const std::ptrdiff_t pl_mask_ringbuf = pl_mask.get_ndarray().extent(0);
    const std::ptrdiff_t pl_mask_read_max = pl_mask_ringbuf / 4;
    std::ptrdiff_t pl_mask_read = -1;
    const int pl_mask_errcode =
        pl_mask.wait_and_claim_readable([&](const std::ptrdiff_t available_elements) {
            using std::min;
            pl_mask_read = min(available_elements, pl_mask_read_max);
            return read_descriptor_t{.claimed = pl_mask_read, .read = pl_mask_read};
        });
    if (pl_mask_errcode < 0)
        return pl_mask_errcode;
    DEBUG("Done waiting for pl_mask input ringbuffer data for frame {:d}; will read {:d} elements",
          gpu_frame_id, pl_mask_read);

    DEBUG("Waiting for pl_expanded_mask output ringbuffer space for frame {:d}...", gpu_frame_id);
    const std::ptrdiff_t pl_expanded_mask_written = pl_mask_read * 2;
    const int pl_expanded_mask_errcode =
        pl_expanded_mask.wait_for_writable(pl_expanded_mask_written);
    if (pl_expanded_mask_errcode < 0)
        return pl_expanded_mask_errcode;
    DEBUG("Done waiting for pl_expanded_mask output ringbuffer data for frame {:d}; will write "
          "{:d} elements",
          gpu_frame_id, pl_expanded_mask_written);

    return 0;
}

cudaEvent_t cudaPLMaskExpander::execute(cudaPipelineState& /*pipestate*/,
                                        const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();
    record_start_event();

    pl_mask.check_metadata();

    pl_expanded_mask.set_metadata(pl_mask.get_metadata());
    const auto& pl_mask_meta = pl_mask.get_metadata();
    const auto& pl_expanded_mask_meta = pl_expanded_mask.get_metadata();
    assert(pl_expanded_mask_meta->get_nfreq() >= 0);

    const std::vector<int> in_time_downsampling_fpga_per_frequency =
        pl_mask_meta->get_time_downsampling_fpga_per_frequency();
    const std::vector<int64_t> in_half_fpga_sample0 = pl_mask_meta->get_half_fpga_sample0();
    std::vector<int> out_time_downsampling_fpga_per_frequency(pl_expanded_mask_meta->get_nfreq());
    std::vector<int64_t> out_half_fpga_sample0(pl_expanded_mask_meta->get_nfreq());

    for (int freq = 0; freq < pl_expanded_mask_meta->get_nfreq(); ++freq) {
        // We would do this if we could start out with 1/4 but we cannot
        // pl_expanded_mask_meta->freq_upchan_factor[freq] *= 4;
        out_time_downsampling_fpga_per_frequency[freq] =
            div_noremainder(in_time_downsampling_fpga_per_frequency[freq], 2);
        out_half_fpga_sample0[freq] = in_half_fpga_sample0[freq]
                                      + out_time_downsampling_fpga_per_frequency[freq]
                                      - in_time_downsampling_fpga_per_frequency[freq];
    }
    pl_expanded_mask_meta->set_time_downsampling_fpga_per_frequency(
        out_time_downsampling_fpga_per_frequency);
    pl_expanded_mask_meta->set_time_downsampling_fpga(
        div_noremainder(pl_mask_meta->get_time_downsampling_fpga(), 2));
    pl_expanded_mask_meta->set_half_fpga_sample0(out_half_fpga_sample0);

    // There is no poison value
    // pl_expanded_mask.set_to_poison(0xff);

    kotekan::uint1x8_t* const pl_mask_memory = pl_mask.get_ndarray().data();
    const kotekan::uint1x8_t* const pl_expanded_mask_memory = pl_expanded_mask.get_ndarray().data();

    const std::ptrdiff_t Tsize_in = pl_mask.get_ndarray().extent(0);
    const std::ptrdiff_t Tmin_in = pl_mask.get_read_valid().begin();
    const std::ptrdiff_t Tsize_out = pl_expanded_mask.get_ndarray().extent(0);
    const std::ptrdiff_t Tmin_out = pl_expanded_mask.get_write_valid().begin();
    const std::ptrdiff_t Tout = pl_expanded_mask.get_write_valid().size() * 64;

    n2k::launch_pl_mask_expander((ulong*)pl_expanded_mask_memory, (const ulong*)pl_mask_memory,
                                 Tmin_in, Tsize_in, Tmin_out, Tsize_out, Tout, num_frequencies,
                                 (num_dishes / 8) * num_polarizations,
                                 device.getStream(cuda_stream_id));

    // There is no poison value
    // pl_expanded_mask.check_for_poison(0xff);

    return record_end_event();
}

void cudaPLMaskExpander::finalize_frame() {
    // Advance the ring buffers
    pl_mask.finish_read();
    pl_expanded_mask.finish_write();

    cudaCommand::finalize_frame();
}
