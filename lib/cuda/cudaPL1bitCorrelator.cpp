#include "Config.hpp"              // for Config
#include "NDArray.hpp"             // for NDArray
#include "bufferContainer.hpp"     // for bufferContainer
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "driver_types.h"          // for cudaEvent_t, CUevent_st, CUstream_st
#include "gpuCommand.hpp"          // for gpuCommandType
#include "kotekanLogging.hpp"      // for DEBUG, ERROR
#include "n2k/pl_kernels.hpp"      // for launch_pl_1bit_correlator

#include "fmt.hpp" // for compile_string_to_view

#include <DataType.hpp>          // for uint1x8_t
#include <NDArrayBuffer.hpp>     // for NDArrayBuffer
#include <NDArrayRingBuffer.hpp> // for NDArrayRingBuffer, extent_t, read_descriptor_t
#include <array>                 // for array
#include <cassert>               // for assert
#include <chordMetadata.hpp>     // for chordMetadata
#include <cstddef>               // for ptrdiff_t
#include <cstdint>               // for int32_t
#include <cudaCommand.hpp>       // for cudaCommand, cudaPipelineState, REGISTER_CUDA_COMMAND
#include <cudaMemsetInt.hpp>     // for cudaMemsetInt
#include <div.hpp>               // for div_noremainder, round_down
#include <functional>            // for function
#include <memory>                // for allocator, shared_ptr, __shared_ptr_access
#include <string>                // for basic_string, string
#include <sys/types.h>           // for uint, ulong
#include <vector>                // for vector

using kotekan::div_noremainder;
using kotekan::round_down;

namespace {} // namespace

/**
 * @class cudaPL1bitCorrelator
 * @brief cudaCommand for the 1-bit correlator generating the N2 counts.
 *
 * @author Erik Schnetter
 *
 * @par GPU Memory
 * @gpu_mem Input expanded PL mask
 *   @gpu_mem_buffer    @c ring
 *   @gpu_mem_quantity  @c pl_mask
 *   @gpu_mem_type      @c uint1x8
 *   @gpu_mem_dim_name  [@c Thi64][@c F][@c P][@c D8][@c Tlo64]
 *   @gpu_mem_shape     [@c num_times / 64][@c num_frequencies][@c num_polarizations]
 *                      [@c num_dishes / 8][@c 64 / 8]
 *   @gpu_mem_metadata  @c chordMetadata
 * @gpu_mem Input RFI mask ringbuffer
 *   @gpu_mem_buffer    @c ring
 *   @gpu_mem_quantity  @c RFImask
 *   @gpu_mem_type      @c uint1x8
 *   @gpu_mem_dim_name  [@c T8hi128][@c F][@c T8lo128]
 *   @gpu_mem_shape     [@c buffer_depth * num_times / 8 / 128][@c num_local_freq][@c 2][@c 128]
 *   @gpu_mem_metadata  @c chordMetadata
 * @gpu_mem Output N2 counts array
 *   @gpu_mem_buffer    @c standard
 *   @gpu_mem_quantity  @c n2k_counts
 *   @gpu_mem_type      @c int32
 *   @gpu_mem_dim_name  [@c Tc][@c F][@c D8Phi][@c D8Plo1[@c D8Plo2]
 *   @gpu_mem_shape     [@c num_subintegrations][@c num_frequencies][@c triangle_num_blocks]
 *                      [@c blocksize][@c blocksize]
 *   @gpu_mem_metadata  @c chordMetadata
 * @conf  buffer_depth           Int.     The number of GPU frames used for pipelining commands.
 * @conf  num_times              Int.     Number of time samples per frame.
 * @conf  num_frequenciees       Int.     Number of frequencies handled by this X-Engine node.
 * @conf  num_polarizations      Int.     Number of polarizations (2)
 * @conf  num_dishes             Int.     Number of dishes
 * @conf  samples_per_data_set   Int.     Number of time samples per Kotekan block.
 * @conf  sub_integration_ntime  Int.     Number of time samples that will be summed into the
 * @conf  pl_expanded_mask_name  String.  Base name for the pl_expanded_mask buffers.
 * @conf  rfi_RFImask_name       String.  Base name for the RFI mask buffers.
 * @conf  n2k_counts_name        String.  Base name for the N2 counts buffers.
 */
class cudaPL1bitCorrelator : public cudaCommand {
public:
    cudaPL1bitCorrelator(kotekan::Config& config, const std::string& unique_name,
                         kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                         const int instance_num);
    virtual ~cudaPL1bitCorrelator();

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
    const int n2k_samples_per_data_set;
    const int n2k_sub_integration_ntime;

    // Kotekan buffer names
    const std::string pl_expanded_mask_name;
    const std::string rfi_RFImask_name;
    const std::string n2k_counts_name;

    // Buffers
    NDArrayRingBuffer<kotekan::uint1x8_t, 5> pl_expanded_mask;
    NDArrayRingBuffer<kotekan::uint1x8_t, 3> rfi_RFImask;
    NDArrayBuffer<std::int32_t, 5> n2k_counts;
};

REGISTER_CUDA_COMMAND(cudaPL1bitCorrelator);

cudaPL1bitCorrelator::cudaPL1bitCorrelator(kotekan::Config& config, const std::string& unique_name,
                                           kotekan::bufferContainer& host_buffers,
                                           cudaDeviceInterface& device, const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "cudaPL1bitCorrelator"),
    // Parameters
    buffer_depth(config.get<int>(unique_name, "buffer_depth")),
    num_times(config.get<int>(unique_name, "num_times")),
    num_frequencies(config.get<int>(unique_name, "num_frequencies")),
    num_polarizations(config.get<int>(unique_name, "num_polarizations")),
    num_dishes(config.get<int>(unique_name, "num_dishes")),
    n2k_samples_per_data_set(config.get<int>(unique_name, "samples_per_data_set")),
    n2k_sub_integration_ntime(config.get<int>(unique_name, "sub_integration_ntime")),
    // Buffer names
    pl_expanded_mask_name(config.get<std::string>(unique_name, "pl_expanded_mask_name")),
    rfi_RFImask_name(config.get<std::string>(unique_name, "rfi_RFImask_name")),
    n2k_counts_name(config.get<std::string>(unique_name, "n2k_counts_name")),
    // Buffers
    pl_expanded_mask(pl_expanded_mask_name, "pl_mask",
                     std::array<std::ptrdiff_t, 5>{buffer_depth * div_noremainder(num_times, 64),
                                                   num_frequencies, num_polarizations,
                                                   div_noremainder(num_dishes, 8), 64 / 8},
                     std::array<std::string, 5>{"Thi64", "F", "P", "D8", "Tlo64"}, *this),
    rfi_RFImask(rfi_RFImask_name, "RFImask",
                std::array<std::ptrdiff_t, 3>{buffer_depth * div_noremainder(num_times, 8 * 128),
                                              num_frequencies, 128},
                std::array<std::string, 3>{"T8hi128", "F", "T8lo128"}, *this),
    n2k_counts([&]() {
        // aka "nt_outer" in n2k.hpp
        const int num_subintegrations =
            div_noremainder(n2k_samples_per_data_set, n2k_sub_integration_ntime);
        const int blocksize = 8;
        const int linear_num_blocks = (num_polarizations * num_dishes / 8 + 1) / blocksize;
        const int triangle_num_blocks = linear_num_blocks * (linear_num_blocks + 1) / 2;
        const std::array<std::ptrdiff_t, 5> n2k_lengths{num_subintegrations, num_frequencies,
                                                        triangle_num_blocks, blocksize, blocksize};
        const std::array<std::string, 5> n2k_dimnames{"Tc", "F", "D8Phi", "D8Plo1", "D8Plo2"};
        return NDArrayBuffer<std::int32_t, 5>(n2k_counts_name, "n2k_counts", n2k_lengths,
                                              n2k_dimnames, *this);
    }())
//
{
    pl_expanded_mask.register_consumer();
    rfi_RFImask.register_consumer();
    n2k_counts.register_producer();

    set_command_type(gpuCommandType::KERNEL);
}

cudaPL1bitCorrelator::~cudaPL1bitCorrelator() {}

int cudaPL1bitCorrelator::wait_on_precondition() {
    // Wait for data to be available in input ringbuffers
    DEBUG("Waiting for pl_expanded_mask input ringbuffer data for frame {:d}...", gpu_frame_id);
    const int pl_expanded_mask_errcode =
        pl_expanded_mask.wait_and_claim_readable([&](const std::ptrdiff_t available_elements) {
            // We measure the expanded pl mask in "coarse" time samples
            const auto pl_samples_per_data_set = div_noremainder(n2k_samples_per_data_set, 64);
            if (available_elements < pl_samples_per_data_set)
                return read_descriptor_t{.claimed = 0, .read = 0};
            else
                return read_descriptor_t{.claimed = pl_samples_per_data_set,
                                         .read = pl_samples_per_data_set};
        });
    if (pl_expanded_mask_errcode < 0)
        return pl_expanded_mask_errcode;
    DEBUG("Done waiting for pl_expanded_mask input ringbuffer data for frame {:d}.", gpu_frame_id);

    DEBUG("Waiting for rfi_RFImask input ringbuffer data for frame {:d}...", gpu_frame_id);
    const int rfi_RFImask_errcode =
        rfi_RFImask.wait_and_claim_readable([&](const std::ptrdiff_t available_elements) {
            // We measure the rfi mask in "coarse" time samples
            const auto rfi_samples_per_data_set =
                div_noremainder(n2k_samples_per_data_set, 8 * 128);
            if (available_elements < rfi_samples_per_data_set)
                return read_descriptor_t{.claimed = 0, .read = 0};
            else
                return read_descriptor_t{.claimed = rfi_samples_per_data_set,
                                         .read = rfi_samples_per_data_set};
        });
    if (rfi_RFImask_errcode < 0)
        return rfi_RFImask_errcode;
    DEBUG("Done waiting for rfi_RFImask input ringbuffer data for frame {:d}.", gpu_frame_id);

    return 0;
}

cudaEvent_t cudaPL1bitCorrelator::execute(cudaPipelineState& /*pipestate*/,
                                          const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();
    record_start_event();

    pl_expanded_mask.check_metadata();
    rfi_RFImask.check_metadata();
    n2k_counts.set_metadata(pl_expanded_mask.get_metadata());

    const std::shared_ptr<const chordMetadata> pl_meta = pl_expanded_mask.get_metadata();
    const std::shared_ptr<const chordMetadata> rfi_meta = rfi_RFImask.get_metadata();
    const std::shared_ptr<chordMetadata> out_meta = n2k_counts.get_metadata();

    // Since we do not use a ring buffer we need to set `meta->sample0_offset`
    // TODO: do this automatically in `NDArrayRingBuffer`
    out_meta->sample0_offset = rfi_RFImask.get_read_valid().begin();

    // The PL mask time_downsampling_factor includes a factor of 64 from
    // the fast time axis which is eaten up by the correlator.
    for (int f = 0; f < out_meta->get_nfreq(); f++) {
        out_meta->time_downsampling_fpga[f] =
            n2k_sub_integration_ntime * div_noremainder(pl_meta->time_downsampling_fpga[f], 64);
    }

    n2k_counts.set_to_poison(0xff);

    // The ringbuffering here is fishy. We should fix the kernel instead.

    // Ensure consistency
    assert(pl_meta->get_nfreq() == rfi_meta->get_nfreq());
    if (!(pl_expanded_mask.get_read_valid().begin() * pl_meta->time_downsampling_fpga[0]
          == rfi_RFImask.get_read_valid().begin() * rfi_meta->time_downsampling_fpga[0])) {
        DEBUG("pl_expanded_mask.get_read_valid().begin()={}",
              pl_expanded_mask.get_read_valid().begin());
        DEBUG("pl_meta->time_downsampling_fpga[0]={}", pl_meta->time_downsampling_fpga[0]);
        DEBUG("rfi_RFImask.get_read_valid().begin()={}", rfi_RFImask.get_read_valid().begin());
        DEBUG("rfi_meta->time_downsampling_fpga[0]={}", rfi_meta->time_downsampling_fpga[0]);
    }
    assert(pl_expanded_mask.get_read_valid().begin() * pl_meta->time_downsampling_fpga[0]
           == rfi_RFImask.get_read_valid().begin() * rfi_meta->time_downsampling_fpga[0]);
    for (int freq = 0; freq < pl_meta->get_nfreq(); ++freq)
        assert(pl_expanded_mask.get_read_valid().begin() * pl_meta->time_downsampling_fpga[freq]
               == rfi_RFImask.get_read_valid().begin() * rfi_meta->time_downsampling_fpga[freq]);

    const std::ptrdiff_t pl_time_offset =
        pl_expanded_mask.get_read_valid().begin() % pl_expanded_mask.get_ndarray().extent(0);
    // Ensure there is no ring-buffer wrap-around
    assert((pl_expanded_mask.get_read_valid().end() - 1) % pl_expanded_mask.get_ndarray().extent(0)
           >= pl_time_offset);
    const kotekan::uint1x8_t* const pl_expanded_mask_memory =
        &pl_expanded_mask.get_ndarray()(pl_time_offset, 0, 0, 0, 0);

    const std::ptrdiff_t rfi_time_offset =
        rfi_RFImask.get_read_valid().begin() % rfi_RFImask.get_ndarray().extent(0);
    assert((rfi_RFImask.get_read_valid().end() - 1) % rfi_RFImask.get_ndarray().extent(0)
           >= rfi_time_offset);
    // Ensure there is no ring-buffer wrap-around
    assert((rfi_RFImask.get_read_valid().end() - 1) % rfi_RFImask.get_ndarray().extent(0)
           >= rfi_time_offset);
    const kotekan::uint1x8_t* const rfi_RFImask_memory =
        &rfi_RFImask.get_ndarray()(rfi_time_offset, 0, 0);

    std::int32_t* const n2k_counts_memory = n2k_counts.get_ndarray().data();

    // This is a "fake" stride, it just needs to be large enough to linearize array indices without
    // overlapping
    const int rfimask_fstride = n2k_samples_per_data_set;
    const int T = n2k_samples_per_data_set;
    const int F = num_frequencies;
    const int Sds = num_dishes / 8 * num_polarizations;
    const int Nds = n2k_sub_integration_ntime;

    if (Sds == 16 || Sds == 128) {
        // These cases are implemented in n2k
        n2k::launch_pl_1bit_correlator(
            n2k_counts_memory, (const ulong*)pl_expanded_mask_memory,
            (const uint*)rfi_RFImask_memory, rfimask_fstride,
            T,   // number of time samples before correlation
            F,   // number of frequency channels
            Sds, // number of stations (after downsampling by 8)
            Nds, // downsampling factor of counts array, relative to baseband
            device.getStream(cuda_stream_id));
    } else {
        // These cases are not yet implemented in n2k. Pretend that there is no packet loss.
        ERROR("The 1-bit correlator calculating the n2k counts is not yet implemented for {:d} "
              "dishes. Pretending there was no packet loss. The n2k counts will be wrong if there "
              "was packet loss.",
              num_dishes);
        cudaMemsetInt(n2k_counts_memory, Nds, n2k_counts.get_ndarray().size());
    }

    n2k_counts.check_for_poison(0xff);

    return record_end_event();
}

void cudaPL1bitCorrelator::finalize_frame() {
    // Advance the ring buffers
    pl_expanded_mask.finish_read();
    rfi_RFImask.finish_read();

    cudaCommand::finalize_frame();
}
