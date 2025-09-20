#include <DataType.hpp>
#include <NDArrayBuffer.hpp>
#include <NDArrayRingBuffer.hpp>
#include <cassert>
#include <chordMetadata.hpp>
#include <cstddef>
#include <cstring>
#include <cudaCommand.hpp>
#include <div.hpp>
#include <memory>
#include <metadata.hpp>
#include <n2k.hpp>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

using kotekan::div_noremainder;
using kotekan::round_down;

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
    const int rfi_downsampling_factor;
    const int rfi_num_times;
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
    rfi_downsampling_factor(config.get<int>(unique_name, "rfi_downsampling_factor")),
    rfi_num_times(config.get<int>(unique_name, "rfi_num_times")),
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
                std::array<std::ptrdiff_t, 3>{buffer_depth * div_noremainder(num_times, 8 * 64),
                                              num_frequencies, 64},
                std::array<std::string, 3>{"T8hi64", "F", "T8lo64"}, *this),
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
            const auto rfi_samples_per_data_set = div_noremainder(n2k_samples_per_data_set, 8 * 64);
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

    // NDArray does not yet support poisoning
    // n2k_counts.set_to_poison(0xff);

    // The ringbuffering here is fishy. We should fix the kernel instead.

    // Ensure consistency
    assert(pl_meta->nfreq == rfi_meta->nfreq);
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
    for (int freq = 0; freq < pl_meta->nfreq; ++freq)
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

    n2k::launch_pl_1bit_correlator(n2k_counts_memory, (const ulong*)pl_expanded_mask_memory,
                                   (const uint*)rfi_RFImask_memory, rfimask_fstride,
                                   T,   // number of time samples before correlation
                                   F,   // number of frequency channels
                                   Sds, // number of stations (after downsampling by 8)
                                   Nds, // downsampling factor of counts array, relative to baseband
                                   device.getStream(cuda_stream_id));

    // NDArray does not yet support poisoning
    // n2k_counts.check_for_poison(0xff);

    return record_end_event();
}

void cudaPL1bitCorrelator::finalize_frame() {
    // Advance the ring buffers
    pl_expanded_mask.finish_read();
    rfi_RFImask.finish_read();

    cudaCommand::finalize_frame();
}
