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

    // Kotekan buffer names
    const std::string pl_expanded_mask_name;
    const std::string rfi_RFImask_name;
    const std::string n2k_counts_name;

    // Buffers
    NDArrayRingBuffer<kotekan::uint1x8_t, 5> pl_expanded_mask;
    NDArrayRingBuffer<kotekan::uint1x8_t, 3> rfi_RFImask;
    NDArrayBuffer<std::int32_t, 6> n2k_counts;
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
    // Buffer names
    pl_expanded_mask_name(config.get<std::string>(unique_name, "pl_expanded_mask_name")),
    rfi_RFImask_name(config.get<std::string>(unique_name, "rfi_RFImask_name")),
    n2k_counts_name(config.get<std::string>(unique_name, "n2k_counts_name")),
    // Buffers
    pl_expanded_mask(
        pl_expanded_mask_name, "pl_expanded_mask",
        std::array<std::ptrdiff_t, 5>{buffer_depth * div_noremainder(num_times, 2 * 64),
                                      num_frequencies / 4, num_polarizations, num_dishes /, 64 / 8},
        std::array<std::string, 5>{"T2hi64", "F4", "P", "D8", "T2lo64"}, *this),
    rfi_RFImask(rfi_RFImask_name, "RFImask",
                std::array<std::ptrdiff_t, 3>{div_noremainder(buffer_depth * num_times, 128),
                                              num_frequencies, 128 / 8},
                std::array<std::string, 3>{"T8hi16", "F", "T8lo16"}, *this),
    n2k_counts([&]() {
        // aka "nt_outer" in n2k.hpp
        const int num_subintegrations =
            div_noremainder(_samples_per_data_set, _sub_integration_ntime);
        const std::array<std::ptrdiff_t, 6> n2k_lengths{
            num_subintegrations, num_frequencies,
            num_polarizations,   div_noremainder(num_dishes, 8),
            num_polarizations,   div_noremainder(num_dishes, 8)};
        return NDArrayBuffer<std::int32_t, 6>(
            n2k_counts_name, "n2k_counts", n2k_lengths,
            std::array<std::string, 6>{"T", "F", "P1", "D81", "P2", "D82"}, *this);
    }()),
//
{
    pl_expanded_mask.register_consumer();
    rfi_RFImask.register_consumer();
    n2k_counts.register_producer();

    set_command_type(gpuCommandType::KERNEL);
}

cudaPL1bitCorrelator::~cudaPL1bitCorrelator() {}

int cudaPL1bitCorrelator::wait_on_precondition() {
#error "NEED rfi_num_times DATA"

    // Wait for data to be available in input ringbuffers
    DEBUG("Waiting for pl_expanded_mask input ringbuffer data for frame {:d}...", gpu_frame_id);
    const std::ptrdiff_t pl_expanded_mask_ringbuf = pl_expanded_mask.get_ndarray().extent(0);
    const std::ptrdiff_t pl_expanded_mask_read_max = pl_expanded_mask_ringbuf / 4;
    std::ptrdiff_t pl_expanded_mask_read = -1;
    const int pl_expanded_mask_errcode =
        pl_expanded_mask.wait_and_claim_readable([&](const std::ptrdiff_t available_elements) {
            // We are downsampling, so we need to process in batches
            // that have a multiple `rfi_downsampling_factor` elements
            using std::min;
            pl_expanded_mask_read =
                round_down(min(available_elements, pl_expanded_mask_read_max), 2);
            return read_descriptor_t{.claimed = pl_expanded_mask_read,
                                     .read = pl_expanded_mask_read};
        });
    if (pl_expanded_mask_errcode < 0)
        return pl_expanded_mask_errcode;
    DEBUG("Done waiting for pl_expanded_mask input ringbuffer data for frame {:d}; will read {:d} "
          "elements",
          gpu_frame_id, pl_expanded_mask_read);

    DEBUG("Waiting for rfi_RFImask input ringbuffer data for frame {:d}...", gpu_frame_id);
    // pl_expanded_mask counts in T64, rfi_RFImask counts in T1024
    const std::ptrdiff_t rfi_RFImask_read = div_noremainder(pl_expanded_mask_read, 16);
    const int rfi_RFImask_errcode =
        rfi_RFImask.wait_and_claim_readable([&](const std::ptrdiff_t available_elements) {
            // Process the same number of elements as from `pl_expanded_mask`
            if (available_elements < rfi_RFImask_read)
                return read_descriptor_t{.claimed = 0, .read = 0};
            else
                return read_descriptor_t{.claimed = rfi_RFImask_read, .read = rfi_RFImask_read};
        });
    if (rfi_RFImask_errcode < 0)
        return rfi_RFImask_errcode;
    DEBUG("Done waiting for rfi_RFImask input ringbuffer data for frame {:d}; will read {:d} "
          "elements",
          gpu_frame_id, rfi_RFImask_read);

    return 0;
}

cudaEvent_t cudaPL1bitCorrelator::execute(cudaPipelineState& /*pipestate*/,
                                          const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();
    record_start_event();

    pl_expanded_mask.check_metadata();
    rfi_RFImask.check_metadata();

    n2k_counts.set_metadata(pl_expanded_mask.get_metadata());
    const auto& n2k_counts_meta = rfi_S012.get_metadata();
    assert(n2k_counts_meta->nfreq >= 0);
    for (int freq = 0; freq < n2k_counts_meta->nfreq; ++freq) {
        n2k_counts_meta->freq_upchan_factor[freq] =
            div_noremainder(n2k_counts_meta->freq_upchan_factor[freq], 64)
            * rfi_downsampling_factor;
        n2k_counts_meta->time_downsampling_fpga[freq] =
            div_noremainder(n2k_counts_meta->time_downsampling_fpga[freq], 64)
            * rfi_downsampling_factor;
    }

    // NDArray does not yet support poisoning
    // n2k_counts.set_to_poison(0xff);

    // Ensure consistency
    assert(pl_expanded_mask.get_read_valid().begin() * 2 == rfi_RFImask.get_read_valid().begin());

#error "NEED TO CHECK THE RFI DOWNSAMPLING AND THE N2K DOWNSAMPLING THEY NEED TO AGREE"
#error "THE RINGBUFFERING IS FISHY HERE FIX THE KERNEL"

    const std::ptrdiff_t pl_time_offset =
        pl_expanded_mask.get_read_valid().begin() % pl_expanded_mask..get_ndarray().extent(0);
    const std::ptrdiff_t rfi_time_offset =
        rfi_RFImask.get_read_valid().begin() % rfi_RFImask..get_ndarray().extent(0);

    // Ensure there is no ring-buffer wrap-around
    assert((pl_expanded_mask.get_read_valid().end() - 1) % pl_expanded_mask.get_ndarray().extent(0)
           >= pl_time_offset);
    assert((rfi_RFImask.get_read_valid().end() - 1) % rfi_RFImask.get_ndarray().extent(0)
           >= rfi_time_offset);

    const kotekan::uint1x8_t* const pl_expanded_mask_memory =
        &pl_expanded_mask.get_ndarray()(pl_time_offset, 0, 0);
    const kotekan::uint1x8_t* const rfi_RFImask_memory =
        &rfi_rfiMask.get_ndarray()(rfi_time_offset, 0, 0);
    std::int32_t* const n2k_counts_memory = n2k_counts.get_ndarray().data();

#error "THIS IS ALL MESSED UP NEED TO FIX INDEXING FOR RFIMASK"
    const int rfimask_fstride = rfi_RFImask.get_ndarray().stride(1);
    const int T = div_noremainder(_samples_per_data_set, _sub_integration_ntime);
    const int F = num_frequencies;
    const int Sds = num_dishes / 8 * num_polarizations;
    const int Nds = n2k_samples_per_data_set;

    n2k::launch_pl_1bit_correlator(n2k_counts_memory, pl_expanded_mask_memory, rfi_RFImask_memory,
                                   rfimask_fstride,
                                   T,   // number of time samples before correlation
                                   F,   // number of frequency channels
                                   Sds, // number of stations (after downsampling by 8)
                                   Nds, // downsampling factor of counts array, relative to baseband
                                   device.getStream(cuda_stream_id));

    // NDArray does not yet support poisoning
    // rfi_S012.check_for_poison(0xff);

    return record_end_event();
}

void cudaPL1bitCorrelator::finalize_frame() {
    // Advance the ring buffers
    pl_expanded_mask.finish_read();
    rfi_RFImask.finish_read();

    cudaCommand::finalize_frame();
}
