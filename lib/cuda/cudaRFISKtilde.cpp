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

class cudaRFISKtilde : public cudaCommand {
public:
    cudaRFISKtilde(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                   const int instance_num);
    virtual ~cudaRFISKtilde();

    int wait_on_precondition() override;
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame() override;

private:
    // Some terminology:

    //    Quantity    Definition
    //     ---------------------
    //    S012        S statistics
    //    SK          SK statistic
    //    RFImask     RFI mask
    //    ...tilde    summed over feeds
    //    ...bar      downsampled in time

    //    Quantity    Producer
    //     -------------------
    //    S012        s0_kernel, s12_kernel
    //    S012tilde   s012_station_downsample_kernel
    //    S012bar     s012_time_downsample_kernel
    //    SKtilde     skKernel
    //    RFImask     skKernel
    //    SKtildebar  skKernel
    //    SKbar       skKernel

    // Parameters
    const int buffer_depth;
    const int num_times;
    const int num_frequencies;
    const int num_polarizations;
    const int num_dishes;
    const int rfi_downsampling_factor;
    const int rfi_num_times;

    // Kotekan buffer names
    const std::string bf_mask_name;
    const std::string rfi_S012_name;
    const std::string rfi_SKtilde_name;
    const std::string rfi_RFImask_name;

    // Buffers
    NDArrayBuffer<std::int8_t, 2> bf_mask;
    NDArrayRingBuffer<std::uint64_t, 5> rfi_S012;
    NDArrayRingBuffer<float, 3> rfi_SKtilde;
    NDArrayRingBuffer<kotekan::uint1x8_t, 3> rfi_RFImask;

    // Kernels
    const n2k::SkKernel skKernel;
};

REGISTER_CUDA_COMMAND(cudaRFISKtilde);

cudaRFISKtilde::cudaRFISKtilde(kotekan::Config& config, const std::string& unique_name,
                               kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                               const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "cudaRFISKtilde"),
    // Parameters
    buffer_depth(config.get<int>(unique_name, "buffer_depth")),
    num_times(config.get<int>(unique_name, "num_times")),
    num_frequencies(config.get<int>(unique_name, "num_frequencies")),
    num_polarizations(config.get<int>(unique_name, "num_polarizations")),
    num_dishes(config.get<int>(unique_name, "num_dishes")),
    rfi_downsampling_factor(config.get<int>(unique_name, "rfi_downsampling_factor")),
    rfi_num_times(config.get<int>(unique_name, "rfi_num_times")),
    // Buffer names
    bf_mask_name(config.get<std::string>(unique_name, "bf_mask_name")),
    rfi_S012_name(config.get<std::string>(unique_name, "rfi_S012_name")),
    rfi_SKtilde_name(config.get<std::string>(unique_name, "rfi_SKtilde_name")),
    rfi_RFImask_name(config.get<std::string>(unique_name, "rfi_RFImask_name")),
    // Buffers
    bf_mask(bf_mask_name, "bf_mask", std::array<std::ptrdiff_t, 2>{num_polarizations, num_dishes},
            std::array<std::string, 2>{"P", "D"}, *this, buffer_type_t::do_once),
    rfi_S012(rfi_S012_name, "S012",
             std::array<std::ptrdiff_t, 5>{buffer_depth * rfi_num_times, num_frequencies, 3,
                                           num_polarizations, num_dishes},
             std::array<std::string, 5>{"Trfi", "F", "S", "P", "D"}, *this),
    rfi_SKtilde(rfi_SKtilde_name, "SKtilde",
                std::array<std::ptrdiff_t, 3>{buffer_depth * rfi_num_times, num_frequencies, 3},
                std::array<std::string, 3>{"Trfi", "F", "SK"}, *this),
    rfi_RFImask(rfi_RFImask_name, "RFImask",
                std::array<std::ptrdiff_t, 3>{div_noremainder(buffer_depth * num_times, 128),
                                              num_frequencies, 16},
                std::array<std::string, 3>{"T16hi8", "F", "T16lo8"}, *this),
    // Kernels
    skKernel(n2k::SkKernel::Params{
        config.get<double>(unique_name, "rfi_sk_rfimask_sigmas"),
        config.get<double>(unique_name, "rfi_single_feed_min_good_frac"),
        config.get<double>(unique_name, "rfi_feed_averaged_min_good_frac"),
        config.get<double>(unique_name, "rfi_mu_min"),
        config.get<double>(unique_name, "rfi_mu_max"),
        rfi_downsampling_factor,
    })
//
{
    rfi_S012.register_consumer();
    rfi_SKtilde.register_producer();
    rfi_RFImask.register_producer();

    set_command_type(gpuCommandType::KERNEL);
}

cudaRFISKtilde::~cudaRFISKtilde() {}

int cudaRFISKtilde::wait_on_precondition() {
    // Wait for data to be available in input ringbuffers
    DEBUG("Waiting for rfi_S012 input ringbuffer data for frame {:d}...", gpu_frame_id);
    const std::ptrdiff_t rfi_S012_ringbuf = rfi_S012.get_ndarray().extent(0);
    const std::ptrdiff_t rfi_S012_read_max = rfi_S012_ringbuf / 4;
    std::ptrdiff_t rfi_S012_read = -1;
    const int rfi_S012_errcode =
        rfi_S012.wait_and_claim_readable([&](const std::ptrdiff_t available_elements) {
            using std::min;
            rfi_S012_read = min(available_elements, rfi_S012_read_max);
            return read_descriptor_t{.claimed = rfi_S012_read, .read = rfi_S012_read};
        });
    if (rfi_S012_errcode < 0)
        return rfi_S012_errcode;
    DEBUG("Done waiting for rfi_S012 input ringbuffer data for frame {:d}; will read {:d} elements",
          gpu_frame_id, rfi_S012_read);

    DEBUG("Waiting for rfi_SKtilde output ringbuffer space for frame {:d}...", gpu_frame_id);
    const std::ptrdiff_t rfi_Sktilde_written = rfi_S012_read;
    const int rfi_SKtilde_errcode = rfi_SKtilde.wait_for_writable(rfi_Sktilde_written);
    if (rfi_SKtilde_errcode < 0)
        return rfi_SKtilde_errcode;
    DEBUG("Done waiting for rfi_SKtilde output ringbuffer space for frame {:d}; "
          "will write {:d} elements",
          gpu_frame_id, rfi_Sktilde_written);

    DEBUG("Waiting for rfi_RFImask output ringbuffer space for frame {:d}...", gpu_frame_id);
    const std::ptrdiff_t rfi_RFImask_written =
        div_noremainder(rfi_S012_read * rfi_downsampling_factor, 128);
    const int rfi_RFImask_errcode = rfi_RFImask.wait_for_writable(rfi_RFImask_written);
    if (rfi_RFImask_errcode < 0)
        return rfi_RFImask_errcode;
    DEBUG("Done waiting for rfi_RFImask output ringbuffer space for frame {:d}; "
          "will write {:d} elements",
          gpu_frame_id, rfi_RFImask_written);

    return 0;
}

cudaEvent_t cudaRFISKtilde::execute(cudaPipelineState& /*pipestate*/,
                                    const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();

    record_start_event();

    rfi_S012.check_metadata();

    rfi_SKtilde.set_metadata(rfi_S012.get_metadata());
    rfi_RFImask.set_metadata(rfi_S012.get_metadata());

    const std::int8_t* const bf_mask_memory = bf_mask.get_ndarray().data();
    const std::uint64_t* const rfi_S012_memory = rfi_S012.get_ndarray().data();
    float* const rfi_SKtilde_memory = rfi_SKtilde.get_ndarray().data();
    kotekan::uint1x8_t* const rfi_RFImask_memory = rfi_RFImask.get_ndarray().data();

    float* const out_sk_feed_averaged = rfi_SKtilde_memory;
    float* const out_sk_single_feed = nullptr;
    uint* const out_rfimask = (uint*)rfi_RFImask_memory;
    const ulong* const in_S012 = rfi_S012_memory;
    const uint8_t* const in_bf_mask = (const uint8_t*)bf_mask_memory;
    const long rfimask_fstride =
        rfi_RFImask.get_ndarray().get_stride(1) / 4; // NOTE: uint32 stride, not bit stride!
    const long T = rfi_S012.get_read_valid().size();
    const long F = rfi_S012.get_ndarray().get_extent(1);
    const long S = rfi_S012.get_ndarray().get_extent(3)
                   * rfi_S012.get_ndarray().get_extent(4); // Number of stations (= 2 * dishes)
    const long S012_Tmin = rfi_S012.get_read_valid().begin();
    const long S012_Tsize = rfi_S012.get_ndarray().get_extent(0);
    const long sk_feed_averaged_Tmin = rfi_SKtilde.get_read_valid().begin();
    const long sk_feed_averaged_Tsize = rfi_SKtilde.get_ndarray().get_extent(0);
    const long rfimask_T128min = rfi_RFImask.get_read_valid().begin();
    const long rfimask_T128size = rfi_RFImask.get_ndarray().get_extent(0);
    const cudaStream_t stream = device.getStream(cuda_stream_id);
    skKernel.launch(out_sk_feed_averaged, out_sk_single_feed, out_rfimask, in_S012, in_bf_mask,
                    rfimask_fstride, T, F, S, S012_Tmin, S012_Tsize, sk_feed_averaged_Tmin,
                    sk_feed_averaged_Tsize, rfimask_T128min, rfimask_T128size, stream);

    return record_end_event();
}

void cudaRFISKtilde::finalize_frame() {
    // Advance the ring buffers
    rfi_S012.finish_read();
    rfi_SKtilde.finish_write();
    rfi_RFImask.finish_write();

    cudaCommand::finalize_frame();
}
