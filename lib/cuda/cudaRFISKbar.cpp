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

class cudaRFISKbar : public cudaCommand {
public:
    cudaRFISKbar(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                 const int instance_num);
    virtual ~cudaRFISKbar();

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
    //    SKbar       skKernel
    //    SKbartilde  skKernel

    // Parameters
    const int buffer_depth;
    const int num_frequencies;
    const int num_polarizations;
    const int num_dishes;
    const int rfi_downsampling_factor;
    const int rfi_second_downsampling_factor;
    const int rfi_num_times_bar;

    // Kotekan buffer names
    const std::string bf_mask_name;
    const std::string rfi_S012bar_name;
    const std::string rfi_SKbar_name;
    const std::string rfi_SKbartilde_name;

    // Buffers
    NDArrayBuffer<std::int8_t, 2> bf_mask;
    NDArrayRingBuffer<std::uint64_t, 5> rfi_S012bar;
    NDArrayRingBuffer<float, 5> rfi_SKbar;
    NDArrayRingBuffer<float, 3> rfi_SKbartilde;

    // Kernels
    const n2k::SkKernel skKernel;
};

REGISTER_CUDA_COMMAND(cudaRFISKbar);

cudaRFISKbar::cudaRFISKbar(kotekan::Config& config, const std::string& unique_name,
                           kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                           const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "cudaRFISKbar"),
    // Parameters
    buffer_depth(config.get<int>(unique_name, "buffer_depth")),
    num_frequencies(config.get<int>(unique_name, "num_frequencies")),
    num_polarizations(config.get<int>(unique_name, "num_polarizations")),
    num_dishes(config.get<int>(unique_name, "num_dishes")),
    rfi_downsampling_factor(config.get<int>(unique_name, "rfi_downsampling_factor")),
    rfi_second_downsampling_factor(config.get<int>(unique_name, "rfi_second_downsampling_factor")),
    rfi_num_times_bar(config.get<int>(unique_name, "rfi_num_times_bar")),
    // Buffer names
    bf_mask_name(config.get<std::string>(unique_name, "bf_mask_name")),
    rfi_S012bar_name(config.get<std::string>(unique_name, "rfi_S012bar_name")),
    rfi_SKbar_name(config.get<std::string>(unique_name, "rfi_SKbar_name")),
    rfi_SKbartilde_name(config.get<std::string>(unique_name, "rfi_SKbartilde_name")),
    // Buffers
    bf_mask(bf_mask_name, "bf_mask", std::array<std::ptrdiff_t, 2>{num_polarizations, num_dishes},
            std::array<std::string, 2>{"P", "D"}, *this, buffer_type_t::do_once),
    rfi_S012bar(rfi_S012bar_name, "S012bar",
                std::array<std::ptrdiff_t, 5>{buffer_depth * rfi_num_times_bar, num_frequencies, 3,
                                              num_polarizations, num_dishes},
                std::array<std::string, 5>{"Trfibar", "F", "S", "P", "D"}, *this),
    rfi_SKbar(rfi_SKbar_name, "SKbar",
              std::array<std::ptrdiff_t, 5>{buffer_depth * rfi_num_times_bar, num_frequencies, 3,
                                            num_polarizations, num_dishes},
              std::array<std::string, 5>{"Trfibar", "F", "SK", "P", "D"}, *this),
    rfi_SKbartilde(
        rfi_SKbartilde_name, "SKbartilde",
        std::array<std::ptrdiff_t, 3>{buffer_depth * rfi_num_times_bar, num_frequencies, 3},
        std::array<std::string, 3>{"Trfibar", "F", "SK"}, *this),
    // Kernels
    skKernel(n2k::SkKernel::Params{
        config.get<double>(unique_name, "rfi_sk_rfimask_sigmas"),
        config.get<double>(unique_name, "rfi_single_feed_min_good_frac"),
        config.get<double>(unique_name, "rfi_feed_averaged_min_good_frac"),
        config.get<double>(unique_name, "rfi_mu_min"),
        config.get<double>(unique_name, "rfi_mu_max"),
        rfi_downsampling_factor * rfi_second_downsampling_factor,
    })
//
{
    rfi_S012bar.register_consumer();
    rfi_SKbar.register_producer();
    rfi_SKbartilde.register_producer();

    set_command_type(gpuCommandType::KERNEL);
}

cudaRFISKbar::~cudaRFISKbar() {}

int cudaRFISKbar::wait_on_precondition() {
    // Wait for data to be available in input ringbuffers
    DEBUG("Waiting for rfi_S012bar input ringbuffer data for frame {:d}...", gpu_frame_id);
    const std::ptrdiff_t rfi_S012bar_ringbuf = rfi_S012bar.get_ndarray().extent(0);
    const std::ptrdiff_t rfi_S012bar_read_max = rfi_S012bar_ringbuf / 4;
    std::ptrdiff_t rfi_S012bar_read = -1;
    const int rfi_S012bar_errcode =
        rfi_S012bar.wait_and_claim_readable([&](const std::ptrdiff_t available_elements) {
            using std::min;
            rfi_S012bar_read = min(available_elements, rfi_S012bar_read_max);
            return read_descriptor_t{.claimed = rfi_S012bar_read, .read = rfi_S012bar_read};
        });
    if (rfi_S012bar_errcode < 0)
        return rfi_S012bar_errcode;
    DEBUG("Done waiting for rfi_S012bar input ringbuffer data for frame {:d}; will read {:d} "
          "elements",
          gpu_frame_id, rfi_S012bar_read);

    DEBUG("Waiting for rfi_SKbar output ringbuffer space for frame {:d}...", gpu_frame_id);
    const std::ptrdiff_t rfi_SKbar_written = rfi_S012bar_read;
    const int rfi_SKbar_errcode = rfi_SKbar.wait_for_writable(rfi_SKbar_written);
    if (rfi_SKbar_errcode < 0)
        return rfi_SKbar_errcode;
    DEBUG("Done waiting for rfi_SKbar output ringbuffer space for frame {:d}; "
          "will write {:d} elements",
          gpu_frame_id, rfi_SKbar_written);

    DEBUG("Waiting for rfi_SKbartilde output ringbuffer space for frame {:d}...", gpu_frame_id);
    const std::ptrdiff_t rfi_SKbartilde_written = rfi_S012bar_read;
    const int rfi_SKbartilde_errcode = rfi_SKbartilde.wait_for_writable(rfi_SKbartilde_written);
    if (rfi_SKbartilde_errcode < 0)
        return rfi_SKbartilde_errcode;
    DEBUG("Done waiting for rfi_SKbartilde output ringbuffer space for frame {:d}; "
          "will write {:d} elements",
          gpu_frame_id, rfi_SKbartilde_written);

    return 0;
}

cudaEvent_t cudaRFISKbar::execute(cudaPipelineState& /*pipestate*/,
                                  const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();
    record_start_event();

    rfi_S012bar.check_metadata();

    rfi_SKbar.set_metadata(rfi_S012bar.get_metadata());
    rfi_SKbartilde.set_metadata(rfi_S012bar.get_metadata());

    rfi_SKbar.set_to_poison(0xff);
    rfi_SKbartilde.set_to_poison(0xff);

    const std::int8_t* const bf_mask_memory = bf_mask.get_ndarray().data();
    const std::uint64_t* const rfi_S012bar_memory = rfi_S012bar.get_ndarray().data();
    float* const rfi_SKbar_memory = rfi_SKbar.get_ndarray().data();
    float* const rfi_SKbartilde_memory = rfi_SKbartilde.get_ndarray().data();

    float* const out_sk_feed_averaged = rfi_SKbartilde_memory;
    float* const out_sk_single_feed = rfi_SKbar_memory;
    uint* const out_rfimask = nullptr;
    const ulong* const in_S012 = rfi_S012bar_memory;
    const uint8_t* const in_bf_mask = (const uint8_t*)bf_mask_memory;
    const long T = rfi_S012bar.get_read_valid().size();
    const long F = rfi_S012bar.get_ndarray().get_extent(1);
    const long S = rfi_S012bar.get_ndarray().get_extent(3)
                   * rfi_S012bar.get_ndarray().get_extent(4); // Number of stations (= 2 * dishes)
    const long S012_Tmin = rfi_S012bar.get_read_valid().begin();
    const long S012_Tsize = rfi_S012bar.get_ndarray().get_extent(0);
    const long sk_feed_averaged_Tmin = rfi_SKbartilde.get_write_valid().begin();
    const long sk_feed_averaged_Tsize = rfi_SKbartilde.get_ndarray().get_extent(0);
    const long sk_single_feed_Tmin = rfi_SKbar.get_write_valid().begin();
    const long sk_single_feed_Tsize = rfi_SKbar.get_ndarray().get_extent(0);
    const long rfimask_T128min = 0;
    const long rfimask_T128size = 0;
    const cudaStream_t stream = device.getStream(cuda_stream_id);
    skKernel.launch(out_sk_feed_averaged, out_sk_single_feed, out_rfimask, in_S012, in_bf_mask, T,
                    F, S, S012_Tmin, S012_Tsize, sk_feed_averaged_Tmin, sk_feed_averaged_Tsize,
                    sk_single_feed_Tmin, sk_single_feed_Tsize, rfimask_T128min, rfimask_T128size,
                    stream);

    rfi_SKbar.check_for_poison(0xff);
    rfi_SKbartilde.check_for_poison(0xff);

    return record_end_event();
}

void cudaRFISKbar::finalize_frame() {
    // Advance the ring buffers
    rfi_S012bar.finish_read();
    rfi_SKbar.finish_write();
    rfi_SKbartilde.finish_write();

    cudaCommand::finalize_frame();
}
