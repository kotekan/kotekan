#include "Config.hpp"              // for Config
#include "NDArray.hpp"             // for NDArray
#include "NDArrayBuffer.hpp"       // for NDArrayBuffer, buffer_type_t
#include "NDArrayRingBuffer.hpp"   // for NDArrayRingBuffer, extent_t, read_descriptor_t
#include "bufferContainer.hpp"     // for bufferContainer
#include "cudaCommand.hpp"         // for cudaCommand, cudaPipelineState, REGISTER_CUDA_COMMAND
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "cudaUtils.hpp"           // for CHECK_CUDA_ERROR
#include "div.hpp"                 // for div_noremainder, round_down
#include "gpuCommand.hpp"          // for gpuCommandType
#include "kotekanLogging.hpp"      // for DEBUG
#include "n2k/rfi_kernels.hpp"     // for SkKernel

#include "fmt.hpp" // for compile_string_to_view

#include <algorithm>          // for min
#include <array>              // for array
#include <cstddef>            // for ptrdiff_t
#include <cstdint>            // for int8_t, uint64_t, uint8_t
#include <cuda_runtime_api.h> // for cudaStreamSynchronize
#include <driver_types.h>     // for cudaEvent_t, CUstream_st, CUevent_st, cudaStream_t
#include <functional>         // for function
#include <memory>             // for allocator, shared_ptr
#include <string>             // for basic_string, string
#include <sys/types.h>        // for uint, ulong
#include <vector>             // for vector

using kotekan::div_noremainder;
using kotekan::round_down;
using kotekan::round_up;

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
    const std::int64_t bf_mask_lifetime_in_samples;
    const int rfi_downsampling_factor;
    const int rfi_second_downsampling_factor;
    const int rfi_num_times_bar;
    const bool poison_buffers;

    const std::int64_t rfi_samples_per_bf_sample;

    // Kotekan buffer names
    const std::string bf_mask_name;
    const std::string rfi_S012bar_name;
    const std::string rfi_SKbar_name;
    const std::string rfi_SKbartilde_name;

    // Buffers
    NDArrayRingBuffer<std::int8_t, 3> bf_mask;
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
    bf_mask_lifetime_in_samples(
        config.get<std::int64_t>(unique_name, "bf_mask_lifetime_in_samples")),
    rfi_downsampling_factor(config.get<int>(unique_name, "rfi_downsampling_factor")),
    rfi_second_downsampling_factor(config.get<int>(unique_name, "rfi_second_downsampling_factor")),
    rfi_num_times_bar(config.get<int>(unique_name, "rfi_num_times_bar")),
    poison_buffers(config.get_default<bool>(unique_name, "poison_buffers", false)),
    rfi_samples_per_bf_sample(div_noremainder(
        bf_mask_lifetime_in_samples, rfi_downsampling_factor * rfi_second_downsampling_factor)),
    // Buffer names
    bf_mask_name(config.get<std::string>(unique_name, "bf_mask_name")),
    rfi_S012bar_name(config.get<std::string>(unique_name, "rfi_S012bar_name")),
    rfi_SKbar_name(config.get<std::string>(unique_name, "rfi_SKbar_name")),
    rfi_SKbartilde_name(config.get<std::string>(unique_name, "rfi_SKbartilde_name")),
    // Buffers
    bf_mask(bf_mask_name, "bf_mask",
            std::array<std::ptrdiff_t, 3>{buffer_depth * 1, num_polarizations, num_dishes},
            std::array<std::string, 3>{"Tbf", "P", "D"},
            std::array<std::ptrdiff_t, 3>{bf_mask_lifetime_in_samples, 1, 1}, *this),
    rfi_S012bar(rfi_S012bar_name, "S012bar",
                std::array<std::ptrdiff_t, 5>{buffer_depth * rfi_num_times_bar, num_frequencies, 3,
                                              num_polarizations, num_dishes},
                std::array<std::string, 5>{"Trfibar", "F", "S", "P", "D"},
                std::array<std::ptrdiff_t, 5>{
                    rfi_downsampling_factor * rfi_second_downsampling_factor, 1, 1, 1, 1},
                *this),
    rfi_SKbar(rfi_SKbar_name, "SKbar",
              std::array<std::ptrdiff_t, 5>{buffer_depth * rfi_num_times_bar, num_frequencies, 3,
                                            num_polarizations, num_dishes},
              std::array<std::string, 5>{"Trfibar", "F", "SK", "P", "D"},
              std::array<std::ptrdiff_t, 5>{
                  rfi_downsampling_factor * rfi_second_downsampling_factor, 1, 1, 1, 1},
              *this),
    rfi_SKbartilde(
        rfi_SKbartilde_name, "SKbartilde",
        std::array<std::ptrdiff_t, 3>{buffer_depth * rfi_num_times_bar, num_frequencies, 3},
        std::array<std::string, 3>{"Trfibar", "F", "SK"},
        std::array<std::ptrdiff_t, 3>{rfi_downsampling_factor * rfi_second_downsampling_factor, 1,
                                      1},
        *this),
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
    if (bf_mask_lifetime_in_samples % (rfi_downsampling_factor * rfi_second_downsampling_factor)
        != 0)
        FATAL_ERROR(
            "rfi_downsampling_factor {:d} x rfi_second_downsampling_factor {:d} must evenly "
            "divide bf_mask_lifetime_in_samples {:d}",
            rfi_downsampling_factor, rfi_second_downsampling_factor, bf_mask_lifetime_in_samples);

    // We read at most this many `rfi_S012bar` elements per frame, and we stop at the end of a
    // bad feed mask element. If a mask lifetime were not a whole number of these batches we
    // would have to read a short batch at the end of each lifetime, which would leave the
    // read position unaligned and could make a later batch straddle the end of the
    // ringbuffer.
    const std::ptrdiff_t rfi_S012bar_read_max = rfi_S012bar.get_ndarray().extent(0) / buffer_depth;
    if (rfi_samples_per_bf_sample % rfi_S012bar_read_max != 0)
        FATAL_ERROR("The bad feed mask lifetime of {:d} rfi_S012bar elements must be a multiple of "
                    "the {:d} elements read per frame",
                    rfi_samples_per_bf_sample, rfi_S012bar_read_max);

    bf_mask.register_consumer();
    rfi_S012bar.register_consumer();
    rfi_SKbar.register_producer();
    rfi_SKbartilde.register_producer();

    set_command_type(gpuCommandType::KERNEL);
}

cudaRFISKbar::~cudaRFISKbar() {}

int cudaRFISKbar::wait_on_precondition() {
    // Wait for data to be available in input ringbuffers

    // Which bad feed mask element covers the data we are about to read? Ask the ringbuffer
    // where our next read will begin. We must not use our own `read_valid` for this: every
    // instance of this command shares one ringbuffer read head, so our own position lags it
    // by whatever the other instances have claimed since our previous frame.
    const std::ptrdiff_t rfi_S012bar_begin = rfi_S012bar.peek_read_head();
    if (rfi_S012bar_begin < 0)
        return -1; // shutting down
    // (`kotekan::div` must be qualified; an unqualified `div` finds C's `::div`)
    const std::ptrdiff_t bf_mask_element =
        kotekan::div(rfi_S012bar_begin, rfi_samples_per_bf_sample);
    const std::ptrdiff_t bf_mask_lifetime_end =
        round_up(rfi_S012bar_begin + 1, rfi_samples_per_bf_sample);

    DEBUG("Waiting for rfi_S012bar input ringbuffer data for frame {:d}...", gpu_frame_id);
    const std::ptrdiff_t rfi_S012bar_ringbuf = rfi_S012bar.get_ndarray().extent(0);
    // Do not overshoot the bf mask lifetime
    using std::min;
    const std::ptrdiff_t rfi_S012bar_read_max =
        min(rfi_S012bar_ringbuf / buffer_depth, bf_mask_lifetime_end - rfi_S012bar_begin);
    assert(rfi_S012bar_read_max > 0);
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
    assert(rfi_S012bar.get_read_valid().begin() == rfi_S012bar_begin);
    assert(rfi_S012bar.get_read_valid().end() <= bf_mask_lifetime_end);

    // Read the bad feed mask element covering these data. We claim it only when we have
    // reached the end of its lifetime, i.e. when this is the last frame that will use it.
    // Until then it stays in the ringbuffer and the following frames read it again.
    //
    // We are holding a claim on `rfi_S012bar` while we wait here. That is safe because the
    // bad feed mask producer does not depend on `rfi_S012bar` being drained: it is fed by
    // its own stage, and all consumers of the bad feed mask advance at the same rate through
    // the data, so none of them can hold the mask ringbuffer's read tail far enough back to
    // keep this element from being written.
    const bool bf_mask_last_use = rfi_S012bar.get_read_valid().end() == bf_mask_lifetime_end;
    DEBUG("Waiting for bf_mask input ringbuffer data for frame {:d}...", gpu_frame_id);
    const int bf_mask_errcode =
        bf_mask.wait_and_claim_readable([&](const std::ptrdiff_t available_elements) {
            if (available_elements < 1)
                return read_descriptor_t{.claimed = 0, .read = 0};
            return read_descriptor_t{.claimed = bf_mask_last_use ? 1 : 0, .read = 1};
        });
    if (bf_mask_errcode < 0)
        return bf_mask_errcode;
    DEBUG("Done waiting for bf_mask input ringbuffer data for frame {:d}; using element {:d}{:s}",
          gpu_frame_id, bf_mask_element, bf_mask_last_use ? " (last use)" : "");
    // The two ringbuffers must agree on which mask element covers these data
    assert(bf_mask.get_read_valid().begin() == bf_mask_element);

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

    bf_mask.check_metadata();
    rfi_S012bar.check_metadata();

    rfi_SKbar.set_metadata(rfi_S012bar.get_metadata());
    rfi_SKbartilde.set_metadata(rfi_S012bar.get_metadata());

    if (poison_buffers) {
        rfi_SKbar.set_to_poison(0xff);
        rfi_SKbartilde.set_to_poison(0xff);
    }

    const std::int8_t* const bf_mask_memory =
        bf_mask.get_ndarray().data()
        + bf_mask.get_ndarray().stride(0)
              * (bf_mask.get_read_valid().begin() % bf_mask.get_ndarray().extent(0));
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
    // S012_Tmin wraps around into actual array index to avoid overflows
    const long S012_Tsize = rfi_S012bar.get_ndarray().get_extent(0);
    const long S012_Tmin = rfi_S012bar.get_read_valid().begin() % S012_Tsize;
    // sk_feed_averaged_Tmin wraps around into actual array index to avoid overflows
    const long sk_feed_averaged_Tsize = rfi_SKbartilde.get_ndarray().get_extent(0);
    const long sk_feed_averaged_Tmin =
        rfi_SKbartilde.get_write_valid().begin() % sk_feed_averaged_Tsize;
    // sk_single_feed_Tmin wraps around into actual array index to avoid overflows
    const long sk_single_feed_Tsize = rfi_SKbar.get_ndarray().get_extent(0);
    const long sk_single_feed_Tmin = rfi_SKbar.get_write_valid().begin() % sk_single_feed_Tsize;
    const long rfimask_T1024min = 0;
    const long rfimask_T1024size = 0;
    const cudaStream_t stream = device.getStream(cuda_stream_id);
    skKernel.launch(out_sk_feed_averaged, out_sk_single_feed, out_rfimask, in_S012, in_bf_mask, T,
                    F, S, S012_Tmin, S012_Tsize, sk_feed_averaged_Tmin, sk_feed_averaged_Tsize,
                    sk_single_feed_Tmin, sk_single_feed_Tsize, rfimask_T1024min, rfimask_T1024size,
                    stream);
#ifdef DEBUGGING
    CHECK_CUDA_ERROR(cudaStreamSynchronize(device.getStream(cuda_stream_id)));
#endif

    if (poison_buffers) {
        rfi_SKbar.check_for_poison(0xff);
        rfi_SKbartilde.check_for_poison(0xff);
    }

    return record_end_event();
}

void cudaRFISKbar::finalize_frame() {
    // Advance the ring buffers. The bad feed mask advances by the one element we claimed if
    // this was the last frame using it, and by nothing otherwise.
    bf_mask.finish_read();
    rfi_S012bar.finish_read();
    rfi_SKbar.finish_write();
    rfi_SKbartilde.finish_write();

    cudaCommand::finalize_frame();
}
