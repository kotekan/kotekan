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

class cudaRFIS012bar : public cudaCommand {
public:
    cudaRFIS012bar(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                   const int instance_num);
    virtual ~cudaRFIS012bar();

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
    const int rfi_second_downsampling_factor;
    const int rfi_num_times;
    const int rfi_num_times_bar;

    // Kotekan buffer names
    const std::string rfi_S012_name;
    const std::string rfi_S012bar_name;

    // Buffers
    NDArrayRingBuffer<std::uint64_t, 5> rfi_S012;
    NDArrayRingBuffer<std::uint64_t, 5> rfi_S012bar;
};

REGISTER_CUDA_COMMAND(cudaRFIS012bar);

cudaRFIS012bar::cudaRFIS012bar(kotekan::Config& config, const std::string& unique_name,
                               kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                               const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "cudaRFIS012bar"),
    // Parameters
    buffer_depth(config.get<int>(unique_name, "buffer_depth")),
    num_frequencies(config.get<int>(unique_name, "num_frequencies")),
    num_polarizations(config.get<int>(unique_name, "num_polarizations")),
    num_dishes(config.get<int>(unique_name, "num_dishes")),
    rfi_second_downsampling_factor(config.get<int>(unique_name, "rfi_second_downsampling_factor")),
    rfi_num_times(config.get<int>(unique_name, "rfi_num_times")),
    rfi_num_times_bar(config.get<int>(unique_name, "rfi_num_times_bar")),
    // Buffer names
    rfi_S012_name(config.get<std::string>(unique_name, "rfi_S012_name")),
    rfi_S012bar_name(config.get<std::string>(unique_name, "rfi_S012bar_name")),
    // Buffers
    rfi_S012(rfi_S012_name, "S012",
             std::array<std::ptrdiff_t, 5>{buffer_depth * rfi_num_times, num_frequencies, 3,
                                           num_polarizations, num_dishes},
             std::array<std::string, 5>{"Trfi", "F", "S", "P", "D"}, *this),
    rfi_S012bar(rfi_S012bar_name, "S012bar",
                std::array<std::ptrdiff_t, 5>{buffer_depth * rfi_num_times_bar, num_frequencies, 3,
                                              num_polarizations, num_dishes},
                std::array<std::string, 5>{"Trfibar", "F", "S", "P", "D"}, *this)
//
{
    rfi_S012.register_consumer();
    rfi_S012bar.register_producer();

    set_command_type(gpuCommandType::KERNEL);
}

cudaRFIS012bar::~cudaRFIS012bar() {}

int cudaRFIS012bar::wait_on_precondition() {
    // Wait for data to be available in input ringbuffers
    DEBUG("Waiting for rfi_S012 input ringbuffer data for frame {:d}...", gpu_frame_id);
    const std::ptrdiff_t rfi_S012_ringbuf = rfi_S012.get_ndarray().extent(0);
    const std::ptrdiff_t rfi_S012_read_max = rfi_S012_ringbuf / 4;
    std::ptrdiff_t rfi_S012_read = -1;
    const int rfi_S012_errcode =
        rfi_S012.wait_and_claim_readable([&](const std::ptrdiff_t available_elements) {
            using std::min;
            rfi_S012_read = round_down(min(available_elements, rfi_S012_read_max),
                                       rfi_second_downsampling_factor);
            return read_descriptor_t{.claimed = rfi_S012_read, .read = rfi_S012_read};
        });
    if (rfi_S012_errcode < 0)
        return rfi_S012_errcode;
    DEBUG("Done waiting for rfi_S012 input ringbuffer data for frame {:d}; will read {:d} elements",
          gpu_frame_id, rfi_S012_read);

    DEBUG("Waiting for rfi_S012bar output ringbuffer space for frame {:d}...", gpu_frame_id);
    const std::ptrdiff_t rfi_S012bar_written =
        div_noremainder(rfi_S012_read, rfi_second_downsampling_factor);
    const int rfi_S012bar_errcode = rfi_S012bar.wait_for_writable(rfi_S012bar_written);
    if (rfi_S012bar_errcode < 0)
        return rfi_S012bar_errcode;
    DEBUG("Done waiting for rfi_S012bar output ringbuffer space for frame {:d}; "
          "will write {:d} elements",
          gpu_frame_id, rfi_S012bar_written);

    return 0;
}

cudaEvent_t cudaRFIS012bar::execute(cudaPipelineState& /*pipestate*/,
                                    const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();
    record_start_event();

    rfi_S012.check_metadata();

    rfi_S012bar.set_metadata(rfi_S012.get_metadata());
    const auto& rfi_S012bar_meta = rfi_S012bar.get_metadata();
    assert(rfi_S012bar_meta->nfreq >= 0);
    for (int freq = 0; freq < rfi_S012bar_meta->nfreq; ++freq) {
        rfi_S012bar_meta->freq_upchan_factor[freq] *= rfi_second_downsampling_factor;
        rfi_S012bar_meta->time_downsampling_fpga[freq] *= rfi_second_downsampling_factor;
    }

    // There is no poison value
    // rfi_S012bar.set_to_poison(0xff);

    const std::uint64_t* const rfi_S012_memory = rfi_S012.get_ndarray().data();
    std::uint64_t* const rfi_S012bar_memory = rfi_S012bar.get_ndarray().data();

    const std::ptrdiff_t Trfi_size = rfi_S012.get_ndarray().extent(0);
    const std::ptrdiff_t Trfi_min = rfi_S012.get_read_valid().begin();
    const std::ptrdiff_t Trfi = rfi_S012.get_read_valid().size();
    DEBUG("Trfi_size={:d} Trfi_min={:d} Trfi={:d}", Trfi_size, Trfi_min, Trfi);

    const std::ptrdiff_t Trfibar_size = rfi_S012bar.get_ndarray().extent(0);
    const std::ptrdiff_t Trfibar_min = rfi_S012bar.get_write_valid().begin();
    const std::ptrdiff_t Trfibar = rfi_S012bar.get_write_valid().size();
    DEBUG("Trfibar_size={:d} Trfibar_min={:d} Trfibar={:d}", Trfibar_size, Trfibar_min, Trfibar);

    ulong* const Sout = rfi_S012bar_memory;
    const ulong* const Sin = rfi_S012_memory;
    const long T = Trfi;
    const long M = rfi_S012.get_ndarray().extent(1) * rfi_S012.get_ndarray().extent(2)
                   * rfi_S012.get_ndarray().extent(3) * rfi_S012.get_ndarray().extent(4);
    const long Nds = rfi_second_downsampling_factor;
    const cudaStream_t stream = device.getStream(cuda_stream_id);

    n2k::launch_s012_time_downsample_kernel(Sout, Sin, T, M, Nds, Trfi_min, Trfi_size, Trfibar_min,
                                            Trfibar_size, stream);

    // There is no poison value
    // rfi_S012bar.check_for_poison(0xff);

    return record_end_event();
}

void cudaRFIS012bar::finalize_frame() {
    // Advance the ring buffers
    rfi_S012.finish_read();
    rfi_S012bar.finish_write();

    cudaCommand::finalize_frame();
}
