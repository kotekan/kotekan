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

class cudaRFIS012tilde : public cudaCommand {
public:
    cudaRFIS012tilde(kotekan::Config& config, const std::string& unique_name,
                     kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                     const int instance_num);
    virtual ~cudaRFIS012tilde();

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

    // Kotekan buffer names
    const std::string bf_mask_name;
    const std::string rfi_S012_name;
    const std::string rfi_S012tilde_name;

    // Buffers
    NDArrayBuffer<std::int8_t, 2> bf_mask;
    NDArrayRingBuffer<std::uint64_t, 5> rfi_S012;
    NDArrayRingBuffer<std::uint64_t, 3> rfi_S012tilde;
};

REGISTER_CUDA_COMMAND(cudaRFIS012tilde);

cudaRFIS012tilde::cudaRFIS012tilde(kotekan::Config& config, const std::string& unique_name,
                                   kotekan::bufferContainer& host_buffers,
                                   cudaDeviceInterface& device, const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "cudaRFIS012tilde"),
    // Parameters
    buffer_depth(config.get<int>(unique_name, "buffer_depth")),
    num_times(config.get<int>(unique_name, "num_times")),
    num_frequencies(config.get<int>(unique_name, "num_frequencies")),
    num_polarizations(config.get<int>(unique_name, "num_polarizations")),
    num_dishes(config.get<int>(unique_name, "num_dishes")),
    rfi_downsampling_factor(config.get<int>(unique_name, "rfi_downsampling_factor")),
    // Buffer names
    bf_mask_name(config.get<std::string>(unique_name, "bf_mask_name")),
    rfi_S012_name(config.get<std::string>(unique_name, "rfi_S012_name")),
    rfi_S012tilde_name(config.get<std::string>(unique_name, "rfi_S012tilde_name")),
    // Buffers
    bf_mask(bf_mask_name, "bf_mask", std::array<std::ptrdiff_t, 2>{num_polarizations, num_dishes},
            std::array<std::string, 2>{"P", "D"}, *this, buffer_type_t::do_once),
    rfi_S012(rfi_S012_name, "S012",
             std::array<std::ptrdiff_t, 5>{
                 div_noremainder(buffer_depth * num_times, rfi_downsampling_factor),
                 num_frequencies, 3, num_polarizations, num_dishes},
             std::array<std::string, 5>{"Tcoarse", "F", "S", "P", "D"}, *this),
    rfi_S012tilde(
        rfi_S012tilde_name, "S012tilde",
        std::array<std::ptrdiff_t, 3>{
            div_noremainder(buffer_depth * num_times, rfi_downsampling_factor), num_frequencies, 3},
        std::array<std::string, 3>{"Tcoarse", "F", "S"}, *this)
//
{
    rfi_S012.register_consumer();
    rfi_S012tilde.register_producer();

    set_command_type(gpuCommandType::KERNEL);
}

cudaRFIS012tilde::~cudaRFIS012tilde() {}

int cudaRFIS012tilde::wait_on_precondition() {
    // Wait for data to be available in input ringbuffers
    DEBUG("Waiting for rfi_S012 input ringbuffer data for frame {:d}...", gpu_frame_id);
    const int rfi_S012_errcode =
        rfi_S012.wait_and_claim_readable([&](const std::ptrdiff_t available_elements) {
            return read_descriptor_t{.claimed = available_elements, .read = available_elements};
        });
    if (rfi_S012_errcode < 0)
        return rfi_S012_errcode;
    const std::ptrdiff_t processed_elements =
        rfi_S012.get_end_read_valid() - rfi_S012.get_begin_read_valid();
    const std::ptrdiff_t produced_elements = processed_elements;
    DEBUG("Done waiting for rfi_S012 input ringbuffer data for frame {:d}; will read {:d} elements",
          gpu_frame_id, processed_elements);

    DEBUG("Waiting for rfi_S012tilde output ringbuffer space for frame {:d}...", gpu_frame_id);
    const int rfi_S012tilde_errcode = rfi_S012tilde.wait_for_writable(produced_elements);
    if (rfi_S012tilde_errcode < 0)
        return rfi_S012tilde_errcode;
    DEBUG("Done waiting for rfi_S012tilde output ringbuffer space for frame {:d}; "
          "will write {:d} elements",
          gpu_frame_id, produced_elements);

    return 0;
}

cudaEvent_t cudaRFIS012tilde::execute(cudaPipelineState& /*pipestate*/,
                                      const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();

    record_start_event();

    rfi_S012.check_metadata();

    rfi_S012tilde.set_metadata(rfi_S012.get_metadata());

    const std::int8_t* const bf_mask_memory = bf_mask.get_ndarray().data();
    const std::uint64_t* const rfi_S012_memory = rfi_S012.get_ndarray().data();
    std::uint64_t* const rfi_S012tilde_memory = rfi_S012tilde.get_ndarray().data();

    const std::ptrdiff_t Tcoarsesize = rfi_S012.get_ndarray().extent(0);
    const std::ptrdiff_t Tcoarsemin = rfi_S012.get_begin_read_valid();
    const std::ptrdiff_t Tcoarse = rfi_S012.get_end_read_valid() - rfi_S012.get_begin_read_valid();
    DEBUG("Tcoarsesize={:d} Tcoarsemin={:d} Tcoarse={:d}", Tcoarsesize, Tcoarsemin, Tcoarse);

    n2k::launch_s012_station_downsample_kernel(
        (ulong*)rfi_S012tilde_memory, (const ulong*)rfi_S012_memory, (const uint8_t*)bf_mask_memory,
        Tcoarse, Tcoarsemin, Tcoarsesize, num_frequencies, num_dishes * num_polarizations,
        device.getStream(cuda_stream_id));

    return record_end_event();
}

void cudaRFIS012tilde::finalize_frame() {
    // Advance the ring buffers
    rfi_S012.finish_read();
    rfi_S012tilde.finish_write();

    cudaCommand::finalize_frame();
}
