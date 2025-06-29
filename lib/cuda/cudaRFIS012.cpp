#include <DataType.hpp>
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

class cudaRFIS012 : public cudaCommand {
public:
    cudaRFIS012(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                const int instance_num);
    virtual ~cudaRFIS012();

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
    const std::string pl_mask_name;
    const std::string voltage_name;
    const std::string rfi_S012_name;

    // Buffers
    NDArrayRingBuffer<kotekan::uint1x8_t, 5> pl_mask;
    NDArrayRingBuffer<kotekan::int4x2chime_t, 4> voltage;
    NDArrayRingBuffer<std::uint64_t, 5> rfi_S012;
};

REGISTER_CUDA_COMMAND(cudaRFIS012);

cudaRFIS012::cudaRFIS012(kotekan::Config& config, const std::string& unique_name,
                         kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                         const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "cudaRFIS012"),
    // Parameters
    buffer_depth(config.get<int>(unique_name, "buffer_depth")),
    num_times(config.get<int>(unique_name, "num_times")),
    num_frequencies(config.get<int>(unique_name, "num_frequencies")),
    num_polarizations(config.get<int>(unique_name, "num_polarizations")),
    num_dishes(config.get<int>(unique_name, "num_dishes")),
    rfi_downsampling_factor(config.get<int>(unique_name, "rfi_downsampling_factor")),
    rfi_num_times(config.get<int>(unique_name, "rfi_num_times")),
    // Buffer names
    pl_mask_name(config.get<std::string>(unique_name, "pl_mask_name")),
    voltage_name(config.get<std::string>(unique_name, "voltage_name")),
    rfi_S012_name(config.get<std::string>(unique_name, "rfi_S012_name")),
    // Buffers
    pl_mask(pl_mask_name, "pl_mask",
            std::array<std::ptrdiff_t, 5>{div_noremainder(buffer_depth * num_times, 128),
                                          num_frequencies / 4, num_polarizations, num_dishes, 8},
            std::array<std::string, 5>{"T16hi8", "F4", "P", "D", "T16lo8"}, *this),
    voltage(voltage_name, "E",
            std::array<std::ptrdiff_t, 4>{buffer_depth * num_times, num_frequencies,
                                          num_polarizations, num_dishes},
            std::array<std::string, 4>{"T", "F", "P", "D"}, *this),
    rfi_S012(rfi_S012_name, "S012",
             std::array<std::ptrdiff_t, 5>{buffer_depth * rfi_num_times, num_frequencies, 3,
                                           num_polarizations, num_dishes},
             std::array<std::string, 5>{"Trfi", "F", "S", "P", "D"}, *this)
//
{
    // For pl_mask_T128_sample_bytes
    if (num_frequencies % 4 != 0)
        FATAL_ERROR("num_frequencies % 4 != 0");

    if (rfi_downsampling_factor % 128 != 0)
        FATAL_ERROR("rfi_downsampling_factor % 128 != 0");

    pl_mask.register_consumer();
    voltage.register_consumer();
    rfi_S012.register_producer();

    set_command_type(gpuCommandType::KERNEL);
}

cudaRFIS012::~cudaRFIS012() {}

int cudaRFIS012::wait_on_precondition() {
    // Wait for data to be available in input ringbuffers
    DEBUG("Waiting for pl_mask input ringbuffer data for frame {:d}...", gpu_frame_id);
    const std::ptrdiff_t pl_mask_ringbuf = pl_mask.get_ndarray().extent(0);
    const std::ptrdiff_t pl_mask_read_max = pl_mask_ringbuf / 4;
    std::ptrdiff_t pl_mask_read = -1;
    const int pl_mask_errcode =
        pl_mask.wait_and_claim_readable([&](const std::ptrdiff_t available_elements) {
            // We are downsampling, so we need to process in batches
            // that have a multiple `rfi_downsampling_factor` elements
            using std::min;
            pl_mask_read = round_down(min(available_elements, pl_mask_read_max),
                                      div_noremainder(rfi_downsampling_factor, 128));
            return read_descriptor_t{.claimed = pl_mask_read, .read = pl_mask_read};
        });
    if (pl_mask_errcode < 0)
        return pl_mask_errcode;
    DEBUG("Done waiting for pl_mask input ringbuffer data for frame {:d}; will read {:d} elements",
          gpu_frame_id, pl_mask_read);

    DEBUG("Waiting for voltage input ringbuffer data for frame {:d}...", gpu_frame_id);
    const std::ptrdiff_t voltage_read = pl_mask_read * 128;
    const int voltage_errcode =
        voltage.wait_and_claim_readable([&](const std::ptrdiff_t available_elements) {
            // Process the same number of elements as from `pl_mask`
            if (available_elements < voltage_read)
                return read_descriptor_t{.claimed = 0, .read = 0};
            else
                return read_descriptor_t{.claimed = voltage_read, .read = voltage_read};
        });
    if (voltage_errcode < 0)
        return voltage_errcode;
    DEBUG("Done waiting for voltage input ringbuffer data for frame {:d}; will read {:d} elements",
          gpu_frame_id, voltage_read);

    DEBUG("Waiting for rfi_S012 output ringbuffer space for frame {:d}...", gpu_frame_id);
    const std::ptrdiff_t rfi_S012_written = div_noremainder(voltage_read, rfi_downsampling_factor);
    const int rfi_S012_errcode = rfi_S012.wait_for_writable(rfi_S012_written);
    if (rfi_S012_errcode < 0)
        return rfi_S012_errcode;
    DEBUG(
        "Done waiting for rfi_S012 input ringbuffer data for frame {:d}; will write {:d} elements",
        gpu_frame_id, rfi_S012_written);

    return 0;
}

cudaEvent_t cudaRFIS012::execute(cudaPipelineState& /*pipestate*/,
                                 const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();

    record_start_event();

    pl_mask.check_metadata();
    voltage.check_metadata();

    rfi_S012.set_metadata(voltage.get_metadata());
    const auto& rfi_S012_meta = rfi_S012.get_metadata();
    assert(rfi_S012_meta->nfreq >= 0);
    for (int freq = 0; freq < rfi_S012_meta->nfreq; ++freq) {
        rfi_S012_meta->freq_upchan_factor[freq] *= rfi_downsampling_factor;
        rfi_S012_meta->time_downsampling_fpga[freq] *= rfi_downsampling_factor;
    }

    const kotekan::uint1x8_t* const pl_mask_memory = pl_mask.get_ndarray().data();
    const kotekan::int4x2chime_t* const voltage_memory = voltage.get_ndarray().data();
    std::uint64_t* const rfi_S012_memory = rfi_S012.get_ndarray().data();

    const std::ptrdiff_t Tsize = voltage.get_ndarray().extent(0);
    assert(Tsize % rfi_downsampling_factor == 0);
    const std::ptrdiff_t Tmin = voltage.get_read_valid().begin();
    assert(Tmin % 128 == 0);
    const std::ptrdiff_t T = voltage.get_read_valid().size();

    const int F_stride = rfi_S012.get_ndarray().stride(1);
    const int S_stride = rfi_S012.get_ndarray().stride(2);
    constexpr bool offset_encoded = true;
    n2k::launch_s0_kernel((ulong*)rfi_S012_memory, (const ulong*)pl_mask_memory, T, Tmin, Tsize,
                          num_frequencies, num_dishes * num_polarizations, rfi_downsampling_factor,
                          F_stride, device.getStream(cuda_stream_id));
    n2k::launch_s12_kernel((ulong*)(rfi_S012_memory + S_stride), (const uint8_t*)voltage_memory, T,
                           Tmin, Tsize, num_frequencies, num_dishes * num_polarizations,
                           rfi_downsampling_factor, F_stride, offset_encoded,
                           device.getStream(cuda_stream_id));

    return record_end_event();
}

void cudaRFIS012::finalize_frame() {
    // Advance the ring buffers
    pl_mask.finish_read();
    voltage.finish_read();
    rfi_S012.finish_write();

    cudaCommand::finalize_frame();
}
