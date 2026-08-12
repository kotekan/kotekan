#include "Config.hpp"              // for Config
#include "NDArray.hpp"             // for NDArray
#include "NDArrayRingBuffer.hpp"   // for NDArrayRingBuffer, extent_t, read_descriptor_t
#include "bufferContainer.hpp"     // for bufferContainer
#include "chordMetadata.hpp"       // for chordMetadata
#include "cudaCommand.hpp"         // for cudaCommand, cudaPipelineState, REGISTER_CUDA_COMMAND
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "cudaUtils.hpp"           // for CHECK_CUDA_ERROR
#include "div.hpp"                 // for div_noremainder, round_down
#include "gpuCommand.hpp"          // for gpuCommandType
#include "kotekanLogging.hpp"      // for DEBUG, FATAL_ERROR
#include "n2k/rfi_kernels.hpp"     // for launch_s012_time_downsample_kernel

#include "fmt.hpp" // for compile_string_to_view

#include <algorithm>          // for min
#include <array>              // for array
#include <cassert>            // for assert
#include <cstddef>            // for ptrdiff_t
#include <cstdint>            // for uint64_t
#include <cuda_runtime_api.h> // for cudaStreamSynchronize
#include <driver_types.h>     // for cudaEvent_t, CUstream_st, CUevent_st, cudaStream_t
#include <functional>         // for function
#include <memory>             // for allocator, shared_ptr, __shared_ptr_access
#include <string>             // for basic_string, string
#include <sys/types.h>        // for ulong
#include <vector>             // for vector

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
    const int rfi_downsampling_factor;
    const int rfi_second_downsampling_factor;
    const int rfi_num_times;
    const int rfi_num_times_bar;
    const bool poison_buffers;

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
    rfi_downsampling_factor(config.get<int>(unique_name, "rfi_downsampling_factor")),
    rfi_second_downsampling_factor(config.get<int>(unique_name, "rfi_second_downsampling_factor")),
    rfi_num_times(config.get<int>(unique_name, "rfi_num_times")),
    rfi_num_times_bar(config.get<int>(unique_name, "rfi_num_times_bar")),
    poison_buffers(config.get_default<bool>(unique_name, "poison_buffers", false)),
    // Buffer names
    rfi_S012_name(config.get<std::string>(unique_name, "rfi_S012_name")),
    rfi_S012bar_name(config.get<std::string>(unique_name, "rfi_S012bar_name")),
    // Buffers
    rfi_S012(rfi_S012_name, "S012",
             std::array<std::ptrdiff_t, 5>{buffer_depth * rfi_num_times, num_frequencies, 3,
                                           num_polarizations, num_dishes},
             std::array<std::string, 5>{"Trfi", "F", "S", "P", "D"},
             std::array<std::ptrdiff_t, 5>{rfi_downsampling_factor, 1, 1, 1, 1}, *this),
    rfi_S012bar(rfi_S012bar_name, "S012bar",
                std::array<std::ptrdiff_t, 5>{buffer_depth * rfi_num_times_bar, num_frequencies, 3,
                                              num_polarizations, num_dishes},
                std::array<std::string, 5>{"Trfibar", "F", "S", "P", "D"},
                std::array<std::ptrdiff_t, 5>{
                    rfi_downsampling_factor * rfi_second_downsampling_factor, 1, 1, 1, 1},
                *this)
//
{
    rfi_S012.register_consumer();
    rfi_S012bar.register_producer();

    // Ensure that this stage can always make progress
    rfi_S012.check_read_progress(rfi_S012.get_ndarray().extent(0) / 4,
                                 rfi_second_downsampling_factor);

    set_command_type(gpuCommandType::KERNEL);

    if (rfi_num_times != rfi_num_times_bar * rfi_second_downsampling_factor) {
        FATAL_ERROR("rfi_num_times: {:d} must be equal to rfi_num_times_bar x  "
                    "rfi_second_downsampling_factor ({:d} x {:d} = {:d})",
                    rfi_num_times, rfi_num_times_bar, rfi_second_downsampling_factor,
                    rfi_num_times_bar * rfi_second_downsampling_factor);
    }
    assert(rfi_num_times == rfi_num_times_bar * rfi_second_downsampling_factor);
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

    // TODO: Set these metadata only once
    rfi_S012bar.set_metadata(rfi_S012.get_metadata());
    const auto& rfi_S012bar_meta = rfi_S012bar.get_metadata();
    rfi_S012bar_meta->set_time_downsampling_fpga(rfi_S012bar_meta->get_time_downsampling_fpga()
                                                 * rfi_second_downsampling_factor);

    // There is no poison value
    // if (poison_buffers)
    //     rfi_S012bar.set_to_poison(0xff);

    const std::uint64_t* const rfi_S012_memory = rfi_S012.get_ndarray().data();
    std::uint64_t* const rfi_S012bar_memory = rfi_S012bar.get_ndarray().data();

    const std::ptrdiff_t Trfi_size = rfi_S012.get_ndarray().extent(0);
    // Trfi_min wraps around into actual array index to avoid overflows
    const std::ptrdiff_t Trfi_min = rfi_S012.get_read_valid().begin() % Trfi_size;
    const std::ptrdiff_t Trfi = rfi_S012.get_read_valid().size();
    DEBUG("Trfi_size={:d} Trfi_min={:d} Trfi={:d}", Trfi_size, Trfi_min, Trfi);

    if (Trfi_min + Trfi > Trfi_size) {
        FATAL_ERROR("Chunk starting at Trfi_min={:d} of size Trfi={:d} runs past end of ringbuffer "
                    "{:s} of size Trfisize={:d}",
                    Trfi_min, Trfi, rfi_S012.get_buffer_name(), Trfi_size);
    }
    assert(Trfi_min + Trfi <= Trfi_size);

    const std::ptrdiff_t Trfibar_size = rfi_S012bar.get_ndarray().extent(0);
    // Trfibar_min wraps around into actual array index to avoid overflows
    const std::ptrdiff_t Trfibar_min = rfi_S012bar.get_write_valid().begin() % Trfibar_size;
    const std::ptrdiff_t Trfibar = rfi_S012bar.get_write_valid().size();
    DEBUG("Trfibar_size={:d} Trfibar_min={:d} Trfibar={:d}", Trfibar_size, Trfibar_min, Trfibar);

    // Time offsets to access S012 buffers
    const std::ptrdiff_t Trfi_offset = Trfi_min * rfi_S012.get_ndarray().stride(0);
    const std::ptrdiff_t Trfibar_offset =
        Trfi_min / rfi_second_downsampling_factor * rfi_S012bar.get_ndarray().stride(0);

    ulong* const Sout = rfi_S012bar_memory + Trfibar_offset;
    const ulong* const Sin = rfi_S012_memory + Trfi_offset;
    const long T = Trfi;
    const long M = rfi_S012.get_ndarray().extent(1) * rfi_S012.get_ndarray().extent(2)
                   * rfi_S012.get_ndarray().extent(3) * rfi_S012.get_ndarray().extent(4);
    const long Nds = rfi_second_downsampling_factor;
    const cudaStream_t stream = device.getStream(cuda_stream_id);

    n2k::launch_s012_time_downsample_kernel(Sout, Sin, T, M, Nds, 0, Trfi_size, 0, Trfibar_size,
                                            stream);
#ifdef DEBUGGING
    CHECK_CUDA_ERROR(cudaStreamSynchronize(device.getStream(cuda_stream_id)));
#endif

    // There is no poison value
    // if (poison_buffers)
    //     rfi_S012bar.check_for_poison(0xff);

    return record_end_event();
}

void cudaRFIS012bar::finalize_frame() {
    // Advance the ring buffers
    rfi_S012.finish_read();
    rfi_S012bar.finish_write();

    cudaCommand::finalize_frame();
}
