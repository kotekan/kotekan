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
#include "kotekanLogging.hpp"      // for DEBUG, FATAL_ERROR
#include "n2k/rfi_kernels.hpp"     // for launch_s012_station_downsample_kernel

#include "fmt.hpp" // for compile_string_to_view

#include <algorithm>          // for min
#include <array>              // for array
#include <assert.h>           // for assert
#include <cstddef>            // for ptrdiff_t
#include <cstdint>            // for uint64_t, int8_t, uint8_t
#include <cuda_runtime_api.h> // for cudaStreamSynchronize
#include <driver_types.h>     // for cudaEvent_t, CUevent_st, CUstream_st
#include <functional>         // for function
#include <memory>             // for allocator, shared_ptr
#include <string>             // for basic_string, string
#include <sys/types.h>        // for ulong
#include <vector>             // for vector

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
    //    SKbar       skKernel
    //    SKbartilde  skKernel

    // Parameters
    const int buffer_depth;
    const int num_times;
    const int num_frequencies;
    const int num_polarizations;
    const int num_dishes;
    const int rfi_downsampling_factor;
    const int rfi_num_times;
    const bool poison_buffers;

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
    rfi_num_times(config.get<int>(unique_name, "rfi_num_times")),
    poison_buffers(config.get_default<bool>(unique_name, "poison_buffers", false)),
    // Buffer names
    bf_mask_name(config.get<std::string>(unique_name, "bf_mask_name")),
    rfi_S012_name(config.get<std::string>(unique_name, "rfi_S012_name")),
    rfi_S012tilde_name(config.get<std::string>(unique_name, "rfi_S012tilde_name")),
    // Buffers
    bf_mask(bf_mask_name, "bf_mask", std::array<std::ptrdiff_t, 2>{num_polarizations, num_dishes},
            std::array<std::string, 2>{"P", "D"}, std::array<std::ptrdiff_t, 2>{1, 1}, *this,
            buffer_type_t::do_once),
    rfi_S012(rfi_S012_name, "S012",
             std::array<std::ptrdiff_t, 5>{buffer_depth * rfi_num_times, num_frequencies, 3,
                                           num_polarizations, num_dishes},
             std::array<std::string, 5>{"Trfi", "F", "S", "P", "D"},
             std::array<std::ptrdiff_t, 5>{rfi_downsampling_factor, 1, 1, 1, 1}, *this),
    rfi_S012tilde(rfi_S012tilde_name, "S012tilde",
                  std::array<std::ptrdiff_t, 3>{buffer_depth * rfi_num_times, num_frequencies, 3},
                  std::array<std::string, 3>{"Trfi", "F", "S"},
                  std::array<std::ptrdiff_t, 3>{rfi_downsampling_factor, 1, 1}, *this)
//
{
    if (num_times % rfi_num_times != 0)
        FATAL_ERROR("num_times {:d} must be a multiple of rfi_num_times {:d}", num_times,
                    rfi_num_times);
    assert(num_times % rfi_num_times == 0);

    rfi_S012.register_consumer();
    rfi_S012tilde.register_producer();

    set_command_type(gpuCommandType::KERNEL);
}

cudaRFIS012tilde::~cudaRFIS012tilde() {}

int cudaRFIS012tilde::wait_on_precondition() {
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

    DEBUG("Waiting for rfi_S012tilde output ringbuffer space for frame {:d}...", gpu_frame_id);
    const std::ptrdiff_t rfi_S012tilde_written = rfi_S012_read;
    const int rfi_S012tilde_errcode = rfi_S012tilde.wait_for_writable(rfi_S012tilde_written);
    if (rfi_S012tilde_errcode < 0)
        return rfi_S012tilde_errcode;
    DEBUG("Done waiting for rfi_S012tilde output ringbuffer space for frame {:d}; "
          "will write {:d} elements",
          gpu_frame_id, rfi_S012tilde_written);

    return 0;
}

cudaEvent_t cudaRFIS012tilde::execute(cudaPipelineState& /*pipestate*/,
                                      const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();
    record_start_event();

    rfi_S012.check_metadata();

    rfi_S012tilde.set_metadata(rfi_S012.get_metadata());

    // There is no poison value
    // if (poison_buffers)
    //     rfi_S012tilde.set_to_poison(0xff);

    const std::int8_t* const bf_mask_memory = bf_mask.get_ndarray().data();
    const std::uint64_t* const rfi_S012_memory = rfi_S012.get_ndarray().data();
    std::uint64_t* const rfi_S012tilde_memory = rfi_S012tilde.get_ndarray().data();

    const std::ptrdiff_t Trfisize = rfi_S012.get_ndarray().extent(0);
    // Trfimin wraps around into actual array index to avoid overflows
    const std::ptrdiff_t Trfimin = rfi_S012.get_read_valid().begin() % Trfisize;
    const std::ptrdiff_t Trfi = rfi_S012.get_read_valid().size();

    // Offsets into rfi_S012 and rfi_S012tilde to start reading/writing.
    if (Trfimin + Trfi > Trfisize) {
        FATAL_ERROR("Chunk starting at Trfimin={:d} of size Trfi={:d} runs past end of ringbuffer "
                    "{:s} of size Trfisize={:d}",
                    Trfimin, Trfi, rfi_S012.get_buffer_name(), Trfisize);
    }
    assert(Trfimin + Trfi <= Trfisize);
    const std::ptrdiff_t Trfi_offset = Trfimin * rfi_S012.get_ndarray().stride(0);
    const std::ptrdiff_t Trfitilde_offset = Trfimin * rfi_S012tilde.get_ndarray().stride(0);
    DEBUG("Trfisize={:d} Trfimin={:d} Trfi={:d}", Trfisize, Trfimin, Trfi);

    n2k::launch_s012_station_downsample_kernel((ulong*)(rfi_S012tilde_memory + Trfitilde_offset),
                                               (const ulong*)(rfi_S012_memory + Trfi_offset),
                                               (const uint8_t*)bf_mask_memory, Trfi, 0, Trfisize,
                                               num_frequencies, num_dishes * num_polarizations,
                                               device.getStream(cuda_stream_id));
#ifdef DEBUGGING
    CHECK_CUDA_ERROR(cudaStreamSynchronize(device.getStream(cuda_stream_id)));
#endif

    // There is no poison value
    // if (poison_buffers)
    //     rfi_S012tilde.check_for_poison(0xff);

    return record_end_event();
}

void cudaRFIS012tilde::finalize_frame() {
    // Advance the ring buffers
    rfi_S012.finish_read();
    rfi_S012tilde.finish_write();

    cudaCommand::finalize_frame();
}
