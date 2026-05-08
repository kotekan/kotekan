// 8-bit FRB beam quantizer for CHIME, using the chromatic CHIME float32 FRB beamformer and CHIME's
// sending-to-Bonsai network format

#include "DataType.hpp"            // for float16_t
#include "NDArray.hpp"             // for NDArray
#include "NDArrayBuffer.hpp"       // for NDArrayBuffer
#include "chordMetadata.hpp"       // for chordMetadata, get_chord_metadata, metadata_is_chord
#include "cudaCommand.hpp"         // for cudaCommand, REGISTER_CUDA_COMMAND, _factory_aliascud...
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "cudaQuantizeKernel8Chime.hpp" // for gpu_quantize8
#include "cudaUtils.hpp"                // for CHECK_CUDA_ERROR
#include "div.hpp"                      // for div_noremainder
#include "gpuCommand.hpp"               // for gpuCommandType
#include "metadata.hpp"                 // for metadataObject

#include <algorithm>          // for max
#include <array>              // for array
#include <cassert>            // for assert
#include <cmath>              //
#include <cstddef>            // for size_t, ptrdiff_t
#include <cstdint>            // for int64_t
#include <cuda_fp16.h>        // for __half2, __half
#include <cuda_runtime_api.h> // for cudaGetLastError, cudaMemcpy
#include <memory>             // for shared_ptr, __shared_ptr_access
#include <stdexcept>          // for runtime_error
#include <string>             // for allocator, basic_string, string, operator+
#include <tuple>              // for tuple, make_tuple
#include <vector>             // for vector

using kotekan::bufferContainer;
using kotekan::Config;

class cudaQuantize8Chime : public cudaCommand {
public:
    cudaQuantize8Chime(kotekan::Config& config, const std::string& unique_name,
                       kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                       int inst);
    ~cudaQuantize8Chime();
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;

private:
    const int _num_beams;
    const int _num_frequencies;
    const int _num_times;

    /// GPU side memory name for the time-stream input
    const std::string _gpu_mem_input;
    /// GPU side memory name for the time-stream output
    const std::string _gpu_mem_beams;
    /// GPU side memory name for mean,stdev output
    const std::string _gpu_mem_beams_offsetscale;

    const NDArrayBuffer<float, 4> input_buffer;
    NDArrayBuffer<std::uint8_t, 8> beam_buffer;
    NDArrayBuffer<float, 7> offsetscale_buffer;
};

REGISTER_CUDA_COMMAND(cudaQuantize8Chime);

cudaQuantize8Chime::cudaQuantize8Chime(Config& config, const std::string& unique_name,
                                       bufferContainer& host_buffers, cudaDeviceInterface& device,
                                       int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst),
    //
    _num_beams(config.get<std::int64_t>(unique_name, "num_beams")),
    _num_frequencies(config.get<std::int64_t>(unique_name, "num_frequencies")),
    _num_times(config.get<std::int64_t>(unique_name, "num_times")),
    //
    _gpu_mem_input(config.get<std::string>(unique_name, "gpu_mem_input")),
    _gpu_mem_beams(config.get<std::string>(unique_name, "gpu_mem_output")),
    _gpu_mem_beams_offsetscale(config.get<std::string>(unique_name, "gpu_mem_offsetscale")),
    //
    input_buffer([&]() {
        const std::array<std::ptrdiff_t, 4> input_lengths{1, _num_beams, _num_frequencies,
                                                          _num_times};
        const std::array<std::string, 4> input_dimnames{"Ttildehi256", "R", "Fbar", "Ttildelo256"};
        return NDArrayBuffer<float, 4>(_gpu_mem_input, "I2", input_lengths, input_dimnames, *this);
    }()),
    beam_buffer([&]() {
        const std::array<std::ptrdiff_t, 8> beam_lengths{1, _num_beams, _num_frequencies,
                                                         _num_times};
        const std::array<std::string, 8> beam_dimnames{"Ttilde256",     "R8",        "Fbar64",
                                                       "Ttilde16_lo16", "Rlo8",      "Fbar16_lo4",
                                                       "Fbarlo16",      "Ttildelo16"};
        return NDArrayBuffer<std::uint8_t, 8>(_gpu_mem_beams, "I3", beam_lengths, beam_dimnames,
                                              *this);
    }()),
    offsetscale_buffer([&]() {
        const std::array<std::ptrdiff_t, 7> offsetscale_lengths{1, _num_beams, _num_frequencies, 2};
        const std::array<std::string, 7> offsetscale_dimnames{
            "Ttilde256", "R8", "Fbar64", "Ttilde16_lo16", "Rlo8", "Fbar16_lo4", "offset/scale"};
        return NDArrayBuffer<float, 7>(_gpu_mem_beams_offsetscale, "I3_offsetscale",
                                       offsetscale_lengths, offsetscale_dimnames, *this);
    }())
//
{
    set_command_type(gpuCommandType::KERNEL);
    set_name("cudaQuantize8Chime");

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_input, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_beams, true, false, true));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_beams_offsetscale, true, false, true));
}

cudaQuantize8Chime::~cudaQuantize8Chime() {}

cudaEvent_t cudaQuantize8Chime::execute(cudaPipelineState&, const std::vector<cudaEvent_t>&) {
    pre_execute();

    record_start_event();

    // Check metadata
    input_buffer.check_metadata();
    const auto& input_meta = input_buffer.get_metadata();

    // Set metadata
    beam_buffer.set_metadata(input_meta);
    offsetscale_buffer.set_metadata(input_meta);

    const auto& input_ndarray = input_buffer.get_ndarray();
    const float* const input = input_ndarray.data();
    assert(input_ndarray.extent(0) == 1);

    auto& offsetscale_ndarray = offsetscale_buffer.get_ndarray();
    float* const outputf = offsetscale_ndarray.data();

    auto& beam_ndarray = beam_buffer.get_ndarray();
    std::uint8_t* const outputi = beam_ndarray.data();

    cudaStream_t stream = device.getStream(cuda_stream_id);
    gpu_quantize8chime(input, outputf, outputi, stream);
    CHECK_CUDA_ERROR(cudaGetLastError());

    return record_end_event();
}
