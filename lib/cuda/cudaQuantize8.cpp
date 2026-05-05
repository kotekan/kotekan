#include "DataType.hpp"            // for float16_t
#include "NDArray.hpp"             // for NDArray
#include "NDArrayBuffer.hpp"       // for NDArrayBuffer
#include "chordMetadata.hpp"       // for chordMetadata, get_chord_metadata, metadata_is_chord
#include "cudaCommand.hpp"         // for cudaCommand, REGISTER_CUDA_COMMAND, _factory_aliascud...
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "cudaQuantizeKernel8.hpp" // for gpu_quantize8
#include "cudaUtils.hpp"           // for CHECK_CUDA_ERROR
#include "div.hpp"                 // for div_noremainder
#include "gpuCommand.hpp"          // for gpuCommandType
#include "metadata.hpp"            // for metadataObject

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

class cudaQuantize8 : public cudaCommand {
public:
    cudaQuantize8(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device, int inst);
    ~cudaQuantize8();
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;

    // These are the exact array sizes supported by the Cuda code
    static constexpr int CHUNK_SIZE = 256;
    static constexpr int FRAME_SIZE = 32;

private:
    const int _num_beams;
    const int _num_frequencies;
    const int _num_times;
    const int64_t _num_chunks;

    /// GPU side memory name for the time-stream input
    const std::string _gpu_mem_input;
    /// GPU side memory name for the time-stream output
    const std::string _gpu_mem_beams;
    /// GPU side memory name for mean,stdev output
    const std::string _gpu_mem_beams_offsetscale;

    const NDArrayBuffer<float16_t, 4> input_buffer;
    NDArrayBuffer<std::uint8_t, 4> beam_buffer;
    NDArrayBuffer<float16_t, 4> offsetscale_buffer;
};

REGISTER_CUDA_COMMAND(cudaQuantize8);

cudaQuantize8::cudaQuantize8(Config& config, const std::string& unique_name,
                             bufferContainer& host_buffers, cudaDeviceInterface& device, int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst),
    //
    _num_beams(config.get<std::int64_t>(unique_name, "num_beams")),
    _num_frequencies(config.get<std::int64_t>(unique_name, "num_frequencies")),
    _num_times(config.get<std::int64_t>(unique_name, "num_times")),
    _num_chunks(
        kotekan::div_noremainder(1LL * _num_beams * _num_frequencies * _num_times, CHUNK_SIZE)),
    //
    _gpu_mem_input(config.get<std::string>(unique_name, "gpu_mem_input")),
    _gpu_mem_beams(config.get<std::string>(unique_name, "gpu_mem_output")),
    _gpu_mem_beams_offsetscale(config.get<std::string>(unique_name, "gpu_mem_offsetscale")),
    //
    input_buffer([&]() {
        const std::array<std::ptrdiff_t, 4> input_lengths{1, _num_beams, _num_frequencies,
                                                          _num_times};
        const std::array<std::string, 4> input_dimnames{"Ttildehi256", "R", "Fbar", "Ttildelo256"};
        return NDArrayBuffer<float16_t, 4>(_gpu_mem_input, "I2", input_lengths, input_dimnames,
                                           *this);
    }()),
    beam_buffer([&]() {
        assert(_num_times % CHUNK_SIZE == 0);
        assert(_num_beams % FRAME_SIZE == 0);
        const std::array<std::ptrdiff_t, 4> beam_lengths{1, _num_beams, _num_frequencies,
                                                         _num_times};
        const std::array<std::string, 4> beam_dimnames{"Ttildehi256", "R", "Fbar", "Ttildelo256"};
        return NDArrayBuffer<std::uint8_t, 4>(_gpu_mem_beams, "I3", beam_lengths, beam_dimnames,
                                              *this);
    }()),
    offsetscale_buffer([&]() {
        assert(_num_times % CHUNK_SIZE == 0);
        assert(_num_beams % FRAME_SIZE == 0);
        const std::array<std::ptrdiff_t, 4> offsetscale_lengths{1, _num_beams, _num_frequencies, 2};
        const std::array<std::string, 4> offsetscale_dimnames{"Ttilde256", "R", "Fbar",
                                                              "offset/scale"};
        return NDArrayBuffer<float16_t, 4>(_gpu_mem_beams_offsetscale, "I3_offsetscale",
                                           offsetscale_lengths, offsetscale_dimnames, *this);
    }())
//
{
    if (_num_chunks % FRAME_SIZE)
        throw std::runtime_error("The num_chunks parameter must be a multiple of 32");

    set_command_type(gpuCommandType::KERNEL);
    set_name("cudaQuantize8");

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_input, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_beams, true, false, true));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_beams_offsetscale, true, false, true));
}

cudaQuantize8::~cudaQuantize8() {}

cudaEvent_t cudaQuantize8::execute(cudaPipelineState&, const std::vector<cudaEvent_t>&) {
    pre_execute();

    record_start_event();

    // Check metadata
    input_buffer.check_metadata();
    const auto& input_meta = input_buffer.get_metadata();

    // Set metadata
    beam_buffer.set_metadata(input_meta);
    offsetscale_buffer.set_metadata(input_meta);

    const auto& input_ndarray = input_buffer.get_ndarray();
    const float16_t* const input = input_ndarray.data();
    const int input_size1 = input_ndarray.extent(3); // time
    const int input_size2 = input_ndarray.extent(2); // freq
    const int input_size3 = input_ndarray.extent(1); // beam
    assert(input_ndarray.extent(0) == 1);
    assert(input_ndarray.stride(3) == 1);              // time
    const int input_stride2 = input_ndarray.stride(2); // freq
    const int input_stride3 = input_ndarray.stride(1); // beam

    auto& offsetscale_ndarray = offsetscale_buffer.get_ndarray();
    float16_t* const outputf = offsetscale_ndarray.data();
    const int outputf_size1 = offsetscale_ndarray.extent(3); // time
    const int outputf_size2 = offsetscale_ndarray.extent(2); // freq
    const int outputf_size3 = offsetscale_ndarray.extent(1); // beam
    assert(offsetscale_ndarray.extent(0) == 1);
    assert(offsetscale_ndarray.stride(3) == 1);                // time
    const int outputf_stride2 = offsetscale_ndarray.stride(2); // freq
    const int outputf_stride3 = offsetscale_ndarray.stride(1); // beam

    auto& beam_ndarray = beam_buffer.get_ndarray();
    std::uint8_t* const outputi = beam_ndarray.data();
    const int outputi_size1 = beam_ndarray.extent(3); // time
    const int outputi_size2 = beam_ndarray.extent(2); // freq
    const int outputi_size3 = beam_ndarray.extent(1); // beam
    assert(beam_ndarray.extent(0) == 1);
    assert(beam_ndarray.stride(3) == 1);                // time
    const int outputi_stride2 = beam_ndarray.stride(2); // freq
    const int outputi_stride3 = beam_ndarray.stride(1); // beam

    cudaStream_t stream = device.getStream(cuda_stream_id);

    gpu_quantize8(input, outputf, outputi,                                                       //
                  input_size1, input_size2, input_size3, input_stride2, input_stride3,           //
                  outputf_size1, outputf_size2, outputf_size3, outputf_stride2, outputf_stride3, //
                  outputi_size1, outputi_size2, outputi_size3, outputi_stride2, outputi_stride3, //
                  stream);
    CHECK_CUDA_ERROR(cudaGetLastError());

    return record_end_event();
}
