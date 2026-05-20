// 8-bit FRB beam quantizer for CHIME, using the float16 FRB beamformer and CHIME's
// sending-to-Bonsai network format

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

    // Input array layout
    static constexpr int in_ntimes = 256;
    static constexpr int in_nfreqs = 256; // 16 coarse channels, upchannelized by 16
    static constexpr int in_nbeams = 1024;

    // Output array layout
    // The output is quantized in "chunks". Each chunk has a different offset and scale.
    static constexpr int out_ntimes_chunk = 16;
    static constexpr int out_nfreqs_chunk = 16;
    // Chunks are combined into packets.
    static constexpr int out_nfreqs_packet = 4;
    static constexpr int out_nbeams_packet = 4;
    // There are several packets.
    static constexpr int out_ntimes_outer = 16;
    static constexpr int out_nfreqs_outer = 4;
    static constexpr int out_nbeams_outer = 256;

    static_assert(out_ntimes_chunk * out_ntimes_outer == in_ntimes);
    static_assert(out_nfreqs_chunk * out_nfreqs_packet * out_nfreqs_outer == in_nfreqs);
    static_assert(out_nbeams_packet * out_nbeams_outer == in_nbeams);

    const NDArrayBuffer<float16_t, 4> input_buffer;
    NDArrayBuffer<std::uint8_t, 8> beam_buffer;
    NDArrayBuffer<float16_t, 7> offsetscale_buffer;
};

REGISTER_CUDA_COMMAND(cudaQuantize8);

cudaQuantize8::cudaQuantize8(Config& config, const std::string& unique_name,
                             bufferContainer& host_buffers, cudaDeviceInterface& device, int inst) :
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
        return NDArrayBuffer<float16_t, 4>(_gpu_mem_input, "I2", input_lengths, input_dimnames,
                                           *this);
    }()),
    beam_buffer([&]() {
        const std::array<std::ptrdiff_t, 8> beam_lengths{1, out_nbeams_outer,
                                                         out_nfreqs_outer, out_ntimes_outer, out_nbeams_packet,
                                                         out_nfreqs_packet, out_nfreqs_chunk, out_ntimes_chunk};
        const std::array<std::string, 8> beam_dimnames{"Ttilde256",     "R4",        "Fbar64",
                                                       "Ttilde16_lo16", "Rlo4",      "Fbar16_lo4",
                                                       "Fbarlo16",      "Ttildelo16"};
        return NDArrayBuffer<std::uint8_t, 8>(_gpu_mem_beams, "I3", beam_lengths, beam_dimnames,
                                              *this);
    }()),
    offsetscale_buffer([&]() {
        const std::array<std::ptrdiff_t, 7> offsetscale_lengths{1, out_nbeams_outer,
                                                         out_nfreqs_outer, out_ntimes_outer, out_nbeams_packet,
                                                         out_nfreqs_packet, 2};
        const std::array<std::string, 7> offsetscale_dimnames{
            "Ttilde256", "R4", "Fbar64", "Ttilde16_lo16", "Rlo4", "Fbar16_lo4", "offset/scale"};
        return NDArrayBuffer<float16_t, 7>(_gpu_mem_beams_offsetscale, "I3_offsetscale",
                                           offsetscale_lengths, offsetscale_dimnames, *this);
    }())
//
{
    set_command_type(gpuCommandType::KERNEL);
    set_name("cudaQuantize8");

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_input, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_beams, true, false, true));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_beams_offsetscale, true, false, true));

    // these sizes are hard-coded in the CUDA kernel
    if (_num_beams != in_nbeams || _num_frequencies != in_nfreqs || _num_times != in_ntimes) {
        FATAL_ERROR("This stage's CUDA kernel hard-codes [num_beams, num_frequencies, num_times] to [{:d}, {:d}, {:d}] and does not support [{:d}, {:d}, {:d}]",
                    in_nbeams, in_nfreqs, in_ntimes, _num_beams, _num_frequencies, _num_times);
    }
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
    assert(input_ndarray.extent(0) == 1);

    auto& offsetscale_ndarray = offsetscale_buffer.get_ndarray();
    float16_t* const outputf = offsetscale_ndarray.data();

    auto& beam_ndarray = beam_buffer.get_ndarray();
    std::uint8_t* const outputi = beam_ndarray.data();

    cudaStream_t stream = device.getStream(cuda_stream_id);
    gpu_quantize8(input, outputf, outputi, stream);
    CHECK_CUDA_ERROR(cudaGetLastError());

    return record_end_event();
}
