#include "cudaQuantize.hpp"

#include "DataType.hpp"            // for uint4x2_t, float16_t
#include "NDArray.hpp"             // for NDArray
#include "NDArrayBuffer.hpp"       // for NDArrayBuffer
#include "chordMetadata.hpp"       // for chordMetadata, get_chord_metadata, metadata_is_chord
#include "cudaCommand.hpp"         // for cudaCommand, REGISTER_CUDA_COMMAND, _factory_aliascud...
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "cudaUtils.hpp"           // for CHECK_CUDA_ERROR
#include "div.hpp"                 // for div_noremainder
#include "gpuCommand.hpp"          // for gpuCommandType
#include "metadata.hpp"            // for metadataObject

#include <algorithm>          // for max
#include <array>              // for array
#include <cassert>            // for assert
#include <cstddef>            // for size_t, ptrdiff_t
#include <cstdint>            // for int64_t
#include <cuda_fp16.h>        // for __half2, __half
#include <cuda_runtime_api.h> // for cudaGetLastError, cudaMemcpy
#include <memory>             // for shared_ptr, __shared_ptr_access
#include <stdexcept>          // for runtime_error
#include <string>             // for allocator, basic_string, string, operator+
#include <tuple>              // for tuple, make_tuple
#include <vector>             // for vector

void gpu_quantize4(const __half* __restrict__ const input, __half* __restrict__ const outputf,
                   kotekan::uint4x2_t* __restrict__ const outputi, const int input_size1,
                   const int input_size2, const int input_size3, const int input_stride2,
                   const int input_stride3, const int outputf_size1, const int outputf_size2,
                   const int outputf_size3, const int outputf_stride2, const int outputf_stride3,
                   const int outputi_size1, const int outputi_size2, const int outputi_size3,
                   const int outputi_stride2, const int outputi_stride3, cudaStream_t stream);

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaQuantize);

cudaQuantize::cudaQuantize(Config& config, const std::string& unique_name,
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
    _gpu_mem_beams_meanstd(config.get<std::string>(unique_name, "gpu_mem_meanstd")),
    //
    input_buffer([&]() {
        const std::array<std::ptrdiff_t, 3> input_lengths{_num_beams, _num_frequencies, _num_times};
        const std::array<std::string, 3> input_dimnames{"R", "Fbar", "Ttilde"};
        return NDArrayBuffer<float16_t, 3>(_gpu_mem_input, "frb2_beams", input_lengths,
                                           input_dimnames, *this);
    }()),
    beam_buffer([&]() {
        // The data are stored as 4-bit integers, 2 values per byte
        assert(_num_times % 2 == 0);
        assert(_num_times % CHUNK_SIZE == 0);
        assert(_num_beams % FRAME_SIZE == 0);
        const std::array<std::ptrdiff_t, 3> beam_lengths{_num_beams, _num_frequencies,
                                                         _num_times / 2};
        const std::array<std::string, 3> beam_dimnames{"R", "Fbar", "Ttilde"};
        return NDArrayBuffer<kotekan::uint4x2_t, 3>(_gpu_mem_beams, "frb3_beams", beam_lengths,
                                                    beam_dimnames, *this);
    }()),
    meanstd_buffer([&]() {
        assert(_num_times % CHUNK_SIZE == 0);
        assert(_num_beams % FRAME_SIZE == 0);
        const std::array<std::ptrdiff_t, 4> meanstd_lengths{_num_beams, _num_frequencies,
                                                            _num_times / CHUNK_SIZE, 2};
        const std::array<std::string, 4> meanstd_dimnames{"R", "Fbar", "Ttilde256", "mean/std"};
        return NDArrayBuffer<float16_t, 4>(_gpu_mem_beams_meanstd, "frb3_beams_meanstd",
                                           meanstd_lengths, meanstd_dimnames, *this);
    }())
//
{
    if (_num_chunks % FRAME_SIZE)
        throw std::runtime_error("The num_chunks parameter must be a multiple of 32");

    set_command_type(gpuCommandType::KERNEL);
    set_name("cudaQuantize");

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_input, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_beams, true, false, true));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_beams_meanstd, true, false, true));
}

cudaQuantize::~cudaQuantize() {}

cudaEvent_t cudaQuantize::execute(cudaPipelineState&, const std::vector<cudaEvent_t>&) {
    pre_execute();

    record_start_event();

    // Check metadata
    input_buffer.check_metadata();
    const auto& input_meta = input_buffer.get_metadata();

    // Set metadata
    beam_buffer.set_metadata(input_meta);
    meanstd_buffer.set_metadata(input_meta);

    const auto& input_ndarray = input_buffer.get_ndarray();
    auto& meanstd_ndarray = meanstd_buffer.get_ndarray();
    auto& beam_ndarray = beam_buffer.get_ndarray();

    const float16_t* const input = input_ndarray.data();
    float16_t* const outputf = meanstd_ndarray.data();
    kotekan::uint4x2_t* const outputi = beam_ndarray.data();
    const int input_size1 = input_ndarray.extent(2);
    const int input_size2 = input_ndarray.extent(1);
    const int input_size3 = input_ndarray.extent(0);
    assert(input_ndarray.stride(2) == 1);
    const int input_stride2 = input_ndarray.stride(1);
    const int input_stride3 = input_ndarray.stride(0);
    assert(meanstd_ndarray.extent(3) == 2);
    const int outputf_size1 = meanstd_ndarray.extent(2) * 2;
    const int outputf_size2 = meanstd_ndarray.extent(1);
    const int outputf_size3 = meanstd_ndarray.extent(0);
    assert(meanstd_ndarray.stride(3) == 1);
    assert(meanstd_ndarray.stride(2) == 2);
    const int outputf_stride2 = meanstd_ndarray.stride(1);
    const int outputf_stride3 = meanstd_ndarray.stride(0);
    const int outputi_size1 = beam_ndarray.extent(2);
    const int outputi_size2 = beam_ndarray.extent(1);
    const int outputi_size3 = beam_ndarray.extent(0);
    assert(beam_ndarray.stride(2) == 1);
    const int outputi_stride2 = beam_ndarray.stride(1);
    const int outputi_stride3 = beam_ndarray.stride(0);
    cudaStream_t stream = device.getStream(cuda_stream_id);

    gpu_quantize4(input, outputf, outputi, input_size1, input_size2, input_size3, input_stride2,
                  input_stride3, outputf_size1, outputf_size2, outputf_size3, outputf_stride2,
                  outputf_stride3, outputi_size1, outputi_size2, outputi_size3, outputi_stride2,
                  outputi_stride3, stream);
    CHECK_CUDA_ERROR(cudaGetLastError());

    return record_end_event();
}
