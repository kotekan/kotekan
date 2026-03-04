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

    const bool _test_kernel;

    /// GPU side memory name for the time-stream input
    const std::string _gpu_mem_input;
    /// GPU side memory name for the time-stream output
    const std::string _gpu_mem_beams;
    /// GPU side memory name for mean,stdev output
    const std::string _gpu_mem_beams_meanstd;

    const NDArrayBuffer<float16_t, 3> input_buffer;
    NDArrayBuffer<std::uint8_t, 3> beam_buffer;
    NDArrayBuffer<float16_t, 4> meanstd_buffer;
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
    _test_kernel(config.get_default<bool>(unique_name, "test_kernel", false)),
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
        assert(_num_times % CHUNK_SIZE == 0);
        assert(_num_beams % FRAME_SIZE == 0);
        const std::array<std::ptrdiff_t, 3> beam_lengths{_num_beams, _num_frequencies, _num_times};
        const std::array<std::string, 3> beam_dimnames{"R", "Fbar", "Ttilde"};
        return NDArrayBuffer<std::uint8_t, 3>(_gpu_mem_beams, "frb3_beams", beam_lengths,
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
    set_name("cudaQuantize8");

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_input, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_beams, true, false, true));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_beams_meanstd, true, false, true));
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
    meanstd_buffer.set_metadata(input_meta);

    const auto& input_ndarray = input_buffer.get_ndarray();
    auto& meanstd_ndarray = meanstd_buffer.get_ndarray();
    auto& beam_ndarray = beam_buffer.get_ndarray();

    const float16_t* const input = input_ndarray.data();
    float16_t* const outputf = meanstd_ndarray.data();
    std::uint8_t* const outputi = beam_ndarray.data();
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

    gpu_quantize8(input, outputf, outputi, input_size1, input_size2, input_size3, input_stride2,
                  input_stride3, outputf_size1, outputf_size2, outputf_size3, outputf_stride2,
                  outputf_stride3, outputi_size1, outputi_size2, outputi_size3, outputi_stride2,
                  outputi_stride3, stream);
    CHECK_CUDA_ERROR(cudaGetLastError());

    if (_test_kernel) {
        // Copy inputs from GPU
        std::vector<float16_t> cpu_input_buffer(input_ndarray.get_size());
        CHECK_CUDA_ERROR(cudaMemcpy(cpu_input_buffer.data(), input,
                                    cpu_input_buffer.size() * sizeof *cpu_input_buffer.data(),
                                    cudaMemcpyDeviceToHost));
        // Copy GPU outputs from GPU
        std::vector<float16_t> gpu_meanstd_buffer(meanstd_ndarray.get_size());
        CHECK_CUDA_ERROR(cudaMemcpy(gpu_meanstd_buffer.data(), outputf,
                                    gpu_meanstd_buffer.size() * sizeof *gpu_meanstd_buffer.data(),
                                    cudaMemcpyDeviceToHost));
        std::vector<std::uint8_t> gpu_beam_buffer(beam_ndarray.get_size());
        CHECK_CUDA_ERROR(cudaMemcpy(gpu_beam_buffer.data(), outputi,
                                    gpu_beam_buffer.size() * sizeof *gpu_beam_buffer.data(),
                                    cudaMemcpyDeviceToHost));
        // Re-calculate results on CPU
        std::vector<float16_t> cpu_meanstd_buffer(meanstd_ndarray.get_size());
        std::vector<std::uint8_t> cpu_beam_buffer(beam_ndarray.get_size());
        cpu_quantize8(cpu_input_buffer.data(), cpu_meanstd_buffer.data(), cpu_beam_buffer.data(),
                      input_size1, input_size2, input_size3, input_stride2, input_stride3,
                      outputf_size1, outputf_size2, outputf_size3, outputf_stride2, outputf_stride3,
                      outputi_size1, outputi_size2, outputi_size3, outputi_stride2,
                      outputi_stride3);
        // Compare results
        for (std::size_t n = 0; n < cpu_meanstd_buffer.size(); ++n) {
            const float16_t x = gpu_meanstd_buffer.at(n);
            assert(isfinite(float(x)));
            assert(x == cpu_meanstd_buffer.at(n));
        }
        for (std::size_t n = 0; n < cpu_beam_buffer.size(); ++n) {
            const std::uint8_t x = gpu_beam_buffer.at(n);
            assert(x != 0 && x != 255);
            assert(x == cpu_beam_buffer.at(n));
        }
    }

    return record_end_event();
}
