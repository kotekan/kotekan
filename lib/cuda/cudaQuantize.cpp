#include "cudaQuantize.hpp"

#include <DataType.hpp>
#include <array>
#include <chordMetadata.hpp>
#include <cmath>
#include <cstdint>
#include <cudaUtils.hpp>
#include <div.hpp>
#include <string>
#include <vector>

void launch_quantize_kernel(cudaStream_t stream, int nframes, const __half2* in_base,
                            __half2* outf_base, unsigned int* outi_base, const int* index_array);

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
    _gpu_mem_index(unique_name + "/index"),
    //
    input_buffer([&]() {
        const std::array<std::ptrdiff_t, 3> input_lengths{_num_beams, _num_frequencies, _num_times};
        const std::array<std::string, 3> input_dimnames{"R", "Fbar", "Ttilde"};
        return NDArrayBuffer<float16_t, 3>(_gpu_mem_input, "frb2_beams", input_lengths,
                                           input_dimnames, *this);
    }()),
    beam_buffer([&]() {
        // The data are stored as 4-bit integers, 2 values per byte
        assert(_num_beams % 2 == 0);
        const std::array<std::ptrdiff_t, 3> beam_lengths{_num_beams / 2, _num_frequencies,
                                                         _num_times};
        const std::array<std::string, 3> beam_dimnames{"R", "Fbar", "Ttilde"};
        return NDArrayBuffer<kotekan::uint4x2_t, 3>(_gpu_mem_beams, "frb3_beams", beam_lengths,
                                                    beam_dimnames, *this);
    }()),
    meanstd_buffer([&]() {
        assert(_num_times % CHUNK_SIZE == 0);
        static_assert(CHUNK_SIZE == 256);
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
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_index, false, true, true));

    const size_t index_array_len = (size_t)_num_chunks * 2 * sizeof(int32_t);
    int32_t* const index_array_memory =
        (int32_t*)device.get_gpu_memory(_gpu_mem_index, index_array_len);

    std::vector<int32_t> index_array_host(index_array_len / sizeof(int32_t), 0);

    for (int f = 0; f < _num_chunks / FRAME_SIZE; f++) {
        for (int i = 0; i < FRAME_SIZE; i++) {
            // offset of the start of the chunk in the input array, for each chunk.
            index_array_host[f * FRAME_SIZE * 2 + i] = (f * FRAME_SIZE + i);
        }
        // offset for the mean/scale outputs per chunk
        index_array_host[f * FRAME_SIZE * 2 + FRAME_SIZE] = f * FRAME_SIZE;
        // offset for the output integers per frame;
        // this is in units of int32s, and the outputs are int4s, hence the divide by 8.
        index_array_host[f * FRAME_SIZE * 2 + FRAME_SIZE + 1] = f * FRAME_SIZE * (CHUNK_SIZE / 8);
    }

    CHECK_CUDA_ERROR(cudaMemcpy(index_array_memory, index_array_host.data(), index_array_len,
                                cudaMemcpyHostToDevice));
}

cudaQuantize::~cudaQuantize() {}

cudaEvent_t cudaQuantize::execute(cudaPipelineState&, const std::vector<cudaEvent_t>&) {
    pre_execute();

    const std::shared_ptr<const metadataObject> input_mc =
        device.get_gpu_memory_array_metadata(_gpu_mem_input + "_buffer", gpu_frame_id);
    assert(input_mc);
    assert(metadata_is_chord(input_mc));
    const std::shared_ptr<const chordMetadata> input_meta = get_chord_metadata(input_mc);
    assert(input_meta->dims == 3);
    assert(input_meta->dim[0] % 2 == 0);
    static_assert(CHUNK_SIZE == 256);

    // TODO: Avoid the index array
    const size_t index_array_len = (size_t)_num_chunks * 2 * sizeof(int32_t);
    const int32_t* const index_array_memory =
        (const int32_t*)device.get_gpu_memory(_gpu_mem_index, index_array_len);

    record_start_event();

    // Check metadata
    input_buffer.check_metadata();

    const void* const input_memory = input_buffer.get_ndarray().data();
    void* const beam_memory = beam_buffer.get_ndarray().data();
    void* const meanstd_memory = meanstd_buffer.get_ndarray().data();

    launch_quantize_kernel(device.getStream(cuda_stream_id), _num_chunks / FRAME_SIZE,
                           (const __half2*)input_memory, (__half2*)meanstd_memory,
                           (unsigned int*)beam_memory, index_array_memory);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Set metadata
    beam_buffer.set_metadata(input_meta);
    meanstd_buffer.set_metadata(input_meta);

    return record_end_event();
}
