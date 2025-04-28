#include "cudaQuantize.hpp"

#include <DataType.hpp>
#include <NDArrayBuffer.hpp>
#include <array>
#include <chordMetadata.hpp>
#include <cmath>
#include <cstdint>
#include <cudaUtils.hpp>
#include <string>
#include <vector>

void launch_quantize_kernel(cudaStream_t stream, int nframes, const __half2* in_base,
                            __half2* outf_base, unsigned int* outi_base, const int* index_array);

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaQuantize);

cudaQuantize::cudaQuantize(Config& config, const std::string& unique_name,
                           bufferContainer& host_buffers, cudaDeviceInterface& device, int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst) {
    _num_chunks = config.get<int64_t>(unique_name, "num_chunks");
    _gpu_mem_input = config.get<std::string>(unique_name, "gpu_mem_input");
    _gpu_mem_output = config.get<std::string>(unique_name, "gpu_mem_output");
    _gpu_mem_meanstd = config.get<std::string>(unique_name, "gpu_mem_meanstd");
    _gpu_mem_index = unique_name + "/index";
    if (_num_chunks % FRAME_SIZE)
        throw std::runtime_error("The num_chunks parameter must be a multiple of 32");

    set_command_type(gpuCommandType::KERNEL);
    set_name("cudaQuantize");

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_input, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_output, true, false, true));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_meanstd, true, false, true));
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_index", false, true, true));

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

    // Check metadata
    const std::shared_ptr<const metadataObject> in_mc =
        device.get_gpu_memory_array_metadata(_gpu_mem_input, gpu_frame_id);
    assert(in_mc);
    assert(metadata_is_chord(in_mc));
    const std::shared_ptr<const chordMetadata> in_meta = get_chord_metadata(in_mc);
    assert(in_meta->get_name() == "I2");
    assert(in_meta->type == kotekan::float16);
    assert(in_meta->dims == 3);
    assert(in_meta->get_dimension_name(0) == "R");
    assert(in_meta->get_dimension_name(1) == "Fbar");
    assert(in_meta->get_dimension_name(2) == "Ttilde");

    const size_t input_frame_len = (size_t)_num_chunks * CHUNK_SIZE * sizeof(float16_t);
    assert(input_frame_len
           == type_total_bytes(in_meta->type) * in_meta->dim[0] * in_meta->dim[1]
                  * in_meta->dim[2]);

    const void* const input_memory = device.get_gpu_memory_array(
        _gpu_mem_input, gpu_frame_id, _gpu_buffer_depth, input_frame_len);
    INFO("Input frame length: {:d} x {:d} x 2 = {:d}", _num_chunks, CHUNK_SIZE, input_frame_len);

    const size_t index_array_len = (size_t)_num_chunks * 2 * sizeof(int32_t);
    const int32_t* const index_array_memory =
        (const int32_t*)device.get_gpu_memory(_gpu_mem_index, index_array_len);

    // The data are stored as 4-bit integers, 2 values per byte
    assert(in_meta->dim[0] % 2 == 0);
    const std::array<std::ptrdiff_t, 3> beam_lengths{in_meta->dim[0] / 2, in_meta->dim[1],
                                                     in_meta->dim[2]};
    const std::array<std::string, 3> beam_dimnames{"R", "Fbar", "Ttilde"};
    NDArrayBuffer<kotekan::uint4x2_t, 3> beam_buffer("frb3_beams", beam_lengths, beam_dimnames,
                                                     device, cuda_stream_id, _gpu_buffer_depth,
                                                     gpu_frame_id);

    assert(in_meta->dim[2] % CHUNK_SIZE == 0);
    static_assert(CHUNK_SIZE == 256);
    const std::array<std::ptrdiff_t, 4> meanstd_lengths{in_meta->dim[0], in_meta->dim[1],
                                                        in_meta->dim[2] / CHUNK_SIZE, 2};
    const std::array<std::string, 4> meanstd_dimnames{"R", "Fbar", "Ttilde256", "mean/std"};
    NDArrayBuffer<float16_t, 4> meanstd_buffer("frb3_beams_meanstd", meanstd_lengths,
                                               meanstd_dimnames, device, cuda_stream_id,
                                               _gpu_buffer_depth, gpu_frame_id);

    void* const beam_memory = beam_buffer.get_ndarray().data();
    void* const meanstd_memory = meanstd_buffer.get_ndarray().data();

    record_start_event();

    launch_quantize_kernel(device.getStream(cuda_stream_id), _num_chunks / FRAME_SIZE,
                           (const __half2*)input_memory, (__half2*)meanstd_memory,
                           (unsigned int*)beam_memory, index_array_memory);
    CHECK_CUDA_ERROR(cudaGetLastError());

    beam_buffer.set_metadata(in_meta);
    meanstd_buffer.set_metadata(in_meta);

    return record_end_event();
}
