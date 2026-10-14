// 8-bit FRB beam quantizer for CHIME, using the float16 FRB beamformer and CHIME's
// sending-to-Bonsai network format

#include "Config.hpp"              // for Config
#include "DataType.hpp"            // for float16_t
#include "NDArray.hpp"             // for NDArray
#include "NDArrayBuffer.hpp"       // for NDArrayBuffer
#include "NDArrayRingBuffer.hpp"   // for NDArrayRingBuffer, read_descriptor_t
#include "bufferContainer.hpp"     // for bufferContainer
#include "chordMetadata.hpp"       // for chordMetadata
#include "cudaCommand.hpp"         // for cudaCommand, cudaPipelineState, REGISTER_CUDA_COMMAND
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "cudaQuantizeKernel8.hpp" // for gpu_quantize8
#include "cudaUtils.hpp"           // for CHECK_CUDA_ERROR
#include "div.hpp"                 // for div_noremainder
#include "driver_types.h"          // for cudaEvent_t, CUevent_st, CUstream_st, cudaStream_t
#include "gpuCommand.hpp"          // for gpuCommandType
#include "kotekanLogging.hpp"      // for FATAL_ERROR
#include "metadata.hpp"            // for metadataObject

#include "fmt.hpp" // for format

#include <algorithm>          // for max
#include <array>              // for array
#include <cassert>            // for assert
#include <cstddef>            // for ptrdiff_t
#include <cstdint>            // for int64_t, uint8_t
#include <cuda_fp16.h>        // for __half
#include <cuda_runtime_api.h> // for cudaGetLastError
#include <stdexcept>          // for runtime_error
#include <string>             // for allocator, basic_string, string
#include <tuple>              // for tuple, make_tuple
#include <vector>             // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::mod;

/**
 * @class cudaQuantize8
 * @brief cudaCommand quantizing the FRB2 beams to 8 bits.
 *
 * The beams are quantized in chunks, each with its own offset and scale. Samples that the packet
 * loss mask `gpu_mem_pl_mask` marks as invalid are skipped when choosing the offset and scale,
 * and are encoded as 0, the code that also represents nan. The mask is the ring buffer written by
 * `cudaPLMaskFRB1`; each frame reads `num_times` of its samples, as `cudaFRBBeamReformer` reads the
 * FRB1 beams.
 *
 * @conf  frb1_max_num_times        Int.  Size of the packet loss mask ring buffer, in samples.
 * @conf  frb1_max_num_frequencies  Int.  Number of frequencies of the packet loss mask.
 * @conf  gpu_mem_pl_mask           String.  Name of the packet loss mask.
 */
class cudaQuantize8 : public cudaCommand {
public:
    cudaQuantize8(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device, int inst);
    ~cudaQuantize8();
    int wait_on_precondition() override;
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame() override;

private:
    const int _frb_downsampling_factor;
    const int _num_beams;
    const int _num_frequencies;
    const int _num_times;
    const int _frb1_max_num_times;
    const int _frb1_max_num_frequencies;

    /// GPU side memory name for the time-stream input
    const std::string _gpu_mem_input;
    /// GPU side memory name for the time-stream output
    const std::string _gpu_mem_beams;
    /// GPU side memory name for mean,stdev output
    const std::string _gpu_mem_beams_offsetscale;
    /// GPU side memory name for the packet loss mask input
    const std::string _gpu_mem_pl_mask;

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
    NDArrayRingBuffer<std::uint8_t, 2> pl_mask_buffer;
    NDArrayBuffer<std::uint8_t, 8> beam_buffer;
    NDArrayBuffer<float16_t, 7> offsetscale_buffer;
};

REGISTER_CUDA_COMMAND(cudaQuantize8);

cudaQuantize8::cudaQuantize8(Config& config, const std::string& unique_name,
                             bufferContainer& host_buffers, cudaDeviceInterface& device, int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst),
    //
    _frb_downsampling_factor(config.get<int>(unique_name, "frb_downsampling_factor")),
    _num_beams(config.get<std::int64_t>(unique_name, "num_beams")),
    _num_frequencies(config.get<std::int64_t>(unique_name, "num_frequencies")),
    _num_times(config.get<std::int64_t>(unique_name, "num_times")),
    _frb1_max_num_times(config.get<int>(unique_name, "frb1_max_num_times")),
    _frb1_max_num_frequencies(config.get<int>(unique_name, "frb1_max_num_frequencies")),
    //
    _gpu_mem_input(config.get<std::string>(unique_name, "gpu_mem_input")),
    _gpu_mem_beams(config.get<std::string>(unique_name, "gpu_mem_output")),
    _gpu_mem_beams_offsetscale(config.get<std::string>(unique_name, "gpu_mem_offsetscale")),
    _gpu_mem_pl_mask(config.get<std::string>(unique_name, "gpu_mem_pl_mask")),
    //
    input_buffer([&]() {
        const std::array<std::ptrdiff_t, 4> input_lengths{1, _num_beams, _num_frequencies,
                                                          _num_times};
        const std::array<std::string, 4> input_dimnames{"Ttildehi256", "R", "Fbar", "Ttildelo256"};
        const std::array<std::ptrdiff_t, 4> input_dimscalings{_frb_downsampling_factor * 256, 1, 1,
                                                              _frb_downsampling_factor};
        return NDArrayBuffer<float16_t, 4>(_gpu_mem_input, "I2", input_lengths, input_dimnames,
                                           input_dimscalings, *this);
    }()),
    pl_mask_buffer(_gpu_mem_pl_mask, "pl_mask_I",
                   std::array<std::ptrdiff_t, 2>{_frb1_max_num_times, _frb1_max_num_frequencies},
                   std::array<std::string, 2>{"Ttilde", "Fbar"},
                   std::array<std::ptrdiff_t, 2>{_frb_downsampling_factor, 1}, *this),
    beam_buffer([&]() {
        const std::array<std::ptrdiff_t, 8> beam_lengths{1,
                                                         out_nbeams_outer,
                                                         out_nfreqs_outer,
                                                         out_ntimes_outer,
                                                         out_nbeams_packet,
                                                         out_nfreqs_packet,
                                                         out_nfreqs_chunk,
                                                         out_ntimes_chunk};
        const std::array<std::string, 8> beam_dimnames{"Ttilde256",     "R4",        "Fbar64",
                                                       "Ttilde16_lo16", "Rlo4",      "Fbar16_lo4",
                                                       "Fbarlo16",      "Ttildelo16"};
        const std::array<std::ptrdiff_t, 8> beam_dimscalings{
            _frb_downsampling_factor * 256, 4, 64, _frb_downsampling_factor * 16, 1, 16, 1,
            _frb_downsampling_factor};
        return NDArrayBuffer<std::uint8_t, 8>(_gpu_mem_beams, "I3", beam_lengths, beam_dimnames,
                                              beam_dimscalings, *this);
    }()),
    offsetscale_buffer([&]() {
        const std::array<std::ptrdiff_t, 7> offsetscale_lengths{1,
                                                                out_nbeams_outer,
                                                                out_nfreqs_outer,
                                                                out_ntimes_outer,
                                                                out_nbeams_packet,
                                                                out_nfreqs_packet,
                                                                2};
        const std::array<std::string, 7> offsetscale_dimnames{
            "Ttilde256", "R4", "Fbar64", "Ttilde16_lo16", "Rlo4", "Fbar16_lo4", "offset/scale"};
        const std::array<std::ptrdiff_t, 7> offsetscale_dimscalings{
            _frb_downsampling_factor * 256, 4, 64, _frb_downsampling_factor * 16, 1, 16, 1};
        return NDArrayBuffer<float16_t, 7>(_gpu_mem_beams_offsetscale, "I3_offsetscale",
                                           offsetscale_lengths, offsetscale_dimnames,
                                           offsetscale_dimscalings, *this);
    }())
//
{
    set_command_type(gpuCommandType::KERNEL);
    set_name("cudaQuantize8");

    input_buffer.register_consumer();
    pl_mask_buffer.register_consumer();
    beam_buffer.register_producer();
    offsetscale_buffer.register_producer();

    // Each frame reads `_num_times` mask samples, which must not wrap around the ring buffer
    if (!(_frb1_max_num_frequencies >= _num_frequencies && _frb1_max_num_times % _num_times == 0))
        FATAL_ERROR("The packet loss mask ring buffer [{:d}, {:d}] must hold at least the {:d} "
                    "frequencies, and a multiple of the {:d} times, of a frame",
                    _frb1_max_num_times, _frb1_max_num_frequencies, _num_frequencies, _num_times);

    // these sizes are hard-coded in the CUDA kernel
    if (_num_beams != in_nbeams || _num_frequencies != in_nfreqs || _num_times != in_ntimes) {
        FATAL_ERROR("This stage's CUDA kernel hard-codes [num_beams, num_frequencies, num_times] "
                    "to [{:d}, {:d}, {:d}] and does not support [{:d}, {:d}, {:d}]",
                    in_nbeams, in_nfreqs, in_ntimes, _num_beams, _num_frequencies, _num_times);
    }
}

cudaQuantize8::~cudaQuantize8() {}

int cudaQuantize8::wait_on_precondition() {
    {
        const int errcode = cudaCommand::wait_on_precondition();
        if (errcode < 0)
            return errcode;
    }

    // Read the packet loss mask for this frame's samples
    const int errcode = pl_mask_buffer.wait_and_claim_readable([&](const std::ptrdiff_t available) {
        if (available < _num_times)
            return read_descriptor_t{.claimed = 0, .read = 0};
        return read_descriptor_t{.claimed = _num_times, .read = _num_times};
    });
    if (errcode < 0)
        return errcode;

    return 0;
}

cudaEvent_t cudaQuantize8::execute(cudaPipelineState&, const std::vector<cudaEvent_t>&) {
    pre_execute();

    record_start_event();

    // Check metadata
    input_buffer.check_metadata();
    const auto& input_meta = input_buffer.get_metadata();
    pl_mask_buffer.check_metadata();
    const auto& pl_mask_meta = pl_mask_buffer.get_metadata();

    // The packet loss mask must describe the input samples
    const std::ptrdiff_t pl_mask_offset = pl_mask_buffer.get_read_valid().begin();
    const std::int64_t pl_mask_fpga_seq_num =
        pl_mask_meta->get_fpga_seq_num()
        + pl_mask_offset * std::int64_t(pl_mask_meta->get_time_downsampling_fpga());
    if (pl_mask_fpga_seq_num != input_meta->get_fpga_seq_num())
        FATAL_ERROR("The packet loss mask {:s} begins at fpga_seq_num {:d}, but the input {:s} "
                    "begins at {:d}",
                    _gpu_mem_pl_mask, pl_mask_fpga_seq_num, _gpu_mem_input,
                    input_meta->get_fpga_seq_num());
    for (int freq = 0; freq < _num_frequencies; ++freq)
        if (pl_mask_meta->get_coarse_freq().at(freq) != input_meta->get_coarse_freq().at(freq)
            || pl_mask_meta->get_freq_upchan_index().at(freq)
                   != input_meta->get_freq_upchan_index().at(freq))
            FATAL_ERROR("The packet loss mask {:s} and the input {:s} describe different "
                        "frequencies at index {:d}",
                        _gpu_mem_pl_mask, _gpu_mem_input, freq);

    // Set metadata
    beam_buffer.set_metadata(input_meta);
    offsetscale_buffer.set_metadata(input_meta);

    const auto& input_ndarray = input_buffer.get_ndarray();
    const float16_t* const input = input_ndarray.data();
    assert(input_ndarray.extent(0) == 1);

    const auto& pl_mask_ndarray = pl_mask_buffer.get_ndarray();
    const int pl_mask_stride = pl_mask_ndarray.stride(0);
    const std::uint8_t* const pl_mask =
        pl_mask_ndarray.data() + mod(pl_mask_offset, pl_mask_ndarray.extent(0)) * pl_mask_stride;

    auto& offsetscale_ndarray = offsetscale_buffer.get_ndarray();
    float16_t* const outputf = offsetscale_ndarray.data();

    auto& beam_ndarray = beam_buffer.get_ndarray();
    std::uint8_t* const outputi = beam_ndarray.data();

    cudaStream_t stream = device.getStream(cuda_stream_id);
    gpu_quantize8(input, pl_mask, pl_mask_stride, outputf, outputi, stream);
    CHECK_CUDA_ERROR(cudaGetLastError());

    return record_end_event();
}

void cudaQuantize8::finalize_frame() {
    pl_mask_buffer.finish_read();

    cudaCommand::finalize_frame();
}
