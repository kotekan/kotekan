/**
 * @file
 * @brief CUDA FRB beamforming kernel, final stage
 *  - cudaFRBBeamReformer : public cudaCommand
 */

#include "Config.hpp"              // for Config
#include "DataType.hpp"            // for float16_t, DataType
#include "buffer.hpp"              // for GenericBuffer
#include "bufferContainer.hpp"     // for bufferContainer
#include "cudaCommand.hpp"         // for cudaCommand, REGISTER_CUDA_COMMAND
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "div.hpp"                 // for div_noremainder, mod
#include "driver_types.h"          // for CUevent_st, cudaEvent_t, CUstream_st
#include "gpuCommand.hpp"          // for gpuCommandType
#include "kotekanLogging.hpp"      // for DEBUG, ERROR
#include "metadata.hpp"            // for metadataObject
#include "ringbuffer.hpp"          // for RingBuffer

#include <algorithm>         // for max
#include <cassert>           // for assert
#include <chordMetadata.hpp> // for chordMetadata, get_chord_metadata, metadata_is_chord
#include <cstddef>           // for size_t, ptrdiff_t
#include <cstdint>           // for int32_t
#include <cstdlib>           // for abort
#include <cublas_api.h>      // for cublasContext, cublasHandle_t
#include <cublas_v2.h>       // for cublasCreate, cublasDestroy, cublasSetStream
#include <driver_types.h>    // for cudaEvent_t
#include <fmt.hpp>           // for compile_string_to_view
#include <memory>            // for shared_ptr, __shared_ptr_access
#include <optional>          // for optional
#include <string>            // for string, basic_string
#include <tuple>             // for tuple, make_tuple
#include <vector>            // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::div_noremainder;
using kotekan::mod;

/**
 * @class cudaFRBBeamReformer
 * @brief cudaCommand for doing final FRB beamforming.  This follows the cudaFRBBeamFormer.
 *
 * Core code was developed by Nada El-Falou in https://github.com/nadafalou/CHORD/blob/main/mmul.cu
 *
 * The phase matrix for the beam locations uses the correct math but
 * with a lot of placeholder assumptions.  This will need to get
 * revisited in post-MVP development.
 */
class cudaFRBBeamReformer : public cudaCommand {
public:
    cudaFRBBeamReformer(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                        int inst);
    ~cudaFRBBeamReformer();
    int wait_on_precondition() override;
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame() override;

private:
    cublasHandle_t handle;

  // Number of time samples per beam packet
  int samples_per_data_set;

  // Number of frequencies to process
  int num_frequencies;

  // Buffers
  NDArrayBuffer<float16_t, 4> phase_buffer;
  NDArrayRingBuffer<float16_t, 4> input_beam_buffer;
  NDArrayBuffer<float16_t, 3> output_beam_buffer;
};




    // Common configuration values (which do not change in a run)
    /// Number of output beams
    const  std::int32_t num_output_beams;
    /// Beam-grid size produced by cudaFRBBeamformer
    int32_t _beam_grid_size_P, _beam_grid_size_Q;
    /// Maximum number of frequencies per data stream sent to each node (determines the buffer
    /// size).
    int32_t _max_num_local_freq;
    /// Number of frequencies per data stream sent to each node (determines which part of the buffer
    /// is used).
    int32_t _num_local_freq;
    /// Total samples in each dataset
    int32_t _Td;

    // Computed values
    /// Number of input beams
    int32_t num_input_beams;

    /// Size in bytes of the input beamgrid array
    size_t beamgrid_len;
    /// Size in bytes of the input phase array
    size_t phase_len;
    /// Size in bytes of the output beam array
    size_t beamout_len;

    /// GPU side memory name for the beam-grid input
    std::string _gpu_mem_beamgrid;
    /// GPU side memory name for the beamforming phases
    std::string _gpu_mem_phase;
    /// GPU side memory name for the beam output
    std::string _gpu_mem_beamout;

    // cublasHgemmBatched -- pre-computed GPU memory locations.
    // [freq batch = stream] = [per-freq pointers]
    std::vector<float16_t**> _gpu_in_pointers;
    std::vector<float16_t**> _gpu_out_pointers;
    std::vector<float16_t**> _gpu_phase_pointers;

    // Signalling ring buffer for the input (raw FRB beams) data
    RingBuffer* input_ringbuf_signal;
    // TODO NDArrayRingBuffer<kotekan::float16, 4> input;

    // Byte count in the ring buffer to read from (may be larger than buffer size)
    // (Set in `wait_for_precondition`, invalid after `finalize_frame`)
    std::ptrdiff_t input_cursor;

    // Byte offset in the ring buffer to read from (modulo buffer size)
    // (Set in `wait_for_precondition`, invalid after `finalize_frame`)
    std::ptrdiff_t input_position;
};

REGISTER_CUDA_COMMAND(cudaFRBBeamReformer);

cudaFRBBeamReformer::cudaFRBBeamReformer(Config& config, const std::string& unique_name,
                                         bufferContainer& host_buffers, cudaDeviceInterface& device,
                                         int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst) {

    // Number of output beams
    _num_beams = config.get<int>(unique_name, "num_beams");
    // Number of input beams
    _beam_grid_size_P = config.get<int>(unique_name, "beam_grid_size_P");
    _beam_grid_size_Q = config.get<int>(unique_name, "beam_grid_size_Q");
    num_input_beams = _beam_grid_size_P * _beam_grid_size_Q;
    // Number of frequencies
    _max_num_local_freq = config.get<int>(unique_name, "max_num_local_freq");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    assert(_num_local_freq <= _max_num_local_freq);
    // Number of time samples
    _Td = config.get<int>(unique_name, "samples_per_data_set");

    // Input and output buffer names
    _gpu_mem_beamgrid = config.get<std::string>(unique_name, "gpu_mem_beamgrid");
    _gpu_mem_phase = config.get<std::string>(unique_name, "gpu_mem_phase");
    _gpu_mem_beamout = config.get<std::string>(unique_name, "gpu_mem_beamout");

    // Calculate buffer sizes (in bytes)
    beamgrid_len = sizeof(float16_t) * num_input_beams * _max_num_local_freq * _Td;
    phase_len = sizeof(float16_t) * num_input_beams * _num_beams * _num_local_freq;
    beamout_len = sizeof(float16_t) * _num_beams * _num_local_freq * _Td;

    // Find input buffer used for signalling ring-buffer state
    input_ringbuf_signal = dynamic_cast<RingBuffer*>(
        host_buffers.get_generic_buffer(config.get<std::string>(unique_name, "in_signal")));
    if (inst == 0)
        input_ringbuf_signal->register_consumer(unique_name);

    // Add Graphviz entries for the GPU buffers used by this kernel
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_beamgrid, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_phase, false, true, true));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_beamout, true, false, true));

    // Kotekan stuff
    set_command_type(gpuCommandType::KERNEL);
    set_name("FRB_beamreformer");

    // Create cuBLAS handle
    cublasStatus_t ierr = cublasCreate(&handle);
    if (ierr != CUBLAS_STATUS_SUCCESS) {
        ERROR("Error at {:s}:{:d}: cublasCreate: {:s}", __FILE__, __LINE__,
              cublasGetStatusString(ierr));
        std::abort();
    }

    // We MUST set the stream -- otherwise it uses the CUDA
    // default stream, which is not our default compute stream!
    DEBUG("Set cublas stream {:d}", cuda_stream_id);
    cublasSetStream(handle, device.getStream(cuda_stream_id));
}

cudaFRBBeamReformer::~cudaFRBBeamReformer() {
    // Destroy cuBLAS handle
    cublasDestroy(handle);
}

int cudaFRBBeamReformer::wait_on_precondition() {
    // Wait for data to be available in input ringbuffer
    DEBUG("Input ring-buffer count: {:d}", samples_per_data_set);
    DEBUG("Waiting for input ringbuffer data for frame {:d}...", gpu_frame_id);
    const std::optional<std::ptrdiff_t> val_in =
        input_ringbuf_signal->wait_and_claim_readable(unique_name, instance_num, input_bytes);
    DEBUG("Finished waiting for input for data frame {:d}.", gpu_frame_id);
    if (!val_in.has_value())
        return -1;
    input_cursor = val_in.value();
    DEBUG("Input ring-buffer offset: {:d}", input_cursor);
    // Mod input cursor by the ringbuffer size

    MOD IN ELEMENTS !

        input_position = mod(input_cursor, input_ringbuf_signal->size);
    // Assert that we don't wrap around!
    assert(input_position + input_bytes <= input_ringbuf_signal->size);
    DEBUG("Modded input ring-buffer byte offset: {:d}", input_position);
    return 0;
}

cudaEvent_t cudaFRBBeamReformer::execute(cudaPipelineState&, const std::vector<cudaEvent_t>&) {
    pre_execute();

    record_start_event();

    // Get buffer pointers
    DEBUG("beamgrid_memory");
    float16_t* const beamgrid_memory =
        (float16_t*)device.get_gpu_memory(_gpu_mem_beamgrid + "_buffer", input_ringbuf_signal->size)
        + div_noremainder(input_position, sizeof(float16_t));
    DEBUG("phase_memory");
    float16_t* const phase_memory =
        (float16_t*)device.get_gpu_memory(_gpu_mem_phase + "_buffer", phase_len);
    DEBUG("beamout_memory");
    float16_t* const beamout_memory = (float16_t*)device.get_gpu_memory_array(
        _gpu_mem_beamout + "_buffer", gpu_frame_id, _gpu_buffer_depth, beamout_len);

    DEBUG("Running CUDA FRB BeamReformer on GPU frame {:d}: F={:d}, T={:d}, B={:d}, "
          "num_input_beams={:d}",
          gpu_frame_id, _num_local_freq, _Td, _num_beams, num_input_beams);

    // Calculate
    //     Iout[T,F,Bout] = Iin[Bin,F0,T] * W[Bin,Bout,F]
    //     C = A^T * B
    //
    //     C[m,n] = A^T[k,m] B[k,n]
    //
    // Thus
    //     m       = T
    //     n       = Bout
    //     k       = Bin
    //     lda     = Bin * F0
    //     ldb     = Bin
    //     ldc     = T * F
    //     strideA = Bin
    //     strideB = Bin * Bout
    //     strideC = T
    //
    // These indices are in Fortran notation (which cuBLAS
    // uses), i.e. the leftmost index is contiguous in memory.
    //
    // The frequency is a spectator index, i.e. we only show
    // one frequency at a time to cuBLAS. This makes the
    // matrices non-contiguous in memory. This is fine, cuBLAS
    // supports this.

    const int m = _Td;
    const int n = _num_beams;
    const int k = num_input_beams;

    const int lda = _max_num_local_freq * num_input_beams;
    const int ldb = num_input_beams;
    const int ldc = _Td * _num_local_freq;

    const std::ptrdiff_t strideA = num_input_beams;
    const std::ptrdiff_t strideB = num_input_beams;
    const std::ptrdiff_t strideC = _Td;

    const float16_t alpha = 1;
    const float16_t beta = 0;

    cublasStatus_t stat = cublasHgemmStridedBatched(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, beamgrid_memory, lda, strideA,
        phase_memory, ldb, strideB, &beta, beamout_memory, ldc, strideC, _num_local_freq);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        ERROR("Error at {:s}:{:d}: cublasHgemmStridedBatched: {:s}", __FILE__, __LINE__,
              cublasGetStatusString(stat));
        std::abort();
    }

    const std::shared_ptr<metadataObject> in_mc = input_ringbuf_signal->get_metadata(0);
    if (metadata_is_chord(in_mc)) {
        const std::shared_ptr<chordMetadata> in_meta = get_chord_metadata(in_mc);
        // Assert that input metadata array shape is as expected.
        DEBUG("Input metadata: array shape {:s}, array type {:s}", in_meta->get_dimensions_string(),
              in_meta->get_type_string());
        assert(in_meta->type == kotekan::float16);
        // Assert Ttilde x Fbar x beamQ x beamP
        assert(in_meta->dims == 4);
        assert(in_meta->dim[0] == _Td);
        if (!(in_meta->dim[1] == _max_num_local_freq))
            ERROR("in dim=[{},{},{},{}] max_num_local_freq={}", in_meta->dim[0], in_meta->dim[1],
                  in_meta->dim[2], in_meta->dim[3], _max_num_local_freq);
        assert(in_meta->dim[1] == _max_num_local_freq);
        assert(in_meta->dim[2] == _beam_grid_size_Q);
        assert(in_meta->dim[3] == _beam_grid_size_P);
        for (int d = in_meta->dims - 1; d >= 0; --d)
            if (d == in_meta->dims - 1)
                assert(in_meta->stride[d] == 1);
            else
                assert(in_meta->stride[d] == in_meta->stride[d + 1] * in_meta->dim[d + 1]);
        // Set metadata on output buffer
        std::shared_ptr<metadataObject> const out_mc = device.create_gpu_memory_array_metadata(
            _gpu_mem_beamout + "_buffer", gpu_frame_id, in_mc->parent_pool);
        std::shared_ptr<chordMetadata> const out_meta = get_chord_metadata(out_mc);
        out_meta->deepCopy(in_meta);
        // Output shape is (Ttilde x Fbar x beam) in float16
        out_meta->set_name("frb2_beams");
        out_meta->type = kotekan::float16;
        out_meta->dims = 3;
        out_meta->set_array_dimension(0, _num_beams, "R");
        out_meta->set_array_dimension(1, _num_local_freq, "Fbar");
        out_meta->set_array_dimension(2, _Td, "Ttilde");
        for (int d = out_meta->dims - 1; d >= 0; --d)
            if (d == out_meta->dims - 1)
                out_meta->stride[d] = 1;
            else
                out_meta->stride[d] = out_meta->stride[d + 1] * out_meta->dim[d + 1];
        DEBUG("Set output metadata: array shape {:s}, array type {:s}",
              out_meta->get_dimensions_string(), out_meta->get_type_string());

        // Since we do not use a ring buffer we need to set `meta->fpga_seq_num`
        assert(input_cursor % in_meta->sample_bytes() == 0);
        out_meta->set_fpga_seq_num(in_meta->get_fpga_seq_num()
                                   + in_meta->get_time_downsampling_fpga()
                                         * div_noremainder(input_cursor, in_meta->sample_bytes()));
    }

    return record_end_event();
}

void cudaFRBBeamReformer::finalize_frame() {
    // Advance the input ringbuffer
    const std::ptrdiff_t input_bytes = beamgrid_len;
    DEBUG("Advancing input ringbuffer by {:d} bytes", input_bytes);
    input_ringbuf_signal->finish_read(unique_name, instance_num, input_bytes);
    cudaCommand::finalize_frame();
}
