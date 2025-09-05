#include "cudaFRBBeamReformer.hpp"

#include <DataType.hpp>
#include <NDArrayBuffer.hpp>
#include <chordMetadata.hpp>
#include <cmath>
#include <cudaUtils.hpp>
#include <div.hpp>
#include <fmt.hpp>
#include <vector>
#include <visUtil.hpp>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::div_noremainder;
using kotekan::mod;

REGISTER_CUDA_COMMAND(cudaFRBBeamReformer);

cudaFRBBeamReformer::cudaFRBBeamReformer(Config& config, const std::string& unique_name,
                                         bufferContainer& host_buffers, cudaDeviceInterface& device,
                                         int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst) {

    // Number of output beams
    _num_beams = config.get<int>(unique_name, "num_beams");
    // Number of input beams
    _beam_grid_size_ns = config.get<int>(unique_name, "beam_grid_size_ns");
    _beam_grid_size_ew = config.get<int>(unique_name, "beam_grid_size_ew");
    num_input_beams = _beam_grid_size_ew * _beam_grid_size_ns;
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
    const std::ptrdiff_t input_bytes = beamgrid_len;
    DEBUG("Input ring-buffer byte count: {:d}", input_bytes);
    DEBUG("Waiting for input ringbuffer data for frame {:d}...", gpu_frame_id);
    const std::optional<std::ptrdiff_t> val_in =
        input_ringbuf_signal->wait_and_claim_readable(unique_name, instance_num, input_bytes);
    DEBUG("Finished waiting for input for data frame {:d}.", gpu_frame_id);
    if (!val_in.has_value())
        return -1;
    input_cursor = val_in.value();
    DEBUG("Input ring-buffer byte offset: {:d}", input_cursor);
    // Mod input cursor by the ringbuffer size
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
        // in_meta->dim[0] is in the ringbuffer
        // in_meta->dim[1] is set wrong
        // if (!(in_meta->dim[1] == _num_local_freq))
        //     ERROR("in dim=[{},{},{},{}] num_local_freq={}", in_meta->dim[0], in_meta->dim[1],
        //           in_meta->dim[2], in_meta->dim[3], _num_local_freq);
        // assert(in_meta->dim[1] == _num_local_freq);
        if (!(in_meta->dim[1] == _max_num_local_freq))
            ERROR("in dim=[{},{},{},{}] max_num_local_freq={}", in_meta->dim[0], in_meta->dim[1],
                  in_meta->dim[2], in_meta->dim[3], _max_num_local_freq);
        assert(in_meta->dim[1] == _max_num_local_freq);
        assert(in_meta->dim[2] == _beam_grid_size_ew);
        assert(in_meta->dim[3] == _beam_grid_size_ns);
        for (int d = in_meta->dims - 1; d >= 0; --d)
            if (d == in_meta->dims - 1)
                assert(in_meta->stride[d] == 1);
            else
                assert(in_meta->stride[d] == in_meta->stride[d + 1] * in_meta->dim[d + 1]);
        // Set metadata on output buffer
        std::shared_ptr<metadataObject> const out_mc = device.create_gpu_memory_array_metadata(
            _gpu_mem_beamout + "_buffer", gpu_frame_id, in_mc->parent_pool);
        std::shared_ptr<chordMetadata> const out_meta = get_chord_metadata(out_mc);
        *out_meta = *in_meta;
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

        // Since we do not use a ring buffer we need to set `meta->sample0_offset`
        assert(input_cursor % in_meta->sample_bytes() == 0);
        out_meta->sample0_offset = div_noremainder(input_cursor, in_meta->sample_bytes());
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
