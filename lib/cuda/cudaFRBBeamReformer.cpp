/**
 * @file
 * @brief CUDA FRB beamforming kernel, final stage
 *  - cudaFRBBeamReformer : public cudaCommand
 */

#include "Config.hpp" // for Config
#include "NDArrayBuffer.hpp"
#include "NDArrayRingBuffer.hpp"
#include "cudaCommand.hpp" // for cudaCommand, REGISTER_CUDA_COMMAND
#include "div.hpp"

#include <array>
#include <cstdlib>
#include <cublas_api.h>   // for cublasContext, cublasHandle_t
#include <cublas_v2.h>    // for cublasCreate, cublasDestroy, cublasSetStream
#include <driver_types.h> // for cudaEvent_t
#include <string>
#include <vector>

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
    const bool poison_buffers;

    const int frb1_max_num_times;
    const int frb1_max_num_frequencies;
    const int frb1_num_beams_P;
    const int frb1_num_beams_Q;

    // Number of time samples per frb2 beam packet
    const int frb2_num_frequencies;
    const int frb2_num_beams;
    const int frb2_num_times;

    // Kotekan buffer names
    const std::string frb2_phase_name;
    const std::string frb1_beams_name;
    const std::string frb2_beams_name;

    // Buffers
    NDArrayBuffer<float16_t, 4> frb2_phase_buffer;
    NDArrayRingBuffer<float16_t, 4> frb1_beams_buffer;
    NDArrayBuffer<float16_t, 3> frb2_beams_buffer;

    cublasHandle_t handle;

    bool did_set_metadata;
};

REGISTER_CUDA_COMMAND(cudaFRBBeamReformer);

cudaFRBBeamReformer::cudaFRBBeamReformer(kotekan::Config& config, const std::string& unique_name,
                                         kotekan::bufferContainer& host_buffers,
                                         cudaDeviceInterface& device, const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "cudaFRBBeamReformer"),

    poison_buffers(config.get_default<bool>(unique_name, "poison_buffers", false)),

    frb1_max_num_times(config.get<int>(unique_name, "frb1_max_num_times")),
    frb1_max_num_frequencies(config.get<int>(unique_name, "frb1_max_num_frequencies")),
    frb1_num_beams_P(config.get<int>(unique_name, "frb1_num_beams_P")),
    frb1_num_beams_Q(config.get<int>(unique_name, "frb1_num_beams_Q")),

    frb2_num_frequencies(config.get<int>(unique_name, "frb2_num_frequencies")),
    frb2_num_beams(config.get<int>(unique_name, "frb2_num_beams")),
    frb2_num_times(config.get<int>(unique_name, "frb2_num_times")),

    frb2_phase_name(config.get<std::string>(unique_name, "frb2_phase_name")),
    frb1_beams_name(config.get<std::string>(unique_name, "frb1_beams_name")),
    frb2_beams_name(config.get<std::string>(unique_name, "frb2_beams_name")),

    frb2_phase_buffer(frb2_phase_name, "W2",
                      std::array<std::ptrdiff_t, 4>{frb2_num_frequencies, frb2_num_beams,
                                                    frb1_num_beams_Q, frb1_num_beams_P},
                      std::array<std::string, 4>{"Fbar", "R", "beamQ", "beamP"}, *this,
                      buffer_type_t::do_once),
    frb1_beams_buffer(frb1_beams_name, "I",
                      std::array<std::ptrdiff_t, 4>{frb1_max_num_times, frb1_max_num_frequencies,
                                                    frb1_num_beams_Q, frb1_num_beams_P},
                      std::array<std::string, 4>{"Ttilde", "Fbar", "beamQ", "beamP"}, *this),
    frb2_beams_buffer(
        frb2_beams_name, "I2",
        std::array<std::ptrdiff_t, 3>{frb2_num_beams, frb2_num_frequencies, frb2_num_times},
        std::array<std::string, 3>{"R", "Fbar", "Ttilde"}, *this),

    did_set_metadata(false)

{
    frb2_phase_buffer.register_consumer();
    frb1_beams_buffer.register_consumer();
    frb2_beams_buffer.register_producer();

    set_command_type(gpuCommandType::KERNEL);

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
    {
        const int errcode = cudaCommand::wait_on_precondition();
        if (errcode < 0)
            return errcode;
    }

    // Wait for data to be available in input ringbuffer
    const std::ptrdiff_t T_ringbuf = frb2_num_times;
    {
        const int errcode =
            frb1_beams_buffer.wait_and_claim_readable([&](const std::ptrdiff_t T_available) {
                if (T_available < T_ringbuf)
                    return read_descriptor_t{.claimed = 0, .read = 0};
                return read_descriptor_t{.claimed = T_ringbuf, .read = T_ringbuf};
            });
        if (errcode < 0)
            return errcode;
    }

    return 0;
}

cudaEvent_t cudaFRBBeamReformer::execute(cudaPipelineState& /*pipestate*/,
                                         const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();
    record_start_event();

    frb2_phase_buffer.check_metadata();
    frb1_beams_buffer.check_metadata();

    if (!did_set_metadata) {
        did_set_metadata = true;
        // Set metadata
        const std::shared_ptr<const chordMetadata> frb1_beams_meta =
            frb1_beams_buffer.get_metadata();
        frb2_beams_buffer.set_metadata(frb1_beams_meta);
        const std::shared_ptr<chordMetadata> frb2_beams_meta = frb2_beams_buffer.get_metadata();
        frb2_beams_meta->set_time_downsampling_fpga(frb1_beams_meta->get_time_downsampling_fpga());
        frb2_beams_meta->set_coarse_freq(frb1_beams_meta->get_coarse_freq());
        frb2_beams_meta->set_freq_upchan_factor(frb1_beams_meta->get_freq_upchan_factor());
        frb2_beams_meta->set_freq_upchan_index(frb1_beams_meta->get_freq_upchan_index());
    }
    frb2_beams_buffer.check_metadata();

    const std::ptrdiff_t frb1_beams_offset = frb1_beams_buffer.get_read_valid().begin();
    const std::ptrdiff_t frb1_beams_extent = frb1_beams_buffer.get_ndarray().get_extent(0);
    const std::ptrdiff_t frb1_beams_stride = frb1_beams_buffer.get_ndarray().get_stride(0);
    // Ensure there is no wrap-around
    if (mod(frb1_beams_offset, frb1_beams_extent) + frb2_num_times > frb1_beams_stride)
        std::abort();

    // Since we do not use a ring buffer we need to set `meta->fpga_seq_num`
    const std::shared_ptr<chordMetadata> frb2_beams_meta = frb2_beams_buffer.get_metadata();
    frb2_beams_meta->set_fpga_seq_num(frb1_beams_offset);

    if (poison_buffers)
        frb2_beams_buffer.set_to_poison(0xff); // 0xffff is a NaN16

    // Get buffer pointers
    const float16_t* const frb2_phase_memory = frb2_phase_buffer.get_ndarray().data();

    const float16_t* const frb1_beams_memory =
        frb1_beams_buffer.get_ndarray().data()
        + frb1_beams_stride * mod(frb1_beams_offset, frb1_beams_extent);

    float16_t* const frb2_beams_memory = frb2_beams_buffer.get_ndarray().data();

    // Calculate
    //     Iout[T,F,Bout] = Iin[Bin,F0,T0] * W[Bin,Bout,F]
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

    // Matrix sizes (times and beams)
    const int m = frb2_num_times;
    const int n = frb2_num_beams;
    const int k = frb1_num_beams_P * frb1_num_beams_Q;

    // Matrix layouts (times and beams)
    const int lda = frb1_beams_buffer.get_ndarray().get_stride(0);
    const int ldb = frb2_phase_buffer.get_ndarray().get_stride(1);
    const int ldc = frb2_beams_buffer.get_ndarray().get_stride(0);

    // Batch strides (frequencies)
    const std::ptrdiff_t strideA = frb1_beams_buffer.get_ndarray().get_stride(1);
    const std::ptrdiff_t strideB = frb2_phase_buffer.get_ndarray().get_stride(2);
    const std::ptrdiff_t strideC = frb2_beams_buffer.get_ndarray().get_stride(1);

    const float16_t alpha = 1;
    const float16_t beta = 0;

    DEBUG("m={} n={} k={} frb1_beams_memory={} lda={} strideA={} frb2_phase_memory={} ldb={} "
          "strideB={} frb2_beams_memory={} ldc={} strideC={} frb2_num_frequencies={}",
          m, n, k, (const void*)frb1_beams_memory, lda, strideA, (const void*)frb2_phase_memory,
          ldb, strideB, (const void*)frb2_beams_memory, ldc, strideC, frb2_num_frequencies);
    cublasStatus_t stat =
        cublasHgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                                  frb1_beams_memory, lda, strideA, frb2_phase_memory, ldb, strideB,
                                  &beta, frb2_beams_memory, ldc, strideC, frb2_num_frequencies);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        ERROR("Error at {:s}:{:d}: cublasHgemmStridedBatched: {:s}", __FILE__, __LINE__,
              cublasGetStatusString(stat));
        std::abort();
    }

    if (poison_buffers)
        frb2_beams_buffer.check_for_poison(0xff); // 0xffff is a NaN16

    return record_end_event();
}

void cudaFRBBeamReformer::finalize_frame() {
    // Advance the input ringbuffer
    frb1_beams_buffer.finish_read();

    cudaCommand::finalize_frame();
}
