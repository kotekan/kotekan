/**
 * @file
 * @brief CUDA FRB beamforming kernel, final stage
 *  - cudaFRBBeamReformer : public cudaCommand
 */

#include "Config.hpp"              // for Config
#include "DataType.hpp"            // for float16_t
#include "NDArray.hpp"             // for NDArray
#include "NDArrayBuffer.hpp"       // for NDArrayBuffer, buffer_type_t
#include "NDArrayRingBuffer.hpp"   // for NDArrayRingBuffer, read_descriptor_t, extent_t
#include "Symbol.hpp"              // for Symbol
#include "bufferContainer.hpp"     // for bufferContainer
#include "chordMetadata.hpp"       // for chordMetadata
#include "cudaCommand.hpp"         // for cudaCommand, cudaPipelineState, REGISTER_CUDA_COMMAND
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "cuda_fp16.h"             // for __half
#include "div.hpp"                 // for mod
#include "gpuCommand.hpp"          // for gpuCommandType
#include "kotekanLogging.hpp"      // for DEBUG, ERROR, FATAL_ERROR

#include "fmt.hpp" // for compile_string_to_view

#include <array>          // for array
#include <cassert>        // for assert
#include <cstddef>        // for ptrdiff_t
#include <cstdlib>        // for abort
#include <cublas_api.h>   // for cublasGetStatusString, CUBLAS_STATUS_SUCCESS, cublasH...
#include <cublas_v2.h>    // for cublasCreate, cublasDestroy, cublasSetStream
#include <driver_types.h> // for cudaEvent_t, CUevent_st, CUstream_st
#include <functional>     // for function
#include <memory>         // for shared_ptr, __shared_ptr_access
#include <string>         // for basic_string, allocator, operator==, string
#include <vector>         // for vector

using kotekan::mod;

/**
 * @class cudaFRBBeamReformer
 * @brief cudaCommand for doing final FRB beamforming.  This follows the cudaFRBBeamFormer.
 *
 * Core code was developed by Nada El-Falou in https://github.com/nadafalou/CHORD/blob/main/mmul.cu
 *
 * The weights matrix for the beam locations uses the correct math but
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

    const int frb_downsampling_factor;

    const int frb1_max_num_times;
    const int frb1_max_num_frequencies;
    const int frb1_num_beams_P;
    const int frb1_num_beams_Q;

    // Number of time samples per frb2 beam packet
    const int frb2_num_frequencies;
    const int frb2_num_beams;
    const int frb2_num_times;

    // Kotekan buffer names
    const std::string frb2_weights_name;
    const std::string frb1_beams_name;
    const std::string frb2_beams_name;

    // Buffers
    NDArrayBuffer<float16_t, 4> frb2_weights_buffer;
    NDArrayRingBuffer<float16_t, 4> frb1_beams_buffer;
    NDArrayBuffer<float16_t, 4> frb2_beams_buffer;

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

    frb_downsampling_factor(config.get<int>(unique_name, "frb_downsampling_factor")),

    frb1_max_num_times(config.get<int>(unique_name, "frb1_max_num_times")),
    frb1_max_num_frequencies(config.get<int>(unique_name, "frb1_max_num_frequencies")),
    frb1_num_beams_P(config.get<int>(unique_name, "frb1_num_beams_P")),
    frb1_num_beams_Q(config.get<int>(unique_name, "frb1_num_beams_Q")),

    frb2_num_frequencies(config.get<int>(unique_name, "frb2_num_frequencies")),
    frb2_num_beams(config.get<int>(unique_name, "frb2_num_beams")),
    frb2_num_times(config.get<int>(unique_name, "frb2_num_times")),

    frb2_weights_name(config.get<std::string>(unique_name, "frb2_weights_name")),
    frb1_beams_name(config.get<std::string>(unique_name, "frb1_beams_name")),
    frb2_beams_name(config.get<std::string>(unique_name, "frb2_beams_name")),

    frb2_weights_buffer(frb2_weights_name, "W2",
                        std::array<std::ptrdiff_t, 4>{frb2_num_frequencies, frb2_num_beams,
                                                      frb1_num_beams_Q, frb1_num_beams_P},
                        std::array<std::string, 4>{"Fbar", "R", "beamQ", "beamP"}, {1, 1, 1, 1},
                        *this, buffer_type_t::do_once),
    frb1_beams_buffer(frb1_beams_name, "I",
                      std::array<std::ptrdiff_t, 4>{frb1_max_num_times, frb1_max_num_frequencies,
                                                    frb1_num_beams_Q, frb1_num_beams_P},
                      std::array<std::string, 4>{"Ttilde", "Fbar", "beamQ", "beamP"},
                      {frb_downsampling_factor, 1, 1, 1}, *this),
    frb2_beams_buffer(
        frb2_beams_name, "I2",
        std::array<std::ptrdiff_t, 4>{1, frb2_num_beams, frb2_num_frequencies, frb2_num_times},
        std::array<std::string, 4>{"Ttildehi256", "R", "Fbar", "Ttildelo256"},
        {frb_downsampling_factor * frb2_num_times, 1, 1, frb_downsampling_factor}, *this),

    did_set_metadata(false)

{
    frb2_weights_buffer.register_consumer();
    frb1_beams_buffer.register_consumer();
    frb2_beams_buffer.register_producer();

    // Ensure that this stage can always make progress
    frb1_beams_buffer.check_read_progress(frb1_beams_buffer.get_ndarray().extent(0),
                                          frb2_num_times);

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

    frb2_weights_buffer.check_metadata();
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
    if (mod(frb1_beams_offset, frb1_beams_extent) + frb2_num_times > frb1_beams_stride) {
        FATAL_ERROR("Inconsistent values for frb1_beams_offset={}, frb1_beams_extent={}, "
                    "frb2_num_times={}, and frb1_beams_stride={}. These would result in a "
                    "wrap-around in the ringbuffer which is not implemented.",
                    frb1_beams_offset, frb1_beams_extent, frb2_num_times, frb1_beams_stride);
        std::abort();
    }

    // Since we do not use a ring buffer we need to set `meta->fpga_seq_num`
    const std::shared_ptr<chordMetadata> frb2_beams_meta = frb2_beams_buffer.get_metadata();
    frb2_beams_meta->set_fpga_seq_num(frb1_beams_offset
                                      * frb2_beams_meta->get_time_downsampling_fpga());

    if (poison_buffers)
        frb2_beams_buffer.set_to_poison(0xff); // 0xffff is a NaN16

    // All arrays here use C index ordering.
    // The "natural" matrix multiplications then looks like
    //     C[n,m] = A[k,m] B[n,k]
    // where k is "on the outside" on the right hand side.
    //
    // We calculate
    //     Iout[Bout,F,T] = W[F,Bout,Bin] * Iin[T,F,Bin]
    //
    // We map this to the matrix multiplication above:
    //     C = A^T * B
    // This is, with indices:
    //     C[n,m] = A^T[m,k] B[n,k]
    // We add an additional spectator index p:
    //     C[p,n,m] = A^T[p,m,k] B[p,n,k]
    //
    // We choose:
    //     A = Iin
    //     B = W
    //     C = Iout
    //     m = T
    //     n = Bout
    //     k = Bin
    //     p = F
    //
    // We read off the arguments for `HgemmStridedBatched`:
    //     transA     = true
    //     transB     = false
    //     M          = T
    //     N          = Bout
    //     K          = Bin
    //     alpha      = 1
    //     A          = Iin
    //     ldA        = T_stride
    //     strideA    = F_stride
    //     B          = W
    //     ldB        = Bout_stride
    //     strideB    = F_stride
    //     beta       = 0
    //     C          = Iout
    //     ldC        = Bout_stride
    //     strideC    = F_stride
    //     batchCount = F
    //
    // The frequency is a spectator index, i.e. we only show
    // one frequency at a time to cuBLAS. This makes the
    // matrices non-contiguous in memory. This is fine, cuBLAS
    // supports this.

    // Matrix transposes
    const auto transA = CUBLAS_OP_T;
    const auto transB = CUBLAS_OP_N;

    // Matrix sizes (times and beams)
    const int M = frb2_num_times;                      // times
    const int N = frb2_num_beams;                      // output beams
    const int K = frb1_num_beams_P * frb1_num_beams_Q; // input beams

    // Matrix A
    const float16_t alpha = 1;
    const float16_t* A = frb1_beams_buffer.get_ndarray().data()
                         + frb1_beams_stride * mod(frb1_beams_offset, frb1_beams_extent);
    assert(std::string(frb1_beams_buffer.get_ndarray().get_dimname(0)) == "Ttilde");
    const int ldA = frb1_beams_buffer.get_ndarray().get_stride(0); // time
    assert(std::string(frb1_beams_buffer.get_ndarray().get_dimname(1)) == "Fbar");
    const std::ptrdiff_t strideA = frb1_beams_buffer.get_ndarray().get_stride(1); // frequency

    // Matrix B
    const float16_t* B = frb2_weights_buffer.get_ndarray().data();
    assert(std::string(frb2_weights_buffer.get_ndarray().get_dimname(1)) == "R");
    const int ldB = frb2_weights_buffer.get_ndarray().get_stride(1); // output beams
    assert(std::string(frb2_weights_buffer.get_ndarray().get_dimname(0)) == "Fbar");
    const std::ptrdiff_t strideB = frb2_weights_buffer.get_ndarray().get_stride(0); // frequency

    // Matrix C
    const float16_t beta = 0;
    float16_t* C = frb2_beams_buffer.get_ndarray().data();
    assert(std::string(frb2_beams_buffer.get_ndarray().get_dimname(1)) == "R");
    const int ldC = frb2_beams_buffer.get_ndarray().get_stride(1); // output beams
    assert(std::string(frb2_beams_buffer.get_ndarray().get_dimname(2)) == "Fbar");
    const std::ptrdiff_t strideC = frb2_beams_buffer.get_ndarray().get_stride(2); // frequency

    // Batch
    const int batchCount = frb2_num_frequencies; // frequencies

    // HgemmStridedBatched(cublasHandle_t handle,
    //                     cublasOperation_t transA, cublasOperation_t transB,
    //                     int M, int N, int K,
    //                     const T* alpha,
    //                     const T* A, int ldA, int strideA,
    //                     const T* B, int ldB, int strideB,
    //                     const T* beta,
    //                     T* C, int ldC, int strideC,
    //                     int batchCount)
    DEBUG("M={} N={} K={} A={} ldA={} strideA={} B={} ldB={} strideB={} C={} ldC={} strideC={} "
          "batchCount={}",
          M, N, K, (const void*)A, ldA, strideA, (const void*)B, ldB, strideB, (void*)C, ldC,
          strideC, batchCount);
    cublasStatus_t stat =
        cublasHgemmStridedBatched(handle, transA, transB, M, N, K, &alpha, A, ldA, strideA, B, ldB,
                                  strideB, &beta, C, ldC, strideC, batchCount);
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
