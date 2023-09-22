/**
 * @file
 * @brief CUDA FRB beamforming kernel, final stage
 *  - cudaFRBBeamReformer : public cudaCommand
 */

#ifndef CUDA_FRB_BEAMREFORMER_HPP
#define CUDA_FRB_BEAMREFORMER_HPP

#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"

#include <cublas_v2.h>

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
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame() override;

protected:
private:
    cublasHandle_t handle;

    // Common configuration values (which do not change in a run)
    int32_t _num_beams;
    /// Beam-grid size produced by cudaFRBBeamformer
    int32_t _beam_grid_size;
    /// Number of frequencies per data stream sent to each node.
    int32_t _num_local_freq;
    /// Total samples in each dataset
    int32_t _Td;

    /// CUDA compute streams to use
    std::vector<int> _cuda_streams;
    std::vector<cudaEvent_t> sync_events;

    // Computed values
    int32_t rho;

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

    bool _batched;

    // cublasHgemmBatched -- pre-computed GPU memory locations.
    // [gpu_frame_id] [freq batch = stream] = [per-freq pointers]
    std::vector<std::vector<__half**>> _gpu_in_pointers;
    std::vector<std::vector<__half**>> _gpu_out_pointers;
    // [freq batch = stream] = [per-freq pointers]
    std::vector<__half**> _gpu_phase_pointers;
};

#endif // CUDA_FRB_BEAMFORMER_HPP
