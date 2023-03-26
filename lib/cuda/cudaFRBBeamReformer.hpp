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
 * Kernel from Nada El-Falou in https://github.com/nadafalou/CHORD/blob/main/mmul.cu
 */
class cudaFRBBeamReformer : public cudaCommand {
public:
    cudaFRBBeamReformer(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device);
    ~cudaFRBBeamReformer();
    cudaEvent_t execute(int gpu_frame_id, const std::vector<cudaEvent_t>& pre_events, bool* quit) override;

protected:
private:
    cublasHandle_t handle;

    // Common configuration values (which do not change in a run)
    int32_t _num_beams;
    /// Beam-grid size produced by cudaFRBBeamformer
    int32_t _beam_grid_size;
    /// Number of frequencies per data stream sent to each node.
    int32_t _num_local_freq;

    /// Total samples in each dataset (before downsampling!)
    int32_t _samples_per_data_set;
    /// Time downsampling factor
    int32_t _time_downsampling;

    // Computed values
    int32_t rho;
    int32_t Td;

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
};

#endif // CUDA_FRB_BEAMFORMER_HPP
