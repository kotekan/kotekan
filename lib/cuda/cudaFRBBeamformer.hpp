/**
 * @file
 * @brief CUDA FRB beamforming kernel
 *  - cudaFRBBeamformer : public cudaCommand
 */

#ifndef CUDA_FRB_BEAMFORMER_HPP
#define CUDA_FRB_BEAMFORMER_HPP

#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"

/**
 * @class cudaFRBBeamformer
 * @brief cudaCommand for doing FRB beamforming.
 *
 * Kernel by Kendrick Smith and Erik Schnetter.
 * https://github.com/eschnett/GPUIndexSpaces.jl/blob/main/output/frb.ptx
 */
class cudaFRBBeamformer : public cudaCommand {
public:
    cudaFRBBeamformer(kotekan::Config& config, const std::string& unique_name,
                      kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device);
    ~cudaFRBBeamformer();
    cudaEvent_t execute(cudaPipelineState& pipestate, const std::vector<cudaEvent_t>& pre_events) override;
    virtual void finalize_frame(int gpu_frame_id) override;

protected:
private:
    // Common configuration values (which do not change in a run)
    /// Number of dishes in the telescope
    int32_t _num_dishes;
    /// Dish grid side length
    int32_t _dish_grid_size;
    /// Number of frequencies per data stream sent to each node.
    int32_t _num_local_freq;
    /// Total samples in each dataset. Must be a value that is a power of 2.
    int32_t _samples_per_data_set;
    /// Time downsampling factor
    int32_t _time_downsampling;

    /// Size in bytes of the dishlayout array
    size_t dishlayout_len;
    /// Size in bytes of the phase array
    size_t phase_len;
    /// Size in bytes of the voltage array
    size_t voltage_len;
    /// Size in bytes of the beamgrid array
    size_t beamgrid_len;
    /// Size in bytes of the info array
    size_t info_len;

    /// GPU side memory name for the dish layout input
    std::string _gpu_mem_dishlayout;
    /// GPU side memory name for the voltage input
    std::string _gpu_mem_voltage;
    /// GPU side memory name for the beamforming phases
    std::string _gpu_mem_phase;
    /// GPU side memory name for the beamformed output
    std::string _gpu_mem_beamgrid;

    /// GPU memory name for the status/info output
    std::string gpu_mem_info;

    // Host-side buffer array for GPU kernel status/info output
    std::vector< std::vector<int32_t> > host_info;

    // frb-v5

    // The bb.yaml file entry like kernel-name = "_Z15julia_frb_...."
    // clang-format off
    const std::string kernel_name = "_Z15julia_frb_1037413CuDeviceArrayI7Int16x2Li1ELi1EES_I9Float16x2Li1ELi1EES_I6Int4x8Li1ELi1EES_IS1_Li1ELi1EES_I5Int32Li1ELi1EE";
    // clang-format on
    // The frb.yaml file entry like
    //    threads: [32, 24]
    const int threads_x = 32;
    const int threads_y = 24;
    // The frb.yaml file entry like
    //    blocks: [256]
    const int blocks_x = 256;
    const int blocks_y = 1;
    // The frb.yaml file entry like
    //    shmem_bytes: #####
    const int shared_mem_bytes = 76896;

    // Compiled-in constants in the CUDA kernel
    const int cuda_samples_per_data_set = 2064;
    const int cuda_time_downsampling = 40;
    const int cuda_num_dishes = 512;
    const int cuda_dish_grid_size = 24;
    const int cuda_num_local_freq = 256;
};

#endif // CUDA_FRB_BEAMFORMER_HPP
