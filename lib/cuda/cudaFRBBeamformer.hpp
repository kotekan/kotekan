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
    cudaEvent_t execute(int gpu_frame_id, cudaEvent_t pre_event) override;

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
    /// GPU side memory name for the status/info output
    std::string _gpu_mem_info;

    // The bb.yaml file entry like kernel-name = "_Z15julia_frb_...."
    // clang-format off
    const std::string kernel_name = "_Z15julia_frb_1046713CuDeviceArrayI7Int16x2Li1ELi1EES_I9Float16x2Li1ELi1EES_I6Int4x8Li1ELi1EES_IS1_Li1ELi1EES_I5Int32Li1ELi1EES_IS2_Li1ELi1EES_IS2_Li1ELi1EES_IS1_Li1ELi1EE";
    // clang-format on
    // The frb.yaml file entry like
    //    threads: [32, 24]
    const int threads_x = 32;
    const int threads_y = 24;
    // The frb.yaml file entry like
    //    blocks: [84]
    const int blocks_x = 84;
    const int blocks_y = 1;
    // The frb.yaml file entry like
    //    shmem_bytes: #####
    const int shared_mem_bytes = 76896;
};

#endif // CUDA_FRB_BEAMFORMER_HPP
