/**
 * @file
 * @brief CUDA baseband beamforming kernel
 *  - cudaBasebandBeamformer : public cudaCommand
 */

#ifndef CUDA_BASEBAND_BEAMFORMER_HPP
#define CUDA_BASEBAND_BEAMFORMER_HPP

#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"

/**
 * @class cudaBasebandBeamformer
 * @brief cudaCommand for doing baseband beamforming.
 *
 * Kernel by Kendrick Smith and Erik Schnetter.
 * https://github.com/eschnett/GPUIndexSpaces.jl/blob/main/output/bb1.ptx
 */
class cudaBasebandBeamformer : public cudaCommand {
public:
    cudaBasebandBeamformer(kotekan::Config& config, const std::string& unique_name,
                           kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device);
    ~cudaBasebandBeamformer();
    cudaEvent_t execute(int gpu_frame_id, cudaEvent_t pre_event) override;

protected:
private:
    // Common configuration values (which do not change in a run)
    /// Number of elements on the telescope
    int32_t _num_elements;
    /// Number of frequencies per data stream sent to each node.
    int32_t _num_local_freq;
    /// Total samples in each dataset. Must be a value that is a power of 2.
    int32_t _samples_per_data_set;
    // Number of beams to form.
    int32_t _num_beams;

    /// GPU side memory name for the voltage input
    std::string _gpu_mem_voltage;
    /// GPU side memory name for the beamforming phases
    std::string _gpu_mem_phase;
    /// GPU side memory name for the output scaling / bit shift
    std::string _gpu_mem_output_scaling;
    /// GPU side memory name for the beamformed output
    std::string _gpu_mem_formed_beams;
    /// GPU side memory name for the status/info output
    std::string _gpu_mem_info;

    // The bb.yaml file entry like kernel-name = "_Z13julia_bb_...."
    // clang-format off
    const std::string kernel_name = "_Z13julia_bb_496913CuDeviceArrayI6Int8x4Li1ELi1EES_I6Int4x8Li1ELi1EES_I5Int32Li1ELi1EES_IS1_Li1ELi1EES_IS2_Li1ELi1EE";
    // clang-format on
    // The bb.yaml file entry like
    //    threads: [32, 24]
    const int threads_x = 32;
    const int threads_y = 24;
    // The bb.yaml file entry like
    //    blocks: [32]
    const int blocks_x = 32;
    const int blocks_y = 1;
    // The bb.yaml file entry like
    //    shmem_bytes: 67712
    const int shared_mem_bytes = 67712;
};

#endif // CUDA_BASEBAND_BEAMFORMER_HPP
