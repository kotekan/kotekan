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
 *
 * @par GPU Memory
 * @gpu_mem  gpu_mem_voltage  Input complex voltages of size samples_per_data_set * num_elements *
 * num_local_freq
 *   @gpu_mem_type   staging
 *   @gpu_mem_format Array of @c int4+4 complex
 * @gpu_mem  gpu_mem_phase  Input complex phases of size num_elements * num_local_freq * num_beams *
 * 2
 *   @gpu_mem_type   staging
 *   @gpu_mem_format Array of @c int8
 * @gpu_mem  gpu_mem_output_scaling  Input number of bits to shift result by; size num_local_freq *
 * num_beams * 2
 *   @gpu_mem_type   staging
 *   @gpu_mem_format Array of @c int32
 * @gpu_mem  gpu_mem_formed_beams  Output beams; size num_local_freq * num_beams *
 * samples_per_data_set * 2
 *   @gpu_mem_type   staging
 *   @gpu_mem_format Array of @c int4+4 complex
 * @gpu_mem  gpu_mem_info  Output status information; size threads_x * threads_y * blocks_x
 *   @gpu_mem_type   staging
 *   @gpu_mem_format Array of @c int32
 *
 * @conf  num_elements          Int.  Number of dishes x polarizations.
 * @conf  num_local_freq        Int.  Number of frequencies in each frame.
 * @conf  samples_per_data_set  Int.  Number of time samples per frame.
 * @conf  num_beams             Int.  Number of beams being formed.
 */
class cudaBasebandBeamformer : public cudaCommand {
public:
    cudaBasebandBeamformer(kotekan::Config& config, const std::string& unique_name,
                           kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device);
    ~cudaBasebandBeamformer();
    cudaEvent_t execute(int gpu_frame_id, const std::vector<cudaEvent_t>& pre_events) override;

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

    // derived gpu array sizes
    size_t voltage_len;
    size_t phase_len;
    size_t shift_len;
    size_t output_len;
    size_t info_len;

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

    // Compiled-in constants in the CUDA kernel -- from bb.yaml
    const int cuda_nbeams = 96;
    const int cuda_nelements = 512 * 2;
    const int cuda_nfreq = 16;
    const int cuda_nsamples = 32768;
};

#endif // CUDA_BASEBAND_BEAMFORMER_HPP
