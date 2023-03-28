/**
 * @file
 * @brief CUDA baseband upchannelizer
 *  - cudaUpchannelize : public cudaCommand
 */

#ifndef CUDA_UPCHANNELIZE16K_HPP
#define CUDA_UPCHANNELIZE16K_HPP

#include "cudaDeviceInterface.hpp"
#include "cudaUpchannelize.hpp"

/**
 * @class cudaUpchannelize
 * @brief cudaCommand for upchannelizing baseband data.
 *
 * Kernel by Kendrick Smith and Erik Schnetter, eg
 * https://github.com/eschnett/IndexSpaces.jl/blob/main/output-A40/upchan.ptx
 *
 * @par GPU Memory
 * @gpu_mem  gpu_mem_input_voltage  Input complex voltages of size samples_per_data_set * num_dishes
 * * 2 polarizations * num_local_freq
 *   @gpu_mem_type   staging
 *   @gpu_mem_format Array of @c int4+4 complex
 * @gpu_mem  gpu_mem_output_voltage  Output complex voltages of size (samples_per_data_set/upchan) *
 * num_dishes * 2 polarizations * (num_local_freq*upchan)
 * 2
 *   @gpu_mem_type   staging
 *   @gpu_mem_format Array of @c int4+4 complex
 * @gpu_mem  gpu_mem_gains  Input per-output-frequency gains, size (num_local_freq * upchan)
 *   @gpu_mem_type   staging
 *   @gpu_mem_format Array of @c float16
 * @gpu_mem  gpu_mem_info  Output status information; size threads_x * threads_y * blocks_x
 *   @gpu_mem_type   staging
 *   @gpu_mem_format Array of @c int32
 *
 * @conf  num_dishes            Int.  Number of dishes.
 * @conf  num_local_freq        Int.  Number of frequencies in each frame.
 * @conf  samples_per_data_set  Int.  Number of time samples per frame.
 * @conf  upchan_factor         Int.  Upchannelization factor.
 */
class cudaUpchannelize16k : public cudaUpchannelize {
public:
    cudaUpchannelize16k(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device);
    ~cudaUpchannelize16k();

    //std::string get_kernel_function_name() override;

protected:
private:
    // The upchan.yaml file entry like kernel-name = "_Z17julia_upchan_..."
    // clang-format off
    //const std::string kernel_name = "_Z17julia_upchan_376513CuDeviceArrayI9Float16x2Li1ELi1EES_I6Int4x8Li1ELi1EES_IS1_Li1ELi1EES_I5Int32Li1ELi1EE";
    //const std::string kernel_name = "_Z17julia_upchan_404813CuDeviceArrayI9Float16x2Li1ELi1EES_I6Int4x8Li1ELi1EES_IS1_Li1ELi1EES_I5Int32Li1ELi1EE";
    
    // // clang-format on
    // // The upchan.yaml file entry like
    // //    threads: [32, 16]
    // const int threads_x = 32;
    // const int threads_y = 16;
    // // The upchan.yaml file entry like
    // //    blocks: [128]
    // const int blocks_x = 128;
    // const int blocks_y = 1;
    // // The upchan.yaml file entry like
    // //    shmem_bytes: 69888
    // const int shared_mem_bytes = 69888;

    // // Compiled-in constants in the CUDA kernel -- from upchan.yaml
    // const int cuda_ndishes = 512;
    // const int cuda_upchan = 16;
    // const int cuda_nfreq = 16;
    //const int cuda_nsamples = 32768;
    const int cuda_nsamples = 16384;
};

#endif // CUDA_UPCHANNELIZE16K_HPP
