/**
 * @file
 * @brief Simple sample CUDA kernel.
 *  - cudaSaxpyKernel : public cudaCommand
 */

#ifndef CUDA_SAXPY_KERNEL_CUH
#define CUDA_SAXPY_KERNEL_CUH

#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"

/**
 * @class cudaSaxpyKernel
 * @brief simple example of how to write a cudaCommand
 *
 * This is a trivial example of a cudaCommand which scales and adds one block of
 * float-formatted input memory to a block of output memory.
 *
 * @par GPU Memory
 * @gpu_mem  input              Input data of size input_frame_len
 *     @gpu_mem_type            static
 *     @gpu_mem_format          Array of @c float
 * @gpu_mem  output             Output data of size output_frame_len
 *     @gpu_mem_type            static
 *     @gpu_mem_format          Array of @c float
 *
 * @conf   saxpy_scale          Float (default 1.0). What to scale the input by.
 * @conf   data_length          Int. How long the data buffers are, in bytes. Must be a multiple of 32B.
 *
 * @author Keith Vanderlinde
 *
 */
class cudaSaxpyKernel : public cudaCommand {
public:
    cudaSaxpyKernel(kotekan::Config& config, const string& unique_name,
                   kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device);
    ~cudaSaxpyKernel();
    cudaEvent_t execute(int gpu_frame_id, cudaEvent_t pre_event) override;

protected:
private:
    // Common configuration values (which do not change in a run)
    /// Float that the input array is scaled by prior to adding to the output.
    float _saxpy_scale;
    /// Length (in bytes) of the input & output arrays.
    int32_t _data_length;
};

#endif // CUDA_SAXPY_KERNEL_CUH
