#ifndef KOTEKAN_CUDA_RECHUNK_HPP
#define KOTEKAN_CUDA_RECHUNK_HPP

#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"

class cudaRechunkState : public cudaCommandState {
public:
    cudaRechunkState(kotekan::Config& config, const std::string& unique_name,
                     kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device) :
        cudaCommandState(config, unique_name, host_buffers, device),
        cols_accumulated(0) {}
    int cols_accumulated;
};

/**
 * @class cudaRechunk
 * @brief cudaCommand for combining GPU frames before further processing.
 *
 * @author Dustin Lang
 *
 * @par GPU Memory
 * @gpu_mem  gpu_mem_input  Input matrix of data (of unspecified type)
 *   @gpu_mem_type   staging
 *   @gpu_mem_format array of raw bytes
 * @gpu_mem  gpu_mem_output Output matrix of data (of unspecified type)
 *   @gpu_mem_type   staging
 *   @gpu_mem_format Array of raw bytes
 *
 * @conf cols_input  - Int
 * @conf cols_output - Int
 * @conf rows        - Int
 *
 * The input array is assumed to be of size (cols_input x rows) bytes.
 *
 * The output array is assumed to be of size (cols_output x rows) bytes.
 *
 * Assuming you have:
 * - Input matrix size R x C, element size S
 * - Number of input matrices to be concatenated = N
 *
 * If you want to "hstack" the input arrays -- that is, keeping the
 * number of rows the same -- then you can set the cols_input,
 * cols_output, and rows values to their intuitive values, where
 * cols_input includes the element size:
 *
 *   cols_input = C * S
 *   rows = R
 *   cols_output = C * S * N
 *
 * If you want to "vstack" the input arrays -- that is, keeping the
 * number of columns the same -- then you can instead think of this as
 * simply concatenating the bytes of your input matrices, and
 * therefore you can set:
 *   cols_input  = C * R * S
 *   cols_output = C * R * S * N
 *   rows = 1
 *
 * (That is, in memory, vstack'ing is just like doing a python ravel()
 * on your arrays and then appending them.)
 *
 * (These equations also work if N is not a whole number, ie, for
 * hstacking, the output column size can be C_out x S rather than
 * (C_in x N x S); for vstacking, the output column size can be (R_out
 * x C x S) rather than (R_in x N x C x S).)
 *
 */
class cudaRechunk : public cudaCommand {
public:
    cudaRechunk(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                std::shared_ptr<cudaCommandState>);
    ~cudaRechunk();
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;

protected:
    cudaRechunkState* get_state();

private:
    size_t _cols_input;
    size_t _cols_output;
    size_t _rows;

    std::string _set_flag;

    /// GPU side memory name for the time-stream input
    std::string _gpu_mem_input;
    /// GPU side memory name for the time-stream output
    std::string _gpu_mem_output;

    /// GPU side memory name for the array where we accumulate the inputs
    std::string gpu_mem_accum;

    /// Optional, name of the field in the pipeline state object that holds the number of input
    /// columns to copy.
    std::string _input_columns_field;
};

#endif // KOTEKAN_CUDA_RECHUNK_HPP
