#ifndef KOTEKAN_CUDA_SYNC_OUTPUT_HPP
#define KOTEKAN_CUDA_SYNC_OUTPUT_HPP

#include "cudaSyncStream.hpp"

#include <vector>

/**
 * @class cudaSyncOutput
 * @brief A synchronization point meant as a shortcut for the common case of waiting
 * on all compute streams before starting the output stream.
 *
 * This command is equivalent to a cudaSyncStream with @c cuda_stream
 * = 1 (the default output stream), waiting on streams >= 2 (the compute streams).
 *
 * @author Dustin Lang
 */
class cudaSyncOutput : public cudaSyncStream {
public:
    cudaSyncOutput(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device, int inst);
    ~cudaSyncOutput();
};


#endif // KOTEKAN_CUDA_SYNC_OUTPUT_HPP
