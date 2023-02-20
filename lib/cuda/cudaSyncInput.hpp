#ifndef KOTEKAN_CUDA_SYNC_INPUT_HPP
#define KOTEKAN_CUDA_SYNC_INPUT_HPP

#include "cudaSyncStream.hpp"

#include <vector>

/**
 * @class cudaSyncInput
 * @brief A synchronization point meant as a shortcut for the common case of waiting
 * on the input stream before starting the default kernel stream.
 *
 * This command is equivalent to a cudaSyncStream with @c cuda_stream
 * = 2 (the default kernel stream), waiting on stream 0 (the default
 * input stream).
 *
 * @author Dustin Lang
 */
class cudaSyncInput : public cudaSyncStream {
public:
    cudaSyncInput(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device);
    ~cudaSyncInput();
};


#endif // KOTEKAN_CUDA_SYNC_INPUT_HPP
