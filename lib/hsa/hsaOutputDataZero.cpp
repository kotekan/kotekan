#include "hsaOutputDataZero.hpp"

#include "Config.hpp"             // for Config
#include "gpuCommand.hpp"         // for gpuCommandType, gpuCommandType::COPY_IN
#include "hsaBase.h"              // for hsa_host_free, hsa_host_malloc
#include "hsaCommand.hpp"         // for REGISTER_HSA_COMMAND, _factory_aliashsaCommand
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface, Config
#include "kotekanLogging.hpp"     // for INFO

#include "fmt.hpp" // for format, fmt

#include <exception> // for exception
#include <regex>     // for match_results<>::_Base_type
#include <string.h>  // for memset
#include <vector>    // for vector

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_HSA_COMMAND(hsaOutputDataZero);

hsaOutputDataZero::hsaOutputDataZero(Config& config, const std::string& unique_name,
                                     bufferContainer& host_buffers, hsaDeviceInterface& device) :
    hsaSubframeCommand(config, unique_name, host_buffers, device, "hsaOutputDataZero", "") {
    command_type = gpuCommandType::COPY_IN;

    int block_size = config.get<int>(unique_name, "block_size");
    int num_elements = config.get<int>(unique_name, "num_elements");
    _num_blocks = (int32_t)(num_elements / block_size) * (num_elements / block_size + 1) / 2.;
    output_len = _num_blocks * block_size * block_size * 2 * sizeof(int32_t);

    output_zeros = hsa_host_malloc(output_len, device.get_gpu_numa_node());
    INFO("hsaOutputDataZero gpu[{:d}], Creating the output zero buffer: {:p}, len: {:d}",
         device.get_gpu_id(), output_zeros, output_len);

    memset(output_zeros, 0, output_len);
}

hsaOutputDataZero::~hsaOutputDataZero() {
    hsa_host_free(output_zeros);
}

hsa_signal_t hsaOutputDataZero::execute(int gpu_frame_id, hsa_signal_t precede_signal) {

    void* gpu_output_ptr = device.get_gpu_memory_array(
                                                       fmt::format(fmt("corr_{:d}"), _sub_frame_index), gpu_frame_id, _gpu_buffer_depth, output_len);

    device.async_copy_host_to_gpu(gpu_output_ptr, output_zeros, output_len, precede_signal,
                                  signals[gpu_frame_id]);

    return signals[gpu_frame_id];
}
