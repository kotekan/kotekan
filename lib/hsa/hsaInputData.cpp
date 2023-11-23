#include "hsaInputData.hpp"

#include "Config.hpp"             // for Config
#include "buffer.hpp"             // for Buffer, mark_frame_empty, register_consumer, wait_for_...
#include "bufferContainer.hpp"    // for bufferContainer
#include "chimeMetadata.hpp"      // for get_first_packet_recv_time
#include "gpuCommand.hpp"         // for gpuCommandType, gpuCommandType::COPY_IN
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface, Config
#include "kotekanLogging.hpp"     // for DEBUG2, INFO
#include "util.h"                 // for e_time

#include <cstdint>    // for int32_t
#include <exception>  // for exception
#include <random>     // for mt19937, random_device, uniform_real_distribution
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error
#include <sys/time.h> // for timeval
#include <unistd.h>   // for usleep
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_HSA_COMMAND(hsaInputData);

hsaInputData::hsaInputData(Config& config, const std::string& unique_name,
                           bufferContainer& host_buffers, hsaDeviceInterface& device) :
    hsaCommand(config, unique_name, host_buffers, device, "hsaInputData", "") {
    command_type = gpuCommandType::COPY_IN;

    int header_size = 0;
    _num_elements = config.get<int32_t>(unique_name, "num_elements");
    _num_local_freq = config.get<int32_t>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int32_t>(unique_name, "samples_per_data_set");
    _enable_delay = config.get_default<bool>(unique_name, "enable_delay", false);
    _delay_max_fraction = config.get_default<float>(unique_name, "delay_max_fraction", 0.5);
    if (_num_elements
        <= 2) // TODO FIND BETTER WORK AROUND FOR DISTINGUISHING VDIF/CHIME INPUT LENGTH
        header_size = 32;
    input_frame_len = (_num_elements * (_num_local_freq + header_size)) * _samples_per_data_set;

    // Generate a constant random delay for each frame after
    // getting it from the network buffer.
    // This is used to help balance out the power usage.
    if (_enable_delay) {
        std::random_device rd; // Seed
        std::mt19937 gen(rd());
        double max_delay = (double)_samples_per_data_set / _sample_arrival_rate;
        std::uniform_real_distribution<> dis(0.0, max_delay * _delay_max_fraction);
        _random_delay = dis(gen);
        INFO("Setting fixed delay to: {:f}", _random_delay);
    }

    network_buf = host_buffers.get_buffer("network_buf");
    network_buf->register_consumer(unique_name);
    network_buffer_id = 0;
    network_buffer_precondition_id = 0;
    network_buffer_finalize_id = 0;
}


hsaInputData::~hsaInputData() {}

int hsaInputData::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;

    // Wait for there to be data in the input (network) buffer.
    uint8_t* frame =
        network_buf->wait_for_full_frame(unique_name, network_buffer_precondition_id);
    if (frame == nullptr)
        return -1;
    // INFO("Got full buffer {:s}[{:d}], gpu[{:d}][{:d}]", network_buf->buffer_name,
    // network_buffer_precondition_id,
    //        device.get_gpu_id(), gpu_frame_id);

    if (_enable_delay) {
        timeval recv_time = get_first_packet_recv_time(network_buf, network_buffer_precondition_id);
        double d_recv_time = (double)recv_time.tv_sec + (double)recv_time.tv_usec / 1000000.0;
        double expected_delay = (double)_samples_per_data_set / _sample_arrival_rate;
        double current_time = e_time();
        // This adjusts the delay to make sure we don't exceed the frame arrive period,
        // in the event we reach this point after the expected time.
        double delay = _random_delay - (current_time - (d_recv_time + expected_delay));
        DEBUG2("frame_time: {:f}, expected_delay: {:f}, current_time: {:f}, random_delay: {:f}, "
               "actual delay: {:f}",
               d_recv_time, expected_delay, current_time, _random_delay, delay);
        // The above forumal shouldn't produce a delay less than _random_dela, unless
        // something is wrong with the time value given, in which case this delay
        // could cause the system to become unstable, so the second condition guards against this.
        if (delay > 0 && delay < (_random_delay + 0.01))
            usleep((int)(delay * 1000000));
    }

    network_buffer_precondition_id = (network_buffer_precondition_id + 1) % network_buf->num_frames;
    return 0;
}

hsa_signal_t hsaInputData::execute(int gpu_frame_id, hsa_signal_t precede_signal) {

    // Get the gpu and cpu memory pointers.
    void* gpu_memory_frame = device.get_gpu_memory_array("input", gpu_frame_id, _gpu_buffer_depth, input_frame_len);
    void* host_memory_frame = (void*)network_buf->frames[network_buffer_id];

    // Do the input data copy.
    device.async_copy_host_to_gpu(gpu_memory_frame, host_memory_frame, input_frame_len,
                                  precede_signal, signals[gpu_frame_id]);

    network_buffer_id = (network_buffer_id + 1) % network_buf->num_frames;

    return signals[gpu_frame_id];
}

void hsaInputData::finalize_frame(int frame_id) {
    hsaCommand::finalize_frame(frame_id);
    network_buf->mark_frame_empty(unique_name, network_buffer_finalize_id);
    network_buffer_finalize_id = (network_buffer_finalize_id + 1) % network_buf->num_frames;
}
