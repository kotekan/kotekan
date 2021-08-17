#include "hsaCommand.hpp"

#include "Config.hpp"             // for Config
#include "hsa/hsa_ext_amd.h"      // for hsa_amd_profiling_async_copy_time_t, hsa_amd_profiling...
#include "hsaBase.h"              // for HSA_CHECK, hsa_host_free, hsa_host_malloc
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface, Config
#include "kotekanLogging.hpp"     // for INFO

#include <assert.h>  // for assert
#include <exception> // for exception
#include <fstream>   // for ifstream, operator|, basic_istream::seekg, ios, basic_...
#include <regex>     // for match_results<>::_Base_type
#include <stdexcept> // for runtime_error
#include <stdlib.h>  // for free, malloc
#include <string.h>  // for memset, size_t
#include <vector>    // for vector

using kotekan::bufferContainer;
using kotekan::Config;

using std::string;

#define MAX_ARGS_LEN 64

hsaCommand::hsaCommand(Config& config_, const std::string& unique_name_,
                       bufferContainer& host_buffers_, hsaDeviceInterface& device_,
                       const std::string& default_kernel_command,
                       const std::string& default_kernel_file_name) :
    gpuCommand(config_, unique_name_, host_buffers_, device_, default_kernel_command,
               default_kernel_file_name),
    device(device_) {
    _gpu_buffer_depth = config.get<int>(unique_name, "buffer_depth");

    // Set the local log level.
    std::string s_log_level = config.get<std::string>(unique_name, "log_level");
    set_log_level(s_log_level);
    set_log_prefix(unique_name);

    signals = (hsa_signal_t*)hsa_host_malloc(_gpu_buffer_depth * sizeof(hsa_signal_t),
                                             device.get_gpu_numa_node());
    assert(signals != nullptr);
    memset(signals, 0, _gpu_buffer_depth * sizeof(hsa_signal_t));

    for (int i = 0; i < _gpu_buffer_depth; ++i) {
        hsa_signal_create(0, 0, nullptr, &signals[i]);
    }

    // Not everyone needs this, maybe move out of constructor
    kernel_args =
        (void**)hsa_host_malloc(_gpu_buffer_depth * sizeof(void*), device.get_gpu_numa_node());
    assert(kernel_args != nullptr);

    // Load the kernel if there is one.
    if (default_kernel_file_name != "") {
        kernel_file_name =
            config.get_default<std::string>(unique_name, "kernel_path", ".") + "/"
            + config.get_default<std::string>(unique_name, "kernel", default_kernel_file_name);
        kernel_command =
            config.get_default<std::string>(unique_name, "command", default_kernel_command);
        // Should this be moved to the base class?
        allocate_kernel_arg_memory(MAX_ARGS_LEN);
        kernel_object = load_hsaco_file(kernel_file_name, kernel_command);
    }
}

hsaCommand::~hsaCommand() {

    hsa_status_t hsa_status;
    for (int i = 0; i < _gpu_buffer_depth; ++i) {
        // DEBUG("Free kernel arg");
        hsa_status = hsa_memory_free(kernel_args[i]);
        HSA_CHECK(hsa_status);
        // DEBUG("Free signal");
        hsa_status = hsa_signal_destroy(signals[i]);
        HSA_CHECK(hsa_status);
    }

    // DEBUG("Free kernel args");
    hsa_host_free(kernel_args);
    // DEBUG("Free signals");
    hsa_host_free(signals);

    // TODO free kernel!!!
}

void hsaCommand::allocate_kernel_arg_memory(int max_size) {
    hsa_status_t hsa_status;
    for (int i = 0; i < _gpu_buffer_depth; ++i) {

        hsa_status = hsa_memory_allocate(device.get_kernarg_region(), max_size, &kernel_args[i]);
        HSA_CHECK(hsa_status);
    }
}

void hsaCommand::finalize_frame(int frame_id) {
    hsa_status_t hsa_status;
    hsa_amd_profiling_dispatch_time_t kernel_time;
    hsa_amd_profiling_async_copy_time_t copy_time;
    uint64_t timestamp_frequency_hz = device.get_hsa_timestamp_freq();

    if (signals[frame_id].handle == 0) {
        return;
    }

    if (command_type == gpuCommandType::KERNEL) {
        hsa_status = hsa_amd_profiling_get_dispatch_time(device.get_gpu_agent(), signals[frame_id],
                                                         &kernel_time);
        last_gpu_execution_time =
            ((double)(kernel_time.end - kernel_time.start)) / (double)timestamp_frequency_hz;
    } else if (command_type == gpuCommandType::COPY_IN
               || command_type == gpuCommandType::COPY_OUT) {
        if (profiling)
            hsa_status = hsa_amd_profiling_get_async_copy_time(signals[frame_id], &copy_time);
            double active_time =
                ((double)(copy_time.end - copy_time.start)) / (double)timestamp_frequency_hz;
            excute_time->add_sample(active_time);
            utilization->add_sample(active_time/frame_arrival_period);
    } else {
        return;
    }

    HSA_CHECK(hsa_status);
}

uint64_t hsaCommand::load_hsaco_file(string& file_name, std::string& kernel_name) {

    hsa_status_t hsa_status;

    // Open file.
    INFO("Loading {:s} {:s}", file_name, kernel_name);
    std::ifstream file(file_name, std::ios::in | std::ios::binary);
    assert(file.is_open() && file.good());

    // Find out file size.
    file.seekg(0, file.end);
    size_t code_object_size = file.tellg();
    file.seekg(0, file.beg);

    // Allocate memory for raw code object.
    char* raw_code_object = (char*)malloc(code_object_size);
    assert(raw_code_object);

    // Read file contents.
    file.read(raw_code_object, code_object_size);

    // Close file.
    file.close();

    // Deserialize code object.
    hsa_code_object_t code_object = {0};
    hsa_status = hsa_code_object_deserialize((void*)raw_code_object, code_object_size, nullptr,
                                             &code_object);
    HSA_CHECK(hsa_status);
    assert(0 != code_object.handle);

    // Create executable.
    hsa_executable_t hsaExecutable;
    hsa_status = hsa_executable_create(HSA_PROFILE_FULL, HSA_EXECUTABLE_STATE_UNFROZEN, nullptr,
                                       &hsaExecutable);
    HSA_CHECK(hsa_status);

    // Load code object.
    hsa_status = hsa_executable_load_code_object(hsaExecutable, device.get_gpu_agent(), code_object,
                                                 nullptr);
    HSA_CHECK(hsa_status);

    // Freeze executable.
    hsa_status = hsa_executable_freeze(hsaExecutable, nullptr);
    HSA_CHECK(hsa_status);

    // Get symbol handle.
    hsa_executable_symbol_t kernelSymbol;
    hsa_status = hsa_executable_get_symbol(hsaExecutable, nullptr, kernel_name.c_str(),
                                           device.get_gpu_agent(), 0, &kernelSymbol);
    HSA_CHECK(hsa_status);

    // Get code handle.
    uint64_t codeHandle;
    hsa_status = hsa_executable_symbol_get_info(
        kernelSymbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &codeHandle);
    HSA_CHECK(hsa_status);

    uint32_t group_segment_size;
    hsa_status = hsa_executable_symbol_get_info(
        kernelSymbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &group_segment_size);
    INFO("Kernel {:s}:{:s} group_segment_size {:d}", file_name, kernel_name, group_segment_size);

    uint32_t priv_segment_size;
    hsa_status = hsa_executable_symbol_get_info(
        kernelSymbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &priv_segment_size);
    INFO("Kernel {:s}:{:s} group_segment_size {:d}", file_name, kernel_name, priv_segment_size);

    // Free raw code object memory.
    free((void*)raw_code_object);

    return codeHandle;
}

void hsaCommand::packet_store_release(uint32_t* packet, uint16_t header, uint16_t rest) {
    __atomic_store_n(packet, ((uint32_t)header) | (((uint32_t)rest) << 16), __ATOMIC_RELEASE);
}

uint16_t hsaCommand::header(hsa_packet_type_t type) {
    uint16_t header = (type << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER);
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
    return header;
}

hsa_signal_t hsaCommand::enqueue_kernel(const kernelParams& params, const int gpu_frame_id) {

    // Get the queue index
    uint64_t packet_id = hsa_queue_add_write_index_scacquire(device.get_queue(), 1);

    // Make sure the queue isn't full
    // Should never hit this condition, but lets be safe.
    // See the HSA docs for details.
    while (packet_id - hsa_queue_load_read_index_relaxed(device.get_queue())
           >= device.get_queue()->size)
        ;

    // Get the packet address
    hsa_kernel_dispatch_packet_t* packet =
        (hsa_kernel_dispatch_packet_t*)device.get_queue()->base_address
        + (packet_id % device.get_queue()->size);

    //    packet->header = HSA_PACKET_TYPE_INVALID;
    packet_store_release((uint32_t*)packet, header(HSA_PACKET_TYPE_INVALID), 0);
    // Zero the packet (see HSA docs)
    memset(((uint8_t*)packet) + 4, 0, sizeof(hsa_kernel_dispatch_packet_t) - 4);

    // Set kernel dims for packet
    packet->workgroup_size_x = params.workgroup_size_x;
    packet->workgroup_size_y = params.workgroup_size_y;
    packet->workgroup_size_z = params.workgroup_size_z;
    packet->grid_size_x = params.grid_size_x;
    packet->grid_size_y = params.grid_size_y;
    packet->grid_size_z = params.grid_size_z;

    // Extra details
    packet->private_segment_size = params.private_segment_size;
    packet->group_segment_size = params.group_segment_size;

    // Set the kernel object (loaded HSACO code)
    packet->kernel_object = this->kernel_object;

    // Add the kernel args
    // Must have been copied before this function is called!
    packet->kernarg_address = (void*)kernel_args[gpu_frame_id];

    // Create the completion signal for this kernel run.
    //    assert(hsa_signal_load_relaxed(signals[gpu_frame_id])==0 && "frame signal not complete.");
    //    hsa_signal_store_relaxed(signals[gpu_frame_id], 1);
    while (0 < hsa_signal_cas_screlease(signals[gpu_frame_id], 0, 1))
        ;
    packet->completion_signal = signals[gpu_frame_id];

    // Create the AQL packet header as an atomic operation,
    // recommended by the HSA docs.
    packet_store_release((uint32_t*)packet, header(HSA_PACKET_TYPE_KERNEL_DISPATCH),
                         params.num_dims << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS);

    // Notify the device there is a new packet in the queue
    hsa_signal_store_screlease(device.get_queue()->doorbell_signal, packet_id);

    return packet->completion_signal;
}

gpuCommandType hsaCommand::get_command_type() {
    return command_type;
}

string hsaCommand::get_kernel_file_name() {
    return kernel_file_name;
}
