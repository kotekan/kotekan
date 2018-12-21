#ifndef GPU_HSA_COMMAND_H
#define GPU_HSA_COMMAND_H

#include "Config.hpp"
#include "buffer.h"
#include "bufferContainer.hpp"
#include "chimeMetadata.h"
#include "errors.h"
#include "factory.hpp"
#include "gpuCommand.hpp"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hsa/hsa_ext_finalize.h"
#include "hsaBase.h"
#include "hsaDeviceInterface.hpp"
#include "kotekanLogging.hpp"

#include <assert.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unistd.h>

struct kernelParams {
    uint16_t workgroup_size_x;
    uint16_t workgroup_size_y;
    uint16_t workgroup_size_z;
    uint32_t grid_size_x;
    uint32_t grid_size_y;
    uint32_t grid_size_z;
    uint16_t num_dims; // Could this be automatically generated from above?
    uint16_t private_segment_size;
    uint16_t group_segment_size;
};

// Note there are _gpu_buffer_depth frames, which can be thought of as distinct
// blocks of input, output, and kernel arg memory for each chain in the
// gpu pipeline which looks something like:
//
// copy-in[0] --v      copy-in[1] --v      copy-in[2] --v
//              |--> kernel[0] --v  |--> kernel[1] --v  |--> kernel[2] --v
//                               |--> copy-out[0]    |--> copy-out[1]    |--> copy-out[2]
//
// This allow us to have host->gpu copies, kernels, and gpu->host copies
// running co-currently, at the expense of some complexity and extra memory.

class hsaCommand : public gpuCommand {
public:
    // Kernel file name is optional.
    hsaCommand(Config& config, const string& unique_name, bufferContainer& host_buffers,
               hsaDeviceInterface& device, const string& default_kernel_command = "",
               const string& default_kernel_file_name = "");
    virtual ~hsaCommand();

    // Adds either a copy or kernel to one of the hardware queues.
    virtual hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) = 0;

    // Should clean any signals used by the command.
    // Note that as a byproduct of this, one shouldn't use the signal after this
    // function has been cased.
    virtual void finalize_frame(int frame_id);

    gpuCommandType get_command_type();

    string get_kernel_file_name();

protected:
    // Extract the code handle for the specified kernelName from the specified fileName
    // Returns a 64-bit code object which can be used with an AQL packet
    uint64_t load_hsaco_file(string& file_name, string& kernel_name);

    // Creates the memory needed for the kernel args.
    void allocate_kernel_arg_memory(int max_size);

    // Requires that kernel_args[frame_id] has been populated with the
    // kernel arguments.
    hsa_signal_t enqueue_kernel(const kernelParams& dims, const int gpu_frame_id);

    // Final signals array
    // Note a value of zero for one of the signals means that
    // it isn't currently set.
    hsa_signal_t* signals;

    hsaDeviceInterface& device;

    // Pointers to kernel args
    // Note, not used for all commands, only kernel commands.
    // It's possible we might want to break this into two classes,
    // but for now this works.
    void** kernel_args;

    // Pointer to the kernel
    uint64_t kernel_object = 0;

    // Helper functions from HSA docs for kernel packet queueing.
    // Some (or all) of this could likely be moved into enqueue_kernel()
    void packet_store_release(uint32_t* packet, uint16_t header, uint16_t rest);
    uint16_t header(hsa_packet_type_t type);
};

// Create a factory for hsaCommands
CREATE_FACTORY(hsaCommand, // const string &, const string &,
               Config&, const string&, bufferContainer&, hsaDeviceInterface&);
#define REGISTER_HSA_COMMAND(newCommand)                                                           \
    REGISTER_NAMED_TYPE_WITH_FACTORY(hsaCommand, newCommand, #newCommand)


#endif // GPU_COMMAND_H
