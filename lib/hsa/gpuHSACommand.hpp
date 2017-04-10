#ifndef GPU_HSA_COMMAND_H
#define GPU_HSA_COMMAND_H

#include "Config.hpp"
#include "errors.h"
#include "assert.h"
#include "buffers.h"
#include "gpuHSADeviceInterface.hpp"
#include "bufferContainer.hpp"

#include <stdio.h>
#include <string>
#include "hsa/hsa.h"
#include "hsa/hsa_ext_finalize.h"
#include "hsa/hsa_ext_amd.h"

using std::string;

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
class gpuHSAcommand
{
public:
    // Kernel file name is optional.
    gpuHSAcommand(const string &kernel_name, const string &kernel_file_name,
                gpuHSADeviceInterface &device, Config &config,
                bufferContainer &host_buffers);
    virtual ~gpuHSAcommand();
    string &get_name();

    // This function blocks on whatever resource is required by this command
    // for example if this command requires a full buffer frame to copy
    // then it should block on that.  It should also block on having any
    // free output buffers that might be referenced by this command.
    virtual void wait_on_precondition(int gpu_frame_id);

    // Adds either a copy or kernel to one of the hardware queues.
    virtual hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq,
                                    hsa_signal_t precede_signal) = 0;

    // Should clean any signals used by the command.
    // Note that as a byproduct of this, one shouldn't use the signal after this
    // function has been cased.
    virtual void finalize_frame(int frame_id);

    virtual void apply_config(const uint64_t &fpga_seq);
protected:

    // Extract the code handle for the specified kernelName from the specified fileName
    // Returns a 64-bit code object which can be used with an AQL packet
    uint64_t load_hsaco_file(string &file_name, string &kernel_name);

    // Creates the memory needed for the kernel args.
    void allocate_kernel_arg_memory(int max_size);

    Config &config;
    gpuHSADeviceInterface &device;
    bufferContainer host_buffers;

    // Final signals array
    // Note a value of zero for one of the signals means that
    // it isn't currently set.
    hsa_signal_t * signals;

    // Pointers to kernel args
    // Note, not used for all commands, only kernel commands.
    // It's possible we might want to break this into two classes,
    // but for now this works.
    void ** kernel_args;

    // Pointer to the kernel
    uint64_t kernel_object = 0;

    // The command or kernel name
    string command_name;

    // The file which containts the kernel, if applicable.
    string kernel_file_name;

    // Config variables
    int _gpu_buffer_depth;
};

#endif // GPU_COMMAND_H

