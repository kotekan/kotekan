#include "clCommand.hpp"

#include <iostream>

using kotekan::bufferContainer;
using kotekan::Config;

using std::string;
using std::to_string;

// default-constructed global so that people don't have to write
// "std::shared_ptr<clCommandState>()" in a bunch of places.  Basically a custom NULL.
std::shared_ptr<clCommandState> no_cl_command_state;

clCommand::clCommand(Config& config_, const std::string& unique_name_,
                     bufferContainer& host_buffers_, clDeviceInterface& device_, int instance_num,
                     std::shared_ptr<gpuCommandState> command_state,
                     const std::string& default_kernel_command,
                     const std::string& default_kernel_file_name) :
    gpuCommand(config_, unique_name_, host_buffers_, device_, instance_num, command_state,
               default_kernel_command, default_kernel_file_name),
    post_event(nullptr), device(device_) {}

clCommand::~clCommand() {
    if (kernel_file_name != "") {
        CHECK_CL_ERROR(clReleaseKernel(kernel));
        DEBUG("kernel Freed");
        CHECK_CL_ERROR(clReleaseProgram(program));
        DEBUG("program Freed");
    }
}

void clCommand::finalize_frame() {
    if (post_event != nullptr) {
        if (profiling) {
            cl_ulong start_time, stop_time;
            CHECK_CL_ERROR(clGetEventProfilingInfo(post_event, CL_PROFILING_COMMAND_START,
                                                   sizeof(start_time), &start_time, nullptr));
            CHECK_CL_ERROR(clGetEventProfilingInfo(post_event, CL_PROFILING_COMMAND_END,
                                                   sizeof(stop_time), &stop_time, nullptr));
            double active_time = (double)(stop_time - start_time) * 1e-9;
            excute_time->add_sample(active_time);
            utilization->add_sample(active_time / frame_arrival_period);
        }

        CHECK_CL_ERROR(clReleaseEvent(post_event));
        post_event = nullptr;
    } else
        FATAL_ERROR("Null OpenCL event!");
}


// Specialist functions:
void clCommand::build() {
    size_t program_size;
    FILE* fp;
    char* program_buffer;
    cl_int err;

    if (kernel_file_name != "") {
        DEBUG2("Building! {:s} from file {:s}", kernel_command, kernel_file_name);
        fp = fopen(kernel_file_name.c_str(), "r");
        if (fp == nullptr) {
            FATAL_ERROR("error loading file: {:s}", kernel_file_name);
        }
        fseek(fp, 0, SEEK_END);
        program_size = ftell(fp);
        rewind(fp);

        program_buffer = (char*)malloc(program_size + 1);
        program_buffer[program_size] = '\0';
        int sizeRead = fread(program_buffer, sizeof(char), program_size, fp);
        if (sizeRead < (int32_t)program_size)
            ERROR("Error reading the file: {:s}", kernel_file_name);
        fclose(fp);
        program =
            clCreateProgramWithSource(((clDeviceInterface*)&device)->get_context(), (cl_uint)1,
                                      (const char**)&program_buffer, &program_size, &err);
        CHECK_CL_ERROR(err);

        program_size = 0;
        free(program_buffer);
        DEBUG2("Built! {:s}", kernel_command);
    }
}

void clCommand::setKernelArg(cl_uint param_ArgPos, cl_mem param_Buffer) {
    CHECK_CL_ERROR(clSetKernelArg(kernel, param_ArgPos, sizeof(void*), (void*)&param_Buffer));
}

std::vector<Buffer*> clCommand::get_buffer_array(const std::string array_name, bool register_buffer, bool producer) {
    std::vector<Buffer*> buffers;
    size_t frame_size = 0;
    for (const auto& buffer_name : config.get_array<std::string>(unique_name, array_name)) {
        Buffer* buf = host_buffers.get_buffer(buffer_name);
        if (buf == nullptr) {
            FATAL_ERROR("Buffer {:s} not found", buffer_name);
        }
        if (frame_size == 0) {
            frame_size = buf->aligned_frame_size;
        } else if (frame_size != buf->aligned_frame_size) {
            FATAL_ERROR("Buffer {:s} has different frame size than previous buffers in array {:s}",
                        buffer_name, array_name);
        }
        // Register the buffer as a producer or consumer
        if (register_buffer) {
            if (producer) {
                register_producer(buf, unique_name.c_str());
            } else {
                register_consumer(buf, unique_name.c_str());
            }
        }
        buffers.push_back(buf);
    }
    return buffers;
}
