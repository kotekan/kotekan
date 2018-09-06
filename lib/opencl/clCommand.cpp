#include "clCommand.hpp"
#include <string.h>
#include <iostream>
#include <string>

using std::string;
using std::to_string;

clCommand::clCommand(
        const string &default_kernel_command,
        const string &default_kernel_file_name,
        Config& config_,
        const string &unique_name_,
        bufferContainer &host_buffers_,
        clDeviceInterface& device_) :
        kernel_command(default_kernel_command),
        kernel_file_name(default_kernel_file_name),
        config(config_),
        unique_name(unique_name_),
        host_buffers(host_buffers_),
        device(device_)
{
    _gpu_buffer_depth = config.get_int(unique_name, "buffer_depth");

    // Set the local log level.
    string s_log_level = config.get_string(unique_name, "log_level");
    set_log_level(s_log_level);
    set_log_prefix(unique_name);

    // Load the kernel if there is one.
    if (default_kernel_file_name != "") {
        kernel_file_name = config.get_string_default(unique_name,"kernel_path",".") + "/" +
                           config.get_string_default(unique_name,"kernel",default_kernel_file_name);
        kernel_command = config.get_string_default(unique_name,"command",default_kernel_command);
    }

    _buffer_depth = config.get_int(unique_name, "buffer_depth");

    post_event = (cl_event*)malloc(_gpu_buffer_depth * sizeof(cl_event));
    CHECK_MEM(post_event);
    for (int j=0;j<_gpu_buffer_depth;++j) post_event[j] = NULL;
}

clCommand::~clCommand()
{
    INFO("KERNEL COMMAND: %s",kernel_command.c_str());
    if (kernel_command != ""){
        CHECK_CL_ERROR( clReleaseKernel(kernel) );
        DEBUG("kernel Freed");
        CHECK_CL_ERROR( clReleaseProgram(program) );
        DEBUG("program Freed");
    }
    free(post_event);
    DEBUG("post_event Freed: %s",unique_name.c_str());
}

int clCommand::wait_on_precondition(int gpu_frame_id) {
    (void)gpu_frame_id;
    return 0;
}

string &clCommand::get_name() {
    return kernel_command;
}

void clCommand::apply_config(const uint64_t& fpga_seq) {
}

void clCommand::build()
{
    size_t program_size;
    FILE *fp;
    char *program_buffer;
    cl_int err;

    if (kernel_command != ""){
        DEBUG2("Building! %s",kernel_command.c_str())
        fp = fopen(kernel_file_name.c_str(), "r");
        if (fp == NULL){
            ERROR("error loading file: %s", kernel_file_name.c_str());
            exit(errno);
        }
        fseek(fp, 0, SEEK_END);
        program_size = ftell(fp);
        rewind(fp);

        program_buffer = (char*)malloc(program_size+1);
        program_buffer[program_size] = '\0';
        int sizeRead = fread(program_buffer, sizeof(char), program_size, fp);
        if (sizeRead < (int32_t)program_size)
            ERROR("Error reading the file: %s", kernel_file_name.c_str());
        fclose(fp);
        program = clCreateProgramWithSource(device.get_context(),
                                        (cl_uint)1,
                                        (const char**)&program_buffer,
                                        &program_size, &err );
        CHECK_CL_ERROR (err);

        program_size = 0;
        free(program_buffer);
        DEBUG2("Built! %s",kernel_command.c_str())
    }
}

cl_event clCommand::execute(int gpu_frame_id, const uint64_t& fpga_seq, cl_event pre_event)
{
    assert(gpu_frame_id<_gpu_buffer_depth);
    assert(gpu_frame_id>=0);

    return NULL;
}

void clCommand::setKernelArg(cl_uint param_ArgPos, cl_mem param_Buffer)
{
    CHECK_CL_ERROR( clSetKernelArg(kernel,
    param_ArgPos,
    sizeof(void*),
    (void*) &param_Buffer) );
}

void clCommand::finalize_frame(int gpu_frame_id) {
    bool profiling = true;
    if (post_event[gpu_frame_id] != NULL){
        if (profiling) {
            cl_ulong start_time, stop_time;
            CHECK_CL_ERROR(clGetEventProfilingInfo (post_event[gpu_frame_id],
                                                    CL_PROFILING_COMMAND_START,
                                                    sizeof(start_time),
                                                    &start_time,
                                                    NULL) );
            CHECK_CL_ERROR(clGetEventProfilingInfo (post_event[gpu_frame_id],
                                                    CL_PROFILING_COMMAND_END,
                                                    sizeof(stop_time),
                                                    &stop_time,
                                                    NULL) );
            last_gpu_execution_time = ((double)(stop_time - start_time)) * 1e-9;
        }

//        cl_uint ref;
//        clGetEventInfo(post_event[gpu_frame_id], CL_EVENT_REFERENCE_COUNT, sizeof(cl_uint), &ref, NULL);
//        INFO("Event Ref Ct: %i", ref);

        CHECK_CL_ERROR(clReleaseEvent(post_event[gpu_frame_id]));
        post_event[gpu_frame_id] = NULL;
    }
    else INFO("*** WTF? Null event!");

}


double clCommand::get_last_gpu_execution_time() {
    return last_gpu_execution_time;
}

clCommandType clCommand::get_command_type() {
    return command_type;
}

