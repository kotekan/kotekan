#include "gpu_command.h"
#include <string.h>
#include <iostream>
#include <string>

using std::string;
using std::to_string;

gpu_command::gpu_command(const char* param_name, Config &param_config, const string &unique_name_) :
    config(param_config), gpuCommandState(0) , gpuKernel(NULL), unique_name(unique_name_)
{
    name = strdup(param_name);
    INFO("Name: %s, %s", param_name, name);
}

gpu_command::gpu_command(const char * param_gpuKernel, const char* param_name, Config &param_config, const string &unique_name_) :
    config(param_config), gpuCommandState(0), gpuKernel(NULL), unique_name(unique_name_)
{
    gpuKernel = new char[strlen(param_gpuKernel)+1];
    strcpy(gpuKernel, param_gpuKernel);
    gpuCommandState=1;
    name = strdup(param_name);
    INFO("Name: %s, %s", param_name, name);
}

gpu_command::~gpu_command()
{
    if (gpuCommandState==1)
        free(gpuKernel);
    free(name);
}

char* gpu_command::get_name()
{
//    INFO("get_name(): %s", name);
    return name;
}

void gpu_command::apply_config(const uint64_t& fpga_seq) {
    (void)fpga_seq;
    _num_adjusted_elements = config.get_int(unique_name, "num_adjusted_elements");
    _num_elements = config.get_int(unique_name, "num_elements");
    _num_local_freq = config.get_int(unique_name, "num_local_freq");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");
    _num_data_sets = config.get_int(unique_name, "num_data_sets");
    _num_adjusted_local_freq = config.get_int(unique_name, "num_adjusted_local_freq");
    _block_size = config.get_int(unique_name, "block_size");
    _num_blocks = config.get_int(unique_name, "num_blocks");
    _buffer_depth = config.get_int(unique_name, "buffer_depth");
}

void gpu_command::build(class device_interface &param_Device)
{
    size_t program_size;
    FILE *fp;
    char *program_buffer;
    cl_int err;

    postEvent = (cl_event*)malloc(param_Device.getInBuf()->num_frames * sizeof(cl_event));
    CHECK_MEM(postEvent);
    for (int j=0;j<param_Device.getInBuf()->num_frames;++j){
        postEvent[j] = NULL;
    }

    if (gpuCommandState==1){
        fp = fopen(gpuKernel, "r");
        if (fp == NULL){
            ERROR("error loading file: %s", gpuKernel);
            exit(errno);
        }
        fseek(fp, 0, SEEK_END);
        program_size = ftell(fp);
        rewind(fp);

        program_buffer = (char*)malloc(program_size+1);
        program_buffer[program_size] = '\0';
        int sizeRead = fread(program_buffer, sizeof(char), program_size, fp);
        if (sizeRead < program_size)
            ERROR("Error reading the file: %s", gpuKernel);
        fclose(fp);
        program = clCreateProgramWithSource(param_Device.getContext(),
                                        (cl_uint)1,
                                        (const char**)&program_buffer,
                                        &program_size, &err );
        CHECK_CL_ERROR (err);

        program_size = 0;
        free(program_buffer);
    }
}

cl_event gpu_command::execute(int param_bufferID, const uint64_t& fpga_seq, device_interface& param_Device, cl_event param_PrecedeEvent)
{
    assert(param_bufferID<param_Device.getInBuf()->num_frames);
    assert(param_bufferID>=0);

    return NULL;
//    DEBUG("Execute kernel: %s", name);
}

void gpu_command::setKernelArg(cl_uint param_ArgPos, cl_mem param_Buffer)
{
    CHECK_CL_ERROR( clSetKernelArg(kernel,
    param_ArgPos,
    sizeof(void*),
    (void*) &param_Buffer) );
}

// TODO This could be on a per command object basis,
// it doesn't really need to be at this level.
string gpu_command::get_cl_options()
{
    string cl_options = "";

    cl_options += "-D ACTUAL_NUM_ELEMENTS=" + to_string(_num_elements);
    cl_options += " -D ACTUAL_NUM_FREQUENCIES=" + to_string(_num_local_freq);
    cl_options += " -D NUM_ELEMENTS=" + to_string(_num_adjusted_elements);
    cl_options += " -D NUM_FREQUENCIES=" + to_string(_num_adjusted_local_freq);
    cl_options += " -D NUM_BLOCKS=" + to_string(_num_blocks);
    cl_options += " -D NUM_TIMESAMPLES=" + to_string(_samples_per_data_set);
    cl_options += " -D NUM_BUFFERS=" + to_string(_buffer_depth);

    DEBUG("kernel: %s cl_options: %s", name, cl_options.c_str());

    return cl_options;
}

void gpu_command::cleanMe(int param_BufferID)
{
    //the events need to be defined as arrays per buffer id

    if (postEvent[param_BufferID] != NULL){
        clReleaseEvent(postEvent[param_BufferID]);
        postEvent[param_BufferID] = NULL;
    }
}

void gpu_command::freeMe()
{
    if (gpuCommandState==1){
        CHECK_CL_ERROR( clReleaseKernel(kernel) );
        DEBUG("kernel Freed\n");
        CHECK_CL_ERROR( clReleaseProgram(program) );
        DEBUG("program Freed\n");
    }
    free(postEvent);
    DEBUG("posteEvent Freed\n");
}


