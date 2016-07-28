#include "gpu_command.h"
#include <string.h>
#include <iostream>

gpu_command::gpu_command(char* param_name)
{
    name = param_name;
}

gpu_command::gpu_command(char * param_gpuKernel, char* param_name)
{
    gpuKernel = new char[strlen(param_gpuKernel)+1];
    strcpy(gpuKernel, param_gpuKernel);
    gpuCommandState=1;
    
    name = param_name;
}

gpu_command::~gpu_command()
{
    if (gpuCommandState==1)
        free(gpuKernel);
}

char* gpu_command::get_name()
{
    return name;
}

void gpu_command::build(Config * param_Config, class device_interface &param_Device)
{
    size_t program_size;
    FILE *fp;
    char *program_buffer;
    cl_int err;

    postEvent = (cl_event*)malloc(param_Device.getInBuf()->num_buffers * sizeof(cl_event));
    CHECK_MEM(postEvent);
    for (int j=0;j<param_Device.getInBuf()->num_buffers;++j){
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

cl_event gpu_command::execute(int param_bufferID, device_interface& param_Device, cl_event param_PrecedeEvent)
{
    assert(param_bufferID<param_Device.getInBuf()->num_buffers);
    assert(param_bufferID>=0);
}

void gpu_command::setKernelArg(cl_uint param_ArgPos, cl_mem param_Buffer)
{
    CHECK_CL_ERROR( clSetKernelArg(kernel,
    param_ArgPos,
    sizeof(void*),
    (void*) &param_Buffer) );
}

char* gpu_command::get_cl_options(Config * param_Config)
{
    char cl_options[1024];
    
    sprintf(cl_options, "-D ACTUAL_NUM_ELEMENTS=%du -D ACTUAL_NUM_FREQUENCIES=%du -D NUM_ELEMENTS=%du -D NUM_FREQUENCIES=%du -D NUM_BLOCKS=%du -D NUM_TIMESAMPLES=%du -D NUM_BUFFERS=%du",
        param_Config->processing.num_elements, param_Config->processing.num_local_freq,
        param_Config->processing.num_adjusted_elements,
        param_Config->processing.num_adjusted_local_freq,
        param_Config->processing.num_blocks,
        param_Config->processing.samples_per_data_set,
        param_Config->processing.buffer_depth);
    
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


