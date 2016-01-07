/*
 * Copyright (c) 2015 <copyright holder> <email>
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#include "device_interface.h"
#include "gpu_command.h"
#include "callbackdata.h"
#include "math.h"

device_interface::device_interface(struct Buffer * param_In_Buf, struct Buffer * param_Out_Buf, Config * param_Config, int param_GPU_ID)
{   
    cl_int err;
  
    in_buf = param_In_Buf;
    out_buf = param_Out_Buf;
    config = param_Config;
    gpu_id = param_GPU_ID;

    accumulate_len = config->processing.num_adjusted_local_freq * 
    config->processing.num_adjusted_elements * 2 * config->processing.num_data_sets * sizeof(cl_int);
    aligned_accumulate_len = PAGESIZE_MEM * (ceil((double)accumulate_len / (double)PAGESIZE_MEM));
    assert(aligned_accumulate_len >= accumulate_len);
        
    // Get a platform.
    CHECK_CL_ERROR( clGetPlatformIDs( 1, &platform_id, NULL ) );

    // Find a GPU device..
    CHECK_CL_ERROR( clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, MAX_GPUS, device_id, NULL) );

    context = clCreateContext( NULL, 1, &device_id[gpu_id], NULL, NULL, &err);
    CHECK_CL_ERROR(err);

    //// TODO explain these numbers/formulas.
    //num_blocks = (config->processing.num_adjusted_elements / config->gpu.block_size) *
    //(config->processing.num_adjusted_elements / config->gpu.block_size + 1) / 2.;

}

int device_interface::getGpuID() const
{
    return gpu_id;
}

Buffer* device_interface::getInBuf() 
{
  return in_buf;
}

Buffer* device_interface::getOutBuf() 
{
  return out_buf;
}

cl_context device_interface::getContext() 
{
  return context;
}

cl_device_id* device_interface::getDeviceID()
{
  return device_id;
}

//cl_mem device_interface::getIDxMap()
//{
  //return * id_x_map;
//}

//cl_mem device_interface::getIDyMap()
//{
  //return * id_y_map;
//}

cl_int* device_interface::getAccumulateZeros()
{
  return accumulate_zeros;
}

int device_interface::getAlignedAccumulateLen() const
{
  return aligned_accumulate_len;
}

//device_interface::defineOutputDataMap(Config *param_Config)
//{
  

//}

void device_interface::prepareCommandQueue()
{
  cl_int err;

  // Create command queues
  for (int i = 0; i < NUM_QUEUES; ++i) {
    queue[i] = clCreateCommandQueue( context, device_id[gpu_id], CL_QUEUE_PROFILING_ENABLE, &err );
    CHECK_CL_ERROR(err);
  }
}    
    
void device_interface::allocateMemory()
{
  //IN THE FUTURE, ANDRE TALKED ABOUT WANTING MEMORY TO BE DYNAMICALLY ALLOCATE BASED ON KERNEL STATES AND ASK FOR MEMORY BY SIZE AND HAVE THAT MEMORY NAMED BY THE KERNEL. KERNELS WOULD THEN BE PASSED INTO DEVICE_INTERFACE
  //TO BE GIVEN MEMORY AND DEVICE_INTERFACE WOULD LOOP THROUGH THE KERNELS MEMORY STATES TO DETERMINE ALL THE MEMORY THAT KERNEL NEEDS.
  
  
  cl_int err;    
    // TODO create a struct to contain all of these (including events) to make this memory allocation cleaner. 

    // Setup device input buffers
    device_input_buffer = (cl_mem *) malloc(in_buf->num_buffers * sizeof(cl_mem));
    CHECK_MEM(device_input_buffer);
    for (int i = 0; i < in_buf->num_buffers; ++i) {
        device_input_buffer[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, in_buf->aligned_buffer_size, NULL, &err);
        CHECK_CL_ERROR(err);
    }

    // Array used to zero the output memory on the device.
    // TODO should this be in it's own function?
    err = posix_memalign((void **) &accumulate_zeros, PAGESIZE_MEM, aligned_accumulate_len);
    if ( err != 0 ) {
        ERROR("Error creating aligned memory for accumulate zeros");
        exit(err);
    }

    // Ask that all pages be kept in memory
    err = mlock((void *) accumulate_zeros, aligned_accumulate_len);
    if ( err == -1 ) {
        ERROR("Error locking memory - check ulimit -a to check memlock limits");
        exit(errno);
    }
    memset(accumulate_zeros, 0, aligned_accumulate_len );

    // Setup device accumulate buffers.
    device_accumulate_buffer = (cl_mem *) malloc(in_buf->num_buffers * sizeof(cl_mem));
    CHECK_MEM(device_accumulate_buffer);
    for (int i = 0; i < in_buf->num_buffers; ++i) {
        device_accumulate_buffer[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, aligned_accumulate_len, NULL, &err);
        CHECK_CL_ERROR(err);
    }

    // Setup device output buffers.
    device_output_buffer = (cl_mem *) malloc(out_buf->num_buffers * sizeof(cl_mem));
    CHECK_MEM(device_output_buffer);
    for (int i = 0; i < out_buf->num_buffers; ++i) {
        device_output_buffer[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, out_buf->aligned_buffer_size, NULL, &err);
        CHECK_CL_ERROR(err);
    }
}

cl_mem device_interface::getInputBuffer(int param_BufferID) 
{
  return device_input_buffer[param_BufferID];
}

cl_mem device_interface::getOutputBuffer(int param_BufferID) 
{
  return device_output_buffer[param_BufferID];
}

cl_mem device_interface::getAccumulateBuffer(int param_BufferID) 
{
  return device_accumulate_buffer[param_BufferID];
}

void device_interface::deallocateResources()
{

    for (int i = 0; i < NUM_QUEUES; ++i) {
        CHECK_CL_ERROR( clReleaseCommandQueue(queue[i]) );
    }

    for (int i = 0; i < in_buf->num_buffers; ++i) {
        CHECK_CL_ERROR( clReleaseMemObject(device_input_buffer[i]) );
    }
    free(device_input_buffer);

    for (int i = 0; i < in_buf->num_buffers; ++i) {
        CHECK_CL_ERROR( clReleaseMemObject(device_accumulate_buffer[i]) );
    }
    free(device_accumulate_buffer);

    for (int i = 0; i < out_buf->num_buffers; ++i) {
        CHECK_CL_ERROR( clReleaseMemObject(device_output_buffer[i]) );
    }
    free(device_output_buffer);

    CHECK_CL_ERROR( clReleaseContext(context) );
}

cl_command_queue* device_interface::getQueue()
{
  return queue;
}

