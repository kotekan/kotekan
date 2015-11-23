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

#include "kernelcorrelator.h"


kernelCorrelator::kernelCorrelator(const char param_gpuKernel):gpu_command(param_gpuKernel)
{
  
}

void kernelCorrelator::build(const Config& param_Config, const device_interface& param_Device)
{
  gpu_command::build(param_Config, param_Device);
    
  cl_int err;
  char cl_options[1024];
  int num_blocks;
  unsigned int num_accumulations;
  
  //Host Buffers
  cl_mem * id_x_map;
  cl_mem * id_y_map;
  
  // Number of compressed accumulations.
  num_accumulations = param_Config->processing.samples_per_data_set/256;
  
  //I GENUINELY DON'T LIKE HOW THIS PORTION OF THE CODE IS DEFINED. WOULD BE NICE TO ENCAPSULATE THE USE OF NUM_BLOCKS AND THE CALL TO DEFINEOUTPUTDATAMAP SOMEWHERE ELSE. 
  // TODO explain these numbers/formulas.
  num_blocks = (param_Config->processing.num_adjusted_elements / param_Config->gpu.block_size) *
  (param_Config->processing.num_adjusted_elements / param_Config->gpu.block_size + 1) / 2.;
  
    sprintf(cl_options, "-D ACTUAL_NUM_ELEMENTS=%du -D ACTUAL_NUM_FREQUENCIES=%du -D NUM_ELEMENTS=%du -D NUM_FREQUENCIES=%du -D NUM_BLOCKS=%du -D NUM_TIMESAMPLES=%du",
	  param_Config->processing.num_elements, param_Config->processing.num_local_freq,
	  param_Config->processing.num_adjusted_elements,
	  param_Config->processing.num_adjusted_local_freq,
	  param_Config->processing.num_blocks, param_Config->processing.samples_per_data_set);
	  
  CHECK_CL_ERROR ( clBuildProgram( program, 1, &param_Device->getDeviceID(), cl_options, NULL, NULL ) );
  
  
  kernel = clCreateKernel( program, "corr", &err );
  CHECK_CL_ERROR(err);
  call defineOutputDataMap(&param_Config, num_blocks, &param_Device, &id_x_map, &id_y_map);
  
      //set other parameters that will be fixed for the kernels (changeable parameters will be set in run loops)
    CHECK_CL_ERROR( clSetKernelArg(kernel, 
                                   2,
                                   sizeof(id_x_map),
                                   (void*) &id_x_map) ); //this should maybe be sizeof(void *)?

    CHECK_CL_ERROR( clSetKernelArg(kernel, 
                                   3,
                                   sizeof(id_y_map),
                                   (void*) &id_y_map) );
    CHECK_CL_ERROR( clSetKernelArg(kernel,
                                   4,
                                   8*8*4 * sizeof(cl_uint),
                                   NULL) );
    
    
    // Correlation kernel global and local work space sizes.
    gws[0] = 8*param_Config->processing.num_data_sets;
    gws[1] = 8*param_Config->processing.num_adjusted_local_freq;
    gws[2] = num_blocks*num_accumulations;

    lws[0] = 8;
    lws[1] = 8;
    lws[2] = 1; 
}

cl_event kernelCorrelator::execute(int param_bufferID, const device_interface& param_Device)
{
  cl_event *postEvent;
    
  postEvent = thisPostEvent[param_bufferID];
  
  
  CHECK_CL_ERROR( clEnqueueNDRangeKernel(param_Device.getQueue()[1],
                                            kernel,
                                            3,
                                            NULL,
                                            gws,
                                            lws,
                                            1,
                                            &preceedEvent,
                                            &postEvent));  
  
  return postEvent;
}
void kernelCorrelator::defineOutputDataMap(const Config & param_Config, int param_num_blocks, const device_interface & param_Device, cl_mem & id_x_map, cl_mem & id_y_map)
{
  cl_int err;    
    // Create lookup tables 

    //upper triangular address mapping --converting 1d addresses to 2d addresses
    unsigned int global_id_x_map[param_num_blocks];
    unsigned int global_id_y_map[param_num_blocks];

    //TODO: p260 OpenCL in Action has a clever while loop that changes 1 D addresses to X & Y indices for an upper triangle.  
    // Time Test kernels using them compared to the lookup tables for NUM_ELEM = 256
    int largest_num_blocks_1D = param_Config->processing.num_adjusted_elements /param_Config->gpu.block_size;
    int index_1D = 0;
    for (int j = 0; j < largest_num_blocks_1D; j++){
        for (int i = j; i < largest_num_blocks_1D; i++){
            global_id_x_map[index_1D] = i;
            global_id_y_map[index_1D] = j;
            index_1D++;
        }
    }

    id_x_map = clCreateBuffer(param_Device->getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    param_num_blocks * sizeof(cl_uint), global_id_x_map, &err);
    if (err){
        printf("Error in clCreateBuffer %i\n", err);
    }

    id_y_map = clCreateBuffer(param_Device->getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    param_num_blocks * sizeof(cl_uint), global_id_y_map, &err);
    if (err){
        printf("Error in clCreateBuffer %i\n", err);
    } 
}

