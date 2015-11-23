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

#include "kerneloffset.h"
#include "device_interface.h"
#include "gpu_command.h"



kernelOffset::kernelOffset(const char param_gpuKernel): gpu_command(param_gpuKernel)
{

}

void kernelOffset::build(const Config& param_Config, const device_interface& param_Device)
{
  gpu_command::build(param_Config, param_Device);

  cl_int err;        
  char cl_options[1024];
  
  sprintf(cl_options, "-D ACTUAL_NUM_ELEMENTS=%du -D ACTUAL_NUM_FREQUENCIES=%du -D NUM_ELEMENTS=%du -D NUM_FREQUENCIES=%du -D NUM_BLOCKS=%du -D NUM_TIMESAMPLES=%du",
	param_Config->processing.num_elements, param_Config->processing.num_local_freq,
	param_Config->processing.num_adjusted_elements,
	param_Config->processing.num_adjusted_local_freq,
	param_Config->processing.num_blocks, param_Config->processing.samples_per_data_set);
	  
  CHECK_CL_ERROR ( clBuildProgram( program, 1, &param_Device->getDeviceID(), cl_options, NULL, NULL ) );
  
  
  kernel = clCreateKernel( program, "offsetAccumulateElements", &err );
  CHECK_CL_ERROR(err);
    
  // Accumulation kernel global and local work space sizes.
  gws[0] = 64*param_Config->processing.num_data_sets;
  gws[1] = (int)ceil(param_Config->processing.num_adjusted_elements *
  param_Config->processing.num_adjusted_local_freq/256.0);
  gws[2] = param_Config->processing.samples_per_data_set/1024;

  lws[0] = 64;
  lws[1] = 1;
  lws[2] = 1;
}

cl_event kernelOffset::execute(int param_bufferID, const device_interface& param_Device)
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
