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

#include "gpu_command.h"

gpu_command::gpu_command(const char param_gpuKernel)
{ 
  gpuKernel = param_gpuKernel;
}

void gpu_command::build(const Config &param_Config, const device_interface &param_Device)
{
    size_t cl_program_size;
  FILE *fp;
  char *cl_program_buffer;
  cl_int err;
  
    fp = fopen(gpuKernel, "r");
        if (fp == NULL){
            ERROR("error loading file: %s", gpuKernel);
            exit(errno);
        }
        fseek(fp, 0, SEEK_END);
        cl_program_size = ftell(fp);
        rewind(fp);
	
	cl_program_buffer = (char*)malloc(cl_program_size+1);
        cl_program_buffer[cl_program_size] = '\0';
        int sizeRead = fread(cl_program_buffer, sizeof(char), cl_program_size, fp);
        if (sizeRead < cl_program_size)
            ERROR("Error reading the file: %s", gpuKernel);
        fclose(fp);
	program = clCreateProgramWithSource( param_Device->getContext(),
					  1,
					  (const char**)cl_program_buffer,
					  cl_program_size, &err );
	CHECK_CL_ERROR (err);
	
	cl_program_size = 0;
	free(cl_program_buffer);
	
	createThisEvent(param_Device);
	
}
void gpu_command::createThisEvent(const device_interface & param_device)
{
      thisPostEvent = malloc(param_device->getInBuf()->num_buffers * sizeof(cl_event));
      CHECK_MEM(thisPostEvent);
}

gpu_command::setKernelArgArg(int param_ArgPos, cl_mem& param_Buffer)
{
  CHECK_CL_ERROR( clSetKernelArg(kernel,
      param_ArgPos,
      sizeof(void*),
      (void*) &param_Buffer) );
}

size_t gpu_command::getGWS() const
{
  return gws;
}

size_t gpu_command::getLWS() const
{
  return lws;
}

gpu_command::setPostEvent(int param_BufferID)
{
  postEvent = thisPostEvent[param_BufferID];
}

cl_event gpu_command::getPostEvent() const
{
  return postEvent;
}

gpu_command::setPreceedEvent(cl_event param_Event)
{
  preceedEvent = param_Event;
}

cl_event gpu_command::getPreceedEvent() const
{
  return preceedEvent;
}

void gpu_command::cleanMe(int param_BufferID)
{
  assert(postEvent[param_BufferID] != NULL);
  assert(preceedEvent[param_BufferID] != NULL);
  
  clReleaseEvent(postEvent[param_BufferID]);
  clReleaseEvent(preceedEvent[param_BufferID]);
}

void gpu_command::freeMe()
{
  CHECK_CL_ERROR( clReleaseKernel(kernel) );
  free(postEvent);
  free(preceedEvent);
  free(program);
}

