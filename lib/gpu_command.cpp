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
#include <string.h>
#include <iostream>

gpu_command::gpu_command()
{
    
}

gpu_command::gpu_command(char * param_gpuKernel)
{ 
    gpuKernel = new char[strlen(param_gpuKernel)+1];
    strcpy(gpuKernel, param_gpuKernel);
    gpuCommandState=1;
}

gpu_command::~gpu_command()
{
    if (gpuCommandState==1)
        free(gpuKernel);
}

void gpu_command::build(Config * param_Config, class device_interface &param_Device)
{
  size_t program_size;
  FILE *fp;
  char *program_buffer;
  cl_int err;
  
    //precedeEvent = (cl_event*)malloc(param_Device.getInBuf()->num_buffers * sizeof(cl_event));
    //CHECK_MEM(precedeEvent);
    
    
    postEvent = (cl_event*)malloc(param_Device.getInBuf()->num_buffers * sizeof(cl_event));
    CHECK_MEM(postEvent);
    for (int j=0;j<param_Device.getInBuf()->num_buffers;++j){
        //precedeEvent[j] = NULL;
        postEvent[j] = NULL;
    }
  
//   FILE * pFile;
//   pFile = fopen("IanTest.txt", "w");
//   
//   if (pFile != NULL){
//       fputs ("I like tofu", pFile);
//       fclose(pFile);
//   }
      
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
	//createThisEvent(param_Device);
	
}
//void gpu_command::createThisEvent(const class device_interface & param_device)
//{
      //postEventArray = malloc(param_device->getInBuf()->num_buffers * sizeof(cl_event));
      //CHECK_MEM(thisPostEvent);
//}

cl_event gpu_command::execute(int param_bufferID, device_interface& param_Device, cl_event param_PrecedeEvent)
{
    assert(param_bufferID<param_Device.getInBuf()->num_buffers);
    assert(param_bufferID>=0);
    
    //precedeEvent[param_bufferID] = param_PrecedeEvent;
}


void gpu_command::setKernelArg(cl_uint param_ArgPos, cl_mem param_Buffer)
{
  CHECK_CL_ERROR( clSetKernelArg(kernel,
      param_ArgPos,
      sizeof(void*),
      (void*) &param_Buffer) );
}

//size_t* gpu_command::getGWS() 
//{
//  return gws;
//}

//size_t* gpu_command::getLWS() 
//{
//  return lws;
//}

// void gpu_command::setPostEvent(int param_BufferID, cl_event param_PostEvent)
// {
//     postEvent[param_BufferID] =param_PostEvent;
// }

//cl_event *gpu_command::getPostEvent()
//{
 // return postEvent;
//}
/*
void gpu_command::setPrecedeEvent(cl_event param_Event)
{
    precedeEvent=param_Event;
}*/

//cl_event *gpu_command::getPreceedEvent()
//{
//  return preceedEvent;
//}

void gpu_command::cleanMe(int param_BufferID)
{
    //the events need to be defined as arrays per buffer id
    
  //assert(postEvent[param_BufferID] != NULL);
  //assert(preceedEvent[param_BufferID] != NULL);
    std::cout << "BufferID = " << param_BufferID << std::endl;
    if (postEvent[param_BufferID] != NULL){
        DEBUG("PostEvent is not null\n");
        std::cout << "PostEvent = " << postEvent[param_BufferID] << std::endl;
        clReleaseEvent(postEvent[param_BufferID]);
        DEBUG("PostEvent released\n");
        postEvent[param_BufferID] = NULL;
        DEBUG("PostEvent set to null\n");
    }
    DEBUG("PostEvent Cleaned\n");
  /*  if (precedeEvent[param_BufferID] != NULL){
        DEBUG("PrecedeEvent is not null\n");
        std::cout << "precedeEvent = " << precedeEvent[param_BufferID] << std::endl;
        CHECK_CL_ERROR( clReleaseEvent(precedeEvent[param_BufferID]));
        DEBUG("PrecedeEvent released\n");
        precedeEvent[param_BufferID] = NULL;
        DEBUG("PrecedeEvent set to null\n");
    }
    DEBUG("PrecedeEvent Cleaned\n");*/

        
  //assert(postEvent[param_BufferID] != NULL);
  //assert(precedeEvent[param_BufferID] != NULL);
  
  //clReleaseEvent(postEvent[param_BufferID]);
  //clReleaseEvent(preceedEvent[param_BufferID]);
  
  
  
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
    //free(precedeEvent);
  
}

