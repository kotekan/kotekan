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

#include "gpu_command_factory.h"
#include <stdio.h>
#include <stdlib.h>
#include "errors.h"
#include <errno.h>


gpu_command_factory::gpu_command_factory()
{
   currentCommandCnt = 0;
}

cl_uint gpu_command_factory::getNumCommands() const
{
  return numCommands;
}

void gpu_command_factory::initializeCommands(const device_interface & param_Device, const Config &param_Config)
{
  numCommands = param_Config->gpu.num_kernels + 2;//SHOULD WE FORCE THE TWO ADDITIONAL COMMANDS TO BE EXPLICITLY DEFINED IN THE CONFIG FILE?
  gpu_command list[numCommands];//UNSURE IF THIS IS LEGAL. CHECK IT LATER.
  listCommands = list;
  const char gpuKernels[param_Config->gpu.num_kernels] = param_Config->gpu.kernels;
  
  for (int i = 0; i < numCommands; i++){
  	//IKT: This needs to be changed later to read the order from the config file
	switch (i)
	{
	  case 0:
	    listCommands[i] = initQueueSequence_command();
	  case 1:
	    listCommands[i] = kernelCorrelator(gpuKernels[0]);
	  case 2:
	    listCommands[i] = kernelOffset(gpuKernels[1]);
	  case 3:
	    listCommands[i] = kernelPreseed(gpuKernels[2]);
	  case 4:
	    listCommands[i] = finalQueueSequence_Command();
	}
	listCommands[i].build(&param_Config, &param_Device);
  }  
  
  //cb_data = malloc(param_Device.getInBuf()->num_buffers * sizeof(struct callBackData));
  //CHECK_MEM(cb_data);
  
  currentCommandCnt = 0;
}

gpu_command gpu_command_factory::getNextCommand(const device_interface & param_Device, int param_BufferID, const cl_event & param_PreceedEvent)
{
  //LEAVE THIS AS IS FOR NOW, BUT LATER WILL WANT TO DYNAMICALLY REQUEST FOR MEMORY BASED ON KERNEL STATE AND SET PRE AND POST CL_EVENT BASED ON EVENTS RETURNED BY INDIVIDUAL KERNAL OBJECTS.
  //KERNELS WILL TRACK SETTING THEIR OWN PRE AND POST EVENTS, BUT WILL RETURN THOSE EVENTS TO BE PASSED TO THE NEXT KERNEL IN THE SEQUENCE
  gpu_command currentCommand;
  
    switch (currentCommandCnt)
    {
      case 0://initQueueSequence_command prep
	listCommands[0].setPreceedEvent(param_PreceedEvent);
	listCommands[0].setPostEvent(param_BufferID);
	       currentCommand = listCommands[0];
      case 1://THIRD KERNEL BY EVENTS SEQUENCE "corr"
	listCommands[1].setKernelArg(0, param_Device.getInputBuffer(param_BufferID));
	listCommands[1].setKernelArg(0, param_Device.getInputBuffer(param_BufferID));
	listCommands[1].setKernelArg(1, param_Device.getOutputBuffer(param_BufferID));
	listCommands[1].setPreceedEvent(param_PreceedEvent);
	listCommands[1].setPostEvent(param_BufferID);
	       currentCommand = listCommands[1];
      case 2://FIRST KERNEL BY EVENTS SEQUENCE "offsetAccumulateElements"
	listCommands[2].setKernelArg(0, param_Device.getInputBuffer(param_BufferID));
	listCommands[2].setKernelArg(1, param_Device.getAccumulateBuffer(param_BufferID));
	listCommands[2].setPreceedEvent(param_PreceedEvent);
	listCommands[2].setPostEvent(param_BufferID);
	       currentCommand = listCommands[2];
      case 3://SECOND KERNEL BY EVENTS SEQUENCE "preseed"
	listCommands[3].setKernelArg(0, param_Device.getAccumulateBuffer(param_BufferID));
	listCommands[3].setKernelArg(1, param_Device.getOutputBuffer(param_BufferID));
	listCommands[3].setPreceedEvent(param_PreceedEvent);
	////MIGHT BE FUN FOR THESE KERNEL SEQUENCE EVENTS TO CREATE AN ARRAY MAINTAINED BY EITHER COMMANDFACTORY OR DEVICEINTERFACE
	////THAT CONTAINS OBJECTS OF THAT CONTAIN THE CL_EVENT REFERENCE AND ARE PLACED INTO PRE AND POST ORDER. HAVE TO DISCUSS.
	listCommands[3].setPostEvent(param_BufferID);
	       currentCommand = listCommands[3];
      case 4:
	 //WILL NEED TO SET THE LIST OF COMMAND OBJECTS TO CB_DATA FOR IT TO LOOP THROUGH FINALIZE METHOD OF EACH TO DEALLOCATE EVENTS FOR EACH.
	listCommands[4].setPreceedEvent(param_PreceedEvent);
	//listCommands[4]->setCBData(&cb_data);
	currentCommand = listCommands[4];
      }
      
      currentCommandCnt++;
      if (currentCommandCnt >= numCommands)
	currentCommandCnt = 0;
  
  return currentCommand;
 
}
void gpu_command_factory::deallocateResources()
{
    for (int i = 0; i < numCommands; i++){
     listCommands[i]->freeMe();
    }
    
    free(cb_data);//THIS IS ALSO DONE IN finalQueueSequence_Command. Should it be done twice? WHich object should manage that reference? I THINK THE 
    //ANSWER MIGHT BE YES, SINCE THE COMMAND OBJECT GETS THE REFERENCE BY BUFFERID, WHILE COMMAND FACTOR HAS ALL OF THE BUFFERIDs DEFINED FOR CB_DATA.
}
