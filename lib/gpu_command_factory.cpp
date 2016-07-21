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

void gpu_command_factory::initializeCommands(class device_interface & param_Device, Config * param_Config)
{
    //PROPER EXECUTION SEQUENCE IS OFFSET, PRESEED, CORRELATE.
  //numCommands = param_Config->gpu.num_kernels + 2;//SHOULD WE FORCE THE TWO ADDITIONAL COMMANDS TO BE EXPLICITLY DEFINED IN THE CONFIG FILE?
  //numCommands = param_Config->gpu.num_kernels + 2;//SHOULD WE FORCE THE TWO ADDITIONAL COMMANDS TO BE EXPLICITLY DEFINED IN THE CONFIG FILE?
    numCommands = 5; //DONE BECUASE CURRENT CONFIG FILE HAS UNDEFINED KERNELS THAT OFFSET THIS COUNT.

  //listCommands =  malloc(numCommands * sizeof(gpu_command));
  listCommands =  new gpu_command * [numCommands];

  char** gpuKernels = param_Config->gpu.kernels;

  for (int i = 0; i < numCommands; i++){
  	//IKT: This needs to be changed later to read the order from the config file
        switch (i)
        {
               case 0:
                   listCommands[i] = new input_data_stage;
                   break;
               case 3:
                   listCommands[i] = new correlator_kernel(gpuKernels[1]);
                   break;
              case 1:
                   listCommands[i] = new offset_kernel(gpuKernels[2]);
                   break;
               case 2:
                   listCommands[i] = new preseed_kernel(gpuKernels[0]);
                   break;
               case 4:
                   listCommands[i] = new output_data_result;
                   break;
        }
        listCommands[i]->build(param_Config, param_Device);
  }

  currentCommandCnt = 0;
}

gpu_command* gpu_command_factory::getNextCommand(class device_interface & param_Device, int param_BufferID)
{
  //LEAVE THIS AS IS FOR NOW, BUT LATER WILL WANT TO DYNAMICALLY REQUEST FOR MEMORY BASED ON KERNEL STATE AND SET PRE AND POST CL_EVENT BASED ON EVENTS RETURNED BY INDIVIDUAL KERNAL OBJECTS.
  //KERNELS WILL TRACK SETTING THEIR OWN PRE AND POST EVENTS, BUT WILL RETURN THOSE EVENTS TO BE PASSED TO THE NEXT KERNEL IN THE SEQUENCE
    gpu_command* currentCommand;

    switch (currentCommandCnt)
    {
           case 0://input_data_stage prep
               currentCommand = listCommands[currentCommandCnt];
               break;
           case 3://THIRD KERNEL BY EVENTS SEQUENCE "corr"
               listCommands[currentCommandCnt]->setKernelArg(0, param_Device.getInputBuffer(param_BufferID));
               listCommands[currentCommandCnt]->setKernelArg(1, param_Device.getOutputBuffer(param_BufferID));
               currentCommand = listCommands[currentCommandCnt];
               break;
           case 1://FIRST KERNEL BY EVENTS SEQUENCE "offsetAccumulateElements"
               listCommands[currentCommandCnt]->setKernelArg(0, param_Device.getInputBuffer(param_BufferID));
               listCommands[currentCommandCnt]->setKernelArg(1, param_Device.getAccumulateBuffer(param_BufferID));
               currentCommand = listCommands[currentCommandCnt];
               break;
           case 2://SECOND KERNEL BY EVENTS SEQUENCE "preseed"
               listCommands[currentCommandCnt]->setKernelArg(0, param_Device.getAccumulateBuffer(param_BufferID));
               listCommands[currentCommandCnt]->setKernelArg(1, param_Device.getOutputBuffer(param_BufferID));
  	////MIGHT BE FUN FOR THESE KERNEL SEQUENCE EVENTS TO CREATE AN ARRAY MAINTAINED BY EITHER COMMANDFACTORY OR DEVICEINTERFACE
  	////THAT CONTAINS OBJECTS OF THAT CONTAIN THE CL_EVENT REFERENCE AND ARE PLACED INTO PRE AND POST ORDER. HAVE TO DISCUSS.
               currentCommand = listCommands[currentCommandCnt];
               break;
           case 4:
  	 //WILL NEED TO SET THE LIST OF COMMAND OBJECTS TO CB_DATA FOR IT TO LOOP THROUGH FINALIZE METHOD OF EACH TO DEALLOCATE EVENTS FOR EACH.
               currentCommand = listCommands[currentCommandCnt];
               break;
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
    DEBUG("CommandsFreed\n");

    delete[] listCommands;
    DEBUG("ListCommandsDeleted\n");
    //free(cb_data);//THIS IS ALSO DONE IN output_data_result. Should it be done twice? WHich object should manage that reference? I THINK THE
    //ANSWER MIGHT BE YES, SINCE THE COMMAND OBJECT GETS THE REFERENCE BY BUFFERID, WHILE COMMAND FACTOR HAS ALL OF THE BUFFERIDs DEFINED FOR CB_DATA.
}
