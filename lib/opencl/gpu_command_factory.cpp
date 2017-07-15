#include "gpu_command_factory.h"
#include <stdio.h>
#include <stdlib.h>
#include "errors.h"
#include <errno.h>
#include <iostream>
#include <vector>

#include "json.hpp"

using namespace std;

gpu_command_factory::gpu_command_factory(class device_interface & device_
                                            , Config &config_
                                            , const string& unique_name_):
    num_commands(0),
    current_command_cnt(0),
    config(config_),
    device(device_),
    unique_name(unique_name_)
{
    vector<json> commands = config.get_json_array(unique_name, "commands");
    num_commands = commands.size();
    use_beamforming = config.get_bool(unique_name, "use_beamforming");
    use_incoh_beamforming = config.get_bool(unique_name, "use_incoh_beamforming");

    list_commands =  new gpu_command * [num_commands];

    for (uint32_t i = 0; i < num_commands; i++){

        if (commands[i]["name"] == "beamform_phase_data" && use_beamforming == 1) {
            list_commands[i] = new beamform_phase_data("beamform_phase_data", config);
        } else if (commands[i]["name"] == "beamform_incoherent_kernel" && use_incoh_beamforming == 1) {
            list_commands[i] = new beamform_incoherent_kernel(commands[i]["kernel"].get<string>().c_str(), "beamform_incoherent_kernel", config);
        } else if (commands[i]["name"] == "beamform_kernel" && use_beamforming == 1) {
            list_commands[i] = new beamform_kernel(commands[i]["kernel"].get<string>().c_str(), "beamform_kernel", config);
        } else if (commands[i]["name"] == "offset_accumulator") {
            list_commands[i] = new offset_kernel(commands[i]["kernel"].get<string>().c_str(), "offset_accumulator", config);
        } else if (commands[i]["name"] == "preseed_multifreq") {
            list_commands[i] = new preseed_kernel(commands[i]["kernel"].get<string>().c_str(), "preseed_multifreq", config);    
        } else if (commands[i]["name"] == "pairwise_correlator") {
            list_commands[i] = new correlator_kernel(commands[i]["kernel"].get<string>().c_str(), "pairwise_correlator", config);
        } else if (commands[i]["name"] == "input_data_stage") {
            list_commands[i] = new input_data_stage("input_data_stage", config);
        } else if (commands[i]["name"] == "output_beamform_incoh_result" && use_incoh_beamforming == 1) {
            list_commands[i] = new output_beamform_incoh_result("output_beamform_incoh_result", config);
        } else if (commands[i]["name"] == "output_beamform_result" && use_beamforming == 1) {
            list_commands[i] = new output_beamform_result("output_beamform_result", config);
        } else if (commands[i]["name"] == "output_data_result") {
            list_commands[i] = new output_data_result("output_data_result", config);
        }

        // TODO This should just be part of the constructor.
        list_commands[i]->build(device);
    }

    current_command_cnt = 0;
}

cl_uint gpu_command_factory::getNumCommands() const
{
    return num_commands;
}

/*
void gpu_command_factory::initializeCommands(class device_interface & param_Device, Config &config)
{
    vector<json> commands = config.get_json_array("/gpu", "commands");
    num_commands = commands.size();
    use_beamforming = config.get_bool("/gpu", "enable_beamforming");

    list_commands =  new gpu_command * [num_commands];

    for (uint32_t i = 0; i < num_commands; i++){

        if (commands[i]["name"] == "beamform_phase_data") {
            list_commands[i] = new beamform_phase_data("beamform_phase_data", config);
        } else if (commands[i]["name"] == "beamform_incoherent_kernel") {
            list_commands[i] = new beamform_incoherent_kernel(commands[i]["kernel"].get<string>().c_str(), "beamform_incoherent_kernel", config);
        } else if (commands[i]["name"] == "beamform_kernel") {
            list_commands[i] = new beamform_kernel(commands[i]["kernel"].get<string>().c_str(), "beamform_kernel", config);
        } else if (commands[i]["name"] == "correlator_kernel") {
            list_commands[i] = new correlator_kernel(commands[i]["kernel"].get<string>().c_str(), "correlator_kernel", config);
        } else if (commands[i]["name"] == "input_data_stage") {
            list_commands[i] = new input_data_stage("input_data_stage", config);
        } else if (commands[i]["name"] == "offset_kernel") {
            list_commands[i] = new offset_kernel(commands[i]["kernel"].get<string>().c_str(), "offset_kernel", config);
        } else if (commands[i]["name"] == "output_beamform_incoh_result") {
            list_commands[i] = new output_beamform_incoh_result("output_beamform_incoh_result", config);
        } else if (commands[i]["name"] == "output_beamform_result") {
            list_commands[i] = new output_beamform_result("output_beamform_result", config);
        } else if (commands[i]["name"] == "output_data_result") {
            list_commands[i] = new output_data_result("output_data_result", config);
        } else if (commands[i]["name"] == "preseed_kernel") {
            list_commands[i] = new preseed_kernel(commands[i]["kernel"].get<string>().c_str(), "preseed_kernel", config);
        }

        // TODO This should just be part of the constructor.
        list_commands[i]->build(param_Device);
    }

    current_command_cnt = 0;
}
*/
gpu_command* gpu_command_factory::getNextCommand(int bufferID)//class device_interface & param_Device, int param_BufferID)
{
      //LEAVE THIS AS IS FOR NOW, BUT LATER WILL WANT TO DYNAMICALLY REQUEST FOR MEMORY BASED ON KERNEL STATE AND SET PRE AND POST CL_EVENT BASED ON EVENTS RETURNED BY INDIVIDUAL KERNAL OBJECTS.
  //KERNELS WILL TRACK SETTING THEIR OWN PRE AND POST EVENTS, BUT WILL RETURN THOSE EVENTS TO BE PASSED TO THE NEXT KERNEL IN THE SEQUENCE

//  TO ADDRESS THE ISSUE OF COMMAND_OBJECTS NEEDING TO PASS BUFFERS TO EACH OTHER (IE BEAMFORM PHASE AND BEAMFORM KERNEL)
//  IT MAY BE A GOOD IDEA TO INTRODUCE AN OBJECT THAT LIVES IN COMMAND_FACTOR CALLED RESOURCE_ALLOCATION. IT WILL SERVE
//  AS AN INTERFACE BETWEEN DEVICE_INTERFACE - RESPONSIBLE FOR ALLOCATING MEMORY, AND COMMAND_OBJECTS - RESPONSIBLE FOR MANAGING
//  MEMORY OBJECTS. MEMORY WILL BE ALLOCATED BY DEVICE_INTERFACE FOR A BUFFER THAT IS STORED AND MAINTAINED BY COMMAND_OBJECT,
//  BUT THE RESOURCE_ALLOCATION OBJECT WILL RECEIVE A REFERENCE TO THESE MEMORY OBJECTS AND WILL BE RESPONSIBLE FOR DISTRIBUTING
//  THOSE MEMORY BUFFERS TO THE DIFFERENT COMMAND_OBJECTS TAHT NEED THEM.
    gpu_command* currentCommand;

    currentCommand = list_commands[current_command_cnt];

    if (currentCommand->get_name() == "input_data_stage"){}//input_data_stage prep
    else if (currentCommand->get_name() == "beamform_phase_data"){}
    else if (currentCommand->get_name() ==  "pairwise_correlator")//THIRD KERNEL BY EVENTS SEQUENCE "corr"
    {
        currentCommand->setKernelArg(0, device.getInputBuffer(bufferID));
        currentCommand->setKernelArg(1, device.getOutputBuffer(bufferID));
    }
    else if (currentCommand->get_name() == "offset_accumulator")//FIRST KERNEL BY EVENTS SEQUENCE "offsetAccumulateElements"
    {
        currentCommand->setKernelArg(0, device.getInputBuffer(bufferID));
        currentCommand->setKernelArg(1, device.getAccumulateBuffer(bufferID));
    }
    else if (currentCommand->get_name() == "preseed_multifreq")//SECOND KERNEL BY EVENTS SEQUENCE "preseed"
    {
        currentCommand->setKernelArg(0, device.getAccumulateBuffer(bufferID));
        currentCommand->setKernelArg(1, device.getOutputBuffer(bufferID));
    }
    else if (currentCommand->get_name() == "beamform_tree_scale")
    {    
    }
    else if (currentCommand->get_name() == "beamform_incoherent")
    {    
    }
    else if (currentCommand->get_name() == "output_data_result"){}
    else if (currentCommand->get_name() == "output_beamform_result"){}
    else if (currentCommand->get_name() == "output_beamform_incoh_result"){}

    current_command_cnt++;
      if (current_command_cnt >= num_commands)
	current_command_cnt = 0;

  return currentCommand;

}
void gpu_command_factory::deallocateResources()
{
    // TODO freeMe should just be a part of the destructor
    for (uint32_t i = 0; i < num_commands; i++){
        list_commands[i]->freeMe();
    }
    DEBUG("CommandsFreed\n");

    delete[] list_commands;
    DEBUG("ListCommandsDeleted\n");
}

