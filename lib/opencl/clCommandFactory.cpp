#include "clCommandFactory.hpp"
#include <stdio.h>
#include <stdlib.h>
#include "errors.h"
#include <errno.h>
#include <iostream>
#include <vector>

#include "json.hpp"

using namespace std;

clCommandFactory::clCommandFactory(device_interface & device_,
                                   Config &config_,
                                   const string& unique_name_):
    num_commands(0),
    current_command_cnt(0),
    config(config_),
    device(device_),
    unique_name(unique_name_)
{
    vector<json> commands = config.get_json_array(unique_name, "commands");

    for (uint32_t i = 0; i < commands.size(); i++){
        auto cmd = create(commands[i]["name"], config, unique_name);
        cmd->build(device);
        list_commands.push_back(cmd);
    }

/*
    num_commands = commands.size();
    use_beamforming = config.get_bool(unique_name, "enable_beamforming");
    //use_incoh_beamforming = false; //config.get_bool(unique_name, "use_incoh_beamforming");

    //list_commands =  new clCommand * [num_commands];

    for (uint32_t i = 0; i < num_commands; i++){

        if (commands[i]["name"] == "beamform_phase_data" && use_beamforming == 1) {
            list_commands.push_back(new beamform_phase_data("beamform_phase_data", config, unique_name));
        //} else if (commands[i]["name"] == "beamform_incoherent_kernel" && use_incoh_beamforming == 1) {
            //list_commands.push_back(new beamform_incoherent_kernel(commands[i]["kernel"].get<string>().c_str(), "beamform_incoherent_kernel", config, unique_name));
        } else if (commands[i]["name"] == "beamform_kernel" && use_beamforming == 1) {
            list_commands.push_back(new beamform_kernel(commands[i]["kernel"].get<string>().c_str(), "beamform_kernel", config, unique_name));
        } else if (commands[i]["name"] == "offset_accumulator") {
            list_commands.push_back(new offset_kernel(commands[i]["kernel"].get<string>().c_str(), "offset_accumulator", config, unique_name));
        } else if (commands[i]["name"] == "preseed_multifreq") {
            list_commands.push_back(new preseed_kernel(commands[i]["kernel"].get<string>().c_str(), "preseed_multifreq", config, unique_name));
        } else if (commands[i]["name"] == "pairwise_correlator") {
            list_commands.push_back(new correlator_kernel(commands[i]["kernel"].get<string>().c_str(), "pairwise_correlator", config, unique_name));
        } else if (commands[i]["name"] == "input_data_stage") {
            list_commands.push_back(new input_data_stage("input_data_stage", config, unique_name));
        //} else if (commands[i]["name"] == "output_beamform_incoh_result" && use_incoh_beamforming == 1) {
            //list_commands.push_back(new output_beamform_incoh_result("output_beamform_incoh_result", config, unique_name));
        } else if (commands[i]["name"] == "output_beamform_result" && use_beamforming == 1) {
            list_commands.push_back(new output_beamform_result("output_beamform_result", config, unique_name));
        } else if (commands[i]["name"] == "output_data_result") {
            list_commands.push_back(new output_data_result("output_data_result", config, unique_name));
        } else if (commands[i]["name"] == "rfi_kernel") {
            list_commands.push_back(new rfi_kernel(commands[i]["kernel"].get<string>().c_str(), "rfi_kernel", config, unique_name));
        } else if (commands[i]["name"] == "output_rfi") {
            list_commands.push_back(new output_rfi("output_rfi", config, unique_name));
        }

        // TODO This should just be part of the constructor.
        list_commands[i]->build(device);
    }

    current_command_cnt = 0;
*/
}

clCommand* clCommandFactory::create(const string &name,
                                      Config& config,
                                      const string &unique_name) const
{
    auto known_commands = clCommandFactoryRegistry::get_registered_commands();
    auto i = known_commands.find(name);
    if (i == known_commands.end())
    {
        ERROR("Unrecognized CL command! (%s)", name.c_str());
        throw std::runtime_error("Unrecognized hsaCommand!");
    }
    clCommandMaker* maker = i->second;
    return maker->create(config,unique_name);
}

/*
cl_uint clCommandFactory::getNumCommands() const
{
    return num_commands;
}
clCommand* clCommandFactory::getNextCommand()//class device_interface & param_Device, int param_BufferID)
{
    clCommand* current_command;

    current_command = list_commands[current_command_cnt];

    current_command_cnt++;
      if (current_command_cnt >= num_commands)
	current_command_cnt = 0;

    return current_command;

}
*/
clCommandFactory::~clCommandFactory()
{
    // TODO freeMe should just be a part of the destructor
    for (uint32_t i = 0; i < num_commands; i++){
        list_commands[i]->freeMe();
    }
    DEBUG("CommandsFreed\n");

    //delete[] list_commands;
    for (auto command : list_commands) {
        delete command;
    }
    DEBUG("ListCommandsDeleted\n");
}

vector<clCommand*>& clCommandFactory::get_commands() {
    return list_commands;
}

void clCommandFactoryRegistry::cl_register_command(const std::string& key, clCommandMaker* cmd)
{
    clCommandFactoryRegistry::instance().cl_reg(key,cmd);
}

std::map<std::string, clCommandMaker*> clCommandFactoryRegistry::get_registered_commands(){
    return clCommandFactoryRegistry::instance()._cl_commands;
}


clCommandFactoryRegistry& clCommandFactoryRegistry::instance() {
    static clCommandFactoryRegistry factory;
    return factory;
}

clCommandFactoryRegistry::clCommandFactoryRegistry(){}

void clCommandFactoryRegistry::cl_reg(const std::string& key, clCommandMaker* cmd)
{
    if (_cl_commands.find(key) != _cl_commands.end())
    {
        ERROR("Multiple OpenCL Commands registered as '%s'!",key.c_str());
        throw std::runtime_error("A clCommand was registered twice!");
    }
    _cl_commands[key] = cmd;
}
