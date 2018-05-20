#include "clCommandFactory.hpp"
#include <stdio.h>
#include <stdlib.h>
#include "errors.h"
#include <errno.h>
#include <iostream>
#include <vector>

#include "json.hpp"

using namespace std;

clCommandFactory::clCommandFactory(Config& config_,
                                     const string &unique_name_,
                                     bufferContainer &host_buffers_,
                                     clDeviceInterface& device_) :
                            config(config_),
                            device(device_),
                            host_buffers(host_buffers_),
                            unique_name(unique_name_)
{
    auto known_commands = clCommandFactoryRegistry::get_registered_commands();
    for (auto &command : known_commands){
        INFO("Registered OpenCL Command: %s",command.first.c_str());
    }

    vector<json> commands = config.get_json_array(unique_name, "commands");

    for (uint32_t i = 0; i < commands.size(); i++){
        auto cmd = create(commands[i]["name"], config, unique_name, host_buffers, device);
        cmd->build();
        list_commands.push_back(cmd);
    }
}

clCommand* clCommandFactory::create(const string &name,
                                    Config& config,
                                    const string &unique_name,
                                    bufferContainer &host_buffers,
                                    clDeviceInterface& device) const
{
    auto known_commands = clCommandFactoryRegistry::get_registered_commands();
    auto i = known_commands.find(name);
    if (i == known_commands.end())
    {
        ERROR("Unrecognized CL command! (%s)", name.c_str());
        throw std::runtime_error("Unrecognized hsaCommand!");
    }
    clCommandMaker* maker = i->second;
    return maker->create(config,unique_name, host_buffers, device);
}

clCommandFactory::~clCommandFactory()
{
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
