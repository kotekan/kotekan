#include "hsaCommandFactory.hpp"
#include <stdio.h>
#include <stdlib.h>
#include "errors.h"
#include <errno.h>
#include <iostream>
#include <vector>

#include "json.hpp"


hsaCommandFactory::hsaCommandFactory(Config& config_,
                                     const string &unique_name_,
                                     bufferContainer &host_buffers_,
                                     hsaDeviceInterface& device_) :
                                        config(config_),
                                        device(device_),
                                        host_buffers(host_buffers_),
                                        unique_name(unique_name_){
    //Dummy constructor to deal with singleton instance for initialization
    if (unique_name == "") return;
    vector<json> commands = config.get_json_array(unique_name, "commands");

    auto fac = hsaCommandFactory::Instance();
    for (uint32_t i = 0; i < commands.size(); i++){
        auto cmd = fac.create(commands[i]["name"], config, unique_name, host_buffers, device);
        list_commands.push_back(cmd);
    }
}

hsaCommandFactory::~hsaCommandFactory() {
    for (auto command : list_commands) {
        delete command;
    }
}

vector<hsaCommand*>& hsaCommandFactory::get_commands() {
    return list_commands;
}

hsaCommandFactory& hsaCommandFactory::Instance() {
    static hsaCommandFactory factory(*new Config(),"",*new bufferContainer(),
                                     *new hsaDeviceInterface(*new Config(), -1, -1));
    return factory;
}

hsaCommand* hsaCommandFactory::create(const string &name,
                                      Config& config,
                                      const string &unique_name,
                                      bufferContainer &host_buffers,
                                      hsaDeviceInterface& device) const
{
    auto i = _hsa_commands.find(name);
    if (i == _hsa_commands.end())
    {
        ERROR("Unrecognized HSA command! (%s)", name.c_str());
    }
    hsaCommandMaker* maker = i->second;
    return maker->create(config,unique_name,host_buffers,device);
}

void hsaCommandFactory::hsaRegisterCommand(const std::string& key, hsaCommandMaker* cmd)
{
    if (_hsa_commands.find(key) != _hsa_commands.end())
    {
        ERROR("Multiple HSA Commands registered as '%s'!",key.c_str());
    }
    _hsa_commands[key] = cmd;
}

