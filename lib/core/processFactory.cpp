#include "processFactory.hpp"
#include "errors.h"

processFactory::processFactory(Config& config,
                               bufferContainer& buffer_container) :
    config(config),
    buffer_container(buffer_container) {

}

processFactory::~processFactory() {
}

map<string, KotekanProcess *> processFactory::build_processes() {
    map<string, KotekanProcess *> processes;

    // Start parsing tree, put the processes in the "processes" vector
    build_from_tree(processes, config.get_full_config_json(), "");

    return processes;
}

void processFactory::build_from_tree(map<string, KotekanProcess *>& processes, json& config_tree, const string& path) {

    auto fac = processFactory::Instance();
    for (json::iterator it = config_tree.begin(); it != config_tree.end(); ++it) {
        // If the item isn't an object we can just ignore it.
        if (!it.value().is_object()) {
            continue;
        }

        // Check if this is a kotekan_process block, and if so create the process.
        string process_name = it.value().value("kotekan_process", "none");
        if (process_name != "none") {
            string unique_name = path + "/" + it.key();
            if (processes.count(unique_name) != 0) {
                throw std::runtime_error("A process with the path " + unique_name + " has been defined more than once!");
            }
//            processes[unique_name] = new_process(process_name, path + "/" + it.key());
            processes[unique_name] = fac.create(process_name, config, path + "/" + it.key(), buffer_container);
            continue;
        }

        // Recursive part.
        // This is a section/scope not a process block.
        build_from_tree(processes, it.value(), path + "/" + it.key());
    }
}


processFactory& processFactory::Instance() {
    static processFactory factory(*new Config(),*new bufferContainer());
    return factory;
}

KotekanProcess* processFactory::create(const string &name,
                                      Config& config,
                                      const string &unique_name,
                                      bufferContainer &host_buffers) const
{
    auto i = _kotekan_processes.find(name);
    if (i == _kotekan_processes.end())
    {
        ERROR("Unrecognized KotekanProcess! (%s)", name.c_str());
    }
    kotekanProcessMaker* maker = i->second;
    return maker->create(config,unique_name,host_buffers);
}

void processFactory::kotekanRegisterProcess(const std::string& key, kotekanProcessMaker* cmd)
{
    if (_kotekan_processes.find(key) != _kotekan_processes.end())
    {
        ERROR("Multiple KotekanProcess-es registered as '%s'!",key.c_str());
    }
    _kotekan_processes[key] = cmd;
}

