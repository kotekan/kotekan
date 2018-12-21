#include "processFactory.hpp"

#include "errors.h"

processFactory::processFactory(Config& config, bufferContainer& buffer_container) :
    config(config),
    buffer_container(buffer_container) {

#ifdef DEBUGGING
    auto known_processes = processFactoryRegistry::get_registered_processes();
    for (auto& process : known_processes) {
        DEBUG("Registered Kotekan Process: %s", process.first.c_str());
    }
#endif
}

processFactory::~processFactory() {}

map<string, KotekanProcess*> processFactory::build_processes() {
    map<string, KotekanProcess*> processes;

    // Start parsing tree, put the processes in the "processes" vector
    build_from_tree(processes, config.get_full_config_json(), "");

    return processes;
}

void processFactory::build_from_tree(map<string, KotekanProcess*>& processes, json& config_tree,
                                     const string& path) {

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
                throw std::runtime_error("A process with the path " + unique_name
                                         + " has been defined more than once!");
            }
            processes[unique_name] =
                create(process_name, config, path + "/" + it.key(), buffer_container);
            continue;
        }

        // Recursive part.
        // This is a section/scope not a process block.
        build_from_tree(processes, it.value(), path + "/" + it.key());
    }
}

KotekanProcess* processFactory::create(const string& name, Config& config,
                                       const string& unique_name,
                                       bufferContainer& host_buffers) const {
    auto known_processes = processFactoryRegistry::get_registered_processes();
    auto i = known_processes.find(name);
    if (i == known_processes.end()) {
        ERROR("Unrecognized KotekanProcess! (%s)", name.c_str());
        throw std::runtime_error("Tried to instantiate a process we don't know about!");
    }
    kotekanProcessMaker* maker = i->second;
    return maker->create(config, unique_name, host_buffers);
}


void processFactoryRegistry::kotekan_register_process(const std::string& key,
                                                      kotekanProcessMaker* proc) {
    processFactoryRegistry::instance().kotekan_reg(key, proc);
}

std::map<std::string, kotekanProcessMaker*> processFactoryRegistry::get_registered_processes() {
    return processFactoryRegistry::instance()._kotekan_processes;
}


processFactoryRegistry& processFactoryRegistry::instance() {
    static processFactoryRegistry factory;
    return factory;
}

processFactoryRegistry::processFactoryRegistry() {}

void processFactoryRegistry::kotekan_reg(const std::string& key, kotekanProcessMaker* proc) {
    if (_kotekan_processes.find(key) != _kotekan_processes.end()) {
        ERROR("Multiple KotekanProcess-es registered as '%s'!", key.c_str());
        throw std::runtime_error("A kotekanProcess was registered twice!");
    }
    _kotekan_processes[key] = proc;
}
