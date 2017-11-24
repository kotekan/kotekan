#ifndef PROCESS_FACTORY_HPP
#define PROCESS_FACTORY_HPP

#include <string>
#include <map>

#include "json.hpp"
#include "KotekanProcess.hpp"

// Name space includes.
using json = nlohmann::json;
using std::string;
using std::map;


class processFactory {

public:

    // One processFactory should be created for each set of config and buffer_container
    processFactory(Config& config, bufferContainer &buffer_container);
    ~processFactory();

    // Creates all the processes listed in the config file, and returns them
    // as a vector of KotekanProcess pointers.
    // This should only be called once.
    map<string, KotekanProcess *> build_processes();

private:

    void build_from_tree(map<string, KotekanProcess *> &processes, json &config_tree, const string &path);
    KotekanProcess * new_process(const string &name, const string &location);

    Config &config;
    bufferContainer &buffer_container;
};

#endif /* PROCESS_FACTORY_HPP */