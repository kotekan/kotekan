/**
 * @file
 * @brief Contains processFactory and associated helper classes / templates.
 *  - kotekanProcessMaker
 *  - processFactory
 *  - processFactoryRegistry
 *  - kotekanProcessMakerTemplate<T>
 */

#ifndef PROCESS_FACTORY_HPP
#define PROCESS_FACTORY_HPP

#include <string>
#include <map>

#include "json.hpp"
#include "Config.hpp"
#include "bufferContainer.hpp"

// Name space includes.
using json = nlohmann::json;
using std::string;
using std::map;

class KotekanProcess;

class kotekanProcessMaker{
public:
    virtual KotekanProcess *create(Config &config, const string &unique_name,
                bufferContainer &host_buffers) const = 0;
 };

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

    Config &config;
    bufferContainer &buffer_container;

    KotekanProcess *create(const string &name,
                       Config& config,
                       const string &unique_name,
                       bufferContainer &host_buffers) const;
};

class processFactoryRegistry {
public:
    //Add the process to the registry.
    static void kotekan_register_process(const std::string& key, kotekanProcessMaker* proc);
    //INFO all the known commands out.
    static std::map<std::string, kotekanProcessMaker*> get_registered_processes();

private:
    processFactoryRegistry();
    void kotekan_reg(const std::string& key, kotekanProcessMaker* proc);
    static processFactoryRegistry& instance();
    std::map<std::string, kotekanProcessMaker*> _kotekan_processes;
};

template<typename T>
class kotekanProcessMakerTemplate : public kotekanProcessMaker
{
    public:
        kotekanProcessMakerTemplate(const std::string& key)
        {
            processFactoryRegistry::kotekan_register_process(key, this);
        }
        virtual KotekanProcess *create(Config &config, const string &unique_name,
                    bufferContainer &host_buffers) const override
        {
            return new T(config, unique_name, host_buffers);
        }
};
#define REGISTER_KOTEKAN_PROCESS(T) static kotekanProcessMakerTemplate<T> maker##T(#T);


#endif /* PROCESS_FACTORY_HPP */