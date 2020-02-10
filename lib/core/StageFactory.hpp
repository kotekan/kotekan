/**
 * @file
 * @brief Contains StageFactory and associated helper classes / templates.
 *  - StageMaker
 *  - StageFactory
 *  - StageFactoryRegistry
 *  - StageMakerTemplate<T>
 */

#ifndef STAGE_FACTORY_HPP
#define STAGE_FACTORY_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "bufferContainer.hpp"

#include "json.hpp" // for json

#include <map>    // for map
#include <string> // for string

namespace kotekan {

class StageMaker {
public:
    virtual Stage* create(Config& config, const std::string& unique_name,
                          bufferContainer& host_buffers) const = 0;
};

class StageFactory {

public:
    // One StageFactory should be created for each set of config and buffer_container
    StageFactory(Config& config, bufferContainer& buffer_container);
    ~StageFactory();

    // Creates all the stages listed in the config file, and returns them
    // as a vector of Stage pointers.
    // This should only be called once.
    std::map<std::string, Stage*> build_stages();

private:
    void build_from_tree(std::map<std::string, Stage*>& stages, const nlohmann::json& config_tree,
                         const std::string& path);

    Config& config;
    bufferContainer& buffer_container;

    Stage* create(const std::string& name, Config& config, const std::string& unique_name,
                  bufferContainer& host_buffers) const;
};

class StageFactoryRegistry {
public:
    // Add the stage to the registry.
    static void kotekan_register_stage(const std::string& key, StageMaker* proc);
    // INFO all the known commands out.
    static std::map<std::string, StageMaker*> get_registered_stages();

private:
    StageFactoryRegistry();
    void kotekan_reg(const std::string& key, StageMaker* proc);
    static StageFactoryRegistry& instance();
    std::map<std::string, StageMaker*> _kotekan_stages;
};

template<typename T>
class StageMakerTemplate : public StageMaker {
public:
    StageMakerTemplate(const std::string& key) {
        StageFactoryRegistry::kotekan_register_stage(key, this);
    }
    virtual Stage* create(Config& config, const std::string& unique_name,
                          bufferContainer& host_buffers) const override {
        return new T(config, unique_name, host_buffers);
    }
};

} // namespace kotekan

#define REGISTER_KOTEKAN_STAGE(T) static ::kotekan::StageMakerTemplate<T> maker##T(#T);

#endif /* STAGE_FACTORY_HPP */
