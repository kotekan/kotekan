#include "StageFactory.hpp"

#include "Config.hpp"         // for Config
#include "kotekanLogging.hpp" // for ERROR_NON_OO

#include "fmt.hpp" // for format, fmt

#include <stdexcept> // for runtime_error
#include <utility>   // for pair


using nlohmann::json;
using std::string;


namespace kotekan {

StageFactory::StageFactory(Config& config, bufferContainer& buffer_container) :
    config(config), buffer_container(buffer_container) {

#ifdef DEBUGGING
    auto known_stages = StageFactoryRegistry::get_registered_stages();
    for (auto& stage : known_stages) {
        DEBUG_NON_OO("Registered Kotekan Stage: {:s}", stage.first);
    }
#endif
}

StageFactory::~StageFactory() {}

std::map<std::string, Stage*> StageFactory::build_stages() {
    std::map<std::string, Stage*> stages;

    // Start parsing tree, put the stages in the "stages" vector
    build_from_tree(stages, config.get_full_config_json(), "");

    return stages;
}

void StageFactory::build_from_tree(std::map<std::string, Stage*>& stages,
                                   const nlohmann::json& config_tree, const std::string& path) {

    for (json::const_iterator it = config_tree.begin(); it != config_tree.end(); ++it) {
        // If the item isn't an object we can just ignore it.
        if (!it.value().is_object()) {
            continue;
        }

        // Check if this is a kotekan_stage block, and if so create the stage.
        string stage_name = it.value().value("kotekan_stage", "none");
        if (stage_name != "none") {
            string unique_name = fmt::format(fmt("{:s}/{:s}"), path, it.key());
            if (stages.count(unique_name) != 0) {
                throw std::runtime_error(
                    fmt::format(fmt("A stage with the path {:s} has been defined more than once!"),
                                unique_name));
            }
            stages[unique_name] =
                create(stage_name, config, fmt::format(fmt("{:s}/{:s}"), path, it.key()),
                       buffer_container);
            continue;
        }

        // Recursive part.
        // This is a section/scope not a stage block.
        build_from_tree(stages, it.value(), fmt::format(fmt("{:s}/{:s}"), path, it.key()));
    }
}

Stage* StageFactory::create(const string& name, Config& config, const string& unique_name,
                            bufferContainer& host_buffers) const {
    auto known_stages = StageFactoryRegistry::get_registered_stages();
    auto i = known_stages.find(name);
    if (i == known_stages.end()) {
        ERROR_NON_OO("Unrecognized Stage! ({:s})", name);
        throw std::runtime_error("Tried to instantiate a stage we don't know about!");
    }
    StageMaker* maker = i->second;
    return maker->create(config, unique_name, host_buffers);
}


void StageFactoryRegistry::kotekan_register_stage(const std::string& key, StageMaker* proc) {
    StageFactoryRegistry::instance().kotekan_reg(key, proc);
}

std::map<std::string, StageMaker*> StageFactoryRegistry::get_registered_stages() {
    return StageFactoryRegistry::instance()._kotekan_stages;
}


StageFactoryRegistry& StageFactoryRegistry::instance() {
    static StageFactoryRegistry factory;
    return factory;
}

StageFactoryRegistry::StageFactoryRegistry() {}

void StageFactoryRegistry::kotekan_reg(const std::string& key, StageMaker* proc) {
    if (_kotekan_stages.find(key) != _kotekan_stages.end()) {
        ERROR_NON_OO("Multiple Kotekan Stage-s registered as '{:s}'!", key);
        throw std::runtime_error("A Stage was registered twice!");
    }
    _kotekan_stages[key] = proc;
}

} // namespace kotekan
