#include "metadataFactory.hpp"
#include "metadata.h"
#include "chimeMetadata.h"
#include "visBuffer.hpp"
#include "Config.hpp"
#include "configEval.hpp"

metadataFactory::metadataFactory(Config& config) : config(config) {
}

metadataFactory::~metadataFactory() {
}

map<string, struct metadataPool *> metadataFactory::build_pools() {
    map<string, struct metadataPool *> pools;

    // Start parsing tree, put the processes in the "pools" vector
    build_from_tree(pools, config.get_full_config_json(), "");

    return pools;
}

void metadataFactory::build_from_tree(map<string, struct metadataPool *> &pools,
                                      json& config_tree, const string& path) {

    for (json::iterator it = config_tree.begin(); it != config_tree.end(); ++it) {
        // If the item isn't an object we can just ignore it.
        if (!it.value().is_object()) {
            continue;
        }

        // Check if this is a kotekan_process block, and if so create the process.
        string pool_type = it.value().value("kotekan_metadata_pool", "none");
        if (pool_type != "none") {
            string unique_path = path + "/" + it.key();
            string name = it.key();
            if (pools.count(name) != 0) {
                throw std::runtime_error("The metadata object named " + name + " has already been defined!");
            }
            pools[name] = new_pool(pool_type, unique_path);
            continue;
        }

        // Recursive part.
        // This is a section/scope not a process block.
        build_from_tree(pools, it.value(), path + "/" + it.key());
    }
}


struct metadataPool* metadataFactory::new_pool(const string &pool_type, const string &location) {

    INFO("Creating metadata pool of type: %s, at config tree path: %s", pool_type.c_str(), location.c_str());

    uint32_t num_metadata_objects = configEval<uint32_t>(
                config, location, "num_metadata_objects").compute_result();

    if (pool_type == "chimeMetadata") {
        return create_metadata_pool(num_metadata_objects, sizeof(struct chimeMetadata));
    }

    if (pool_type == "visMetadata") {
        return create_metadata_pool(num_metadata_objects, sizeof(struct visMetadata));
    }
    // No metadata found
    throw std::runtime_error("No metadata object named: " + pool_type);
}
