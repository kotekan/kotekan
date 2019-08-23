#include "metadataFactory.hpp"

#include "Config.hpp"
#include "chimeMetadata.h"
#include "metadata.h"
#include "visBuffer.hpp"

#include "fmt.hpp"

using json = nlohmann::json;
using std::map;
using std::string;

namespace kotekan {

metadataFactory::metadataFactory(Config& config) : config(config) {}

metadataFactory::~metadataFactory() {}

map<string, struct metadataPool*> metadataFactory::build_pools() {
    map<string, struct metadataPool*> pools;

    // Start parsing tree, put the metadata_pool's in the "pools" vector
    build_from_tree(pools, config.get_full_config_json(), "");

    return pools;
}

void metadataFactory::build_from_tree(map<string, struct metadataPool*>& pools, json& config_tree,
                                      const string& path) {

    for (json::iterator it = config_tree.begin(); it != config_tree.end(); ++it) {
        // If the item isn't an object we can just ignore it.
        if (!it.value().is_object()) {
            continue;
        }

        // Check if this is a kotekan_metadata_pool block, and if so create the metadata_pool.
        string pool_type = it.value().value("kotekan_metadata_pool", "none");
        if (pool_type != "none") {
            string unique_path = fmt::format(fmt("{:s}/{:s}"), path, it.key());
            string name = it.key();
            if (pools.count(name) != 0) {
                throw std::runtime_error(fmt::format(
                    fmt("The metadata object named {:s} has already been defined!"), name));
            }
            pools[name] = new_pool(pool_type, unique_path);
            continue;
        }

        // Recursive part.
        // This is a section/scope not a kotekan_metadata_pool block.
        build_from_tree(pools, it.value(), fmt::format(fmt("{:s}/{:s}"), path, it.key()));
    }
}


struct metadataPool* metadataFactory::new_pool(const string& pool_type, const string& location) {

    INFO_NON_OO("Creating metadata pool of type: {:s}, at config tree path: {:s}", pool_type,
                location);

    uint32_t num_metadata_objects = config.get<uint32_t>(location, "num_metadata_objects");

    if (pool_type == "chimeMetadata") {
        return create_metadata_pool(num_metadata_objects, sizeof(struct chimeMetadata));
    }

    if (pool_type == "visMetadata") {
        return create_metadata_pool(num_metadata_objects, sizeof(struct visMetadata));
    }
    // No metadata found
    throw std::runtime_error(fmt::format(fmt("No metadata object named: {:s}"), pool_type));
}

} // namespace kotekan
