#include "metadataFactory.hpp"

#include "BeamMetadata.hpp"   // for BeamMetadata
#include "Config.hpp"         // for Config
#include "HFBMetadata.hpp"    // for HFBMetadata
#include "chimeMetadata.hpp"  // for chimeMetadata
#include "kotekanLogging.hpp" // for INFO_NON_OO
#include "metadata.h"         // for create_metadata_pool
#include "visBuffer.hpp"      // for VisMetadata

#include "fmt.hpp" // for format, fmt

#include <exception> // for exception
#include <regex>     // for match_results<>::_Base_type
#include <stdexcept> // for runtime_error
#include <stdint.h>  // for uint32_t
#include <vector>    // for vector

using json = nlohmann::json;
using std::map;
using std::string;

namespace kotekan {

metadataFactory::metadataFactory(Config& config) : config(config) {}

metadataFactory::~metadataFactory() {}

map<std::string, struct metadataPool*> metadataFactory::build_pools() {
    map<std::string, struct metadataPool*> pools;

    // Start parsing tree, put the metadata_pool's in the "pools" vector
    build_from_tree(pools, config.get_full_config_json(), "");

    return pools;
}

void metadataFactory::build_from_tree(map<std::string, struct metadataPool*>& pools,
                                      const json& config_tree, const std::string& path) {

    for (json::const_iterator it = config_tree.begin(); it != config_tree.end(); ++it) {
        // If the item isn't an object we can just ignore it.
        if (!it.value().is_object()) {
            continue;
        }

        // Check if this is a kotekan_metadata_pool block, and if so create the metadata_pool.
        std::string pool_type = it.value().value("kotekan_metadata_pool", "none");
        if (pool_type != "none") {
            std::string unique_path = fmt::format(fmt("{:s}/{:s}"), path, it.key());
            std::string name = it.key();
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


struct metadataPool* metadataFactory::new_pool(const std::string& pool_type,
                                               const std::string& location) {

    INFO_NON_OO("Creating metadata pool of type: {:s}, at config tree path: {:s}", pool_type,
                location);

    uint32_t num_metadata_objects = config.get<uint32_t>(location, "num_metadata_objects");

    if (pool_type == "chimeMetadata") {
        return create_metadata_pool(num_metadata_objects, sizeof(struct chimeMetadata));
    }

    if (pool_type == "VisMetadata") {
        return create_metadata_pool(num_metadata_objects, sizeof(struct VisMetadata));
    }

    if (pool_type == "HFBMetadata") {
        return create_metadata_pool(num_metadata_objects, sizeof(struct HFBMetadata));
    }

    if (pool_type == "BeamMetadata") {
        return create_metadata_pool(num_metadata_objects, sizeof(struct BeamMetadata));
    }
    if (pool_type == "MergedBeamMetadata"){
        return create_metadata_pool(num_metadata_objects, sizeof(struct MergedBeamMetadata));
    }
    // No metadata found
    throw std::runtime_error(fmt::format(fmt("No metadata object named: {:s}"), pool_type));
}

} // namespace kotekan
