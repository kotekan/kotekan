// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Kotekan Developers

/****************************************************
 * @file metadatFactory.cpp
 * @brief This file implements the factory for making
 *        metadata structures, registered previously.
 *
 * @author Mehdi Najafi
 * @date   28 AUG 2022
 *****************************************************/

#include "metadataFactory.hpp"

#include "Config.hpp"         // for Config
#include "kotekanLogging.hpp" // for INFO_NON_OO
#include "metadata.h"         // for create_metadata_pool

// include all metadata header files available here
#include "BasebandMetadata.hpp"
#include "BeamMetadata.hpp"
#include "FrequencyAssembledMetadata.hpp"
#include "HFBMetadata.hpp"
#include "chimeMetadata.hpp"

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

} // namespace kotekan
