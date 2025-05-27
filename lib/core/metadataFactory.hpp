#ifndef METADATA_FACTORY_HPP
#define METADATA_FACTORY_HPP

#include <map>           // for map
#include <string>        // for string
#include <memory>        // for shared_ptr

#include "Config.hpp"    // for Config
#include "metadata.hpp"  // for metadataPool
#include "json.hpp"      // for json

namespace kotekan {

class metadataFactory {

public:
    // One metadataFactory should be created for each set of config and buffer_container
    metadataFactory(Config& config);
    ~metadataFactory();

    std::map<std::string, std::shared_ptr<metadataPool>> build_pools();

private:
    void build_from_tree(std::map<std::string, std::shared_ptr<metadataPool>>& pools,
                         const nlohmann::json& config_tree, const std::string& path);
    std::shared_ptr<metadataPool> new_pool(const std::string& pool_type,
                                           const std::string& location);

    Config& config;
};

} // namespace kotekan

#endif /* METADATA_FACTORY_HPP */
