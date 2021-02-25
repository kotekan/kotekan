#ifndef METADATA_FACTORY_HPP
#define METADATA_FACTORY_HPP

#include "Config.hpp" // for Config
#include "metadata.h" // for metadataPool // IWYU pragma: keep

#include <json.hpp> // for json
#include <map>      // for map
#include <string>   // for string

namespace kotekan {

class metadataFactory {

public:
    // One metadataFactory should be created for each set of config and buffer_container
    metadataFactory(Config& config);
    ~metadataFactory();

    std::map<std::string, struct metadataPool*> build_pools();

private:
    void build_from_tree(std::map<std::string, struct metadataPool*>& pools,
                         const nlohmann::json& config_tree, const std::string& path);
    struct metadataPool* new_pool(const std::string& pool_type, const std::string& location);

    Config& config;
};

} // namespace kotekan

#endif /* METADATA_FACTORY_HPP */
