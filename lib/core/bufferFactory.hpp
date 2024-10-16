#ifndef BUFFER_FACTORY_HPP
#define BUFFER_FACTORY_HPP

#include "Config.hpp" // for Config
#include "buffer.hpp" // for Buffer // IWYU pragma: keep
#include "metadata.h" // for metadataPool // IWYU pragma: keep

#include "json.hpp" // for json

#include <map>    // for map
#include <string> // for string

namespace kotekan {

class bufferFactory {

public:
    // One bufferFactory should be created for each set of config and buffer_container
    bufferFactory(Config& config, std::map<std::string, metadataPool*>& metadataPools);
    ~bufferFactory();

    std::map<std::string, Buffer*> build_buffers();

private:
    void build_from_tree(std::map<std::string, Buffer*>& buffers, const nlohmann::json& config_tree,
                         const std::string& path);
    Buffer* new_buffer(const std::string& type_name, const std::string& name,
                       const std::string& location);

    Config& config;
    std::map<std::string, metadataPool*>& metadataPools;
};

} // namespace kotekan

#endif /* BUFFER_FACTORY_HPP */
