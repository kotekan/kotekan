#ifndef BUFFER_FACTORY_HPP
#define BUFFER_FACTORY_HPP

#include <map>           // for map
#include <string>        // for string
#include <memory>        // for shared_ptr

#include "Config.hpp"    // for Config
#include "buffer.hpp"    // for GenericBuffer
#include "metadata.hpp"  // for metadataPool
#include "json.hpp"      // for json

namespace kotekan {

class bufferFactory {

public:
    // One bufferFactory should be created for each set of config and buffer_container
    bufferFactory(Config& config,
                  std::map<std::string, std::shared_ptr<metadataPool>>& metadataPools);
    ~bufferFactory();

    std::map<std::string, GenericBuffer*> build_buffers();

private:
    void build_from_tree(std::map<std::string, GenericBuffer*>& buffers,
                         const nlohmann::json& config_tree, const std::string& path);
    GenericBuffer* new_buffer(const std::string& type_name, const std::string& name,
                              const std::string& location);

    Config& config;
    std::map<std::string, std::shared_ptr<metadataPool>>& metadataPools;
};

} // namespace kotekan

#endif /* BUFFER_FACTORY_HPP */
