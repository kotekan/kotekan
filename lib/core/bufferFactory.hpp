#ifndef BUFFER_FACTORY_HPP
#define BUFFER_FACTORY_HPP

#include "Config.hpp"
#include "buffer.h"
#include "metadata.h"

#include "json.hpp"

#include <map>
#include <string>

namespace kotekan {

class bufferFactory {

public:
    // One bufferFactory should be created for each set of config and buffer_container
    bufferFactory(Config& config, std::map<std::string, struct metadataPool*>& metadataPools);
    ~bufferFactory();

    std::map<std::string, struct Buffer*> build_buffers();

private:
    void build_from_tree(std::map<std::string, struct Buffer*>& buffers,
                         nlohmann::json& config_tree, const std::string& path);
    struct Buffer* new_buffer(const std::string& type_name, const std::string& name,
                              const std::string& location);

    Config& config;
    std::map<std::string, struct metadataPool*>& metadataPools;
};

} // namespace kotekan

#endif /* BUFFER_FACTORY_HPP */
