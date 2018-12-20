#ifndef BUFFER_FACTORY_HPP
#define BUFFER_FACTORY_HPP

#include "Config.hpp"
#include "buffer.h"
#include "metadata.h"

#include "json.hpp"

#include <map>
#include <string>

// Name space includes.
using json = nlohmann::json;
using std::map;
using std::string;

class bufferFactory {

public:
    // One processFactory should be created for each set of config and buffer_container
    bufferFactory(Config& config, map<string, struct metadataPool*>& metadataPools);
    ~bufferFactory();

    map<string, struct Buffer*> build_buffers();

private:
    void build_from_tree(map<string, struct Buffer*>& buffers, json& config_tree,
                         const string& path);
    struct Buffer* new_buffer(const string& type_name, const string& name, const string& location);

    Config& config;
    map<string, struct metadataPool*>& metadataPools;
};

#endif /* BUFFER_FACTORY_HPP */