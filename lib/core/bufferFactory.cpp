#include "bufferFactory.hpp"

#include "Config.hpp"
#include "metadata.h"
#include "visBuffer.hpp"

bufferFactory::bufferFactory(Config& _config, map<string, struct metadataPool*>& _metadataPools) :
    config(_config),
    metadataPools(_metadataPools) {}

bufferFactory::~bufferFactory() {}

map<string, struct Buffer*> bufferFactory::build_buffers() {
    map<string, struct Buffer*> buffers;

    // Start parsing tree, put the processes in the "pools" vector
    build_from_tree(buffers, config.get_full_config_json(), "");

    return buffers;
}

void bufferFactory::build_from_tree(map<string, struct Buffer*>& buffers, json& config_tree,
                                    const string& path) {

    for (json::iterator it = config_tree.begin(); it != config_tree.end(); ++it) {
        // If the item isn't an object we can just ignore it.
        if (!it.value().is_object()) {
            continue;
        }

        // Check if this is a kotekan_process block, and if so create the process.
        string buffer_type = it.value().value("kotekan_buffer", "none");
        if (buffer_type != "none") {
            string name = it.key();
            if (buffers.count(name) != 0) {
                throw std::runtime_error("The buffer named " + name + " has already been defined!");
            }
            buffers[name] = new_buffer(buffer_type, name, path + "/" + it.key());
            continue;
        }

        // Recursive part.
        // This is a section/scope not a process block.
        build_from_tree(buffers, it.value(), path + "/" + it.key());
    }
}

struct Buffer* bufferFactory::new_buffer(const string& type_name, const string& name,
                                         const string& location) {

    // DEBUG("Creating buffer of type: %s, at config tree path: %s", name.c_str(),
    // location.c_str());
    uint32_t num_frames = config.get<uint32_t>(location, "num_frames");
    string metadataPool_name = config.get<std::string>(location, "metadata_pool");
    if (metadataPools.count(metadataPool_name) != 1) {
        throw std::runtime_error("The buffer " + name + " is requesting metadata pool named "
                                 + metadataPool_name + " but no pool exists.");
    }
    struct metadataPool* pool = metadataPools[metadataPool_name];

    if (type_name == "standard") {
        uint32_t frame_size = config.get<uint32_t>(location, "frame_size");
        INFO("Creating standard buffer named %s, with %d frames, frame_size of %d, and metadata "
             "pool %s",
             name.c_str(), num_frames, frame_size, metadataPool_name.c_str());
        int32_t numa_node = config.get_default<int32_t>(location, "numa_node", 0);
        return create_buffer(num_frames, frame_size, pool, name.c_str(), numa_node);
    }

    if (type_name == "vis") {
        int num_elements = config.get<int>(location, "num_elements");
        int num_ev = config.get<int>(location, "num_ev");
        int num_prod = config.get_default<int>(location, "num_prod", -1);

        if (num_prod < 0) {
            num_prod = num_elements * (num_elements + 1) / 2;
        }

        auto layout = visFrameView::calculate_buffer_layout(num_elements, num_prod, num_ev);
        uint32_t frame_size = layout.first;

        INFO("Creating visBuffer named %s with %d frames, frame size of %d and metadata pool %s",
             name.c_str(), num_frames, frame_size, metadataPool_name.c_str());
        return create_buffer(num_frames, frame_size, pool, name.c_str(), 0);
    }

    // No metadata found
    throw std::runtime_error("No buffer type named: " + name);
}
