#include "bufferFactory.hpp"

#include "Config.hpp"         // for Config
#include "buffer.h"           // for create_buffer
#include "kotekanLogging.hpp" // for INFO_NON_OO
#include "metadata.h"         // for metadataPool // IWYU pragma: keep
#include "visBuffer.hpp"      // for VisFrameView

#include "fmt.hpp" // for format, fmt

#include <cstdint>   // for int32_t, uint32_t
#include <exception> // for exception
#include <regex>     // for match_results<>::_Base_type
#include <stdexcept> // for runtime_error
#include <utility>   // for pair
#include <vector>    // for vector

using json = nlohmann::json;
using std::map;
using std::string;

namespace kotekan {

bufferFactory::bufferFactory(Config& _config, map<string, struct metadataPool*>& _metadataPools) :
    config(_config),
    metadataPools(_metadataPools) {}

bufferFactory::~bufferFactory() {}

map<string, struct Buffer*> bufferFactory::build_buffers() {
    map<string, struct Buffer*> buffers;

    // Start parsing tree, put the buffers in the "pools" vector
    build_from_tree(buffers, config.get_full_config_json(), "");

    return buffers;
}

void bufferFactory::build_from_tree(map<string, struct Buffer*>& buffers, const json& config_tree,
                                    const string& path) {

    for (json::const_iterator it = config_tree.begin(); it != config_tree.end(); ++it) {
        // If the item isn't an object we can just ignore it.
        if (!it.value().is_object()) {
            continue;
        }

        // Check if this is a kotekan_buffer block, and if so create the buffer.
        string buffer_type = it.value().value("kotekan_buffer", "none");
        if (buffer_type != "none") {
            string name = it.key();
            if (buffers.count(name) != 0) {
                throw std::runtime_error(
                    fmt::format(fmt("The buffer named {:s} has already been defined!"), name));
            }
            buffers[name] =
                new_buffer(buffer_type, name, fmt::format(fmt("{:s}/{:s}"), path, it.key()));
            continue;
        }

        // Recursive part.
        // This is a section/scope not a buffer block.
        build_from_tree(buffers, it.value(), fmt::format(fmt("{:s}/{:s}"), path, it.key()));
    }
}

struct Buffer* bufferFactory::new_buffer(const string& type_name, const string& name,
                                         const string& location) {

    // DEBUG("Creating buffer of type: {:s}, at config tree path: {:s}", name, location);
    uint32_t num_frames = config.get<uint32_t>(location, "num_frames");
    string metadataPool_name = config.get_default<std::string>(location, "metadata_pool", "none");
    int32_t numa_node = config.get_default<int32_t>(location, "numa_node", 0);

    struct metadataPool* pool = nullptr;
    if (metadataPool_name != "none") {
        if (metadataPools.count(metadataPool_name) != 1) {
            throw std::runtime_error(fmt::format(
                fmt("The buffer {:s} is requesting metadata pool named {:s} but no pool exists."),
                name, metadataPool_name));
        }
        pool = metadataPools[metadataPool_name];
    }

    if (type_name == "standard") {
        uint32_t frame_size = config.get<uint32_t>(location, "frame_size");
        INFO_NON_OO("Creating standard buffer named {:s}, with {:d} frames, frame_size of {:d}, "
                    "and metadata pool {:s} on numa_node {:d}",
                    name, num_frames, frame_size, metadataPool_name, numa_node);

        return create_buffer(num_frames, frame_size, pool, name.c_str(), numa_node);
    }

    if (type_name == "vis") {
        int num_elements = config.get<int>(location, "num_elements");
        int num_ev = config.get<int>(location, "num_ev");
        int num_prod = config.get_default<int>(location, "num_prod", -1);

        if (num_prod < 0) {
            num_prod = num_elements * (num_elements + 1) / 2;
        }

        auto layout = VisFrameView::calculate_buffer_layout(num_elements, num_prod, num_ev);
        uint32_t frame_size = layout.first;

        INFO_NON_OO("Creating VisFrameView named {:s} with {:d} frames, frame size of {:d} and "
                    "metadata pool {:s}",
                    name, num_frames, frame_size, metadataPool_name);
        return create_buffer(num_frames, frame_size, pool, name.c_str(), numa_node);
    }

    // No metadata found
    throw std::runtime_error(fmt::format(fmt("No buffer type named: {:s}"), name));
}

} // namespace kotekan
