#include "bufferFactory.hpp"

#include "Config.hpp"         // for Config
#include "HFBFrameView.hpp"   // for HFBFrameView
#include "buffer.hpp"         // for create_buffer
#include "kotekanLogging.hpp" // for INFO_NON_OO
#include "metadata.hpp"       // for metadataPool // IWYU pragma: keep
#include "ringbuffer.hpp"
#include "visBuffer.hpp"      // for VisFrameView
#include "N2FrameView.hpp"    // for N2FrameView

#include "fmt.hpp" // for format, fmt

#include <cstdint>   // for int32_t, uint32_t
#include <exception> // for exception
#include <regex>     // for match_results<>::_Base_type
#include <stddef.h>  // for size_t
#include <stdexcept> // for runtime_error
#include <vector>    // for vector

using json = nlohmann::json;
using std::map;
using std::string;

namespace kotekan {

bufferFactory::bufferFactory(Config& _config,
                             map<string, std::shared_ptr<metadataPool>>& _metadataPools) :
    config(_config),
    metadataPools(_metadataPools) {}

bufferFactory::~bufferFactory() {}

map<string, GenericBuffer*> bufferFactory::build_buffers() {
    map<string, GenericBuffer*> buffers;
    // Start parsing tree, put the buffers in the "buffers" map
    build_from_tree(buffers, config.get_full_config_json(), "");
    return buffers;
}

void bufferFactory::build_from_tree(map<string, GenericBuffer*>& buffers, const json& config_tree,
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

GenericBuffer* bufferFactory::new_buffer(const string& type_name, const string& name,
                                         const string& location) {
    // DEBUG("Creating buffer of type: {:s}, at config tree path: {:s}", name, location);
    string metadataPool_name = config.get_default<std::string>(location, "metadata_pool", "none");
    int32_t numa_node = config.get_default<int32_t>(location, "numa_node", 0);
    std::string s_log_level = config.get<std::string>(location, "log_level");

    std::shared_ptr<metadataPool> pool;
    if (metadataPool_name != "none") {
        if (metadataPools.count(metadataPool_name) != 1) {
            throw std::runtime_error(fmt::format(
                fmt("The buffer {:s} is requesting metadata pool named {:s} but no pool exists."),
                name, metadataPool_name));
        }
        pool = metadataPools[metadataPool_name];
    }

    // See also buffer::is_frame_buffer(), which looks for these three strings ("standard", "vis",
    // "hfb")
    size_t frame_size = 0;
    if (type_name == "standard") {
        frame_size = config.get<size_t>(location, "frame_size");
    } else if (type_name == "vis") {
        frame_size = VisFrameView::calculate_frame_size(config, location);
    } else if (type_name == "N2") {
        frame_size = N2FrameView::calculate_frame_size(config, location);
    } else if (type_name == "hfb") {
        frame_size = HFBFrameView::calculate_frame_size(config, location);
    }

    GenericBuffer* buf;
    if (frame_size != 0) {
        uint32_t num_frames = config.get<uint32_t>(location, "num_frames");
        bool use_hugepages = config.get_default<bool>(location, "use_hugepages", false);
        bool mlock_frames = config.get_default<bool>(location, "mlock_frames", true);
        bool zero_new_frames = config.get_default<bool>(location, "zero_new_frames", true);
        INFO_NON_OO("Creating {:s}Buffer named {:s} with {:d} frames, frame size of {:d} and "
                    "metadata pool {:s} on numa_node {:d}",
                    type_name, name, num_frames, frame_size, metadataPool_name, numa_node);
        buf = new Buffer(num_frames, frame_size, pool, name, type_name, numa_node, use_hugepages,
                         mlock_frames, zero_new_frames);

    } else if (type_name == "ring") {
        size_t ringbuf_size = config.get<size_t>(location, "ring_buffer_size");
        INFO_NON_OO("Creating {:s}Buffer named {:s} with ring buffer size of {:d} and "
                    "metadata pool {:s} on numa_node {:d}",
                    type_name, name, ringbuf_size, metadataPool_name, numa_node);
        buf = new RingBuffer(ringbuf_size, pool, name, type_name);

    } else {
        // Unknown buffer type
        throw std::runtime_error(
            fmt::format(fmt("No buffer type named {:s} or buffer size is 0"), type_name));
    }

    buf->set_log_level(s_log_level);
    return buf;
}

} // namespace kotekan
