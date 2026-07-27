#include "kotekanMode.hpp"

#include "Config.hpp"            // for Config
#include "FrameDesc.hpp"         // for FrameDesc
#include "PipelineGraph.hpp"     // for PipelineGraph, GraphNode
#include "Stage.hpp"             // for Stage
#include "StageFactory.hpp"      // for StageFactory
#include "Telescope.hpp"         // for Telescope
#include "buffer.hpp"            // for StageInfo, Buffer, GenericBuffer
#include "bufferFactory.hpp"     // for bufferFactory
#include "configTracker.hpp"     // for ConfigTracker
#include "configUpdater.hpp"     // for configUpdater
#include "datasetManager.hpp"    // for datasetManager
#include "kotekanLogging.hpp"    // for INFO_NON_OO, FATAL_ERROR_NON_OO
#include "kotekanTrackers.hpp"   // for KotekanTrackers
#include "metadata.hpp"          // for metadataObject
#include "metadataFactory.hpp"   // for metadataFactory
#include "modp_b64.hpp"          // for modp_b64_encode, modp_b64_encode_len, MODP_B64_ERROR
#include "prometheusMetrics.hpp" // for Metrics
#include "restServer.hpp"        // for restServer, connectionInstance
#include "version.h"             // for get_cmake_build_options, get_git_branch, get_git_commit...

#include "fmt.hpp"  // for compile_string_to_view, format, format_string
#include "json.hpp" // for json, basic_json

#include <exception>  // for exception
#include <functional> // for bind, function, _1
#include <limits>     // for numeric_limits
#include <memory>     // for shared_ptr
#include <stdint.h>   // for uint8_t, uint16_t
#include <string>     // for basic_string, string, stoll
#include <utility>    // for pair, move
#include <vector>     // for vector

using namespace std::placeholders;

namespace kotekan {

kotekanMode::kotekanMode(Config& config_) : config(config_) {

    restServer::instance().register_get_callback("/config", [&](connectionInstance& conn) {
        conn.send_json_reply(config.get_full_config_json());
    });

#ifdef WITH_SSL
    restServer::instance().register_get_callback("/config_md5sum", [&](connectionInstance& conn) {
        nlohmann::json reply;
        reply["md5sum"] = config.get_md5sum();
        conn.send_json_reply(reply);
    });
#endif

    restServer::instance().add_aliases_from_config(config);
    restServer::instance().set_cors_from_config(config);
}

kotekanMode::~kotekanMode() {

    // Log which config items were used (and by which stage) if usage tracking
    // was enabled. Done before tearing down stages, while config is intact.
    config.log_access_summary();

    configUpdater::instance().reset();
    restServer::instance().remove_get_callback("/config");
    restServer::instance().remove_get_callback("/buffers");
    restServer::instance().remove_get_callback("/pipeline_dot");
    for (auto const& endpoint : frame_peek_endpoints)
        restServer::instance().remove_get_callback(endpoint);
    restServer::instance().remove_all_aliases();

    KotekanTrackers::instance().set_kotekan_mode_ptr(nullptr);

#if !defined(MAC_OSX)
    // Stop (and join) the CPU monitor before deleting the stages its
    // tracking thread reads. Normally already done in stop_stages(), but
    // teardown on a failed start skips that path.
    cpu_monitor.stop();
#endif

    for (auto const& stage : stages) {
        if (stage.second != nullptr) {
            delete stage.second;
            prometheus::Metrics::instance().remove_stage_metrics(stage.first);
        }
    }

    for (auto const& buf : buffers) {
        if (buf.second != nullptr) {
            delete buf.second;
        }
    }

    metadata_pools.clear();
}

void kotekanMode::initalize_stages() {

    // Create Config Updater
    configUpdater& config_updater = configUpdater::instance();
    config_updater.apply_config(config);

    // Apply config to datasetManager
    if (config.exists("/", "dataset_manager"))
        datasetManager::instance(config);

    // Apply config for Telescope class
    Telescope::instance(config);

    // Create and register kotekan trackers before stages created
    KotekanTrackers::instance(config).register_with_server(&restServer::instance());
    KotekanTrackers::instance().set_kotekan_mode_ptr(this);

    // Create Metadata Pool
    metadataFactory metadata_factory(config);
    metadata_pools = metadata_factory.build_pools();

    // Create Buffers
    bufferFactory buffer_factory(config, metadata_pools);
    buffers = buffer_factory.build_buffers();
    buffer_container.set_buffer_map(buffers);

    // ConfigTracker setup. Disabled, unless a /config_tracker block exists.
    // This enables the tracker, unless explicitly disabled with `enabled: false`.
    // Done before stages are built so stages can read the tracker's enabled
    // state at construction; the tracker setup does not depend on any stage.
    bool ct_enabled = false;
    if (config.exists("/", "config_tracker")) {
        const nlohmann::json ct_node = config.get_value("/", "config_tracker");
        if (!ct_node.is_object()) {
            FATAL_ERROR_NON_OO("kotekanMode: /config_tracker must be an object (e.g. {{enabled: "
                               "true}}); bare-bool form is not supported.");
        }
        ct_enabled = config.get_default<bool>("/config_tracker", "enabled", true);
    }
    ConfigTracker::instance().set_enabled(ct_enabled);

    if (ct_enabled) {
        // Register ConfigTracker endpoints and set the local startup config
        // so the tracker can propagate configs to downstream instances.
        // The local config is identified by its content alone; downstream
        // peers re-key it under whatever (host, port) they actually dialed
        // when they fetch /config_tracker_local.
        ConfigTracker::instance().register_with_server(&restServer::instance());
        try {
            ConfigTracker::instance().setLocalConfig(
                config.get_full_config_json(), get_kotekan_version(), get_git_branch(),
                get_git_commit_hash(), get_cmake_build_options());
        } catch (const std::exception& e) {
            FATAL_ERROR_NON_OO("Failed to set local config in ConfigTracker: {:s}", e.what());
        }

        // Apply the rest of the /config_tracker block: upstream fetch
        // policy and (optionally) the FPGA controller snapshot.
        ConfigTracker::instance().applyConfig(config);
    }

    // Create Stages
    StageFactory stage_factory(config, buffer_container);
    stages = stage_factory.build_stages();

    // Update REST server
    restServer::instance().set_server_affinity(config);

    // Register pipeline status callbacks
    restServer::instance().register_get_callback(
        "/buffers", std::bind(&kotekanMode::buffer_data_callback, this, _1));

    restServer::instance().register_get_callback(
        "/pipeline_dot", std::bind(&kotekanMode::pipeline_dot_graph_callback, this, _1));

    // Register frame-peek endpoints for every frame-holding buffer, so the
    // newest full frame in any buffer can be inspected without config changes.
    for (auto& buf : buffer_container.get_buffer_map()) {
        Buffer* frame_buf = dynamic_cast<Buffer*>(buf.second);
        if (frame_buf == nullptr)
            continue;

        std::string endpoint = fmt::format(fmt("/buffer/{:s}/frame"), buf.first);
        restServer::instance().register_get_callback(
            endpoint, std::bind(&kotekanMode::buffer_frame_callback, this, frame_buf, _1));
        frame_peek_endpoints.push_back(endpoint);
    }
}

void kotekanMode::join() {
    for (auto const& stage : stages) {
        INFO_NON_OO("Joining kotekan_stage: {:s}...", stage.first);
        stage.second->join();
    }
}

void kotekanMode::start_stages() {
    for (auto const& stage : stages) {
        INFO_NON_OO("Starting kotekan_stage: {:s}...", stage.first);
        stage.second->start();
    }

#if !defined(MAC_OSX)
    if (config.get_default<bool>("/cpu_monitor", "enabled", false)) {
        cpu_monitor.set_track_len(config.get_default<uint16_t>("/cpu_monitor", "track_length", 2));
        cpu_monitor.save_stages(stages);
        cpu_monitor.start();
        cpu_monitor.set_affinity(config);
    }
#endif
}

void kotekanMode::stop_stages() {
#if !defined(MAC_OSX)
    cpu_monitor.stop();
#endif
    // First set the shutdown variable on all stages
    for (auto const& stage : stages)
        stage.second->stop();

    // Then send shutdown signal to buffers which
    // should wake up stages which are blocked.
    for (auto const& buf : buffers) {
        INFO_NON_OO("Sending shutdown signal to buffer: {:s}", buf.first);
        buf.second->send_shutdown_signal();
    }
}

nlohmann::json kotekanMode::get_buffer_json() {
    nlohmann::json buffer_json = {};
    for (auto& buf : buffer_container.get_buffer_map()) {
        nlohmann::json buf_info = {};
        buf.second->json_description(buf_info);
        buffer_json[buf.first] = buf_info;
    }

    return buffer_json;
}

void kotekanMode::buffer_data_callback(connectionInstance& conn) {
    conn.send_json_reply(get_buffer_json());
}

PipelineGraph kotekanMode::get_pipeline_graph() {
    PipelineGraph graph;
    graph.header_comments.push_back(
        "This is a DOT formatted pipeline graph, use the graphviz package to plot.");

    // Buffer nodes.
    for (auto& buf : buffer_container.get_buffer_map()) {
        auto& node = graph.add_node(buf.first);
        node.label_lines = buf.second->dot_label_lines();
        node.set_attr("shape", "ellipse").set_attr("color", "blue");
    }

    // Stage nodes. Every stage gets its node here, so that the edges below always
    // have something to land on, and so a stage overriding add_graph_details()
    // only has to describe what is particular to it.
    for (auto& stage : stages) {
        auto& node = graph.add_node(stage.first);
        node.add_line(stage.first);
        node.set_attr("shape", "box").set_attr("color", "darkgreen");
    }
    for (auto& stage : stages)
        stage.second->add_graph_details(graph);

    // Producer/consumer relations. Registration names are stage unique names, but
    // a stage may register a helper under another name (and a stage can be torn
    // down while its registration lives on), so give any unrecognised name a node
    // of its own rather than losing the edge.
    auto endpoint = [&graph](const std::string& name) -> const std::string& {
        if (!graph.has_node(name))
            graph.add_node(name)
                .add_line(name)
                .set_attr("shape", "box")
                .set_attr("style", "dashed");
        return name;
    };
    for (auto& buf : buffer_container.get_buffer_map()) {
        for (auto& cit : buf.second->consumers)
            graph.add_edge(buf.first, endpoint(cit.second.name));
        for (auto& pit : buf.second->producers)
            graph.add_edge(endpoint(pit.second.name), buf.first);
    }

    return graph;
}

void kotekanMode::pipeline_dot_graph_callback(connectionInstance& conn) {
    conn.send_text_reply(get_pipeline_graph().to_dot());
}

void kotekanMode::buffer_frame_callback(Buffer* buf, connectionInstance& conn) {
    // Optional `len` query parameter: absent means copy the whole frame,
    // `len=0` requests metadata only.
    size_t max_len = std::numeric_limits<size_t>::max();
    auto query = conn.get_query();
    auto len_arg = query.find("len");
    if (len_arg != query.end()) {
        try {
            size_t pos = 0;
            const long long parsed = std::stoll(len_arg->second, &pos);
            if (pos != len_arg->second.size() || parsed < 0)
                throw std::invalid_argument("not a non-negative integer");
            max_len = (size_t)parsed;
        } catch (const std::exception&) {
            conn.send_error("'len' query parameter must be a non-negative integer",
                            HTTP_RESPONSE::BAD_REQUEST);
            return;
        }
    }

    std::vector<uint8_t> data;
    std::shared_ptr<metadataObject> frame_metadata;
    const int frame_id = buf->peek_newest_full_frame(data, max_len, frame_metadata);
    if (frame_id < 0) {
        conn.send_error(
            fmt::format(fmt("no full frame currently in buffer {:s}, try again"), buf->buffer_name),
            HTTP_RESPONSE::REQUEST_FAILED);
        return;
    }

    nlohmann::json reply;
    reply["buffer"] = buf->buffer_name;
    reply["frame_id"] = frame_id;
    reply["frame_size"] = buf->frame_size;
    reply["data_length"] = data.size();
    reply["metadata"] = frame_metadata ? frame_metadata->to_json() : nlohmann::json();
    auto frame_desc = buf->get_frame_desc();
    reply["frame_desc"] = frame_desc ? frame_desc->to_json() : nlohmann::json();
    if (!data.empty()) {
        std::string encoded(modp_b64_encode_len(data.size()), '\0');
        const size_t enc_len =
            modp_b64_encode(&encoded[0], reinterpret_cast<const char*>(data.data()), data.size());
        if (enc_len == MODP_B64_ERROR) {
            conn.send_error("base64 encoding of frame data failed", HTTP_RESPONSE::INTERNAL_ERROR);
            return;
        }
        // Trim the encode-buffer allocation down to the actual encoded length.
        encoded.resize((data.size() + 2) / 3 * 4);
        reply["data"] = std::move(encoded);
        reply["encoding"] = "base64";
    }
    conn.send_json_reply(reply);
}

} // namespace kotekan
