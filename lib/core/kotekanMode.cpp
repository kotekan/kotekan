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
    restServer::instance().remove_get_callback("/pipeline_json");
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

    restServer::instance().register_get_callback(
        "/pipeline_json", std::bind(&kotekanMode::pipeline_json_graph_callback, this, _1));

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

/// The config section a stage was declared in ("/gpu/gpu_0" -> "gpu"), or an
/// empty string for a stage declared at the top level.
static std::string config_section(const std::string& unique_name) {
    const size_t first = unique_name.find_first_not_of('/');
    if (first == std::string::npos)
        return "";
    const size_t slash = unique_name.find('/', first);
    if (slash == std::string::npos)
        return ""; // no section: the stage is a top level config block
    return unique_name.substr(first, slash - first);
}

/// Adds a colour key, so the graph can be read without knowing this code.
static void add_legend(PipelineGraph& graph) {
    auto& legend = graph.add_cluster("__legend");
    legend.label = "legend";
    legend.set_attr("style", "rounded").set_attr("color", "gray70");
    // One rank keeps the key as a compact block instead of a band stretched
    // across the whole drawing.
    legend.set_attr("rank", "same");

    struct Entry {
        const char* id;
        GraphCategory category;
        const char* text;
    };
    static const Entry entries[] = {
        {"buffer", GraphCategory::Buffer, "buffer: name / type · metadata / layout / size / fill"},
        {"compute", GraphCategory::Compute, "stage on the CPU"},
        {"gpu", GraphCategory::Gpu, "GPU stage and its commands"},
        {"io", GraphCategory::Io, "stage moving data on or off this machine"},
        {"memory", GraphCategory::Memory, "device memory (not a kotekan buffer)"},
        {"endpoint", GraphCategory::Endpoint, "the far end: a socket, a port, a file"},
    };
    for (const auto& entry : entries) {
        auto& node = graph.add_node(std::string("__legend/") + entry.id);
        node.add_line(entry.text);
        node.cluster = legend.id;
        node.set_category(entry.category);
    }
}

PipelineGraph kotekanMode::get_pipeline_graph(const GraphOptions& options) {
    PipelineGraph graph;
    graph.options = options;
    graph.header_comments.push_back(
        "This is a DOT formatted pipeline graph, use the graphviz package to plot.");
    // Pipelines are long and thin, so they read left to right; the extra
    // crossing-minimisation passes are cheap next to the cost of untangling the
    // result by eye, and newrank lines the clusters up across the whole graph.
    graph.graph_attrs["rankdir"] = options.rankdir;
    graph.graph_attrs["nodesep"] = "0.3";
    graph.graph_attrs["ranksep"] = "0.9";
    graph.graph_attrs["mclimit"] = "6";
    graph.graph_attrs["newrank"] = "true";
    graph.node_attrs["fontname"] = "Helvetica";
    graph.node_attrs["fontsize"] = "11";
    graph.edge_attrs["fontname"] = "Helvetica";
    graph.edge_attrs["fontsize"] = "8";

    // Buffer nodes.
    for (auto& buf : buffer_container.get_buffer_map()) {
        auto& node = graph.add_node(buf.first);
        node.label_lines = buf.second->dot_label_lines(options);
        node.set_category(GraphCategory::Buffer);
        if (options.runtime)
            node.set_buffer_state(buf.second->dot_buffer_state());
        // A rendered graph is then a way in: click a buffer to see the frame it
        // is holding. Only frame buffers have somewhere to go.
        if (options.urls && dynamic_cast<Buffer*>(buf.second) != nullptr)
            node.set_attr("URL", fmt::format(fmt("/buffer/{:s}/frame"), buf.first));
        node.set_attr("tooltip",
                      fmt::format(fmt("{:s} ({:s} buffer)"), buf.first, buf.second->buffer_type));
    }

#if !defined(MAC_OSX)
    const std::map<std::string, double> cpu_usage = cpu_monitor.get_stage_cpu_usage();
#else
    const std::map<std::string, double> cpu_usage;
#endif

    // Stage nodes, grouped by the config section they were declared in. Every
    // stage gets its node here, so that the edges below always have something to
    // land on, and so a stage overriding add_graph_details() only has to describe
    // what is particular to it.
    for (auto& stage : stages) {
        // The type is what the stage *is*; the config path is only where it was
        // declared. Read it back from the config, so no stage has to report it.
        const std::string type = config.get_default<std::string>(stage.first, "kotekan_stage", "");
        auto& node = graph.add_node(stage.first);
        node.add_line(leaf_name(stage.first));
        node.add_line(type.empty() ? "" : "<" + type + ">");
        node.set_category(stage_category(type));

        // What the stage is costing, and where it is allowed to run: the two
        // things one goes looking for when a pipeline will not keep up.
        const std::vector<pid_t> tids = stage.second->get_tids();
        const std::vector<int> affinity = stage.second->get_cpu_affinity();
        if (options.runtime) {
            std::string running =
                fmt::format(fmt("{:d} thread{:s}"), tids.size(), tids.size() == 1 ? "" : "s");
            auto usage = cpu_usage.find(stage.first);
            if (usage != cpu_usage.end())
                running += fmt::format(fmt(" · cpu {:.0f}%"), usage->second);
            if (stage.second->is_stopping())
                running += " · stopping";
            node.add_line(running);
        }
        if (!affinity.empty()) {
            std::string cores;
            for (int core : affinity)
                cores += (cores.empty() ? "" : ",") + std::to_string(core);
            node.add_line("cores " + cores);
        }
        // The node shows the leaf name; the full config path is a hover away.
        node.set_attr("tooltip", stage.first);

        const std::string section = config_section(stage.first);
        if (options.cluster && !section.empty()) {
            auto& cluster = graph.add_cluster(section);
            cluster.label = section;
            cluster.set_attr("style", "rounded").set_attr("color", "gray70");
            node.cluster = cluster.id;
        }
    }
    // Stages may re-parent their own node (a device draws itself as a region
    // inside its section), so this has to run before anything reads the layout.
    for (auto& stage : stages)
        stage.second->add_graph_details(graph);

    // Producer/consumer relations. Registration names are stage unique names, but
    // a stage may register a helper under another name (and a stage can be torn
    // down while its registration lives on), so give any unrecognised name a node
    // of its own rather than losing the edge.
    auto endpoint = [&graph](const std::string& name) -> std::string {
        if (graph.has_node(name))
            return name;
        // A stage may register under a name of its own making below its unique
        // name -- a GPU command is "<stage>/commands/<n>" -- and that name has no
        // node when the graph was asked not to draw the stage's internals. Hang
        // the edge on the nearest enclosing thing that *is* drawn.
        std::string prefix = name;
        while (true) {
            const size_t slash = prefix.find_last_of('/');
            if (slash == std::string::npos || slash == 0)
                break;
            prefix = prefix.substr(0, slash);
            if (graph.has_node(prefix))
                return prefix;
        }
        graph.add_node(name)
            .add_line(name)
            .set_category(GraphCategory::Compute)
            .set_attr("style", "rounded,dashed");
        return name;
    };
    for (auto& buf : buffer_container.get_buffer_map()) {
        // A buffer with one producer and one consumer is a link in a chain, and
        // the chain is the thing to keep straight; everything else can bend
        // around it.
        const bool in_chain =
            buf.second->producers.size() == 1 && buf.second->consumers.size() == 1;
        const std::string weight = in_chain ? "4" : "1";

        std::vector<std::string> touched_by;
        for (auto& cit : buf.second->consumers) {
            const std::string node = endpoint(cit.second.name);
            graph.add_edge(buf.first, node).set_attr("weight", weight);
            touched_by.push_back(node);
        }
        for (auto& pit : buf.second->producers) {
            const std::string node = endpoint(pit.second.name);
            graph.add_edge(node, buf.first).set_attr("weight", weight);
            touched_by.push_back(node);
        }
        // Draw the buffer wherever everything using it lives: private buffers
        // move inside the device or section that owns them, and only the genuine
        // hand-off points are left crossing a boundary.
        if (!touched_by.empty())
            graph.add_node(buf.first).cluster = graph.common_cluster(touched_by);
    }

    // The metadata pools, which no buffer node can show on its own: a pool is
    // shared, and its object size is what every frame of every buffer using it
    // carries. (There is no occupancy to report -- a pool hands out objects on
    // demand rather than holding a fixed set.)
    if (options.pools && !metadata_pools.empty()) {
        auto& pools = graph.add_cluster("__pools");
        pools.label = "metadata pools";
        pools.set_attr("style", "rounded").set_attr("color", "gray70");
        pools.set_attr("rank", "same");
        for (auto& pool : metadata_pools) {
            const std::string id = "__pool/" + pool.first;
            auto& node = graph.add_node(id);
            node.add_line(pool.first);
            if (pool.second) {
                node.add_line(pool.second->type_name);
                node.add_line(human_bytes(pool.second->metadata_object_size) + " per frame");
            }
            node.cluster = pools.id;
            node.set_category(GraphCategory::Buffer);
        }
    }

    if (options.legend)
        add_legend(graph);
    return graph;
}

/// Reads the graph options out of a request's query arguments.
static GraphOptions graph_options_from_query(connectionInstance& conn) {
    GraphOptions options;
    const std::map<std::string, std::string> query = conn.get_query();

    auto flag = [&query](const std::string& name, bool& value) {
        auto arg = query.find(name);
        if (arg == query.end())
            return;
        // Accept the forms people actually type, and treat a bare `?legend` as
        // asking for it.
        const std::string& text = arg->second;
        value = text.empty() || text == "1" || text == "true" || text == "yes" || text == "on";
    };
    flag("cluster", options.cluster);
    flag("legend", options.legend);
    flag("pools", options.pools);
    flag("kernels", options.kernels);
    flag("runtime", options.runtime);
    flag("urls", options.urls);

    auto rankdir = query.find("rankdir");
    if (rankdir != query.end()
        && (rankdir->second == "LR" || rankdir->second == "TB" || rankdir->second == "RL"
            || rankdir->second == "BT"))
        options.rankdir = rankdir->second;

    return options;
}

void kotekanMode::pipeline_dot_graph_callback(connectionInstance& conn) {
    conn.send_text_reply(get_pipeline_graph(graph_options_from_query(conn)).to_dot(),
                         "text/vnd.graphviz");
}

void kotekanMode::pipeline_json_graph_callback(connectionInstance& conn) {
    conn.send_json_reply(get_pipeline_graph(graph_options_from_query(conn)).to_json());
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
