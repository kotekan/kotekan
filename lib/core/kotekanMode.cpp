#include "kotekanMode.hpp"

#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "StageFactory.hpp"      // for StageFactory
#include "Telescope.hpp"         // for Telescope
#include "buffer.hpp"            // for StageInfo, GenericBuffer
#include "bufferFactory.hpp"     // for bufferFactory
#include "configTracker.hpp"     // for ConfigTracker
#include "configUpdater.hpp"     // for configUpdater
#include "datasetManager.hpp"    // for datasetManager
#include "kotekanLogging.hpp"    // for INFO_NON_OO, ERROR_NON_OO
#include "kotekanTrackers.hpp"   // for KotekanTrackers
#include "metadataFactory.hpp"   // for metadataFactory
#include "prometheusMetrics.hpp" // for Metrics
#include "restServer.hpp"        // for restServer, connectionInstance
#include "version.h"             // for get_cmake_build_options, get_git_branch, get_git_commit...

#include "fmt.hpp"  // for compile_string_to_view, format, format_string
#include "json.hpp" // for json, basic_json

#include <exception>  // for exception
#include <functional> // for bind, function, _1
#include <string>     // for string
#include <utility>    // for pair

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
}

kotekanMode::~kotekanMode() {

    configUpdater::instance().reset();
    restServer::instance().remove_get_callback("/config");
    restServer::instance().remove_get_callback("/buffers");
    restServer::instance().remove_get_callback("/pipeline_dot");
    restServer::instance().remove_all_aliases();

    KotekanTrackers::instance().set_kotekan_mode_ptr(nullptr);

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

    // ConfigTracker setup. The optional /config_tracker block (an object)
    // configures the tracker; defaults apply when absent.
    const std::string ct_path = "/config_tracker";
    bool ct_enabled = true;
    if (config.exists("/", "config_tracker")) {
        const nlohmann::json ct_node = config.get_value("/", "config_tracker");
        if (!ct_node.is_object()) {
            FATAL_ERROR_NON_OO("kotekanMode: /config_tracker must be an object (e.g. {{enabled: "
                               "true}}); the legacy bare-bool form is no longer supported.");
        }
        ct_enabled = config.get_default<bool>(ct_path, "enabled", true);
    }

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

        // HTTP retry/timeout policy. Applies to every upstream fetch (both
        // the one-shot FPGA fetch below and every getUpstreamConfigs call).
        // FATAL after retries are exhausted.
        const int upstream_retries = config.get_default<int>(ct_path, "upstream_fetch_retries", 2);
        const int upstream_timeout =
            config.get_default<int>(ct_path, "upstream_fetch_timeout_seconds", 10);
        ConfigTracker::instance().setUpstreamFetchPolicy(upstream_retries, upstream_timeout);

        // Optionally fetch the upstream FPGA controller's config + timing.
        // If fpga_host_info is set, both endpoints must succeed at startup
        // (otherwise ConfigTracker FATALs).
        if (config.exists(ct_path, "fpga_host_info")) {
            const std::string fpga_ref = config.get<std::string>(ct_path, "fpga_host_info");
            const std::string host = config.get<std::string>(fpga_ref, "host");
            const int port_int = config.get<int>(fpga_ref, "port");
            if (port_int <= 0 || port_int > 65535) {
                FATAL_ERROR_NON_OO("kotekanMode: invalid {}.port: {}", fpga_ref, port_int);
            }
            // The endpoint paths live on the controller block itself so the
            // Telescope (which reads ``timing_endpoint`` from the same block
            // via ``gps_host_info``) and the ConfigTracker share one source.
            const std::string config_endpoint =
                config.get_default<std::string>(fpga_ref, "config_endpoint", "/config");
            const std::string timing_endpoint =
                config.get_default<std::string>(fpga_ref, "timing_endpoint", "/get-frame0-time");

            ConfigTracker::instance().fetchAndRegisterFpgaTracking(
                host, static_cast<uint16_t>(port_int), config_endpoint, timing_endpoint);
        }
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

void kotekanMode::pipeline_dot_graph_callback(connectionInstance& conn) {
    const std::string prefix = "    ";
    std::string dot =
        "# This is a DOT formatted pipeline graph, use the graphviz package to plot.\n";
    dot += "digraph pipeline {\n";

    // Setup buffer nodes
    for (auto& buf : buffer_container.get_buffer_map()) {
        std::string label = buf.second->get_dot_node_label();
        dot += fmt::format("{:s}\"{:s}\" [label=<{:s}> shape=ellipse, color=blue];\n", prefix,
                           buf.first, label);
    }

    // Setup stage nodes
    for (auto& stage : stages) {
        dot += stage.second->dot_string(prefix);
    }

    // Generate graph edges (producer/consumer relations)
    for (auto& buf : buffer_container.get_buffer_map()) {
        for (auto& cit : buf.second->consumers)
            dot += fmt::format("{:s}\"{:s}\" -> \"{:s}\";\n", prefix, buf.first, cit.second.name);
        for (auto& pit : buf.second->producers)
            dot += fmt::format("{:s}\"{:s}\" -> \"{:s}\";\n", prefix, pit.second.name, buf.first);
    }

    dot += "}\n";
    conn.send_text_reply(dot);
}

} // namespace kotekan
