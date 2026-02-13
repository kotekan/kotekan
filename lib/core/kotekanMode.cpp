#include "kotekanMode.hpp"

#include "Config.hpp"         // for Config
#include "Stage.hpp"          // for Stage
#include "StageFactory.hpp"   // for StageFactory
#include "Telescope.hpp"      // for Telescope
#include "buffer.h"           // for Buffer, StageInfo, get_num_full_frames, delete_buffer
#include "bufferFactory.hpp"  // for bufferFactory
#include "configUpdater.hpp"  // for configUpdater
#include "datasetManager.hpp" // for datasetManager
#include "kotekanLogging.hpp" // for INFO_NON_OO
#include "kotekanTrackers.hpp"
#include "metadata.h"            // for delete_metadata_pool
#include "metadataFactory.hpp"   // for metadataFactory
#include "prometheusMetrics.hpp" // for Metrics
#include "restServer.hpp"        // for restServer, connectionInstance

#include "fmt.hpp"  // for format
#include "json.hpp" // for basic_json<>::object_t, basic_json<>::value_type, json

#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, _Placeholder, bind, _1, placeholders
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error
#include <stdlib.h>   // for free
#include <utility>    // for pair
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
            delete_buffer(buf.second);
            free(buf.second);
        }
    }

    for (auto const& metadata_pool : metadata_pools) {
        if (metadata_pool.second != nullptr) {
            delete_metadata_pool(metadata_pool.second);
            free(metadata_pool.second);
        }
    }
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
        send_shutdown_signal(buf.second);
    }
}

nlohmann::json kotekanMode::get_buffer_json() {
    nlohmann::json buffer_json = {};

    for (auto& buf : buffer_container.get_buffer_map()) {
        nlohmann::json buf_info = {};
        buf_info["consumers"];
        for (int i = 0; i < MAX_CONSUMERS; ++i) {
            if (buf.second->consumers[i].in_use) {
                std::string consumer_name = buf.second->consumers[i].name;
                buf_info["consumers"][consumer_name] = {};
                buf_info["consumers"][consumer_name]["last_frame_acquired"] =
                    buf.second->consumers[i].last_frame_acquired;
                buf_info["consumers"][consumer_name]["last_frame_released"] =
                    buf.second->consumers[i].last_frame_released;
                for (int f = 0; f < buf.second->num_frames; ++f) {
                    buf_info["consumers"][consumer_name]["marked_frame_empty"].push_back(
                        buf.second->consumers_done[f][i]);
                }
            }
        }
        buf_info["producers"];
        for (int i = 0; i < MAX_PRODUCERS; ++i) {
            if (buf.second->producers[i].in_use) {
                std::string producer_name = buf.second->producers[i].name;
                buf_info["producers"][producer_name] = {};
                buf_info["producers"][producer_name]["last_frame_acquired"] =
                    buf.second->producers[i].last_frame_acquired;
                buf_info["producers"][producer_name]["last_frame_released"] =
                    buf.second->producers[i].last_frame_released;
                for (int f = 0; f < buf.second->num_frames; ++f) {
                    buf_info["producers"][producer_name]["marked_frame_empty"].push_back(
                        buf.second->producers_done[f][i]);
                }
            }
        }
        buf_info["frames"];
        for (int i = 0; i < buf.second->num_frames; ++i) {
            buf_info["frames"].push_back(buf.second->is_full[i]);
        }

        buf_info["num_full_frame"] = get_num_full_frames(buf.second);
        buf_info["num_frames"] = buf.second->num_frames;
        buf_info["frame_size"] = buf.second->frame_size;
        buf_info["last_frame_arrival_time"] = buf.second->last_arrival_time;
        buf_info["type"] = buf.second->buffer_type;

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
        dot += fmt::format(
            "{:s}\"{:s}\" [label=<{:s}<BR/>{:d}/{:d} ({:.1f}%)> shape=ellipse, color=blue];\n",
            prefix, buf.first, buf.first, get_num_full_frames(buf.second), buf.second->num_frames,
            (float)get_num_full_frames(buf.second) / buf.second->num_frames * 100);
    }

    // Setup stage nodes
    for (auto& stage : stages) {
        dot += stage.second->dot_string(prefix);
    }

    // Generate graph edges (producer/consumer relations)
    for (auto& buf : buffer_container.get_buffer_map()) {
        for (int i = 0; i < MAX_CONSUMERS; ++i) {
            if (buf.second->consumers[i].in_use) {
                dot += fmt::format("{:s}\"{:s}\" -> \"{:s}\";\n", prefix, buf.first,
                                   buf.second->consumers[i].name);
            }
        }
        for (int i = 0; i < MAX_PRODUCERS; ++i) {
            if (buf.second->producers[i].in_use) {
                dot += fmt::format("{:s}\"{:s}\" -> \"{:s}\";\n", prefix,
                                   buf.second->producers[i].name, buf.first);
            }
        }
    }

    dot += "}\n";
    conn.send_text_reply(dot);
}

} // namespace kotekan
