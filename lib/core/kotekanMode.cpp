#include "kotekanMode.hpp"

#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "StageFactory.hpp"      // for StageFactory
#include "buffer.h"              // for delete_buffer, send_shutdown_signal
#include "bufferFactory.hpp"     // for bufferFactory
#include "configUpdater.hpp"     // for configUpdater
#include "datasetManager.hpp"    // for datasetManager
#include "kotekanLogging.hpp"    // for INFO_NON_OO
#include "metadata.h"            // for delete_metadata_pool
#include "metadataFactory.hpp"   // for metadataFactory
#include "prometheusMetrics.hpp" // for Metrics
#include "restServer.hpp"        // for restServer, connectionInstance

#include "json.hpp" // for basic_json<>::object_t, json, basic_json<>::value_type

#include <stdlib.h> // for free
#include <utility>  // for pair

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

    // Create Metadata Pool
    metadataFactory metadata_factory(config);
    metadata_pools = metadata_factory.build_pools();

    // Create Config Updater
    configUpdater& config_updater = configUpdater::instance();
    config_updater.apply_config(config);

    // Create Buffers
    bufferFactory buffer_factory(config, metadata_pools);
    buffers = buffer_factory.build_buffers();
    buffer_container.set_buffer_map(buffers);

    // Apply config to datasetManager
    if (config.exists("/", "dataset_manager"))
        datasetManager::instance(config);

    // Create Stages
    StageFactory stage_factory(config, buffer_container);
    stages = stage_factory.build_stages();


    restServer::instance().register_get_callback("/buffers", [&](connectionInstance& conn) {
        nlohmann::json buffer_json = {};

        for (auto& buf : buffer_container.get_buffer_map()) {
            json buf_info = {};
            buf_info["consumers"];
            for (int i = 0; i < MAX_CONSUMERS; ++i) {
                if (buf.second->consumers[i].in_use) {
                    buf_info["consumers"].push_back(buf.second->consumers[i].name);
                }
            }
            buf_info["producers"];
            for (int i = 0; i < MAX_PRODUCERS; ++i) {
                if (buf.second->producers[i].in_use) {
                    buf_info["producers"].push_back(buf.second->producers[i].name);
                }
            }
            buf_info["frames"];
            for (int i = 0; i < buf.second->num_frames; ++i) {
                buf_info["frames"].push_back(buf.second->is_full[i]);
            }

            buffer_json[buf.first] = buf_info;
        }

        conn.send_json_reply(buffer_json);
    });

    restServer::instance().register_get_callback("/pipeline_dot", [&](connectionInstance& conn) {
        string dot = "# This is a DOT formated pipeline graph, use the graphviz package to plot.\n";
        dot += "digraph pipeline {\n";

        // Setup buffer nodes
        for (auto& buf : buffer_container.get_buffer_map()) {
            dot += "    \"" + buf.first + "\" [shape=doubleoctagon, color=blue];\n";
        }

        // Setup stage nodes
        for (auto& stage : stages) {
            dot += stage.second->dot_string("    ");
        }

        // Generate graph edges (producer/consumer relations)
        for (auto& buf : buffer_container.get_buffer_map()) {
            for (int i = 0; i < MAX_CONSUMERS; ++i) {
                if (buf.second->consumers[i].in_use) {
                    dot += "    \"" + buf.first + "\" -> \"" + string(buf.second->consumers[i].name)
                           + "\";\n";
                }
            }
            for (int i = 0; i < MAX_PRODUCERS; ++i) {
                if (buf.second->producers[i].in_use) {
                    dot += "    \"" + string(buf.second->producers[i].name) + "\" -> \"" + buf.first
                           + "\";\n";
                }
            }
        }

        dot += "}\n";
        conn.send_text_reply(dot);
    });


    // Update REST server
    restServer::instance().set_server_affinity(config);
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
}

void kotekanMode::stop_stages() {
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

} // namespace kotekan
