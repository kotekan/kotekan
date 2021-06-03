#include "kotekanMode.hpp"

#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "StageFactory.hpp"      // for StageFactory
#include "Telescope.hpp"         // for Telescope
#include "buffer.h"              // for Buffer, StageInfo, get_num_full_frames, delete_buffer
#include "bufferFactory.hpp"     // for bufferFactory
#include "configUpdater.hpp"     // for configUpdater
#include "datasetManager.hpp"    // for datasetManager
#include "kotekanLogging.hpp"    // for INFO_NON_OO
#include "metadata.h"            // for delete_metadata_pool
#include "metadataFactory.hpp"   // for metadataFactory
#include "prometheusMetrics.hpp" // for Metrics
#include "restServer.hpp"        // for restServer, connectionInstance

#include "fmt.hpp"  // for format
#include "json.hpp" // for basic_json<>::object_t, basic_json<>::value_type, json

#include <functional> // for _Bind_helper<>::type, _Placeholder, bind, _1, placeholders
#include <stdlib.h>   // for free
#include <utility>    // for pair
#include <pthread.h>
#include <fstream>

using namespace std::placeholders;
using namespace std::chrono_literals;

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

    // Create Config Updater
    configUpdater& config_updater = configUpdater::instance();
    config_updater.apply_config(config);

    // Apply config to datasetManager
    if (config.exists("/", "dataset_manager"))
        datasetManager::instance(config);

    // Apply config for Telescope class
    Telescope::instance(config);

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

void kotekanMode::buffer_data_callback(connectionInstance& conn) {
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

    conn.send_json_reply(buffer_json);
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

CpuMonitor::CpuMonitor() {};

void CpuMonitor::start() {
    pthread_t pid;
    pthread_create(&pid, NULL, CpuMonitor::track_cpu, NULL);
    pthread_detach(pid);

    ERROR_NON_OO("After pthread_creat");

    // Register CPU usage callback
    restServer::instance().register_get_callback(
        "/cpu_ult", std::bind(&CpuMonitor::cpu_ult_call_back, this, _1));
}

void* CpuMonitor::track_cpu(void *) {
    ERROR_NON_OO("entered track_cpu");

    // Read total CPU stat from /proc/stat first line
    std::string stat;
    std::ifstream cpu_stat("./proc/stat", std::ifstream::in);
    getline(cpu_stat, stat);
    std::istringstream iss(stat);

    // Parse and get total cpu time
    char prefix[10];
    iss >> prefix;
    uint32_t num = 0;
    uint32_t cpu_time = 0;
    for (int i = 0; i < 10; i++) {
        iss >> num;
        cpu_time += num;
    }

    // Read CPU stat for each thread
    for(auto element : thread_list) {
        char fname[100];
        snprintf(fname, sizeof(fname), "/proc/self/task/%d/stat", element.second);
        FILE *fp = fopen(fname, "r");

        if (fp) {
            // Get the 14th (utime) and the 15th (stime) numbers
            uint32_t utime = 0, stime = 0;
            fscanf(fp, "%*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %d %d", &utime, &stime);

            auto itr = ult_list.find(element.first);
            if (itr != ult_list.end()) {
                // Compute usr and sys CPU usage
                itr->second.utime_usage =
                    100 * (utime - itr->second.prev_utime) / (cpu_time - prev_cpu_time);
                itr->second.stime_usage =
                    100 * (stime - itr->second.prev_stime) / (cpu_time - prev_cpu_time);
            }

            // Update thread usr and sys time
            itr->second.prev_utime = utime;
            itr->second.prev_stime = stime;
        }
    }
    prev_cpu_time = cpu_time;

    // Check each stage periodically
    std::this_thread::sleep_for(2000ms);
}

void CpuMonitor::cpu_ult_call_back(connectionInstance& conn) {
    nlohmann::json cpu_ult_json = {};

    for (auto element : ult_list) {
        nlohmann::json thread_cpu_ult = {};
        thread_cpu_ult["usr_cpu_ult"] = element.second.utime_usage;
        thread_cpu_ult["sys_cpu_ult"] = element.second.stime_usage;

        cpu_ult_json[element.first] = thread_cpu_ult;
    }

    conn.send_json_reply(cpu_ult_json);
}


} // namespace kotekan
