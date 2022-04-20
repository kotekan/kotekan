#include "gpuProcess.hpp"

#include "Config.hpp"             // for Config
#include "gpuCommand.hpp"         // for gpuCommand, gpuCommandType, gpuCommandType::COPY_IN
#include "gpuDeviceInterface.hpp" // for gpuDeviceInterface, Config
#include "gpuEventContainer.hpp"  // for gpuEventContainer
#include "kotekanLogging.hpp"     // for INFO, DEBUG2, DEBUG
#include "restServer.hpp"         // for restServer, connectionInstance
#include "util.h"                 // for e_time

#include "fmt.hpp"  // for format, fmt
#include "json.hpp" // for json, basic_json<>::object_t, basic_json<>::value_type

#include <algorithm>   // for max
#include <atomic>      // for atomic_bool
#include <exception>   // for exception
#include <functional>  // for _Bind_helper<>::type, _Placeholder, bind, ref, _1, fun...
#include <iosfwd>      // for std
#include <pthread.h>   // for pthread_setaffinity_np
#include <regex>       // for match_results<>::_Base_type
#include <sched.h>     // for cpu_set_t, CPU_SET, CPU_ZERO
#include <stdexcept>   // for runtime_error
#include <sys/types.h> // for uint

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

using kotekan::connectionInstance;
using kotekan::restServer;

using namespace std;

using nlohmann::json;

// TODO Remove the GPU_ID from this constructor
gpuProcess::gpuProcess(Config& config_, const std::string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container, std::bind(&gpuProcess::main_thread, this)) {
    log_profiling = config.get_default<bool>(unique_name, "log_profiling", false);

    _gpu_buffer_depth = config.get<int>(unique_name, "buffer_depth");
    gpu_id = config.get<int>(unique_name, "gpu_id");

    frame_arrival_period = config.get_default<double>(unique_name, "frame_arrival_period", 0.0);

    json in_bufs = config.get_value(unique_name, "in_buffers");
    for (json::iterator it = in_bufs.begin(); it != in_bufs.end(); ++it) {
        std::string internal_name = it.key();
        std::string global_buffer_name = it.value();
        struct Buffer* buf = buffer_container.get_buffer(global_buffer_name);
        local_buffer_container.add_buffer(internal_name, buf);
    }

    json out_bufs = config.get_value(unique_name, "out_buffers");
    for (json::iterator it = out_bufs.begin(); it != out_bufs.end(); ++it) {
        std::string internal_name = it.key();
        std::string global_buffer_name = it.value();
        struct Buffer* buf = buffer_container.get_buffer(global_buffer_name);
        local_buffer_container.add_buffer(internal_name, buf);
    }
    INFO("GPU Process Starting...");
}

gpuProcess::~gpuProcess() {
    restServer::instance().remove_get_callback(fmt::format(fmt("/gpu_profile/{:d}"), gpu_id));
    for (auto& command : commands)
        delete command;
    for (auto& event : final_signals)
        delete event;

    delete dev;
}

void gpuProcess::init() {
    for (uint i = 0; i < _gpu_buffer_depth; i++) {
        final_signals.push_back(create_signal());
    }

    std::string g_log_level = config.get<string>(unique_name, "log_level");
    std::string s_log_level =
        config.get_default<string>(unique_name, "device_interface_log_level", g_log_level);
    dev->set_log_level(s_log_level);
    dev->set_log_prefix(fmt::format(fmt("GPU[{:d}] device interface"), gpu_id));

    vector<json> cmds = config.get<std::vector<json>>(unique_name, "commands");
    int i = 0;
    for (json cmd : cmds) {
        std::string unique_path = fmt::format(fmt("{:s}/commands/{:d}"), unique_name, i++);
        std::string command_name = cmd["name"];
        commands.push_back(create_command(command_name, unique_path));
    }

    for (auto& buf : local_buffer_container.get_buffer_map()) {
        register_host_memory(buf.second);
    }
}

void gpuProcess::profile_callback(connectionInstance& conn) {
    json reply;

    reply["copy_in"] = json::array();
    reply["kernel"] = json::array();
    reply["copy_out"] = json::array();

    double total_copy_in_time = 0;
    double total_copy_out_time = 0;
    double total_kernel_time = 0;

    for (auto& cmd : commands) {
        double time = cmd->get_last_gpu_execution_time();
        double utilization = time / frame_arrival_period;
        if (cmd->get_command_type() == gpuCommandType::KERNEL) {
            reply["kernel"].push_back(
                {{"name", cmd->get_name()}, {"time", time}, {"utilization", utilization}});
            total_kernel_time += cmd->get_last_gpu_execution_time();
        } else if (cmd->get_command_type() == gpuCommandType::COPY_IN) {
            reply["copy_in"].push_back(
                {{"name", cmd->get_name()}, {"time", time}, {"utilization", utilization}});
            total_copy_in_time += cmd->get_last_gpu_execution_time();
        } else if (cmd->get_command_type() == gpuCommandType::COPY_OUT) {
            reply["copy_out"].push_back(
                {{"name", cmd->get_name()}, {"time", time}, {"utilization", utilization}});
            total_copy_out_time += cmd->get_last_gpu_execution_time();
        } else {
            continue;
        }
    }

    reply["copy_in_total_time"] = total_copy_in_time;
    reply["kernel_total_time"] = total_kernel_time;
    reply["copy_out_total_time"] = total_copy_out_time;
    reply["copy_in_utilization"] = total_copy_in_time / frame_arrival_period;
    reply["kernel_utilization"] = total_kernel_time / frame_arrival_period;
    reply["copy_out_utilization"] = total_copy_out_time / frame_arrival_period;

    conn.send_json_reply(reply);
}


void gpuProcess::main_thread() {
    restServer& rest_server = restServer::instance();
    rest_server.register_get_callback(
        fmt::format(fmt("/gpu_profile/{:d}"), gpu_id),
        std::bind(&gpuProcess::profile_callback, this, std::placeholders::_1));

    // Start with the first GPU frame;
    int gpu_frame_id = 0;
    bool first_run = true;

    while (!stop_thread) {
        // Wait for all the required preconditions
        // This is things like waiting for the input buffer to have data
        // and for there to be free space in the output buffers.
        // INFO("Waiting on preconditions for GPU[{:d}][{:d}]", gpu_id, gpu_frame_id);
        for (auto& command : commands) {
            if (command->wait_on_precondition(gpu_frame_id) != 0) {
                INFO("Received exit signal from GPU command precondition (Command '{:s}')",
                     command->get_name());
                goto exit_loop;
            }
        }

        DEBUG("Waiting for free slot for GPU[{:d}][{:d}]", gpu_id, gpu_frame_id);
        // We make sure we aren't using a gpu frame that's currently in-flight.
        final_signals[gpu_frame_id]->wait_for_free_slot();
        queue_commands(gpu_frame_id);
        if (first_run) {
            results_thread_handle = std::thread(&gpuProcess::results_thread, std::ref(*this));

            // Requires Linux, this could possibly be made more general someday.
            // TODO Move to config
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            for (auto& i : config.get<std::vector<int>>(unique_name, "cpu_affinity"))
                CPU_SET(i, &cpuset);
            pthread_setaffinity_np(results_thread_handle.native_handle(), sizeof(cpu_set_t),
                                   &cpuset);
            first_run = false;
        }

        gpu_frame_id = (gpu_frame_id + 1) % _gpu_buffer_depth;
    }
exit_loop:
    for (auto& sig_container : final_signals) {
        sig_container->stop();
    }
    INFO("Waiting for GPU packet queues to finish up before freeing memory.");
    if (results_thread_handle.joinable())
        results_thread_handle.join();
}


void gpuProcess::results_thread() {
    // Start with the first GPU frame;
    int gpu_frame_id = 0;

    while (true) {
        // Wait for a signal to be completed
        DEBUG2("Waiting for signal for gpu[{:d}], frame {:d}, time: {:f}", gpu_id, gpu_frame_id,
               e_time());
        if (final_signals[gpu_frame_id]->wait_for_signal() == -1) {
            // If wait_for_signal returns -1, then we don't have a signal to wait on,
            // but we have been given a shutdown request, so break this loop.
            break;
        }
        DEBUG2("Got final signal for gpu[{:d}], frame {:d}, time: {:f}", gpu_id, gpu_frame_id,
               e_time());

        for (auto& command : commands) {
            // Note the fact that we don't run `finalize_frame()` when the shutdown
            // signal is set, means that we cannot use it to free memory.
            // In theory this shouldn't be a problem, but it might be an issue for
            // some GPU APIs which require a memory clean up step after each run.
            // Two ways around this would be to have a different call for memory freeing
            // which is always called, or make sure that all finalize_frame calls can
            // run even when there is a shutdown in progress.
            if (!stop_thread)
                command->finalize_frame(gpu_frame_id);
        }
        DEBUG2("Finished finalizing frames for gpu[{:d}][{:d}]", gpu_id, gpu_frame_id);

        if (log_profiling) {
            std::string output = "";
            for (uint32_t i = 0; i < commands.size(); ++i) {
                output = fmt::format(fmt("{:s}command: {:s} metrics: {:s}; \n"), output,
                                     commands[i]->get_unique_name(),
                                     commands[i]->get_performance_metric_string());
            }
            INFO("GPU[{:d}] Profiling: \n{:s}", gpu_id, output);
        }

        final_signals[gpu_frame_id]->reset();

        gpu_frame_id = (gpu_frame_id + 1) % _gpu_buffer_depth;
    }
}

std::string gpuProcess::dot_string(const std::string& prefix) const {
    std::string dot = fmt::format("{:s}subgraph \"cluster_{:s}\" {{\n", prefix, get_unique_name());

    dot += fmt::format("{:s}{:s}style=filled;\n", prefix, prefix);
    dot += fmt::format("{:s}{:s}color=lightgrey;\n", prefix, prefix);
    dot += fmt::format("{:s}{:s}node [style=filled,color=white];\n", prefix, prefix);
    dot += fmt::format("{:s}{:s}label = \"{:s}\";\n", prefix, prefix, get_unique_name());

    for (auto& command : commands) {
        std::string shape;
        switch (command->get_command_type()) {
            case gpuCommandType::COPY_IN:
                shape = "trapezium";
                break;
            case gpuCommandType::KERNEL:
                shape = "box";
                break;
            case gpuCommandType::BARRIER:
                shape = "parallelogram";
                break;
            case gpuCommandType::COPY_OUT:
                shape = "invtrapezium";
                break;
            default:
                // Hopefully one notices the type wasn't set with this shape.
                shape = "diamond";
                break;
        }
        dot += fmt::format("{:s}{:s}\"{:s}\" [shape={:s},label=\"{:s}\"];\n", prefix, prefix,
                           command->get_unique_name(), shape, command->get_name());
    }

    bool first_item = true;
    std::string last_item = "";
    for (auto& command : commands) {
        if (first_item) {
            last_item = command->get_unique_name();
            first_item = false;
            continue;
        }
        dot += fmt::format("{:s}{:s}\"{:s}\" -> \"{:s}\" [style=dotted];\n", prefix, prefix,
                           last_item, command->get_unique_name());
        last_item = command->get_unique_name();
    }

    dot += fmt::format("{:s}}}\n", prefix);

    return dot;
}