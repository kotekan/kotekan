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
#include <set>         // for set
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
        Buffer* buf = buffer_container.get_buffer(global_buffer_name);
        local_buffer_container.add_buffer(internal_name, buf);
    }

    json out_bufs = config.get_value(unique_name, "out_buffers");
    for (json::iterator it = out_bufs.begin(); it != out_bufs.end(); ++it) {
        std::string internal_name = it.key();
        std::string global_buffer_name = it.value();
        Buffer* buf = buffer_container.get_buffer(global_buffer_name);
        local_buffer_container.add_buffer(internal_name, buf);
    }
    INFO("GPU Process Starting...");
}

gpuProcess::~gpuProcess() {
    restServer::instance().remove_get_callback(fmt::format(fmt("/gpu_profile/{:d}"), gpu_id));
    for (auto& command : commands)
        for (auto& c : command)
            delete c;
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
        // The multiple gpuCommand instances share a StatsTracker object, so we only need
        // ask the first one for its stats.
        double time = cmd[0]->excute_time->get_avg(); //->get_last_gpu_execution_time();
        double utilization = time / frame_arrival_period;
        if (cmd[0]->get_command_type() == gpuCommandType::KERNEL) {
            reply["kernel"].push_back(
                {{"name", cmd[0]->get_name()}, {"time", time}, {"utilization", utilization}});
            total_kernel_time += isnan(time) ? 0. : time;
        } else if (cmd[0]->get_command_type() == gpuCommandType::COPY_IN) {
            reply["copy_in"].push_back(
                {{"name", cmd[0]->get_name()}, {"time", time}, {"utilization", utilization}});
            total_copy_in_time += isnan(time) ? 0. : time;
        } else if (cmd[0]->get_command_type() == gpuCommandType::COPY_OUT) {
            reply["copy_out"].push_back(
                {{"name", cmd[0]->get_name()}, {"time", time}, {"utilization", utilization}});
            total_copy_out_time += isnan(time) ? 0. : time;
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
    dev->set_thread_device();

    restServer& rest_server = restServer::instance();
    rest_server.register_get_callback(
        fmt::format(fmt("/gpu_profile/{:d}"), gpu_id),
        std::bind(&gpuProcess::profile_callback, this, std::placeholders::_1));

    // Start with the first GPU frame;
    int gpu_frame_counter = 0;
    bool first_run = true;

    while (!stop_thread) {

        for (auto& command : commands) {
            int ic = gpu_frame_counter % command.size();
            command[ic]->start_frame(gpu_frame_counter);
        }

        // Wait for all the required preconditions
        // This is things like waiting for the input buffer to have data
        // and for there to be free space in the output buffers.
        // INFO("Waiting on preconditions for GPU[{:d}][{:d}]", gpu_id, gpu_frame_id);
        for (auto& command : commands) {
            int ic = gpu_frame_counter % command.size();
            if (command[ic]->wait_on_precondition() != 0) {
                INFO("Received exit signal from GPU command precondition (Command '{:s}')",
                     command[ic]->get_name());
                goto exit_loop;
            }
        }

        DEBUG("Waiting for free slot for GPU[{:d}] frame {:d}", gpu_id, gpu_frame_counter);
        // We make sure we aren't using a gpu frame that's currently in-flight.
        int ic = gpu_frame_counter % final_signals.size();
        final_signals[ic]->wait_for_free_slot();
        queue_commands(gpu_frame_counter);
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

        gpu_frame_counter++;
    }
exit_loop:
    for (auto& sig_container : final_signals)
        sig_container->stop();
    INFO("Waiting for GPU packet queues to finish up before freeing memory.");
    if (results_thread_handle.joinable())
        results_thread_handle.join();
}


void gpuProcess::results_thread() {
    dev->set_thread_device();

    // Start with the first GPU frame;
    int gpu_frame_counter = 0;

    while (true) {
        // Wait for a signal to be completed
        DEBUG2("Waiting for signal for gpu[{:d}], frame {:d}, time: {:f}", gpu_id,
               gpu_frame_counter, e_time());
        int ic = gpu_frame_counter % final_signals.size();
        if (final_signals[ic]->wait_for_signal() == -1) {
            // If wait_for_signal returns -1, then we don't have a signal to wait on,
            // but we have been given a shutdown request, so break this loop.
            break;
        }
        DEBUG2("Got final signal for gpu[{:d}], frame {:d}, time: {:f}", gpu_id, gpu_frame_counter,
               e_time());

        for (auto& command : commands) {
            // Note the fact that we don't run `finalize_frame()` when the shutdown
            // signal is set, means that we cannot use it to free memory.
            // In theory this shouldn't be a problem, but it might be an issue for
            // some GPU APIs which require a memory clean up step after each run.
            // Two ways around this would be to have a different call for memory freeing
            // which is always called, or make sure that all finalize_frame calls can
            // run even when there is a shutdown in progress.
            if (!stop_thread) {
                ic = gpu_frame_counter % command.size();
                command[ic]->finalize_frame();
            }
        }
        DEBUG2("Finished finalizing frames for gpu[{:d}][{:d}]", gpu_id, gpu_frame_counter);

        if (log_profiling) {
            std::string output = "";
            for (size_t i = 0; i < commands.size(); ++i) {
                ic = gpu_frame_counter % commands[i].size();
                output = fmt::format(fmt("{:s}command: {:s} ({:30s}) metrics: {:s}; \n"), output,
                                     commands[i][ic]->get_unique_name(),
                                     commands[i][ic]->get_name(),
                                     commands[i][ic]->get_performance_metric_string());
            }
            INFO("GPU[{:d}] frame {:d} Profiling: \n{:s}", gpu_id, gpu_frame_counter, output);
        }

        ic = gpu_frame_counter % final_signals.size();
        final_signals[ic]->reset();
        gpu_frame_counter++;
    }
}

std::string gpuProcess::dot_string(const std::string& prefix) const {
    std::string dot = fmt::format("{:s}subgraph \"cluster_{:s}\" {{\n", prefix, get_unique_name());

    dot += fmt::format("{:s}{:s}style=filled;\n", prefix, prefix);
    dot += fmt::format("{:s}{:s}color=lightgrey;\n", prefix, prefix);
    dot += fmt::format("{:s}{:s}node [style=filled,color=white];\n", prefix, prefix);
    dot += fmt::format("{:s}{:s}label = \"{:s}\";\n", prefix, prefix, get_unique_name());

    // Draw a node for each gpuCommand
    for (auto& command : commands) {
        std::string shape;
        switch (command[0]->get_command_type()) {
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
                           command[0]->get_unique_name(), shape, command[0]->get_name());
    }

    // Draw edges between gpuCommands
    dot += fmt::format("{:s}{:s}// start gpu command edges\n", prefix, prefix);
    bool first_item = true;
    std::string last_item = "";
    for (auto& command : commands) {
        if (first_item) {
            last_item = command[0]->get_unique_name();
            first_item = false;
            continue;
        }
        dot += fmt::format("{:s}{:s}\"{:s}\" -> \"{:s}\" [style=dotted];\n", prefix, prefix,
                           last_item, command[0]->get_unique_name());
        last_item = command[0]->get_unique_name();
    }
    dot += fmt::format("{:s}{:s}// end gpu command edges\n", prefix, prefix);

    // Draw GPU buffers (non-array)
    std::set<std::string> gpu_buffers;
    std::set<std::string> gpu_buffer_arrays;
    for (auto& command : commands) {
        auto buffs = command[0]->get_gpu_buffers();
        for (auto& buff : buffs)
            if (std::get<1>(buff))
                gpu_buffer_arrays.insert(std::get<0>(buff));
            else
                gpu_buffers.insert(std::get<0>(buff));
    }
    dot += fmt::format("{:s}subgraph \"cluster_{:s}_mem\" {{\n", prefix, get_unique_name());
    for (std::string name : gpu_buffer_arrays) {
        // shape="box3d"
        dot += fmt::format("{:s}{:s}\"{:s}\" [shape=\"oval\",color=\"hotpink3\",label=\"{:s}\"];\n",
                           prefix, prefix, name, name);
    }

    for (std::string name : gpu_buffers) {
        // shape="rect"
        dot += fmt::format("{:s}{:s}\"{:s}\" [shape=\"oval\",color=\"hotpink\",label=\"{:s}\"];\n",
                           prefix, prefix, name, name);
    }
    dot += fmt::format("{:s} }}\n", prefix);

    // Draw I/O edges on GPU buffers
    for (auto& command : commands) {
        auto buffs = command[0]->get_gpu_buffers();
        for (auto& buff : buffs) {
            std::string buffname = std::get<0>(buff);
            if (std::get<2>(buff))
                // Read
                dot += fmt::format("{:s}{:s}\"{:s}\" -> \"{:s}\" [style=solid];\n", prefix, prefix,
                                   buffname, command[0]->get_unique_name());
            if (std::get<3>(buff))
                // Write
                dot += fmt::format("{:s}{:s}\"{:s}\" -> \"{:s}\" [style=solid];\n", prefix, prefix,
                                   command[0]->get_unique_name(), buffname);
        }
    }

    // Add any extra DOT commands...
    for (auto& command : commands) {
        dot += command[0]->get_extra_dot(prefix);
    }

    dot += fmt::format("{:s}}}\n", prefix);

    return dot;
}
