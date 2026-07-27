#include "gpuProcess.hpp"

#include "Config.hpp"             // for Config
#include "PipelineGraph.hpp"      // for PipelineGraph, GraphNode
#include "gpuCommand.hpp"         // for gpuCommand, gpuCommandType
#include "gpuDeviceInterface.hpp" // for gpuDeviceInterface
#include "gpuEventContainer.hpp"  // for gpuEventContainer
#include "kotekanLogging.hpp"     // for DEBUG2, INFO
#include "restServer.hpp"         // for restServer, connectionInstance
#include "util.h"                 // for e_time
#include "visUtil.hpp"            // for StatTracker

#include "fmt.hpp"  // for format, compile_string_to_view, format_string, fmt
#include "json.hpp" // for json_ref, basic_json, json, iter_impl

#include <assert.h>    // for assert
#include <cmath>       // for isnan
#include <functional>  // for bind, ref, function, _1
#include <map>         // for operator!=, map, _Rb_tree_const_iterator, _Rb_tree_ite...
#include <memory>      // for __shared_ptr_access, shared_ptr
#include <pthread.h>   // for pthread_setaffinity_np
#include <sched.h>     // for cpu_set_t, CPU_SET, CPU_ZERO
#include <set>         // for set
#include <sstream>     // for basic_ostringstream, basic_ostream, ostringstream
#include <sys/types.h> // for uint
#include <tuple>       // for get, tuple
#include <utility>     // for pair

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
        GenericBuffer* buf = buffer_container.get_generic_buffer(global_buffer_name);
        local_buffer_container.add_buffer(internal_name, buf);
    }

    json out_bufs = config.get_value(unique_name, "out_buffers");
    for (json::iterator it = out_bufs.begin(); it != out_bufs.end(); ++it) {
        std::string internal_name = it.key();
        std::string global_buffer_name = it.value();
        GenericBuffer* buf = buffer_container.get_generic_buffer(global_buffer_name);
        local_buffer_container.add_buffer(internal_name, buf);
    }
    INFO("GPU Process Starting...");
}

gpuProcess::~gpuProcess() {
    // unique_name starts with "/", matching the path registered in main_thread()
    restServer::instance().remove_get_callback(fmt::format(fmt("/gpu_profile{:s}"), unique_name));
    for (auto& command : commands)
        for (auto& c : command)
            delete c;
    for (auto& event : final_signals)
        delete event;
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
        if (is_frame_buffer(buf.second))
            register_host_memory(dynamic_cast<Buffer*>(buf.second));
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
    // unique_name starts with "/", so this path becomes something like
    // "/gpu_profile/gpuB/gpu_0" for pipeline B running on GPU 0.
    rest_server.register_get_callback(
        fmt::format(fmt("/gpu_profile{:s}"), unique_name),
        std::bind(&gpuProcess::profile_callback, this, std::placeholders::_1));

    // Start with the first GPU frame;
    int gpu_frame_counter = 0;
    bool first_run = true;

    while (!stop_thread) {
        const int ic = gpu_frame_counter % final_signals.size();

        DEBUG2("Waiting for free slot for GPU[{:d}][{:d}] {:s}", gpu_id, gpu_frame_counter,
               unique_name);

        // We make sure we aren't using a gpu frame that's currently in-flight.
        final_signals[ic]->wait_for_free_slot();

        // Update the gpu_frame_counter and perform any reset actions on the command object
        // for this frame.

        for (auto& command : commands) {
            assert(command.size() == final_signals.size());
            command[ic]->start_frame(gpu_frame_counter);
        }

        // Wait for all the required preconditions
        // This is things like waiting for the input buffer to have data
        // and for there to be free space in the output buffers.
        DEBUG2("Waiting on preconditions for GPU[{:d}][{:d}] {:s}", gpu_id, gpu_frame_counter,
               unique_name);
        for (auto& command : commands) {
            if (command[ic]->wait_on_precondition() != 0) {
                INFO("Received exit signal from GPU command precondition (Command '{:s}')",
                     command[ic]->get_name());
                goto exit_loop;
            }
        }

        DEBUG2("Preconditions met for GPU[{:d}][{:d}] {:s}, queuing commands", gpu_id,
               gpu_frame_counter, unique_name);
        // Queue the commands for this frame.  This calls execute on each commandObject.
        queue_commands(gpu_frame_counter);

        // Launch the results thread if it hasn't been launched yet.
        if (first_run) {
            results_thread_handle = std::thread(&gpuProcess::results_thread, std::ref(*this));

            // Set the CPU affinity for the results thread, uses the "cpu_affinity" from the
            // gpuProcess config.
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
        const int ic = gpu_frame_counter % final_signals.size();
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
                assert(command.size() == final_signals.size());
                command[ic]->finalize_frame();
            }
        }
        DEBUG2("Finished finalizing frames for gpu[{:d}][{:d}]", gpu_id, gpu_frame_counter);

        if (log_profiling) {
            std::ostringstream output;
            for (auto& command : commands) {
                assert(command.size() == final_signals.size());
                output << fmt::format(fmt("command: {:s} ({:30s}) metrics: {:s}; \n"),
                                      command[ic]->get_unique_name(), command[ic]->get_name(),
                                      command[ic]->get_performance_metric_string());
            }
            INFO("GPU[{:d}] frame {:d} Profiling: \n{:s}", gpu_id, gpu_frame_counter, output.str());
        }

        DEBUG2("Resetting signal for gpu[{:d}][{:d}]", gpu_id, gpu_frame_counter);
        final_signals[ic]->reset();
        gpu_frame_counter++;
    }
}

std::string gpuProcess::gpu_mem_node_prefix(const std::string& stage_name) {
    return fmt::format("{:s}/mem/", stage_name);
}

void gpuProcess::add_graph_details(kotekan::PipelineGraph& graph) const {
    const std::string name = get_unique_name();

    // The device region, holding this stage's commands and GPU memory.
    auto& stage_node = graph.add_node(name);
    auto& device = graph.add_cluster(name);
    device.label = fmt::format(fmt("GPU {:d}"), gpu_id);
    device.set_attr("style", "rounded,filled")
        .set_attr("fillcolor", kotekan::graph_device_fill)
        .set_attr("color", "gray60");
    // Keep whatever grouping the pipeline put this stage in as the region's own
    // parent, so a device sits inside its config section rather than beside it.
    device.parent = stage_node.cluster;

    // The stage node itself belongs in the region: the host buffer edges are
    // added centrally and point at the stage, so without a node inside the box
    // they would end on an empty one drawn beside it.
    stage_node.cluster = device.id;

    // A node per gpuCommand, chained in execution order after the stage node.
    // Only the first instance of each command is drawn; the others are the same
    // step of the pipeline operating on another frame.
    std::string previous = name;
    for (auto& command : commands) {
        std::string shape;
        std::string kind;
        switch (command[0]->get_command_type()) {
            case gpuCommandType::COPY_IN:
                shape = "trapezium";
                kind = "copy in";
                break;
            case gpuCommandType::KERNEL:
                shape = "box";
                kind = "kernel";
                break;
            case gpuCommandType::BARRIER:
                shape = "parallelogram";
                kind = "barrier";
                break;
            case gpuCommandType::COPY_OUT:
                shape = "invtrapezium";
                kind = "copy out";
                break;
            default:
                // Hopefully one notices the type wasn't set with this shape.
                shape = "diamond";
                kind = "type not set";
                break;
        }
        const std::string id = command[0]->get_unique_name();
        auto& node = graph.add_node(id);
        node.add_line(command[0]->get_name());
        node.add_line(kind);
        node.cluster = device.id;
        node.set_category(kotekan::GraphCategory::Gpu)
            .set_attr("shape", shape)
            .set_attr("style", "filled");
        graph.add_edge(previous, id).set_attr("style", "dotted");
        previous = id;
    }

    // GPU memory, in a region of its own inside the device.
    auto& mem = graph.add_cluster(fmt::format("{:s}/mem", name));
    mem.parent = device.id;
    mem.label = "device memory";
    mem.set_attr("style", "rounded").set_attr("color", "gray70");
    // GPU memory names are local to a gpuProcess ("voltage" on one device is not
    // the "voltage" on the next), so the node ids carry the stage they belong to.
    const std::string mem_prefix = gpu_mem_node_prefix(name);
    std::set<std::string> gpu_buffers;
    std::set<std::string> gpu_buffer_arrays;
    for (auto& command : commands) {
        for (auto& buff : command[0]->get_gpu_buffers()) {
            if (std::get<1>(buff))
                gpu_buffer_arrays.insert(std::get<0>(buff));
            else
                gpu_buffers.insert(std::get<0>(buff));
        }
    }
    // Arrays are per-frame (one region per buffer_depth slot); the rest is a
    // single region shared by every frame in flight.
    for (const auto& buffer_name : gpu_buffer_arrays) {
        auto& node = graph.add_node(mem_prefix + buffer_name);
        node.add_line(buffer_name);
        node.add_line(fmt::format(fmt("array ×{:d}"), _gpu_buffer_depth));
        node.cluster = mem.id;
        node.set_category(kotekan::GraphCategory::Memory);
    }
    for (const auto& buffer_name : gpu_buffers) {
        auto& node = graph.add_node(mem_prefix + buffer_name);
        node.add_line(buffer_name);
        node.cluster = mem.id;
        node.set_category(kotekan::GraphCategory::Memory);
    }

    // Which commands read and write which GPU memory.
    for (auto& command : commands) {
        for (auto& buff : command[0]->get_gpu_buffers()) {
            const std::string buffer_id = mem_prefix + std::get<0>(buff);
            if (std::get<2>(buff)) // read
                graph.add_edge(buffer_id, command[0]->get_unique_name()).set_attr("style", "solid");
            if (std::get<3>(buff)) // write
                graph.add_edge(command[0]->get_unique_name(), buffer_id).set_attr("style", "solid");
        }
    }

    // Anything else a command wants to say about its GPU memory.
    for (auto& command : commands)
        command[0]->add_graph_details(graph, mem_prefix);
}
