#include "clProcess.hpp"
#include "unistd.h"
#include "util.h"

#include <iostream>
#include <sys/time.h>

using namespace std;

REGISTER_KOTEKAN_PROCESS(clProcess);

// TODO Remove the GPU_ID from this constructor
clProcess::clProcess(Config& config_,
        const string& unique_name,
        bufferContainer &buffer_container):
    gpuProcess(config_, unique_name, buffer_container)
{
//    final_signals.resize(_gpu_buffer_depth);
    device = new clDeviceInterface(config_, gpu_id, _gpu_buffer_depth);
    ((clDeviceInterface*)device)->prepareCommandQueue(true); //yes profiling
    init();
}

clProcess::~clProcess() {
}

gpuEventContainer *clProcess::create_signal(){
    return new clEventContainer();
}

gpuCommand *clProcess::create_command(json cmd_info){
    auto cmd = FACTORY(clCommand)::create_bare(cmd_info["name"], config,
                                               unique_name, local_buffer_container,
                                               *((clDeviceInterface*)device));
    cmd->build();
    std::string name = cmd_info["name"];
    DEBUG("Command added: %s",name.c_str());
    return cmd;
}

void clProcess::profile_callback(connectionInstance& conn) {
    DEBUG(" *** *** *** Profile call made.");

    double frame_arrival_period=0.5;

    json reply;
    // Move to this class?
//    vector<clCommand *> &commands = factory->get_commands();

    reply["copy_in"] = json::array();
    reply["kernel"] = json::array();
    reply["copy_out"] = json::array();

    double total_copy_in_time = 0;
    double total_copy_out_time = 0;
    double total_kernel_time = 0;

    for (auto &command : commands) {
        clCommand *cmd = (clCommand*)command;
        double time = cmd->get_last_gpu_execution_time();
        double utilization = time/frame_arrival_period;
        if (cmd->get_command_type() == gpuCommandType::KERNEL) {
            reply["kernel"].push_back({{"name", cmd->get_name()},
                                        {"time", time},
                                        {"utilization", utilization} });
            total_kernel_time += cmd->get_last_gpu_execution_time();
        } else if (cmd->get_command_type() == gpuCommandType::COPY_IN) {

            reply["copy_in"].push_back({{"name", cmd->get_name()},
                                        {"time", time},
                                        {"utilization", utilization} });
            total_copy_in_time += cmd->get_last_gpu_execution_time();
        } else if (cmd->get_command_type() == gpuCommandType::COPY_OUT) {

            reply["copy_out"].push_back({{"name", cmd->get_name()},
                                        {"time", time},
                                        {"utilization", utilization} });
            total_copy_out_time += cmd->get_last_gpu_execution_time();
        } else {
            continue;
        }
    }

    reply["copy_in_total_time"] = total_copy_in_time;
    reply["kernel_total_time"] = total_kernel_time;
    reply["copy_out_total_time"] = total_copy_out_time;
    reply["copy_in_utilization"] = total_copy_in_time/frame_arrival_period;
    reply["kernel_utilization"] = total_kernel_time/frame_arrival_period;
    reply["copy_out_utilization"] = total_copy_out_time/frame_arrival_period;

    conn.send_json_reply(reply);
}


void clProcess::main_thread()
{
    restServer &rest_server = restServer::instance();
    rest_server.register_get_callback("/gpu_profile/"+ std::to_string(gpu_id),
            std::bind(&clProcess::profile_callback, this, std::placeholders::_1));

//    vector<clCommand *> &commands = factory->get_commands();

    // Start with the first GPU frame;
    int gpu_frame_id = 0;
    bool first_run = true;
    cl_event signal;

    while (!stop_thread) {
        // Wait for all the required preconditions
        // This is things like waiting for the input buffer to have data
        // and for there to be free space in the output buffers.
        //INFO("Waiting on preconditions for GPU[%d][%d]", gpu_id, gpu_frame_id);
        for (auto &command : commands) {
            if (command->wait_on_precondition(gpu_frame_id) != 0){
                INFO("Received exit in OpenCL command precondition! (Command '%s')",command->get_name().c_str());
                break;
            }
        }

        INFO("Waiting for free slot for GPU[%d][%d]", gpu_id, gpu_frame_id);
        // We make sure we aren't using a gpu frame that's currently in-flight.
        final_signals[gpu_frame_id]->wait_for_free_slot();
        signal = NULL;
        for (auto &command : commands) {
            // Feed the last signal into the next operation
            signal = ((clCommand*)command)->execute(gpu_frame_id, 0, signal);
            //usleep(10);
        }
        final_signals[gpu_frame_id]->set_signal(signal);
        INFO("Commands executed.");

        if (first_run) {
            results_thread_handle = std::thread(&clProcess::results_thread, std::ref(*this));
            first_run = false;
        }

        gpu_frame_id = (gpu_frame_id + 1) % _gpu_buffer_depth;
    }
    for (auto &sig_container : final_signals) {
        sig_container->stop();
    }
    INFO("Waiting for CL packet queues to finish up before freeing memory.");
    results_thread_handle.join();
}


void clProcess::results_thread() {
//    vector<clCommand *> &commands = factory->get_commands();

    // Start with the first GPU frame;
    int gpu_frame_id = 0;

    while (true) {
        // Wait for a signal to be completed
        DEBUG2("Waiting for signal for gpu[%d], frame %d, time: %f", gpu_id, gpu_frame_id, e_time());
        if (final_signals[gpu_frame_id]->wait_for_signal() == -1) {
            // If wait_for_signal returns -1, then we don't have a signal to wait on,
            // but we have been given a shutdown request, so break this loop.
            break;
        }
        DEBUG2("Got final signal for gpu[%d], frame %d, time: %f", gpu_id, gpu_frame_id, e_time());

        for (auto &command : commands) {
            ((clCommand*)command)->finalize_frame(gpu_frame_id);
        }
        DEBUG2("Finished finalizing frames for gpu[%d][%d]", gpu_id, gpu_frame_id);

        bool log_profiling = true;
        if (log_profiling) {
            string output = "";
            for (auto &command : commands) {
                clCommand *cmd = (clCommand*)command;
                output += "kernel: " + cmd->get_name() +
                          " time: " + std::to_string(cmd->get_last_gpu_execution_time()) + "; \n";
            }
            INFO("GPU[%d] Profiling: %s", gpu_id, output.c_str());
        }

        final_signals[gpu_frame_id]->reset();

        gpu_frame_id = (gpu_frame_id + 1) % _gpu_buffer_depth;
    }
}