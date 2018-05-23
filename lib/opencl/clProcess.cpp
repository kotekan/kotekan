#include "clProcess.hpp"
#include "clCommand.hpp"
#include "callbackdata.h"
#include "unistd.h"
#include "vdif_functions.h"
#include "fpga_header_functions.h"
#include "KotekanProcess.hpp"
#include "util.h"

#include <iostream>
#include <sys/time.h>

using namespace std;

REGISTER_KOTEKAN_PROCESS(clProcess);

// TODO Remove the GPU_ID from this constructor
clProcess::clProcess(Config& config_,
        const string& unique_name,
        bufferContainer &buffer_container):
    KotekanProcess(config_, unique_name, buffer_container, std::bind(&clProcess::main_thread, this))
{
    _gpu_buffer_depth = config.get_int(unique_name, "buffer_depth");
    gpu_id = config.get_int(unique_name, "gpu_id");

    final_signals.resize(_gpu_buffer_depth);

//    frame_arrival_period = config.get_double_eval(unique_name, "frame_arrival_period");
//    _use_beamforming = config.get_bool(unique_name, "enable_beamforming");

    json in_bufs = config.get_value(unique_name, "in_buffers");
    for (json::iterator it = in_bufs.begin(); it != in_bufs.end(); ++it) {
        string internal_name = it.key();
        string global_buffer_name = it.value();
        struct Buffer * buf = buffer_container.get_buffer(global_buffer_name);
        local_buffer_container.add_buffer(internal_name, buf);
        register_consumer(buf, unique_name.c_str());
    }

    json out_bufs = config.get_value(unique_name, "out_buffers");
    for (json::iterator it = out_bufs.begin(); it != out_bufs.end(); ++it) {
        string internal_name = it.key();
        string global_buffer_name = it.value();
        struct Buffer * buf = buffer_container.get_buffer(global_buffer_name);
        local_buffer_container.add_buffer(internal_name, buf);
        register_producer(buf, unique_name.c_str());
    }

    device = new clDeviceInterface(config_, gpu_id, _gpu_buffer_depth);

    string g_log_level = config.get_string(unique_name, "log_level");
    string s_log_level = config.get_string_default(unique_name, "device_interface_log_level", g_log_level);
    device->set_log_level(s_log_level);
    device->set_log_prefix("GPU[" + std::to_string(gpu_id) + "] device interface");
    device->prepareCommandQueue(true); //yes profiling

    factory = new clCommandFactory(config, unique_name, local_buffer_container, *device);
    INFO("Starting...");
}

void clProcess::apply_config(uint64_t fpga_seq) {
}

clProcess::~clProcess() {
    restServer::instance().remove_get_callback("/gpu_profile/"+ std::to_string(gpu_id));
    delete factory;
    delete device;
}


void clProcess::profile_callback(connectionInstance& conn) {
    DEBUG(" *** *** *** Profile call made.");

    double frame_arrival_period=0.5;

    json reply;
    // Move to this class?
    vector<clCommand *> &commands = factory->get_commands();

    reply["copy_in"] = json::array();
    reply["kernel"] = json::array();
    reply["copy_out"] = json::array();

    double total_copy_in_time = 0;
    double total_copy_out_time = 0;
    double total_kernel_time = 0;

    for (uint32_t i = 0; i < commands.size(); ++i) {
        double time = commands[i]->get_last_gpu_execution_time();
        double utilization = time/frame_arrival_period;
        if (commands[i]->get_command_type() == clCommandType::KERNEL) {
            reply["kernel"].push_back({{"name", commands[i]->get_name()},
                                        {"time", time},
                                        {"utilization", utilization} });
            total_kernel_time += commands[i]->get_last_gpu_execution_time();
        } else if (commands[i]->get_command_type() == clCommandType::COPY_IN) {

            reply["copy_in"].push_back({{"name", commands[i]->get_name()},
                                        {"time", time},
                                        {"utilization", utilization} });
            total_copy_in_time += commands[i]->get_last_gpu_execution_time();
        } else if (commands[i]->get_command_type() == clCommandType::COPY_OUT) {

            reply["copy_out"].push_back({{"name", commands[i]->get_name()},
                                        {"time", time},
                                        {"utilization", utilization} });
            total_copy_out_time += commands[i]->get_last_gpu_execution_time();
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

    vector<clCommand *> &commands = factory->get_commands();

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
        final_signals[gpu_frame_id].wait_for_free_slot();
        signal = NULL;
        for (auto &command : commands) {
            // Feed the last signal into the next operation
            signal = command->execute(gpu_frame_id, 0, signal);
            //usleep(10);
        }
        final_signals[gpu_frame_id].set_signal(signal);
        INFO("Commands executed.");

        if (first_run) {
            results_thread_handle = std::thread(&clProcess::results_thread, std::ref(*this));
            first_run = false;
        }

        gpu_frame_id = (gpu_frame_id + 1) % _gpu_buffer_depth;
    }
    for (auto &sig_container : final_signals) {
        sig_container.stop();
    }
    INFO("Waiting for CL packet queues to finish up before freeing memory.");
    results_thread_handle.join();
}


void clProcess::results_thread() {
    vector<clCommand *> &commands = factory->get_commands();

    // Start with the first GPU frame;
    int gpu_frame_id = 0;

    while (true) {
        // Wait for a signal to be completed
        DEBUG2("Waiting for signal for gpu[%d], frame %d, time: %f", gpu_id, gpu_frame_id, e_time());
        if (final_signals[gpu_frame_id].wait_for_signal() == -1) {
            // If wait_for_signal returns -1, then we don't have a signal to wait on,
            // but we have been given a shutdown request, so break this loop.
            break;
        }
        DEBUG2("Got final signal for gpu[%d], frame %d, time: %f", gpu_id, gpu_frame_id, e_time());

        for (auto &command : commands) {
            command->finalize_frame(gpu_frame_id);
        }
        DEBUG2("Finished finalizing frames for gpu[%d][%d]", gpu_id, gpu_frame_id);

        bool log_profiling = true;
        if (log_profiling) {
            string output = "";
            for (uint32_t i = 0; i < commands.size(); ++i) {
                output += "kernel: " + commands[i]->get_name() +
                          " time: " + std::to_string(commands[i]->get_last_gpu_execution_time()) + "; \n";
            }
            INFO("GPU[%d] Profiling: %s", gpu_id, output.c_str());
        }

        final_signals[gpu_frame_id].reset();

        gpu_frame_id = (gpu_frame_id + 1) % _gpu_buffer_depth;
    }
}