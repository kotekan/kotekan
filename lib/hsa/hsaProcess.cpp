#include "hsaProcess.hpp"
#include "unistd.h"
#include "vdif_functions.h"
#include "fpga_header_functions.h"
#include "KotekanProcess.hpp"
#include "util.h"

#include <iostream>
#include <sys/time.h>

REGISTER_KOTEKAN_PROCESS(hsaProcess);

hsaProcess::hsaProcess(Config& config, const string& unique_name,
                     bufferContainer &buffer_container):
        KotekanProcess(config, unique_name, buffer_container,
                     std::bind(&hsaProcess::main_thread, this)) {

    apply_config(0);

    final_signals.resize(_gpu_buffer_depth);

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

    log_profiling = config.get_bool_default(unique_name, "log_profiling", false);

    device = new hsaDeviceInterface(config, gpu_id, _gpu_buffer_depth);

    string g_log_level = config.get_string(unique_name, "log_level");
    string s_log_level = config.get_string_default(unique_name, "device_interface_log_level", g_log_level);
    device->set_log_level(s_log_level);
    device->set_log_prefix("GPU[" + std::to_string(gpu_id) + "] device interface");

    factory = new hsaCommandFactory(config, unique_name, local_buffer_container, *device);

    endpoint = "/gpu_profile/" + std::to_string(gpu_id);
}

void hsaProcess::apply_config(uint64_t fpga_seq) {
    (void)fpga_seq;
    _gpu_buffer_depth = config.get_int(unique_name, "buffer_depth");
    gpu_id = config.get_int(unique_name, "gpu_id");

    frame_arrival_period = config.get_double_eval(unique_name, "frame_arrival_period");
}

hsaProcess::~hsaProcess() {
    restServer::instance().remove_get_callback(endpoint);

    delete factory;
    delete device;
}

void hsaProcess::profile_callback(connectionInstance& conn) {

    DEBUG("Profile call made.");

    json reply;
    // Move to this class?
    vector<hsaCommand *> &commands = factory->get_commands();

    reply["copy_in"] = json::array();
    reply["kernel"] = json::array();
    reply["copy_out"] = json::array();

    double total_copy_in_time = 0;
    double total_copy_out_time = 0;
    double total_kernel_time = 0;

    for (uint32_t i = 0; i < commands.size(); ++i) {
        double time = commands[i]->get_last_gpu_execution_time();
        double utilization = time/frame_arrival_period;
        if (commands[i]->get_command_type() == CommandType::KERNEL) {

            reply["kernel"].push_back({{"name", commands[i]->get_kernel_file_name()},
                                        {"time", time},
                                        {"utilization", utilization} });
            total_kernel_time += commands[i]->get_last_gpu_execution_time();
        } else if (commands[i]->get_command_type() == CommandType::COPY_IN) {

            reply["copy_in"].push_back({{"name", commands[i]->get_name()},
                                        {"time", time},
                                        {"utilization", utilization} });
            total_copy_in_time += commands[i]->get_last_gpu_execution_time();
        } else if (commands[i]->get_command_type() == CommandType::COPY_OUT) {

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

void hsaProcess::main_thread()
{
    vector<hsaCommand *> &commands = factory->get_commands();

//    using namespace std::placeholders;
    restServer &rest_server = restServer::instance();
    rest_server.register_get_callback(endpoint,
            std::bind(&hsaProcess::profile_callback, this, std::placeholders::_1));

    // Start with the first GPU frame;
    int gpu_frame_id = 0;
    bool first_run = true;

    while (!stop_thread) {

        // Wait for all the required preconditions
        // This is things like waiting for the input buffer to have data
        // and for there to be free space in the output buffers.
        //INFO("Waiting on preconditions for GPU[%d][%d]", gpu_id, gpu_frame_id);
        for (uint32_t i = 0; i < commands.size(); ++i) {
            if (commands[i]->wait_on_precondition(gpu_frame_id) != 0){
                INFO("Received exit in HSA command precondition! (Command %i, '%s')",i,commands[i]->get_name().c_str());
                break;
            }
        }

        //INFO("Waiting for free slot for GPU[%d][%d]", gpu_id, gpu_frame_id);
        // We make sure we aren't using a gpu frame that's currently in-flight.
        final_signals[gpu_frame_id].wait_for_free_slot();

        hsa_signal_t signal;
        signal.handle = 0;
        //INFO("Adding commands to GPU[%d][%d] queues", gpu_id, gpu_frame_id);

        for (uint32_t i = 0; i < commands.size(); i++) {
            // Feed the last signal into the next operation
            signal = commands[i]->execute(gpu_frame_id, 0, signal);
            //usleep(10);
        }
        final_signals[gpu_frame_id].set_signal(signal);

        if (first_run) {
            results_thread_handle = std::thread(&hsaProcess::results_thread, std::ref(*this));

            // Requires Linux, this could possibly be made more general someday.
            // TODO Move to config
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            for (int j = 4; j < 12; j++)
                CPU_SET(j, &cpuset);
            pthread_setaffinity_np(results_thread_handle.native_handle(),
                                    sizeof(cpu_set_t), &cpuset);
            first_run = false;
        }

        gpu_frame_id = (gpu_frame_id + 1) % _gpu_buffer_depth;
    }
    for (signalContainer &sig_container : final_signals) {
        sig_container.stop();
    }
    INFO("Waiting for HSA packet queues to finish up before freeing memory.");
    results_thread_handle.join();
}

void hsaProcess::results_thread() {

    vector<hsaCommand *> &commands = factory->get_commands();

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

        for (uint32_t i = 0; i < commands.size(); ++i) {
            commands[i]->finalize_frame(gpu_frame_id);
        }
        DEBUG2("Finished finalizing frames for gpu[%d][%d]", gpu_id, gpu_frame_id);

        if (log_profiling) {
            string output = "";
            for (uint32_t i = 0; i < commands.size(); ++i) {
                if (commands[i]->get_command_type() == CommandType::KERNEL) {
                    output += "kernel: " + commands[i]->get_name() +
                              " time: " + std::to_string(commands[i]->get_last_gpu_execution_time()) + "; ";
                }
            }
            INFO("GPU[%d] Profiling: %s", gpu_id, output.c_str());
        }
        gpu_frame_id = (gpu_frame_id + 1) % _gpu_buffer_depth;
    }
}
