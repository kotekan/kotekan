#include "hsaProcess.hpp"
#include "unistd.h"
#include "vdif_functions.h"
#include "fpga_header_functions.h"
#include "KotekanProcess.hpp"
#include "util.h"

#include <iostream>
#include <sys/time.h>

using namespace std;

hsaProcess::hsaProcess(Config& config, const string& unique_name,
                     bufferContainer &buffer_container):
        KotekanProcess(config, unique_name, buffer_container,
                     std::bind(&hsaProcess::main_thread, this)) {

    apply_config(0);

    final_signals.resize(_gpu_buffer_depth);

    // TODO move this into the config.
    local_buffer_container.add_buffer("network_buf", get_buffer("network_buffer"));
    register_consumer(get_buffer("network_buffer"), unique_name.c_str());
    local_buffer_container.add_buffer("output_buf", get_buffer("output_buffer"));
    register_producer(get_buffer("output_buffer"), unique_name.c_str());

    device = new hsaDeviceInterface(config, gpu_id);
    factory = new hsaCommandFactory(config, *device, local_buffer_container, unique_name);
}

void hsaProcess::apply_config(uint64_t fpga_seq) {
    (void)fpga_seq;
    _gpu_buffer_depth = config.get_int(unique_name, "buffer_depth");
    gpu_id = config.get_int(unique_name, "gpu_id");
}

hsaProcess::~hsaProcess() {
    delete factory;
    delete device;
}

void hsaProcess::main_thread()
{

    vector<hsaCommand *> &commands = factory->get_commands();

    // Start with the first GPU frame;
    int gpu_frame_id = 0;
    bool first_run = true;

    for(;;) {

        // Wait for all the required preconditions
        // This is things like waiting for the input buffer to have data
        // and for there to be free space in the output buffers.
        //INFO("Waiting on preconditions for GPU[%d][%d]", gpu_id, gpu_frame_id);
        for (uint32_t i = 0; i < commands.size(); ++i) {
            commands[i]->wait_on_precondition(gpu_frame_id);
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
    // TODO Make the exiting process actually work here.
    results_thread_handle.join();
}

void hsaProcess::results_thread() {

    vector<hsaCommand *> &commands = factory->get_commands();

    // Start with the first GPU frame;
    int gpu_frame_id = 0;

    for(;;) {
        // Wait for a signal to be completed
        //INFO("Waiting for signal for gpu[%d], frame %d, time: %f", gpu_id, gpu_frame_id, e_time());
        final_signals[gpu_frame_id].wait_for_signal();
        //INFO("Got final signal for gpu[%d], frame %d, time: %f", gpu_id, gpu_frame_id, e_time());

        for (uint32_t i = 0; i < commands.size(); ++i) {
            commands[i]->finalize_frame(gpu_frame_id);
        }
        //INFO("Finished finalizing frames for gpu[%d][%d]", gpu_id, gpu_frame_id);

        final_signals[gpu_frame_id].reset();

        gpu_frame_id = (gpu_frame_id + 1) % _gpu_buffer_depth;
    }
}
