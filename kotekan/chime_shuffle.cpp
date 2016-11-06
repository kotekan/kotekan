#include "chime_shuffle.hpp"
#include "buffers.h"
#include "gpu_thread.h"
#include "chrxUplink.hpp"
#include "gpuPostProcess.hpp"
#include "networkOutputSim.hpp"
#include "nullProcess.hpp"
#include "vdifStream.hpp"
#include "network_dpdk.h"
#include "hccGPUThread.hpp"
#include "util.h"

#include <vector>
#include <string>

using std::string;
using std::vector;

void chime_shuffle_setup(Config &config) {

    std::vector<KotekanProcess *> processes;

    // Config values:
    int32_t num_gpus = config.get_int("/gpu/num_gpus");
    int32_t num_total_freq = config.get_int("/processing/num_total_freq");
    int32_t num_elements = config.get_int("/processing/num_elements");
    int32_t num_adjusted_local_freq = config.get_int("/processing/num_adjusted_local_freq");
    int32_t num_adjusted_elements = config.get_int("/processing/num_adjusted_elements");
    int32_t block_size = config.get_int("/gpu/block_size");
    int32_t num_blocks = config.get_int("/gpu/num_blocks");
    int32_t num_data_sets = config.get_int("/processing/num_data_sets");
    int32_t samples_per_data_set = config.get_int("/processing/samples_per_data_set");
    int32_t buffer_depth = config.get_int("/processing/buffer_depth");
    int32_t num_fpga_links = config.get_int("/dpdk/num_links");
    int32_t network_buffer_depth = config.get_int("/ch_master_network/network_buffer_depth");
    vector<int32_t> link_map = config.get_int_array("/gpu/link_map");
    bool enable_upload = config.get_bool("/ch_master_network/enable_upload");
    bool no_network_test = config.get_bool("/processing/no_network_test");
    int32_t network_test_pattern = config.get_int("/processing/network_test_pattern");

    // Create buffers.
    struct Buffer gpu_input_buffer[num_gpus];
    struct Buffer gpu_output_buffer[num_gpus];

    cl_int output_len = num_adjusted_local_freq * num_blocks * (block_size*block_size)*2.;

    // Create the shared pool of buffer info objects; used for recording information about a
    // given frame and past between buffers as needed.
    struct InfoObjectPool pool[num_gpus];

    hccGPUThread * gpu_threads[num_gpus];

    INFO("Starting GPU threads...");
    char buffer_name[100];

    for (int i = 0; i < num_gpus; ++i) {

        DEBUG("Creating buffers...");

        int links_per_gpu = config.num_links_per_gpu(i);

        INFO("Num links for gpu[%d] = %d", i, links_per_gpu);

        create_info_pool(&pool[i], 2 * links_per_gpu * buffer_depth,
                                    num_adjusted_local_freq,
                                    num_adjusted_elements);

        snprintf(buffer_name, 100, "gpu_input_buffer_%d", i);
        create_buffer(&gpu_input_buffer[i],
                      links_per_gpu * buffer_depth,
                      samples_per_data_set * num_adjusted_elements *
                      num_adjusted_local_freq * num_data_sets,
                      4,
                      1,
                      &pool[i],
                      buffer_name);

        snprintf(buffer_name, 100, "gpu_output_buffer_%d", i);
        create_buffer(&gpu_output_buffer[i],
                      links_per_gpu * buffer_depth,
                      output_len * num_data_sets * sizeof(cl_int),
                      1,
                      1,
                      &pool[i],
                      buffer_name);

        // TODO better management of the buffers so this list doesn't have to change size...
        gpu_threads[i] = new hccGPUThread(config, gpu_input_buffer[i], gpu_output_buffer[i], i);

        DEBUG("Setting up GPU thread %d\n", i);

        gpu_threads[i]->start();
        processes.push_back((KotekanProcess*)gpu_threads[i]);

        INFO("GPU thread %d ready.", i);
    }

    //sleep(5);
    INFO("Starting up network threads...");

    // Create network threads
    pthread_t network_dpdk_t;
    struct networkDPDKArg network_dpdk_args;
    struct Buffer *** tmp_buffer;
    tmp_buffer = (struct Buffer ***)malloc(4 * sizeof(struct Buffer *));
    for (int port = 0; port < 4; ++port) {
        tmp_buffer[port] = (struct Buffer **)malloc(4 * sizeof(struct Buffer *));
        for (int freq = 0; freq < 4; ++freq) {
            tmp_buffer[port][freq] = (struct Buffer *)0xdeadbeef;
        }
    }

    // TODO move to function
    int current_gpu_id = 0;
    int current_link_id = 0;
    int32_t link_ids[num_fpga_links];
    for (int i = 0; i < num_fpga_links; ++i) {
        if (current_gpu_id != link_map[i]) {
            current_gpu_id = link_map[i];
            current_link_id = 0;
        }
        link_ids[i] = current_link_id++;
        INFO("link_ids[%d] = %d", i, link_ids[i]);
    }

    if (no_network_test == 0) {
        // Start DPDK
        for (int i = 0; i < num_fpga_links; ++i) {
            for (int freq = 0; freq < NUM_FREQ; ++freq) {
                tmp_buffer[i][freq] = &gpu_input_buffer[freq];
                INFO ("tmp_buffer[%d][%d] = %p", i, freq, tmp_buffer[i][freq]);
            }
            network_dpdk_args.num_links_in_group[i] = config.num_links_per_gpu(i);
            network_dpdk_args.link_id[i] = link_ids[i];
        }
        network_dpdk_args.buf = tmp_buffer;
        network_dpdk_args.vdif_buf = NULL;
        network_dpdk_args.num_links = num_fpga_links;
        network_dpdk_args.timesamples_per_packet = config.get_int("/dpdk/timesamples_per_packet");
        network_dpdk_args.samples_per_data_set = samples_per_data_set;
        network_dpdk_args.num_data_sets = num_data_sets;
        network_dpdk_args.num_gpu_frames = config.get_int("/processing/num_gpu_frames");
        network_dpdk_args.udp_packet_size = config.get_int("/dpdk/udp_packet_size");
        network_dpdk_args.num_lcores = 4;
        network_dpdk_args.num_links_per_lcore = 1;
        network_dpdk_args.port_offset[0] = 0;
        network_dpdk_args.port_offset[1] = 1;
        network_dpdk_args.port_offset[2] = 2;
        network_dpdk_args.port_offset[3] = 3;

        CHECK_ERROR( pthread_create(&network_dpdk_t, NULL, &network_dpdk_thread,
                                    (void *)&network_dpdk_args ) );

    } else {
        // Start network simulation
        for (int i = 0; i < num_fpga_links; ++i) {
            networkOutputSim * network_output_sim =
                    new networkOutputSim(config,
                                         gpu_input_buffer[link_map[i]],
                                         config.num_links_per_gpu(link_map[i]),
                                         link_ids[i],
                                         network_test_pattern,
                                         i);
            network_output_sim->start();
            processes.push_back((KotekanProcess*)network_output_sim);
        }
    }

    struct Buffer network_output_buffer;
    struct Buffer gated_output_buffer;

    int num_values = ((num_elements * (num_elements + 1)) / 2 ) * num_total_freq;

    const int tcp_buffer_size = sizeof(struct tcp_frame_header) +
        num_values * sizeof(complex_int_t) +
        num_total_freq * sizeof(struct per_frequency_data) +
        num_total_freq * num_elements * sizeof(struct per_element_data) +
        num_values * sizeof(uint8_t);

    const int gate_buffer_size = sizeof(struct gate_frame_header)
                            + num_values * sizeof(complex_int_t);

    create_buffer(&network_output_buffer, network_buffer_depth, tcp_buffer_size,
                  1, 1, pool, "network_output_buffer");

    create_buffer(&gated_output_buffer, network_buffer_depth, gate_buffer_size,
                  1, 1, pool, "gated_output_buffer");

    // The thread which creates output frame.
    gpuPostProcess * gpu_post_process = new gpuPostProcess(config,
                                                            gpu_output_buffer,
                                                            network_output_buffer,
                                                            gated_output_buffer);

    gpu_post_process->start();
    processes.push_back((KotekanProcess*)gpu_post_process);

    if (enable_upload) {
        chrxUplink * chrx_uplink = new chrxUplink(config, network_output_buffer, gated_output_buffer);
        chrx_uplink->start();
        processes.push_back((KotekanProcess*)chrx_uplink);
    } else {
        // Drop the data.
        nullProcess * null_process1 = new nullProcess(config, network_output_buffer);
        null_process1->start();
        processes.push_back((KotekanProcess*)null_process1);

        nullProcess * null_process2 = new nullProcess(config, gated_output_buffer);
        null_process2->start();
        processes.push_back((KotekanProcess*)null_process2);
    }

    // Join the threads.
    for (auto&& process : processes)
        process->join();
}