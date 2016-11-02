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
#include "packet_cap.hpp"
#include "fullPacketDump.hpp"

#include <vector>
#include <string>
#include <unistd.h>

using std::string;
using std::vector;

void packet_cap(Config &config) {

    INFO("starting packet cap");

    std::vector<KotekanProcess *> processes;

    // Config values:
    int32_t fpga_packet_lenght = config.get_int("/fpga_network/udp_packet_size");
    int32_t num_data_sets = config.get_int("/processing/num_data_sets");
    int32_t samples_per_data_set = config.get_int("/processing/samples_per_data_set");
    int32_t buffer_depth = config.get_int("/processing/buffer_depth");
    int32_t num_fpga_links = config.get_int("/fpga_network/num_links");

    // Create buffers.
    struct Buffer network_input_buffer[num_fpga_links];

    // Create the shared pool of buffer info objects; used for recording information about a
    // given frame and past between buffers as needed.
    struct InfoObjectPool pool[num_fpga_links];

    INFO("Creating input buffers...");
    char buffer_name[100];

    for (int i = 0; i < num_fpga_links; ++i) {
        create_info_pool(&pool[i], 2 * buffer_depth, 0, 0);

        snprintf(buffer_name, 100, "network_input_buffer_%d", i);
        create_buffer(&network_input_buffer[i],
                      buffer_depth,
                      fpga_packet_lenght * samples_per_data_set,
                      1,
                      1,
                      &pool[i],
                      buffer_name);
    }

    INFO("Starting up network threads...");

    // Create network threads
    pthread_t network_dpdk_t;
    struct networkDPDKArg network_dpdk_args;
    struct Buffer *** tmp_buffer;
    tmp_buffer = (struct Buffer ***)malloc(num_fpga_links * sizeof(struct Buffer *));
    for (int port = 0; port < num_fpga_links; ++port) {
        tmp_buffer[port] = (struct Buffer **)malloc(NUM_FREQ * sizeof(struct Buffer *));
    }

    // TODO move to function
    int32_t link_ids[num_fpga_links];
    for (int i = 0; i < num_fpga_links; ++i) {
        link_ids[i] = 0;
    }

    // Start DPDK
    for (int i = 0; i < num_fpga_links; ++i) {
        for (int freq = 0; freq < NUM_FREQ; ++freq) {
            tmp_buffer[i][freq] = &network_input_buffer[i];
            INFO ("tmp_buffer[%d][%d] = %p", i, freq, tmp_buffer[i][freq]);
        }
        network_dpdk_args.num_links_in_group[i] = 1;
        network_dpdk_args.link_id[i] = link_ids[i];
    }
    network_dpdk_args.buf = tmp_buffer;
    network_dpdk_args.vdif_buf = NULL;
    network_dpdk_args.num_links = num_fpga_links;
    network_dpdk_args.timesamples_per_packet = config.get_int("/fpga_network/timesamples_per_packet");
    network_dpdk_args.samples_per_data_set = samples_per_data_set;
    network_dpdk_args.num_data_sets = num_data_sets;
    network_dpdk_args.num_gpu_frames = config.get_int("/processing/num_gpu_frames");
    network_dpdk_args.udp_packet_size = fpga_packet_lenght;
    network_dpdk_args.num_lcores = 4;
    network_dpdk_args.num_links_per_lcore = 1;
    network_dpdk_args.port_offset[0] = 0;
    network_dpdk_args.port_offset[1] = 1;
    network_dpdk_args.port_offset[2] = 2;
    network_dpdk_args.port_offset[3] = 3;
    network_dpdk_args.enable_shuffle = 0;
    network_dpdk_args.dump_full_packets = 1;

    CHECK_ERROR( pthread_create(&network_dpdk_t, NULL, &network_dpdk_thread,
                                (void *)&network_dpdk_args ) );


    // The thread which creates output frame.
    for (int i = 0; i < num_fpga_links; ++i) {
        fullPacketDump * full_packet_dump = new fullPacketDump(config,
                                                            network_input_buffer[i],
                                                            i);
        full_packet_dump->start();
        processes.push_back((KotekanProcess*)full_packet_dump);
    }

    // Join the threads.

    processes[0]->join();

}