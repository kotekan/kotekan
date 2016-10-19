#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <sys/time.h>
#include <math.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <getopt.h>
#include <assert.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <iostream>
#include <fstream>

extern "C" {
#include <pthread.h>
}

// DPDK!
extern "C" {
#include <rte_config.h>
#include <rte_common.h>
#include <rte_log.h>
#include <rte_memory.h>
#include <rte_memcpy.h>
#include <rte_memzone.h>
#include <rte_eal.h>
#include <rte_per_lcore.h>
#include <rte_launch.h>
#include <rte_atomic.h>
#include <rte_cycles.h>
#include <rte_prefetch.h>
#include <rte_lcore.h>
#include <rte_per_lcore.h>
#include <rte_branch_prediction.h>
#include <rte_interrupts.h>
#include <rte_pci.h>
#include <rte_random.h>
#include <rte_debug.h>
#include <rte_ether.h>
#include <rte_ethdev.h>
#include <rte_ring.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
}

#include "errors.h"
#include "buffers.h"
#include "gpu_thread.h"
#include "error_correction.h"
#include "chrxUplink.hpp"
#include "Config.hpp"
#include "gpuPostProcess.hpp"
#include "beamformingPostProcess.hpp"
#include "vdifStream.hpp"
#include "nullProcess.hpp"
#include "util.h"
#include "network_dpdk.h"
#include "version.h"
#include "raw_cap.hpp"
#include "networkOutputSim.hpp"
#include "SampleProcess.hpp"
#include "json.hpp"
#include "gpuPostProcess.hpp"
#include "beamformingPostProcess.hpp"

using json = nlohmann::json;

void print_help() {
    printf("usage: kotekan [opts]\n\n");
    printf("Options:\n");
    printf("    --config (-c) [file]            The local JSON config file to use\n");
    printf("    --log-to-screen (-s)            Output log messages to stderr\n");
    printf("    --log-level (-l) [level]        0 = errors only, 1 = +warnings, 2 = +info, 3 = +debug\n\n");
    printf("    --daemon -d [root]              Run in daemon mode, disables output to stderr");
    printf("Testing Options:\n");
    printf("    --no-network-test (-t)          Generates fake data for testing.\n");
}

void dpdk_setup() {

    char  arg0[] = "./kotekan";
    char  arg1[] = "-n";
    char  arg2[] = "4";
    char  arg3[] = "-c";
    char  arg4[] = "F";
    char* argv2[] = { &arg0[0], &arg1[0], &arg2[0], &arg3[0], &arg4[0], NULL };
    int   argc2   = (int)(sizeof(argv2) / sizeof(argv2[0])) - 1;

    /* Initialize the Environment Abstraction Layer (EAL). */
    int ret2 = rte_eal_init(argc2, argv2);
    if (ret2 < 0)
        exit(EXIT_FAILURE);

    // Can we see ports here?
    int nb_ports = rte_eth_dev_count();
    fprintf(stdout, "kotekan; Number of ports: %d\n", nb_ports);

}

int main(int argc, char ** argv) {

    dpdk_setup();

    int opt_val = 0;
    int no_network_test = 0;
    int network_sim_pattern = 0;
    char * config_file_name = (char *)"kotekan.conf";
    int log_options = LOG_CONS | LOG_PID | LOG_NDELAY;
    int make_daemon = 0;
    char * kotekan_root_dir = (char *)"./";

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int j = 6; j < 12; j++)
        CPU_SET(j, &cpuset);

    cpu_set_t cpuset_gpu;
    CPU_ZERO(&cpuset_gpu);
    for (int j = 4; j < 6; j++)
        CPU_SET(j, &cpuset_gpu);

    for (;;) {
        static struct option long_options[] = {
            {"no-network-test", required_argument, 0, 't'},
            {"read-file", required_argument, 0, 'f'},
            {"config", required_argument, 0, 'c'},
            {"write-local", no_argument, 0, 'w'},
            {"log-level", required_argument, 0, 'l'},
            {"daemon", required_argument, 0, 'd'},
            {"help", no_argument, 0, 'h'},
            {0, 0, 0, 0}
        };

        int option_index = 0;
        int option_value = 0;

        opt_val = getopt_long (argc, argv, "ht:wsc:l:d:",
                               long_options, &option_index);

        // End of args
        if (opt_val == -1) {
            break;
        }

        switch (opt_val) {
            case 'h':
                print_help();
                return 0;
                break;
            case 't':
                no_network_test = 1;
                network_sim_pattern = atoi(optarg);
                break;
            case 's':
                log_options = log_options | LOG_PERROR;
                break;
            case 'c':
                config_file_name = strdup(optarg);
                break;
            case 'd':
                make_daemon = 1;
                kotekan_root_dir = strdup(optarg);
                break;
            case 'l':
                option_value = atoi(optarg);
                switch (option_value) {
                    case 3:
                        log_level_debug = 1;
                    case 2:
                        log_level_info = 1;
                    case 1:
                        log_level_warn = 1;
                    default:
                        break;
                }
                break;
            default:
                printf("Invalid option, run with -h to see options");
                return -1;
                break;
        }
    }

    if (make_daemon) {
        // Do not use LOG_CONS or LOG_PERROR, since stderr does not exist in daemon mode.
        openlog ("kotekan", log_options, LOG_PID | LOG_NDELAY);
        if ((chdir(kotekan_root_dir)) < 0) {
            ERROR("Cannot change directory to %s", kotekan_root_dir);
            exit(EXIT_FAILURE);
        }
    } else {
        // Setup syslog
        openlog ("kotekan", log_options, LOG_LOCAL1);
    }

    // Load configuration file.
    //INFO("Kotekan starting with config file %s", config_file_name);
    const char git_hash[] = GIT_COMMIT_HASH;
    const char git_branch[] = GIT_BRANCH;
    INFO("Kotekan %f starting build: %s, on branch: %s, with config file: %s",
            KOTEKAN_VERSION, git_hash, git_branch, config_file_name);

    Config config;

    config.parse_file(config_file_name, 0);
    config.generate_extra_options();
    config.dump_config();

    // TODO: allow raw capture in line with the correlator.
    //if (config.get_bool("/raw_capture/enabled")) {
    //    return raw_cap(&config);
    //}

    std::vector<KotekanProcess *> processes;

    // Config values:
    int32_t num_gpus = config.get_int("/gpu/num_gpus");
    int32_t num_local_freq = config.get_int("/processing/num_local_freq");
    int32_t num_total_freq = config.get_int("/processing/num_total_freq");
    int32_t num_elements = config.get_int("/processing/num_elements");
    int32_t num_adjusted_local_freq = config.get_int("/processing/num_adjusted_local_freq");
    int32_t num_adjusted_elements = config.get_int("/processing/num_adjusted_elements");
    int32_t block_size = config.get_int("/gpu/block_size");
    int32_t num_blocks = config.get_int("/gpu/num_blocks");
    int32_t num_data_sets = config.get_int("/processing/num_data_sets");
    int32_t samples_per_data_set = config.get_int("/processing/samples_per_data_set");
    int32_t buffer_depth = config.get_int("/processing/buffer_depth");
    int32_t num_fpga_links = config.get_int("/fpga_network/num_links");
    int32_t network_buffer_depth = config.get_int("/ch_master_network/network_buffer_depth");
    vector<int32_t> link_map = config.get_int_array("/gpu/link_map");
    bool enable_upload = config.get_bool("/ch_master_network/enable_upload");
    bool enable_beamforming = config.get_bool("/gpu/enable_beamforming");

    // Create buffers.
    struct Buffer gpu_input_buffer[num_gpus];
    struct Buffer gpu_output_buffer[num_gpus];
    struct Buffer gpu_beamform_output_buffer[num_gpus];
    struct Buffer gpu_incoh_beamform_output_buffer[num_gpus];

    cl_int output_len = num_adjusted_local_freq * num_blocks* (block_size*block_size)*2.;

    // Create the shared pool of buffer info objects; used for recording information about a
    // given frame and past between buffers as needed.
    struct InfoObjectPool pool[num_gpus];

    gpuThread * gpu_threads[num_gpus];

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
                      1,
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

         snprintf(buffer_name, 100, "gpu_beamform_output_buffer_%d", i);
         create_buffer(&gpu_beamform_output_buffer[i],
                       links_per_gpu * buffer_depth,
                       samples_per_data_set * num_data_sets *
                       num_local_freq * 2,
                       1,
                       1,
                       &pool[i],
                       buffer_name);

        // TODO better management of the buffers so this list doesn't have to change size...
        gpu_threads[i] = new gpuThread(config,
                                        gpu_input_buffer[i],
                                        gpu_output_buffer[i],
                                        gpu_beamform_output_buffer[i],
                                        gpu_incoh_beamform_output_buffer[i], i);

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
    struct Buffer * tmp_buffer[num_fpga_links];

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
            tmp_buffer[i] = &gpu_input_buffer[link_map[i]];
            network_dpdk_args.num_links_in_group[i] = config.num_links_per_gpu(i);
            network_dpdk_args.link_id[i] = link_ids[i];
        }
        network_dpdk_args.buf = tmp_buffer;
        network_dpdk_args.vdif_buf = NULL;
        network_dpdk_args.num_links = num_fpga_links;
        network_dpdk_args.timesamples_per_packet = config.get_int("/fpga_network/timesamples_per_packet");
        network_dpdk_args.samples_per_data_set = samples_per_data_set;
        network_dpdk_args.num_data_sets = num_data_sets;
        network_dpdk_args.num_gpu_frames = config.get_int("/processing/num_gpu_frames");
        network_dpdk_args.udp_packet_size = config.get_int("/fpga_network/udp_packet_size");
        network_dpdk_args.num_lcores = 4;
        network_dpdk_args.num_links_per_lcore = 2;
        network_dpdk_args.port_offset[0] = 0;
        network_dpdk_args.port_offset[1] = 2;
        network_dpdk_args.port_offset[2] = 4;
        network_dpdk_args.port_offset[3] = 6;

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
                                         network_sim_pattern,
                                         i);
            network_output_sim->start();
            processes.push_back((KotekanProcess*)network_output_sim);
        }
    }

    struct Buffer network_output_buffer;
    struct Buffer gated_output_buffer;
    struct Buffer vdif_output_buffer;

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

    // The beamforming output thread
    if (enable_beamforming) {
        INFO("Creating vdif output threads")

        create_buffer(&vdif_output_buffer, network_buffer_depth, 625*16*5032,
                      1, 1, pool, "vdif_output_buffer");

        // The thread which creates output frame.
        beamformingPostProcess * beamform_post_process = new beamformingPostProcess(config,
                                                            gpu_beamform_output_buffer,
                                                            vdif_output_buffer);
        beamform_post_process->start();
        processes.push_back((KotekanProcess*)beamform_post_process);

        // The thread which sends it with UDP to the VDIF collection server
        vdifStream * vdif_stream = new vdifStream(config, vdif_output_buffer);
        vdif_stream->start();
        processes.push_back((KotekanProcess*)vdif_stream);
    }

    // Join the threads.

    processes[0]->join();

    INFO("kotekan shutdown successfully.");

    closelog();

    return 0;
}
