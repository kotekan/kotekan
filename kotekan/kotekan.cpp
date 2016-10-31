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
#include "chime_shuffle.hpp"

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

    chime_shuffle_setup(config);

    INFO("kotekan shutdown successfully.");

    closelog();

    return 0;
}
