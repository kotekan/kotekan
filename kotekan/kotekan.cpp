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
#include "restServer.hpp"
#include "packet_cap.hpp"

using json = nlohmann::json;

void print_help() {
    printf("usage: kotekan [opts]\n\n");
    printf("Options:\n");
    printf("    --config (-c) [file]            The local JSON config file to use\n\n");
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
    char * config_file_name = (char *)"kotekan.conf";
    int log_options = LOG_CONS | LOG_PID | LOG_NDELAY | LOG_PERROR;

    for (;;) {
        static struct option long_options[] = {
            {"config", required_argument, 0, 'c'},
            {"help", no_argument, 0, 'h'},
            {0, 0, 0, 0}
        };

        int option_index = 0;

        opt_val = getopt_long (argc, argv, "hc:",
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
            case 'c':
                config_file_name = strdup(optarg);
                break;
            default:
                printf("Invalid option, run with -h to see options");
                return -1;
                break;
        }
    }

    openlog ("kotekan", log_options, LOG_LOCAL1);

    // Load configuration file.
    //INFO("Kotekan starting with config file %s", config_file_name);
    const char git_hash[] = GIT_COMMIT_HASH;
    const char git_branch[] = GIT_BRANCH;
    INFO("Kotekan %f starting build: %s, on branch: %s, with config file: %s",
            KOTEKAN_VERSION, git_hash, git_branch, config_file_name);

    Config config;

    restServer *rest_server = get_rest_server();
    rest_server->start();

    config.parse_file(config_file_name, 0);
    config.generate_extra_options();
    config.dump_config();

    // Adjust the log level
    int log_level = config.get_int("/system/log_level");

    log_level_warn = 0;
    log_level_debug = 0;
    log_level_info = 0;
    switch (log_level) {
        case 3:
            log_level_debug = 1;
        case 2:
            log_level_info = 1;
        case 1:
            log_level_warn = 1;
        default:
            break;
    }

    string mode = config.get_string("/system/mode");

    if (mode == "packet_cap") {
        packet_cap(config);
    } else if (mode == "chime_shuffle") {
        chime_shuffle_setup(config);
    } else {
        ERROR("Mode = %s does not exist", mode.c_str());
    }

    INFO("kotekan shutdown successfully.");

    closelog();

    return 0;
}
