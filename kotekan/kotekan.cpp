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
#include <atomic>
#include <mutex>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>

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
#include <rte_ring.h>
}

#include "errors.h"
#include "buffers.h"

#include "Config.hpp"
#include "util.h"
#include "version.h"
#include "SampleProcess.hpp"
#include "json.hpp"
#include "restServer.hpp"

#ifdef WITH_HSA
    #include "chimeShuffleMode.hpp"
    #include "gpuTestMode.hpp"
#endif
#ifdef WITH_OPENCL
    #include "clProcess.hpp"
#endif

#include "packetCapMode.hpp"
#include "singleDishMode.hpp"

using json = nlohmann::json;

kotekanMode * kotekan_mode = nullptr;
bool running = false;
std::mutex kotekan_state_lock;

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
#ifdef DPDK_VDIF_MODE
    char  arg4[] = "FF";
#else
    char  arg4[] = "F";
#endif
    char  arg5[] = "-m";
    char  arg6[] = "256";
    char* argv2[] = { &arg0[0], &arg1[0], &arg2[0], &arg3[0], &arg4[0], &arg5[0], &arg6[0], NULL };
    int   argc2   = (int)(sizeof(argv2) / sizeof(argv2[0])) - 1;

    /* Initialize the Environment Abstraction Layer (EAL). */
    int ret2 = rte_eal_init(argc2, argv2);
    if (ret2 < 0)
        exit(EXIT_FAILURE);
}

std::string exec(const std::string &cmd) {
    std::array<char, 256> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() for the command " + cmd + " failed!");
    while (!feof(pipe.get())) {
        if (fgets(buffer.data(), 256, pipe.get()) != NULL)
            result += buffer.data();
    }
    return result;
}

void update_log_levels(Config &config) {
    // Adjust the log level
    int log_level = config.get_int("/system/", "log_level");

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
}

int start_new_kotekan_mode(Config &config) {

    config.dump_config();
    update_log_levels(config);

    string mode = config.get_string("/system", "mode");

    if (mode == "packet_cap") {
        kotekan_mode = (kotekanMode *) new packetCapMode(config);
    } else if (mode == "chime_shuffle") {
        #ifdef WITH_HSA
            kotekan_mode = (kotekanMode *) new chimeShuffleMode(config);
        #else
        return -1;
        #endif
    } else if (mode == "gpu_test") {
        #ifdef WITH_HSA
            kotekan_mode = (kotekanMode *) new gpuTestMode(config);
        #else
        return -1;
        #endif
    } else if (mode == "single_dish") {
        kotekan_mode = (kotekanMode *) new singleDishMode(config);
    } else {
        return -1;
    }
    kotekan_mode->initalize_processes();
    kotekan_mode->start_processes();
    running = true;

    return 0;
}

int main(int argc, char ** argv) {

    dpdk_setup();
    json config_json;

    int opt_val = 0;
    char * config_file_name = (char *)"none";
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
    INFO("Kotekan %f starting build: %s, on branch: %s",
            KOTEKAN_VERSION, git_hash, git_branch);

    Config config;

    restServer *rest_server = get_rest_server();
    rest_server->start();


    if (string(config_file_name) != "none") {
        // TODO should be in a try catch block, to make failures cleaner.
        std::lock_guard<std::mutex> lock(kotekan_state_lock);
        //config.parse_file(config_file_name, 0);
        std::string json_string = exec("python ../../scripts/yaml_to_json.py " + std::string(config_file_name));
        config_json = json::parse(json_string.c_str());
        config.update_config(config_json, 0);
        if (start_new_kotekan_mode(config) == -1) {
            ERROR("Mode not supported");
            return -1;
        }
    }

    // Main REST callbacks.
    rest_server->register_json_callback("/start", [&] (connectionInstance &conn, json& json_config) {
        std::lock_guard<std::mutex> lock(kotekan_state_lock);
        if (running) {
            conn.send_error("Already running", STATUS_REQUEST_FAILED);
        }

        config.update_config(json_config, 0);

        try {
            if (!start_new_kotekan_mode(config)) {
                conn.send_error("Mode not supported", STATUS_BAD_REQUEST);
                return;
            }
        } catch (std::out_of_range ex) {
            DEBUG("Out of range exception %s", ex.what());
            delete kotekan_mode;
            kotekan_mode = nullptr;
            conn.send_error(ex.what(), STATUS_BAD_REQUEST);
            return;
        } catch (std::runtime_error ex) {
            DEBUG("Runtime error %s", ex.what());
            delete kotekan_mode;
            kotekan_mode = nullptr;
            conn.send_error(ex.what(), STATUS_BAD_REQUEST);
            return;
        } catch (std::exception ex) {
            DEBUG("Generic exception %s", ex.what());
            delete kotekan_mode;
            kotekan_mode = nullptr;
            conn.send_error(ex.what(), STATUS_BAD_REQUEST);
            return;
        }
        conn.send_empty_reply(STATUS_OK);
    });

    rest_server->register_json_callback("/stop", [&](connectionInstance &conn, json &json_request) {
        std::lock_guard<std::mutex> lock(kotekan_state_lock);
        if (!running) {
            conn.send_error("kotekan is already stopped", STATUS_REQUEST_FAILED);
            return;
        }
        assert(kotekan_mode != nullptr);
        kotekan_mode->stop_processes();
        // TODO should we have three states (running, shutting down, and stoped)?
        // This would prevent this function from blocking on join.
        kotekan_mode->join();
        delete kotekan_mode;
        kotekan_mode = nullptr;
        conn.send_empty_reply(STATUS_OK);
    });

    rest_server->register_json_callback("/status", [&](connectionInstance &conn, json &json_request){
        std::lock_guard<std::mutex> lock(kotekan_state_lock);
        json reply;
        reply["running"] = running;
        conn.send_json_reply(reply);
    });

    for(EVER){
        // Note you cannot actaully kill kotekan from the REST interface, it's always running.
        // Maybe we should transfer control to the reserver loop here, but this isn't expensive,
        // and might be a useful loop for other things.
        sleep(1000);
    }

    INFO("kotekan shutdown successfully.");

    closelog();

    return 0;
}
