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
#include <strings.h>
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
#include <csignal>
#include "configEval.hpp"

extern "C" {
#include <pthread.h>
}

#include "errors.h"
#include "buffer.h"

#include "Config.hpp"
#include "util.h"
#include "version.h"
#include "json.hpp"
#include "restServer.hpp"
#include "kotekanMode.hpp"
#include "fpga_header_functions.h"
#include "gpsTime.h"
#include "KotekanProcess.hpp"
#include "prometheusMetrics.hpp"
#include "basebandRequestManager.hpp"
#include "processFactory.hpp"

#ifdef WITH_HSA
#include "hsaBase.h"
#endif

using json = nlohmann::json;

kotekanMode * kotekan_mode = nullptr;
bool running = false;
std::mutex kotekan_state_lock;
volatile std::sig_atomic_t sig_value = 0;

void signal_handler(int signal)
{
    sig_value = signal;
}

void print_help() {
    printf("usage: kotekan [opts]\n\n");
    printf("Options:\n");
    printf("    --config (-c) [file]        The local JSON config file to use.\n");
    printf("    --config-daemon (-d) [file] Same as -c, but uses installed yaml->json script\n");
    printf("    --gps-time (-g)             Used with -c, try to get GPS time (CHIME cmd line runs only).\n");
    printf("    --syslog (-s)               Send a copy of the output to syslog.\n");
    printf("    --no-stderr (-n)            Disables output to std error if syslog (-s) is enabled.\n");
    printf("    --version (-v)              Prints the kotekan version and build details.\n\n");
    printf("If no options are given then kotekan runs in daemon mode and\n");
    printf("expects to get it configuration via the REST endpoint '/start'.\n");
    printf("In daemon mode output is only sent to syslog.\n\n");
}

void print_version() {
    printf("Kotekan version %s\n", get_kotekan_version());
    printf("Build branch: %s\n", get_git_branch());
    printf("Git commit hash: %s\n\n", get_git_commit_hash());
    printf("CMake build settings: \n%s\n", get_cmake_build_options());

    printf("Available kotekan processes:\n");
    std::map<std::string, kotekanProcessMaker*> known_processes = processFactoryRegistry::get_registered_processes();
    for (auto &process_maker : known_processes) {
        if (process_maker.first != known_processes.rbegin()->first) {
            printf("%s, ", process_maker.first.c_str());
        } else {
            printf("%s\n\n", process_maker.first.c_str());
        }
    }
}

json get_json_version_into() {
    // Create version information
    json version_json;
    version_json["kotekan_version"] = get_kotekan_version();
    version_json["branch"] = get_git_branch();
    version_json["git_commit_hash"] = get_git_commit_hash();
    version_json["cmake_build_settings"] = get_cmake_build_options();
    vector<string> available_processes;
    std::map<std::string, kotekanProcessMaker*> known_processes = processFactoryRegistry::get_registered_processes();
    for (auto &process_maker : known_processes)
        available_processes.push_back(process_maker.first);
    version_json["available_processes"] = available_processes;
    return version_json;
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
    string s_log_level = config.get_string("/", "log_level");
    logLevel log_level;

    if (strcasecmp(s_log_level.c_str(), "off") == 0) {
        log_level = logLevel::OFF;
    } else if (strcasecmp(s_log_level.c_str(), "error") == 0) {
        log_level = logLevel::ERROR;
    } else if (strcasecmp(s_log_level.c_str(), "warn") == 0) {
        log_level = logLevel::WARN;
    } else if (strcasecmp(s_log_level.c_str(), "info") == 0) {
        log_level = logLevel::INFO;
    } else if (strcasecmp(s_log_level.c_str(), "debug") == 0) {
        log_level = logLevel::DEBUG;
    } else if (strcasecmp(s_log_level.c_str(), "debug2") == 0) {
        log_level = logLevel::DEBUG2;
    } else {
        throw std::runtime_error("The value given for log_level: '" + s_log_level + "is not valid! " +
                "(It should be one of 'off', 'error', 'warn', 'info', 'debug', 'debug2')");
    }

    __log_level = static_cast<std::underlying_type<logLevel>::type>(log_level);
}

/**
 * @brief Sets the global GPS time reference
 *
 * @param config config file containing the GPS time.
 * @return True if the config contained a GPS time, and false if not.
 */
bool set_gps_time(Config &config) {
    if (config.exists("/", "gps_time") &&
        !config.exists("/gps_time", "error") &&
        config.exists("/gps_time", "frame0_nano")) {

        uint64_t frame0 = config.get_uint64("/gps_time", "frame0_nano");
        set_global_gps_time(frame0);
        INFO("Set FPGA frame 0 time to %" PRIu64 " nanoseconds since Unix Epoch\n", frame0);
        return true;
    }

    if (config.exists("/gps_time", "error")) {
        string error_message = config.get_string("/gps_time", "error");
        ERROR("*****\nGPS time lookup failed with reason: \n %s\n ******\n",
                error_message.c_str());
    } else {
        WARN("No GPS time set, using system clock.");
    }
    return false;
}

/**
 * @brief Starts a new kotekan mode (config instance)
 *
 * @param config The config to generate the instance from
 * @param requires_gps_time If set to true, then the config must provide a valid time
 *                          otherwise an error is thrown.
 */
void start_new_kotekan_mode(Config &config, bool requires_gps_time) {
    config.dump_config();
    update_log_levels(config);
    if(!set_gps_time(config)) {
        if (requires_gps_time) {
            ERROR("GPS time was expected to be provided!");
            throw std::runtime_error("GPS time required but not set.");
        }
    }

    kotekan_mode = new kotekanMode(config);

    kotekan_mode->initalize_processes();
    kotekan_mode->start_processes();
    running = true;
}

int main(int argc, char ** argv) {

    std::signal(SIGINT, signal_handler);

    int opt_val = 0;
    char * config_file_name = (char *)"none";
    int log_options = LOG_CONS | LOG_PID | LOG_NDELAY;
    bool opt_d_set = false;
    bool gps_time = false;
    bool enable_stderr = true;
    // We disable syslog to start.
    // If only --config is provided, then we only send messages to stderr
    // If --syslog is added, then output is to both syslog and stderr
    // If no options are given then stderr is disabled, and syslog is enabled.
    // The no options mode is the default daemon mode where it expects a remote config
    __enable_syslog = 0;

    for (;;) {
        static struct option long_options[] = {
            {"config", required_argument, 0, 'c'},
            {"config-daemon", required_argument, 0, 'd'},
            {"gps-time", no_argument, 0, 'g'},
            {"help", no_argument, 0, 'h'},
            {"syslog", no_argument, 0, 's'},
            {"no-stderr", no_argument, 0, 'n'},
            {"version", no_argument, 0, 'v'},
            {0, 0, 0, 0}
        };

        int option_index = 0;

        opt_val = getopt_long (argc, argv, "ghc:d:snv",
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
            case 'd':
                config_file_name = strdup(optarg);
                opt_d_set = true;
                break;
            case 'g':
                gps_time = true;
                break;
            case 's':
                __enable_syslog = 1;
                break;
            case 'n':
                enable_stderr = false;
                break;
            case 'v':
                print_version();
                return 0;
                break;
            default:
                printf("Invalid option, run with -h to see options");
                return -1;
                break;
        }
    }

#ifdef WITH_HSA
    kotekan_hsa_start();
#endif

    if (string(config_file_name) == "none") {
        __enable_syslog = 1;
        fprintf(stderr, "Kotekan running in daemon mode, output is to syslog only.\n");
        fprintf(stderr, "Configuration should be provided via the `/start` REST endpoint.\n");
    }

    if (string(config_file_name) != "none" && enable_stderr) {
        log_options |= LOG_PERROR;
    }

    if (__enable_syslog == 1) {
        openlog ("kotekan", log_options, LOG_LOCAL1);
        if (!enable_stderr)
            fprintf(stderr, "Kotekan logging to syslog only!");
    }

    // Load configuration file.
    INFO("Kotekan version %s starting...",
            get_kotekan_version());

    Config config;

    restServer &rest_server = restServer::instance();

    if (string(config_file_name) != "none") {
        // TODO should be in a try catch block, to make failures cleaner.
        std::lock_guard<std::mutex> lock(kotekan_state_lock);
        INFO("Opening config file %s", config_file_name);

        string exec_script;
        string exec_base;
        if (gps_time) {
            INFO("Getting GPS time from ch_master, this might take some time...");
            exec_script = "gps_yaml_to_json.py ";
        } else {
            exec_script = "yaml_to_json.py ";
        }
        if (opt_d_set) {
            exec_base = "/usr/local/bin/";
        } else {
            exec_base = "../../scripts/";
        }
        string exec_command = "python " + exec_base + exec_script + std::string(config_file_name);
        std::string json_string = exec(exec_command.c_str());
        json config_json = json::parse(json_string.c_str());
        config.update_config(config_json);
        try {
            start_new_kotekan_mode(config, gps_time);
        } catch (const std::exception &ex) {
            ERROR("Failed to start kotekan with config file %s, error message: %s",
                  config_file_name, ex.what());
            ERROR("Exiting...");
            exit(-1);
        }
    }

    // Main REST callbacks.
    rest_server.register_post_callback("/start", [&] (connectionInstance &conn, json& json_config) {
        std::lock_guard<std::mutex> lock(kotekan_state_lock);
        if (running) {
            conn.send_error("Already running", HTTP_RESPONSE::REQUEST_FAILED);
        }

        config.update_config(json_config);

        try {
            start_new_kotekan_mode(config, false);
        } catch (const std::out_of_range &ex) {
            ERROR("Out of range exception %s", ex.what());
            delete kotekan_mode;
            kotekan_mode = nullptr;
            conn.send_error(ex.what(), HTTP_RESPONSE::BAD_REQUEST);
            return;
        } catch (const std::runtime_error &ex) {
            ERROR("Runtime error %s", ex.what());
            delete kotekan_mode;
            kotekan_mode = nullptr;
            conn.send_error(ex.what(), HTTP_RESPONSE::BAD_REQUEST);
            return;
        } catch (const std::exception &ex) {
            ERROR("Generic exception %s", ex.what());
            delete kotekan_mode;
            kotekan_mode = nullptr;
            conn.send_error(ex.what(), HTTP_RESPONSE::BAD_REQUEST);
            return;
        }
        conn.send_empty_reply(HTTP_RESPONSE::OK);
    });

    rest_server.register_get_callback("/stop", [&](connectionInstance &conn) {
        std::lock_guard<std::mutex> lock(kotekan_state_lock);
        if (!running) {
            conn.send_error("kotekan is already stopped", HTTP_RESPONSE::REQUEST_FAILED);
            return;
        }
        assert(kotekan_mode != nullptr);
        kotekan_mode->stop_processes();
        // TODO should we have three states (running, shutting down, and stopped)?
        // This would prevent this function from blocking on join.
        kotekan_mode->join();
        delete kotekan_mode;
        kotekan_mode = nullptr;
        running = false;
        conn.send_empty_reply(HTTP_RESPONSE::OK);
    });

    rest_server.register_get_callback("/kill", [&](connectionInstance &conn) {
        raise(SIGINT);
        conn.send_empty_reply(HTTP_RESPONSE::OK);
    });

    rest_server.register_get_callback("/status", [&](connectionInstance &conn){
        std::lock_guard<std::mutex> lock(kotekan_state_lock);
        json reply;
        reply["running"] = running;
        conn.send_json_reply(reply);
    });

    json version_json = get_json_version_into();

    rest_server.register_get_callback("/version", [&](connectionInstance &conn) {
        conn.send_json_reply(version_json);
    });

    prometheusMetrics &metrics = prometheusMetrics::instance();
    metrics.register_with_server(&rest_server);

    basebandRequestManager &baseband = basebandRequestManager::instance();
    baseband.register_with_server(&rest_server);

    for(EVER){
        sleep(1);
        // Update running state
        {
            std::lock_guard<std::mutex> lock(kotekan_state_lock);
            metrics.add_process_metric("kotekan_running", "main", running);
        }

        if (sig_value == SIGINT) {
            INFO("Got SIGINT, shutting down kotekan...");
            std::lock_guard<std::mutex> lock(kotekan_state_lock);
            if (kotekan_mode != nullptr) {
                INFO("Attempting to stop and join kotekan_processes...");
                kotekan_mode->stop_processes();
                kotekan_mode->join();
                delete kotekan_mode;
            }
            break;
        }
    }

    INFO("kotekan shutdown successfully.");

    closelog();

    return 0;
}
