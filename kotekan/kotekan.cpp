#include <array>
#include <assert.h>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <errno.h>
#include <fcntl.h>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <math.h>
#include <memory.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <strings.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

extern "C" {
#include <pthread.h>
}

#include "Config.hpp"
#include "Stage.hpp"
#include "StageFactory.hpp"
#include "basebandApiManager.hpp"
#include "buffer.h"
#include "errors.h"
#include "fpga_header_functions.h"
#include "gpsTime.h"
#include "kotekanLogging.hpp"
#include "kotekanMode.hpp"
#include "prometheusMetrics.hpp"
#include "restServer.hpp"
#include "util.h"
#include "version.h"
#include "visUtil.hpp"

#include "fmt.hpp"
#include "json.hpp"

#ifdef WITH_HSA
#include "hsaBase.h"
#endif

using json = nlohmann::json;
using namespace kotekan;

// Embedded script for converting the YAML config to json
const std::string yaml_to_json = R"(
import yaml, json, sys, os, subprocess

file_name = sys.argv[1]
gps_server = ""
if len(sys.argv) == 3:
    gps_server = sys.argv[2]

# Lint the YAML file, helpful for finding errors
try:
    output = subprocess.Popen(["yamllint",
                               "-d",
                               "{extends: relaxed, \
                                 rules: {line-length: {max: 100}, \
                                        commas: disable, \
                                        brackets: disable, \
                                        trailing-spaces: {level: warning}}}" ,
                                 file_name],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    response,stderr = output.communicate()
    if response != "":
        sys.stderr.write("yamllint warnings/errors for: ")
        sys.stderr.write(str(response))
except OSError as e:
    if e.errno == os.errno.ENOENT:
        sys.stderr.write("yamllint not installed, skipping pre-validation\n")
    else:
        sys.stderr.write("error with yamllint, skipping pre-validation\n")

with open(file_name, "r") as stream:
    try:
        config_json = yaml.load(stream)
    except yaml.YAMLError as exc:
        sys.stderr.write(exc)

# Get the GPS server time if a server was given
if gps_server != "":
    import requests
    try:
        gps_request = requests.get(gps_server)
        gps_request.raise_for_status()
    except requests.exceptions.HTTPError as rex:
        config_json["gps_time"] = {}
        config_json["gps_time"]["error"] = str(rex)
        sys.stdout.write(json.dumps(config_json))
        quit()
    except requests.exceptions.RequestException as rex:
        config_json["gps_time"] = {}
        config_json["gps_time"]["error"] = str(rex)
        sys.stdout.write(json.dumps(config_json))
        quit()

    try:
        config_json["gps_time"] = gps_request.json()
    except:
        config_json["gps_time"] = {}
        config_json["gps_time"]["error"] = "Server did not return valid JSON"

sys.stdout.write(json.dumps(config_json))
)";

// The default location for getting the GPS time reference
// TODO This entire GPS time system might be moved out of kotekan.cpp entirely
// since it's a very CHIME specific system.
const std::string default_gps_source = "http://carillon.chime:54321/get-frame0-time";

kotekanMode* kotekan_mode = nullptr;
bool running = false;
std::mutex kotekan_state_lock;
volatile std::sig_atomic_t sig_value = 0;

void signal_handler(int signal) {
    sig_value = signal;
}

void print_help() {
    printf("usage: kotekan [opts]\n\n");
    printf("Options:\n");
    printf("    --config (-c) [file]           The local JSON config file to use.\n");
    printf("    --bind-address (-b) [ip:port]  The IP address and port to bind"
           " (default 0.0.0.0:12048)\n");
    printf("    --gps-time (-g)                Used with -c, try to get GPS time"
           " (CHIME cmd line runs only).\n");
    printf("    --gps-time-source (-t)         URL for GPS server (used with -g) default: %s\n",
           default_gps_source.c_str());
    printf("    --syslog (-s)                  Send a copy of the output to syslog.\n");
    printf("    --no-stderr (-n)               Disables output to std error if syslog (-s) is "
           "enabled.\n");
    printf("    --version (-v)                 Prints the kotekan version and build details.\n\n");
    printf("If no options are given then kotekan runs in daemon mode and\n");
    printf("expects to get it configuration via the REST endpoint '/start'.\n");
    printf("In daemon mode output is only sent to syslog.\n\n");
}

void print_version() {
    printf("Kotekan version %s\n", get_kotekan_version());
    printf("Build branch: %s\n", get_git_branch());
    printf("Git commit hash: %s\n\n", get_git_commit_hash());
    printf("CMake build settings: \n%s\n", get_cmake_build_options());

    printf("Available kotekan stages:\n");
    std::map<std::string, StageMaker*> known_stages = StageFactoryRegistry::get_registered_stages();
    for (auto& stage_maker : known_stages) {
        if (stage_maker.first != known_stages.rbegin()->first) {
            printf("%s, ", stage_maker.first.c_str());
        } else {
            printf("%s\n\n", stage_maker.first.c_str());
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
    vector<string> available_stages;
    std::map<std::string, StageMaker*> known_stages = StageFactoryRegistry::get_registered_stages();
    for (auto& stage_maker : known_stages)
        available_stages.push_back(stage_maker.first);
    version_json["available_stages"] = available_stages;
    return version_json;
}

std::string exec(const std::string& cmd) {
    std::array<char, 256> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe)
        throw std::runtime_error(fmt::format(fmt("popen() for the command {:s} failed!"), cmd));
    while (!feof(pipe.get())) {
        if (fgets(buffer.data(), 256, pipe.get()) != NULL)
            result += buffer.data();
    }
    return result;
}

void update_log_levels(Config& config) {
    // Adjust the log level
    string s_log_level = config.get<std::string>("/", "log_level");
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
        throw std::runtime_error(
            fmt::format(fmt("The value given for log_level: '{:s}' is not valid! (It should be one "
                            "of 'off', 'error', 'warn', 'info', 'debug', 'debug2')"),
                        s_log_level));
    }

    _global_log_level = static_cast<std::underlying_type<logLevel>::type>(log_level);
}

/**
 * @brief Sets the global GPS time reference
 *
 * @param config config file containing the GPS time.
 * @return True if the config contained a GPS time, and false if not.
 */
bool set_gps_time(Config& config) {
    if (config.exists("/", "gps_time") && !config.exists("/gps_time", "error")
        && config.exists("/gps_time", "frame0_nano")) {

        uint64_t frame0 = config.get<uint64_t>("/gps_time", "frame0_nano");
        set_global_gps_time(frame0);
        INFO_NON_OO("Set FPGA frame 0 time to {:d} nanoseconds since Unix Epoch\n", frame0);
        return true;
    }

    if (config.exists("/gps_time", "error")) {
        string error_message = config.get<std::string>("/gps_time", "error");
        ERROR_NON_OO("*****\nGPS time lookup failed with reason: \n {:s}\n ******\n",
                     error_message);
    } else {
        WARN_NON_OO("No GPS time set, using system clock.");
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
void start_new_kotekan_mode(Config& config, bool requires_gps_time) {
    config.dump_config();
    update_log_levels(config);
    if (!set_gps_time(config)) {
        if (requires_gps_time) {
            ERROR_NON_OO("GPS time was expected to be provided!");
            throw std::runtime_error("GPS time required but not set.");
        }
    }

    kotekan_mode = new kotekanMode(config);

    kotekan_mode->initalize_stages();
    kotekan_mode->start_stages();
    running = true;
}

int main(int argc, char** argv) {

    std::signal(SIGINT, signal_handler);

    int opt_val = 0;
    char* config_file_name = (char*)"none";
    int log_options = LOG_CONS | LOG_PID | LOG_NDELAY;
    bool gps_time = false;
    bool enable_stderr = true;
    std::string gps_time_source = default_gps_source;
    std::string bind_address = "0.0.0.0:12048";
    // We disable syslog to start.
    // If only --config is provided, then we only send messages to stderr
    // If --syslog is added, then output is to both syslog and stderr
    // If no options are given then stderr is disabled, and syslog is enabled.
    // The no options mode is the default daemon mode where it expects a remote config
    __enable_syslog = 0;

    for (;;) {
        static struct option long_options[] = {{"config", required_argument, 0, 'c'},
                                               {"bind-address", required_argument, 0, 'b'},
                                               {"gps-time", no_argument, 0, 'g'},
                                               {"gps-time-source", required_argument, 0, 't'},
                                               {"help", no_argument, 0, 'h'},
                                               {"syslog", no_argument, 0, 's'},
                                               {"no-stderr", no_argument, 0, 'n'},
                                               {"version", no_argument, 0, 'v'},
                                               {0, 0, 0, 0}};

        int option_index = 0;

        opt_val = getopt_long(argc, argv, "gt:hc:b:snv", long_options, &option_index);

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
            case 'b':
                bind_address = string(optarg);
                break;
            case 't':
                gps_time_source = string(optarg);
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
        openlog("kotekan", log_options, LOG_LOCAL1);
        if (!enable_stderr)
            fprintf(stderr, "Kotekan logging to syslog only!");
    }

    // Load configuration file.
    INFO_NON_OO("Kotekan version {:s} starting...", get_kotekan_version());

    Config config;

    restServer& rest_server = restServer::instance();
    std::vector<std::string> address_parts = regex_split(bind_address, ":");
    // TODO validate IP and port
    rest_server.start(address_parts.at(0), std::stoi(address_parts.at(1)));

    if (string(config_file_name) != "none") {
        // TODO should be in a try catch block, to make failures cleaner.
        std::lock_guard<std::mutex> lock(kotekan_state_lock);
        INFO_NON_OO("Opening config file {:s}", config_file_name);

        std::string exec_command;
        if (gps_time) {
            INFO_NON_OO("Getting GPS time from server ({:s}), this might take some time...",
                        gps_time_source);
            exec_command = fmt::format(fmt("python -c '{:s}' {:s} {:s}"), yaml_to_json,
                                       config_file_name, gps_time_source);
        } else {
            exec_command =
                fmt::format(fmt("python -c '{:s}' {:s}"), yaml_to_json, config_file_name);
        }
        std::string json_string = exec(exec_command.c_str());
        json config_json = json::parse(json_string.c_str());
        config.update_config(config_json);
        try {
            start_new_kotekan_mode(config, gps_time);
        } catch (const std::exception& ex) {
            ERROR_NON_OO("Failed to start kotekan with config file {:s}, error message: {:s}",
                         config_file_name, ex.what());
            ERROR_NON_OO("Exiting...");
            exit(-1);
        }
    }

    // Main REST callbacks.
    rest_server.register_post_callback("/start", [&](connectionInstance& conn, json& json_config) {
        std::lock_guard<std::mutex> lock(kotekan_state_lock);
        if (running) {
            WARN_NON_OO(
                "/start was called, but the system is already running, ignoring start request.");
            conn.send_error("Already running", HTTP_RESPONSE::REQUEST_FAILED);
            return;
        }

        config.update_config(json_config);

        try {
            INFO_NON_OO("Starting new kotekan mode using POSTed config.");
            start_new_kotekan_mode(config, false);
        } catch (const std::out_of_range& ex) {
            delete kotekan_mode;
            kotekan_mode = nullptr;
            conn.send_error(ex.what(), HTTP_RESPONSE::BAD_REQUEST);
            // TODO This exit shouldn't be required, but some stages aren't able
            // to fully clean up on system failure.  This results in the system
            // getting into a bad state if the posted config is invalid.
            // See ticket: #464
            // The same applies to exit (raise) statements in other parts of
            // this try statement.
            FATAL_ERROR_NON_OO("Provided config had an out of range exception: {:s}", ex.what());
            return;
        } catch (const std::runtime_error& ex) {
            delete kotekan_mode;
            kotekan_mode = nullptr;
            conn.send_error(ex.what(), HTTP_RESPONSE::BAD_REQUEST);
            FATAL_ERROR_NON_OO("Provided config failed to start with runtime error: {:s}",
                               ex.what());
            return;
        } catch (const std::exception& ex) {
            delete kotekan_mode;
            kotekan_mode = nullptr;
            conn.send_error(ex.what(), HTTP_RESPONSE::BAD_REQUEST);
            FATAL_ERROR_NON_OO("Provided config failed with exception: {:s}", ex.what());
            return;
        }
        conn.send_empty_reply(HTTP_RESPONSE::OK);
    });

    rest_server.register_get_callback("/stop", [&](connectionInstance& conn) {
        std::lock_guard<std::mutex> lock(kotekan_state_lock);
        if (!running) {
            WARN_NON_OO("/stop called, but the system is already stopped, ignoring stop request.");
            conn.send_error("kotekan is already stopped", HTTP_RESPONSE::REQUEST_FAILED);
            return;
        }
        INFO_NON_OO("/stop endpoint called, shutting down current config.");
        assert(kotekan_mode != nullptr);
        kotekan_mode->stop_stages();
        // TODO should we have three states (running, shutting down, and stopped)?
        // This would prevent this function from blocking on join.
        kotekan_mode->join();
        delete kotekan_mode;
        kotekan_mode = nullptr;
        running = false;
        conn.send_empty_reply(HTTP_RESPONSE::OK);
    });

    rest_server.register_get_callback("/kill", [&](connectionInstance& conn) {
        ERROR_NON_OO(
            "/kill endpoint called, raising SIGINT to shutdown the kotekan system process.");
        kotekan::kotekanLogging::set_error_message("/kill endpoint called.");
        exit_kotekan(ReturnCode::CLEAN_EXIT);
        conn.send_empty_reply(HTTP_RESPONSE::OK);
    });

    rest_server.register_get_callback("/status", [&](connectionInstance& conn) {
        std::lock_guard<std::mutex> lock(kotekan_state_lock);
        json reply;
        reply["running"] = running;
        conn.send_json_reply(reply);
    });

    json version_json = get_json_version_into();

    rest_server.register_get_callback(
        "/version", [&](connectionInstance& conn) { conn.send_json_reply(version_json); });

    auto& metrics = prometheus::Metrics::instance();
    metrics.register_with_server(&rest_server);
    auto& kotekan_running_metric = metrics.add_gauge("kotekan_running", "main");
    kotekan_running_metric.set(running);

    basebandApiManager& baseband = basebandApiManager::instance();
    baseband.register_with_server(&rest_server);

    for (EVER) {
        sleep(1);
        // Update running state
        {
            std::lock_guard<std::mutex> lock(kotekan_state_lock);
            kotekan_running_metric.set(running);
        }

        if (sig_value == SIGINT) {
            INFO_NON_OO("Got SIGINT, shutting down kotekan...");
            std::lock_guard<std::mutex> lock(kotekan_state_lock);
            if (kotekan_mode != nullptr) {
                INFO_NON_OO("Attempting to stop and join kotekan_stages...");
                kotekan_mode->stop_stages();
                kotekan_mode->join();
                delete kotekan_mode;
            }
            break;
        }
    }

    INFO_NON_OO("kotekan shutdown with status: {:s}", get_exit_code_string(get_exit_code()));

    // Print error message if there is one.
    if (string(get_error_message()) != "not set") {
        INFO_NON_OO("Fatal error message was: {:s}", get_error_message());
    }

    closelog();

    return get_exit_code();
}
