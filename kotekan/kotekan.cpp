#include "Config.hpp"             // for Config
#include "StageFactory.hpp"       // for StageFactoryRegistry, StageMaker
#include "basebandApiManager.hpp" // for basebandApiManager
#include "errors.h"               // for get_error_message, get_exit_code, __enable_syslog, exi...
#include "kotekanLogging.hpp"     // for INFO_NON_OO, logLevel, ERROR_NON_OO, FATAL_ERROR_NON_OO
#include "kotekanMode.hpp"        // for kotekanMode
#include "prometheusMetrics.hpp"  // for Metrics, Gauge
#include "restServer.hpp"         // for connectionInstance, HTTP_RESPONSE, restServer, HTTP_RE...
#include "util.h"                 // for EVER
#include "version.h"              // for get_kotekan_version, get_cmake_build_options, get_git_...
#include "visUtil.hpp"            // for regex_split

#include "fmt.hpp"  // for format, fmt
#include "json.hpp" // for basic_json<>::object_t, basic_json<>::value_type, json

#include <algorithm>   // for max
#include <array>       // for array
#include <assert.h>    // for assert
#include <csignal>     // for signal, SIGINT, sig_atomic_t
#include <exception>   // for exception
#include <getopt.h>    // for no_argument, getopt_long, required_argument, option
#include <iostream>    // for endl, basic_ostream, cout, ostream
#include <iterator>    // for reverse_iterator
#include <map>         // for map
#include <memory>      // for allocator, shared_ptr
#include <mutex>       // for mutex, lock_guard
#include <stdexcept>   // for runtime_error, out_of_range
#include <stdio.h>     // for printf, fprintf, feof, fgets, popen, stderr, pclose
#include <stdlib.h>    // for exit, free
#include <string.h>    // for strdup
#include <string>      // for string, basic_string, operator!=, operator<<, operator==
#include <strings.h>   // for strcasecmp
#include <syslog.h>    // for closelog, openlog, LOG_CONS, LOG_LOCAL1, LOG_NDELAY
#include <type_traits> // for underlying_type, underlying_type<>::type
#include <unistd.h>    // for optarg, sleep
#include <utility>     // for pair
#include <vector>      // for vector


#ifdef WITH_HSA
#include "hsaBase.h"
#endif

using std::string;
using json = nlohmann::json;
using namespace kotekan;

// Embedded script for converting the YAML config to json
const std::string yaml_to_json = R"(
import yaml, json, sys, os, subprocess, errno

file_name = sys.argv[1]

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
# TODO: change to checking for OSError subtypes when Python 2 support is removed
except OSError as e:
    if e.errno == errno.ENOENT:
        sys.stderr.write("yamllint not installed, skipping pre-validation\n")
    else:
        sys.stderr.write("error with yamllint, skipping pre-validation\n")

with open(file_name, "r") as stream:
    try:
        config_json = yaml.load(stream)
    except yaml.YAMLError as exc:
        sys.stderr.write(exc)

sys.stdout.write(json.dumps(config_json))
)";

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
    printf("    --syslog (-s)                  Send a copy of the output to syslog.\n");
    printf("    --no-stderr (-n)               Disables output to std error if syslog (-s) is "
           "enabled.\n");
    printf("    --version (-v)                 Prints the kotekan version and build details.\n");
    printf("    --print-config (-p)            Prints the config file being used.\n\n");
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


std::vector<std::string> split_string(const std::string& s, const std::string& delimiter) {

    std::vector<std::string> tokens;

    size_t start = 0;

    while (start <= s.size()) {
        size_t end = s.find(delimiter, start);

        // If no match was found, then we should select to the end of the string
        if (end == std::string::npos)
            end = s.size();

        // If a match was found at the start then we shouldn't add anything
        if (end != start)
            tokens.push_back(s.substr(start, end - start));

        start = end + delimiter.size();
    }

    return tokens;
}

std::string trim(std::string& s) {
    s.erase(0, s.find_first_not_of(' '));
    s.erase(s.find_last_not_of(' ') + 1);
    return s;
}

json parse_cmake_options() {
    auto options = split_string(get_cmake_build_options(), "\n");

    json j;

    for (auto opt : options) {

        // Trim off the indent from any nested options
        if (opt[1] == '-') {
            opt = opt.substr(2, opt.size() - 2);
        }

        auto t = split_string(opt, ":");
        auto key = trim(t[0]);
        auto val = trim(t[1]);

        j[key] = val;
    }
    return j;
}

json get_json_version_info() {
    // Create version information
    json version_json;
    version_json["kotekan_version"] = get_kotekan_version();
    version_json["branch"] = get_git_branch();
    version_json["git_commit_hash"] = get_git_commit_hash();
    version_json["cmake_build_settings"] = parse_cmake_options();
    std::vector<std::string> available_stages;
    std::map<std::string, StageMaker*> known_stages = StageFactoryRegistry::get_registered_stages();
    for (auto& stage_maker : known_stages)
        available_stages.push_back(stage_maker.first);
    version_json["available_stages"] = available_stages;
    return version_json;
}

void print_json_version() {
    std::cout << get_json_version_info().dump(2) << std::endl;
}

std::string exec(const std::string& cmd) {
    std::array<char, 256> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe)
        throw std::runtime_error(fmt::format(fmt("popen() for the command {:s} failed!"), cmd));
    while (!feof(pipe.get())) {
        if (fgets(buffer.data(), 256, pipe.get()) != nullptr)
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
 * @brief Starts a new kotekan mode (config instance)
 *
 * @param config The config to generate the instance from
 * @param dump_config If set to true, then the config file is printed to stdout.
 */
void start_new_kotekan_mode(Config& config, bool dump_config) {

    if (dump_config)
        config.dump_config();
    update_log_levels(config);

    kotekan_mode = new kotekanMode(config);

    kotekan_mode->initalize_stages();
    kotekan_mode->start_stages();
    running = true;
}

int main(int argc, char** argv) {

    std::signal(SIGINT, signal_handler);

    char* config_file_name = (char*)"none";
    int log_options = LOG_CONS | LOG_PID | LOG_NDELAY;
    bool enable_stderr = true;
    bool dump_config = false;
    std::string bind_address = "0.0.0.0:12048";
    // We disable syslog to start.
    // If only --config is provided, then we only send messages to stderr
    // If --syslog is added, then output is to both syslog and stderr
    // If no options are given then stderr is disabled, and syslog is enabled.
    // The no options mode is the default daemon mode where it expects a remote config
    __enable_syslog = 0;

    for (;;) {
        static struct option long_options[] = {{"config", required_argument, nullptr, 'c'},
                                               {"bind-address", required_argument, nullptr, 'b'},
                                               {"help", no_argument, nullptr, 'h'},
                                               {"syslog", no_argument, nullptr, 's'},
                                               {"no-stderr", no_argument, nullptr, 'n'},
                                               {"version", no_argument, nullptr, 'v'},
                                               {"version-json", no_argument, nullptr, 'j'},
                                               {"print-config", no_argument, nullptr, 'p'},
                                               {nullptr, 0, nullptr, 0}};

        int option_index = 0;

        int opt_val = getopt_long(argc, argv, "hc:b:snvp", long_options, &option_index);

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
            case 'j':
                print_json_version();
                return 0;
                break;
            case 'p':
                dump_config = true;
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

        std::string exec_command =
            fmt::format(fmt("python -c '{:s}' {:s}"), yaml_to_json, config_file_name);

        std::string json_string = exec(exec_command.c_str());
        json config_json = json::parse(json_string);
        config.update_config(config_json);
        try {
            start_new_kotekan_mode(config, dump_config);
        } catch (const std::exception& ex) {
            ERROR_NON_OO("Failed to start kotekan with config file {:s}, error message: {:s}",
                         config_file_name, ex.what());
            ERROR_NON_OO("Exiting...");
            exit(-1);
        }
        free(config_file_name);
        config_file_name = nullptr;
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
            start_new_kotekan_mode(config, dump_config);
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

    json version_json = get_json_version_info();

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
            rest_server.stop();
            break;
        }
    }

    metrics.remove_stage_metrics("/rest_server");

    INFO_NON_OO("kotekan shutdown with status: {:s}", get_exit_code_string(get_exit_code()));

    // Print error message if there is one.
    if (string(get_error_message()) != "not set") {
        INFO_NON_OO("Fatal error message was: {:s}", get_error_message());
    }

    closelog();

    return get_exit_code();
}
