#include "Config.hpp"             // for Config
#include "StageFactory.hpp"       // for StageFactoryRegistry, StageMaker
#include "basebandApiManager.hpp" // for basebandApiManager
#include "errors.h"               // for get_error_message, get_exit_code, __enable_syslog, exi...
#include "kotekanLogging.hpp"     // for INFO_NON_OO, logLevel, ERROR_NON_OO, FATAL_ERROR_NON_OO
#include "kotekanMode.hpp"        // for kotekanMode
#include "kotekanTrackers.hpp"    // for KotekanTrackers
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
#include <csignal>     // for signal, SIGINT, SIGTERM, sig_atomic_t
#include <exception>   // for exception
#include <getopt.h>    // for no_argument, getopt_long, required_argument, option
#include <iostream>    // for endl, basic_ostream, cout, ostream
#include <iterator>    // for reverse_iterator
#include <map>         // for map
#include <memory>      // for allocator_traits<>::value_type
#include <mutex>       // for mutex, lock_guard
#include <stdexcept>   // for runtime_error, out_of_range
#include <stdio.h>     // for printf, fprintf, feof, fgets, fdopen, stderr, fclose, STDOUT_FILENO
#include <stdlib.h>    // for exit, free
#include <string.h>    // for strdup
#include <string>      // for string, basic_string, operator!=, operator<<, operator==
#include <strings.h>   // for strcasecmp
#include <sys/wait.h>  // for waitpid
#include <syslog.h>    // for closelog, openlog, LOG_CONS, LOG_LOCAL1, LOG_NDELAY
#include <type_traits> // for underlying_type, underlying_type<>::type
#include <unistd.h>    // for optarg, sleep, pipe, fork, execvp, dup2
#include <utility>     // for pair
#include <vector>      // for vector


#ifdef WITH_HSA
#include "hsaBase.h"
#endif

using std::string;
using json = nlohmann::json;
using namespace kotekan;

// Embedded script for converting the YAML config to json
// Copied from python/scripts/config_to_json.py
// TODO copy this in automatically at compile time.
const std::string yaml_to_json = R"(
import argparse
import errno
import json
import os
import subprocess
import sys

try:
    import yaml
except ImportError as err:
    sys.stderr.write(
        "Missing pyyaml, run: pip3 install -r python/requirements.txt\n"
        + "Error message: "
        + str(err)
        + "\n"
    )
    exit(-1)

# Setup arg parser
parser = argparse.ArgumentParser(description="Convert YAML or Jinja files into JSON")
parser.add_argument("name", help="Config file name", type=str)
parser.add_argument(
    "-d", "--dump", help="Dump the yaml, useful with .j2 files", action="store_true"
)
parser.add_argument(
    "-e", "--variables", help="Add extra jinja variables, JSON format", type=str
)
args = parser.parse_args()

options = args.variables

# Split the file name into the name, directory path, and extension
file_name_full = args.name
file_ext = os.path.splitext(file_name_full)[1]
directory, file_name = os.path.split(file_name_full)

# Treat all files as pure YAML, unless it is a ".j2" file, then run jinja.
if file_ext != ".j2":
    # Lint the YAML file, helpful for finding errors
    try:
        output = subprocess.Popen(
            [
                "yamllint",
                "-d",
                "{extends: relaxed, \
                                     rules: {line-length: {max: 100}, \
                                            commas: disable, \
                                            brackets: disable, \
                                            trailing-spaces: {level: warning}}}",
                file_name_full,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        response, stderr = output.communicate()
        if response != "":
            sys.stderr.write("yamllint warnings/errors for: ")
            sys.stderr.write(str(response))
    # TODO: change to checking for OSError subtypes when Python 2 support is removed
    except OSError as e:
        if e.errno == errno.ENOENT:
            sys.stderr.write("yamllint not installed, skipping pre-validation\n")
        else:
            sys.stderr.write("error with yamllint, skipping pre-validation\n")

    try:
        with open(file_name_full, "r") as stream:
            config_yaml = yaml.safe_load(stream)
    except IOError as err:
        sys.stderr.write("Error reading file " + file_name_full + ": " + str(err))
        sys.exit(-1)
    except yaml.YAMLError as err:
        sys.stderr.write("Error parsing yaml: \n" + str(err) + "\n")
        sys.exit(-1)

    if args.dump:
        sys.stderr.write(yaml.dump(config_yaml) + "\n")

    sys.stdout.write(json.dumps(config_yaml))

else:
    try:
        from jinja2 import FileSystemLoader, Environment, select_autoescape
        from jinja2 import TemplateNotFound
    except ImportError as err:
        sys.stderr.write(
            "Jinja2 required for '.j2' files, run pip3 install -r python/requirements.txt"
            + "\nError message: "
            + str(err)
            + "\n"
        )
        exit(-1)

    # Load the template
    env = Environment(
        loader=FileSystemLoader(directory), autoescape=select_autoescape()
    )
    try:
        template = env.get_template(file_name)
    except TemplateNotFound as err:
        sys.stderr.write("Could not open the file: " + file_name_full + "\n")
        exit(-1)

    # Parse the optional variables (if any)
    options_dict = {}
    if options:
        options_dict = json.loads(str(options))

    # Convert to yaml
    config_yaml_raw = template.render(options_dict)

    # Dump the rendered yaml file if requested
    if args.dump:
        sys.stderr.write(config_yaml_raw + "\n")

    # TODO Should we also lint the output of the template?
    try:
        config_yaml = yaml.safe_load(config_yaml_raw)
    except yaml.YAMLError as err:
        sys.stderr.write("Error parsing yaml: \n" + str(err) + "\n")
        sys.exit(-1)

    sys.stdout.write(json.dumps(config_yaml))
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
    printf("    --variables (-e) [json]        Add Jinja config variables in JSON format\n");
    printf("    --version (-v)                 Prints the kotekan version and build details.\n");

    printf("    --print-json (-p)              Prints the json version of the config.\n");
    printf("    --print-yaml (-y)              Prints the yaml version of the config.\n\n");
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

std::string exec(std::vector<std::string>& cmd) {
    // create a pipe for interprocess communication
    int pipefds[2];
    if (pipe(pipefds))
        throw std::runtime_error("Could not create a pipe");

    // fork
    pid_t pid = fork();
    if (pid < 0)
        throw std::runtime_error("Unable to fork!");

    if (pid == 0) {
        // In child process

        // close the output side of the pipe and redirect
        // stdout to the input side
        close(pipefds[0]);
        dup2(pipefds[1], STDOUT_FILENO);

        // Convert cmd to a C string array.  The C array has to be writeable,
        // so we need to strdup.
        char** args = new char*[cmd.size() + 1];
        size_t i;
        for (i = 0; i < cmd.size(); ++i) {
            args[i] = strdup(cmd[i].c_str());
        }
        args[i] = nullptr;

        // exec to subprocess.  On success, this does not return.
        execvp(args[0], args);

        // exec-ing failed
        ERROR_NON_OO("exec to {:s} failed!", cmd[0]);
        exit(1);

        // Child can't get here
    }
    // In parent process

    // close the input side of the pipe and open a stream for the
    // output side
    close(pipefds[1]);
    FILE* pipe = fdopen(pipefds[0], "r");
    if (!pipe)
        throw std::runtime_error("Could not create stream for exec pipe");

    // Read from the child until it exits
    std::array<char, 256> buffer;
    std::string result;
    try {
        while (!feof(pipe)) {
            if (fgets(buffer.data(), 256, pipe) != nullptr)
                result += buffer.data();
        }
    } catch (...) {
        fclose(pipe);
        throw std::runtime_error("Could not read from the exec pipe");
    }
    fclose(pipe);

    // Reap the child
    int exitcode;
    waitpid(pid, &exitcode, 0);

    if (!WIFEXITED(exitcode) || WEXITSTATUS(exitcode) != 0)
        return "";
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
    std::signal(SIGTERM, signal_handler);

    char* config_file_name = (char*)"none";
    int log_options = LOG_CONS | LOG_PID | LOG_NDELAY;
    bool enable_stderr = true;
    bool dump_config = false;
    bool dump_yaml = false;
    std::string bind_address = "0.0.0.0:12048";
    std::string jinja_variables = "";
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
                                               {"variables", required_argument, nullptr, 'e'},
                                               {"version", no_argument, nullptr, 'v'},
                                               {"version-json", no_argument, nullptr, 'j'},
                                               {"print-json", no_argument, nullptr, 'p'},
                                               {"print-yaml", no_argument, nullptr, 'y'},
                                               {nullptr, 0, nullptr, 0}};

        int option_index = 0;

        int opt_val = getopt_long(argc, argv, "hc:b:sne:vjpy", long_options, &option_index);

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
            case 'e':
                jinja_variables = string(optarg);
                break;
            case 'v':
                print_version();
                return 0;
                break;
            case 'j':
                print_json_version();
                return 0;
                break;
            case 'y':
                dump_yaml = true;
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
        std::lock_guard<std::mutex> lock(kotekan_state_lock);
        INFO_NON_OO("Opening config file {:s}", config_file_name);

        // Create the command line, adding the yaml dump, and extra vars if needed
        std::vector<std::string> exec_command;
        exec_command.push_back("python3.7");
        exec_command.push_back("-c");
        exec_command.push_back(yaml_to_json);

        if (dump_yaml) {
            exec_command.push_back("-d");
        }
        if (jinja_variables != "") {
            exec_command.push_back("-e");
            exec_command.push_back(jinja_variables);
        }
        exec_command.push_back(config_file_name);

        std::string json_string = exec(exec_command);
        if (json_string == "") {
            ERROR_NON_OO("Unable to load config from {:s}", config_file_name);
            exit(-1);
        }
        json config_json = {};
        try {
            config_json = json::parse(json_string);
        } catch (const json::exception& exp) {
            ERROR_NON_OO("Unable to parse JSON");
            ERROR_NON_OO("Error {:s}", exp.what());
            exit(-1);
        }
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
            "/kill endpoint called, raising SIGTERM to shutdown the kotekan system process.");
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

        if (sig_value == SIGINT || sig_value == SIGTERM) {
            INFO_NON_OO("Got SIGINT or SIGTERM, shutting down kotekan...");
            std::lock_guard<std::mutex> lock(kotekan_state_lock);
            if (kotekan_mode != nullptr) {
                INFO_NON_OO("Attempting to stop and join kotekan_stages...");
                kotekan_mode->stop_stages();
                kotekan_mode->join();
                if (string(get_error_message()) != "not set") {
                    KotekanTrackers::instance().dump_trackers();
                }
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
