#define BOOST_TEST_MODULE "test_gnss_host_rows"

// The row contract of the GNSS host stages (gnss::host_rows) and their REST registration
// (gnss::rest_register_unique). Each FATAL is checked in a forked child, since FATAL_ERROR raises
// SIGTERM and the row names and REST endpoints are process-wide.

#include "Config.hpp"                // for Config
#include "GnssCoherentCombiner.hpp"  // for GnssCoherentCombiner
#include "GnssGpuRecordAssemble.hpp" // for GnssGpuRecordAssemble
#include "GnssTelemPack.hpp"         // for GnssTelemPack
#include "bufferContainer.hpp"       // for bufferContainer
#include "gnssHostRows.hpp"          // for host_rows, rest_register_unique
#include "kotekanLogging.hpp"        // for FatalError
#include "restServer.hpp"            // for restServer, connectionInstance

#include "json.hpp" // for json

#include <boost/test/included/unit_test.hpp>
#include <csignal>
#include <functional>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using json = nlohmann::json;
using kotekan::connectionInstance;

namespace {

/// Runs @p fn in a forked child. Returns "" if it threw no FatalError, else the FatalError's
/// message; "<exception>" for any other exception.
std::string fatal_of(const std::function<void()>& fn) {
    int fd[2];
    BOOST_REQUIRE(pipe(fd) == 0);
    const pid_t pid = fork();
    BOOST_REQUIRE(pid >= 0);
    if (pid == 0) {
        close(fd[0]);
        std::signal(SIGTERM, SIG_IGN); // FATAL_ERROR raises it before it throws
        std::string msg;
        try {
            fn();
        } catch (const FatalError& e) {
            msg = e.what();
        } catch (const std::exception& e) {
            msg = "<exception>";
        }
        if (write(fd[1], msg.data(), msg.size()) != (ssize_t)msg.size())
            _exit(2);
        _exit(0);
    }
    close(fd[1]);
    std::string msg;
    char b[256];
    ssize_t n;
    while ((n = read(fd[0], b, sizeof(b))) > 0)
        msg.append(b, (size_t)n);
    close(fd[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    BOOST_REQUIRE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    return msg;
}

json keys_block(const std::vector<std::string>& keys) {
    json b = json::object();
    for (const auto& k : keys)
        b[k] = 1;
    return b;
}

/// A config with host block @p host at /host, the stage settings kotekan::Stage reads at the root.
json config_with(const json& host) {
    return {{"log_level", "warn"}, {"cpu_affinity", {0}}, {"host", host}};
}

template<typename Host>
void construct(const json& cfg) {
    kotekan::Config config;
    config.update_config(cfg);
    kotekan::bufferContainer bc;
    Host h(config, "/host", bc);
}

/// For each of @p row_keys, a host whose block sets every key but whose second row lacks that one
/// FATALs naming it.
template<typename Host>
void check_row_keys(const std::vector<std::string>& row_keys) {
    for (const auto& k : row_keys) {
        json host = keys_block(row_keys);
        json lacking = keys_block(row_keys);
        lacking.erase(k);
        host["chains"] = {{"gnss0_a", keys_block(row_keys)}, {"gnss0_b", lacking}};
        const std::string msg = fatal_of([&] { construct<Host>(config_with(host)); });
        BOOST_CHECK_MESSAGE(msg == "/host: row gnss0_b does not set " + k, k << ": " << msg);
    }
}

void noop_get(connectionInstance&) {}
void noop_post(connectionInstance&, json&) {}

} // namespace

// Claims its row names in this process, so they differ from the other cases'.
BOOST_AUTO_TEST_CASE(rows_and_paths) {
    kotekan::Config config;
    config.update_config(config_with(
        {{"n_prn", 4}, {"chains", {{"gnss0_q", {{"n_prn", 6}}}, {"gnss0_p", {{"n_prn", 5}}}}}}));
    const std::vector<gnss::HostRow> rows =
        gnss::host_rows(config, "/host", "_n2combine", {"n_prn"}, {});
    BOOST_REQUIRE_EQUAL(rows.size(), 2);
    BOOST_CHECK_EQUAL(rows[0].key, "gnss0_p");
    BOOST_CHECK_EQUAL(rows[0].cfg_path, "/host/chains/gnss0_p");
    BOOST_CHECK_EQUAL(rows[0].name, "/gnss0_p_n2combine");
    BOOST_CHECK_EQUAL(rows[1].name, "/gnss0_q_n2combine");
    BOOST_CHECK_EQUAL(config.get<int>(rows[1].cfg_path, "n_prn"), 6);

    // Without chains: one row named and configured as the stage itself.
    const std::vector<gnss::HostRow> one =
        gnss::host_rows(config, "/gnss0_p_n2assemble", "_n2assemble", {"prns"}, {});
    BOOST_REQUIRE_EQUAL(one.size(), 1);
    BOOST_CHECK(one[0].key.empty());
    BOOST_CHECK_EQUAL(one[0].cfg_path, "/gnss0_p_n2assemble");
    BOOST_CHECK_EQUAL(one[0].name, "/gnss0_p_n2assemble");
}

BOOST_AUTO_TEST_CASE(row_missing_key_is_fatal) {
    check_row_keys<GnssGpuRecordAssemble>(GnssGpuRecordAssemble::ROW_KEYS);
    check_row_keys<GnssCoherentCombiner>(GnssCoherentCombiner::ROW_KEYS);
    check_row_keys<GnssTelemPack>(GnssTelemPack::ROW_KEYS);
}

BOOST_AUTO_TEST_CASE(row_only_key_on_host_is_fatal) {
    for (const auto& k : GnssGpuRecordAssemble::ROW_ONLY_KEYS) {
        json host = keys_block(GnssGpuRecordAssemble::ROW_KEYS);
        host[k] = 1;
        host["chains"] = {{"gnss0_a", keys_block(GnssGpuRecordAssemble::ROW_KEYS)}};
        const std::string msg =
            fatal_of([&] { construct<GnssGpuRecordAssemble>(config_with(host)); });
        BOOST_CHECK_MESSAGE(msg == "/host: " + k + " is set per row in chains, not on the host",
                            k << ": " << msg);
    }
    json host = keys_block(GnssCoherentCombiner::ROW_KEYS);
    host["out_buf"] = "x";
    host["chains"] = {{"gnss0_a", keys_block(GnssCoherentCombiner::ROW_KEYS)}};
    BOOST_CHECK_EQUAL(fatal_of([&] { construct<GnssCoherentCombiner>(config_with(host)); }),
                      "/host: out_buf is set per row in chains, not on the host");
}

BOOST_AUTO_TEST_CASE(empty_chains_is_fatal) {
    BOOST_CHECK_EQUAL(
        fatal_of([] { construct<GnssTelemPack>(config_with({{"chains", json::object()}})); }),
        "/host: chains must be a non-empty map of rows");
}

BOOST_AUTO_TEST_CASE(duplicate_row_name_is_fatal) {
    const json cfg = config_with({{"chains", {{"gnss0_a", {{"n_prn", 1}}}}}});
    const json other = {{"chains", {{"gnss0_a", {{"n_prn", 1}}}}}};
    // Two hosts with the same row, and a row and the one-row stage it stands for.
    const std::string two_hosts = fatal_of([&] {
        kotekan::Config config;
        json c = cfg;
        c["other"] = other;
        config.update_config(c);
        gnss::host_rows(config, "/host", "_n2combine", {"n_prn"}, {});
        gnss::host_rows(config, "/other", "_n2combine", {"n_prn"}, {});
    });
    BOOST_CHECK_EQUAL(two_hosts, "/other: row name /gnss0_a_n2combine is already in use");
    const std::string row_and_stage = fatal_of([&] {
        kotekan::Config config;
        config.update_config(cfg);
        gnss::host_rows(config, "/host", "_n2combine", {"n_prn"}, {});
        gnss::host_rows(config, "/gnss0_a_n2combine", "_n2combine", {"n_prn"}, {});
    });
    BOOST_CHECK_EQUAL(row_and_stage,
                      "/gnss0_a_n2combine: row name /gnss0_a_n2combine is already in use");
    // The same row key under another host type is another name.
    BOOST_CHECK_EQUAL(fatal_of([&] {
                          kotekan::Config config;
                          json c = cfg;
                          c["other"] = other;
                          config.update_config(c);
                          gnss::host_rows(config, "/host", "_n2combine", {"n_prn"}, {});
                          gnss::host_rows(config, "/other", "_n2assemble", {"n_prn"}, {});
                      }),
                      "");
}

BOOST_AUTO_TEST_CASE(duplicate_rest_path_is_fatal) {
    // Two hosts registering one path, as GET then GET and as GET then POST.
    BOOST_CHECK_EQUAL(fatal_of([] {
                          gnss::rest_register_unique("/gnss0_a_n2combine/get_status", noop_get);
                          gnss::rest_register_unique("/gnss0_b_n2combine/get_status", noop_get);
                      }),
                      "");
    BOOST_CHECK_EQUAL(fatal_of([] {
                          gnss::rest_register_unique("/gnss0_a_n2combine/get_status", noop_get);
                          gnss::rest_register_unique("/gnss0_a_n2combine/get_status", noop_get);
                      }),
                      "REST endpoint /gnss0_a_n2combine/get_status is already registered");
    BOOST_CHECK_EQUAL(fatal_of([] {
                          gnss::rest_register_unique("/gnss0_a_n2assemble/x", noop_get);
                          gnss::rest_register_unique("/gnss0_a_n2assemble/x", noop_post);
                      }),
                      "REST endpoint /gnss0_a_n2assemble/x is already registered");
}

BOOST_AUTO_TEST_CASE(has_callback) {
    BOOST_CHECK_EQUAL(fatal_of([] {
                          kotekan::restServer& rest = kotekan::restServer::instance();
                          if (rest.has_callback("/hc/get") || rest.has_callback("/hc/post"))
                              throw FatalError("registered before");
                          rest.register_get_callback("/hc/get", noop_get);
                          rest.register_post_callback("hc/post", noop_post);
                          if (!rest.has_callback("/hc/get") || !rest.has_callback("hc/get")
                              || !rest.has_callback("/hc/post") || rest.has_callback("/hc"))
                              throw FatalError("lookup");
                      }),
                      "");
}
