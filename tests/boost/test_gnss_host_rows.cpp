#define BOOST_TEST_MODULE "test_gnss_host_rows"

// The row contract of the GNSS host stages (gnss::host_rows) and their REST registration
// (gnss::rest_register_unique), and their release when a host is destroyed. Each case runs in a
// forked child, since FATAL_ERROR raises SIGTERM and the row names and REST endpoints are
// process-wide.

#include "Config.hpp"                // for Config
#include "GnssCoherentCombiner.hpp"  // for GnssCoherentCombiner
#include "GnssGpuRecordAssemble.hpp" // for GnssGpuRecordAssemble
#include "GnssTelemPack.hpp"         // for GnssTelemPack
#include "bufferContainer.hpp"       // for bufferContainer
#include "gnssGpuRows.hpp"           // for gpu_rows
#include "gnssHostRows.hpp"          // for host_rows, rest_register_unique
#include "kotekanLogging.hpp"        // for FatalError
#include "kotekanMode.hpp"           // for kotekanMode
#include "prometheusMetrics.hpp"     // for Metrics
#include "restServer.hpp"            // for restServer, connectionInstance
#include "util.h"                    // for string_tail

#include "json.hpp" // for json

#include <boost/test/included/unit_test.hpp>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <sched.h>
#include <set>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <thread>
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

/// Starts @p cfg twice in this process, stopping and deleting the first run as /stop does. Each
/// run must serve every one of @p paths, and none may be left once it is deleted.
void start_stop_start(const json& cfg, const std::vector<std::string>& paths) {
    kotekan::Config config;
    config.update_config(cfg);
    kotekan::restServer& rest = kotekan::restServer::instance();
    for (int run = 0; run < 2; ++run) {
        kotekan::kotekanMode* mode = new kotekan::kotekanMode(config);
        mode->initalize_stages();
        mode->start_stages();
        for (const auto& p : paths) {
            const auto t0 = std::chrono::steady_clock::now();
            while (!rest.has_callback(p)) {
                if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(10))
                    throw FatalError("run " + std::to_string(run) + " does not serve " + p);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        mode->stop_stages();
        mode->join();
        delete mode;
        for (const auto& p : paths)
            if (rest.has_callback(p))
                throw FatalError("run " + std::to_string(run) + " left " + p);
    }
}

/// A config with buffers @p bufs of one record each and the combiner block @p host at /<name>.
json combiner_config(const std::vector<std::string>& bufs, const std::string& name,
                     const json& host) {
    json cfg = {{"log_level", "warn"},
                {"cpu_affinity", {0}},
                {"telescope", {{"num_polarizations", 2}, {"num_dishes", 2}}}};
    for (const auto& b : bufs)
        cfg[b] = {{"kotekan_buffer", "standard"}, {"num_frames", 2}, {"frame_size", 1 << 20}};
    cfg[name] = host;
    cfg[name]["kotekan_stage"] = "GnssCoherentCombiner";
    cfg[name]["n_elements"] = 1;
    cfg[name]["fft_len"] = 16384;
    return cfg;
}

/// A record host block at /gnss0_records whose rows @p keys read in_buf epl_<key> and write
/// rec_<key>, with the buffers they name, and @p workers.
json record_config(const std::vector<std::string>& keys, const json& workers) {
    json cfg = {{"log_level", "warn"},
                {"cpu_affinity", {0}},
                {"telescope", {{"num_polarizations", 2}, {"num_dishes", 2}}}};
    json host = {{"kotekan_stage", "GnssGpuRecordAssemble"},
                 {"n_elements", 2},
                 {"sample_rate", 3.2e9},
                 {"spectrum_window_samples", 1e9},
                 {"workers", workers}};
    for (const auto& k : keys) {
        host["chains"][k] = {
            {"in_buf", "epl_" + k}, {"out_buf", "rec_" + k}, {"prns", {1}}, {"channel_ids", {100}}};
        for (const std::string b : {"epl_", "rec_"})
            cfg[b + k] = {
                {"kotekan_buffer", "standard"}, {"num_frames", 2}, {"frame_size", 1 << 16}};
    }
    cfg["gnss0_records"] = host;
    return cfg;
}

/// A combiner host block at /gnss<g>_combine, g the half of the first of @p keys, whose rows
/// @p keys (gnss<g>_<chain>) read in_bufs [rec_<key>] and carry chain <chain>, with the buffers
/// they name, and @p workers.
json combine_config(const std::vector<std::string>& keys, const json& workers) {
    json host = {{"workers", workers}};
    std::vector<std::string> bufs;
    for (const auto& k : keys) {
        host["chains"][k] = {{"in_bufs", {"rec_" + k}}, {"n_prn", 1}, {"chain", k.substr(6)}};
        bufs.push_back("rec_" + k);
    }
    return combiner_config(bufs, keys[0].substr(0, 5) + "_combine", host);
}

/// The FatalError message of starting @p cfg's stages, "" for none.
std::string start_fatal(const json& cfg) {
    return fatal_of([&] {
        kotekan::Config config;
        config.update_config(cfg);
        kotekan::kotekanMode mode(config);
        mode.initalize_stages();
    });
}

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

// A GPU half's gnss_rows: two rows of one shape (32 PRN x 6 channels), as gnss0_gps_l5 and
// gnss0_gal_e5a on a node.
static json gpu_rows_block() {
    const size_t b = gnss_gpu::ctl_bytes(32, 6);
    json rows = json::array();
    for (int i = 0; i < 2; ++i)
        rows.push_back({{"name", i ? "gnss0_gal_e5a" : "gnss0_gps_l5"},
                        {"chain", i ? "gal_e5a" : "gps_l5"},
                        {"chain_id", i + 1},
                        {"row", i},
                        {"channels", {326, 327, 328, 329, 330, 331}},
                        {"lane_base", 64 * i},
                        {"lanes", 64},
                        {"fg_offset", 6 * i},
                        {"ctl_offset", b * i},
                        {"ctl_bytes", b},
                        {"n_prn", 32}});
    return {{"gnss_rows", rows}};
}

// Shape checks cannot tell two rows of one shape apart, so gnss_rows must be the arrays'
// concatenation in list order: a swap of two rows' ctl_offset, fg_offset or row, or a wrong
// ctl_bytes, is refused.
BOOST_AUTO_TEST_CASE(gpu_rows_refuse_a_swap) {
    auto rows_of = [](const json& host) {
        kotekan::Config config;
        config.update_config(config_with(host));
        return gnss_gpu::gpu_rows(config, "/host");
    };
    BOOST_CHECK_EQUAL(fatal_of([&] { BOOST_REQUIRE(rows_of(gpu_rows_block()).size() == 2); }), "");
    for (const char* k : {"ctl_offset", "fg_offset", "row"}) {
        json h = gpu_rows_block();
        std::swap(h["gnss_rows"][0][k], h["gnss_rows"][1][k]);
        BOOST_CHECK_MESSAGE(fatal_of([&] { rows_of(h); }).find("the rows before it put it at")
                                != std::string::npos,
                            k);
    }
    json h = gpu_rows_block();
    h["gnss_rows"][1]["ctl_bytes"] = gnss_gpu::ctl_bytes(32, 7);
    BOOST_CHECK(fatal_of([&] { rows_of(h); }).find("ctl_bytes") != std::string::npos);
    h = gpu_rows_block();
    h["gnss_rows"][1]["chain_id"] = 1;
    BOOST_CHECK(fatal_of([&] { rows_of(h); }).find("not unique") != std::string::npos);
    h = gpu_rows_block();
    h["gnss_rows"][1].erase("lanes");
    BOOST_CHECK_EQUAL(fatal_of([&] { rows_of(h); }), "<exception>");
    // Rows that share a channel must write disjoint lanes.
    h = gpu_rows_block();
    h["gnss_rows"][1]["lane_base"] = 32;
    BOOST_CHECK(fatal_of([&] { rows_of(h); }).find("overlapping lanes") != std::string::npos);
    h["gnss_rows"][1]["channels"] = {332, 333, 334, 335, 336, 337};
    BOOST_CHECK_EQUAL(fatal_of([&] { rows_of(h); }), "");
}

// A record row reads in_buf, or tiles_buf and ctl_buf, each set at its own path.
BOOST_AUTO_TEST_CASE(record_row_input) {
    const json none = json::object();
    const json both = {{"in_buf", "x"}, {"tiles_buf", "x"}, {"ctl_buf", "x"}};
    const json tiles = {{"tiles_buf", "x"}};
    const json ctl = {{"ctl_buf", "x"}};
    for (const json& in : {none, both, tiles, ctl}) {
        json row = keys_block(GnssGpuRecordAssemble::ROW_KEYS);
        row.update(in);
        json host = keys_block(GnssGpuRecordAssemble::ROW_KEYS);
        host["chains"] = {{"gnss0_a", row}};
        BOOST_CHECK_EQUAL(fatal_of([&] { construct<GnssGpuRecordAssemble>(config_with(host)); }),
                          "/gnss0_a_n2assemble: set in_buf, or tiles_buf and ctl_buf");
    }
    // gnss_row and fg_total come together, and only with tiles_buf.
    for (const json& in : {json{{"tiles_buf", "x"}, {"ctl_buf", "x"}, {"gnss_row", 1}},
                           json{{"in_buf", "x"}, {"gnss_row", 1}, {"fg_total", 1}}}) {
        json row = keys_block(GnssGpuRecordAssemble::ROW_KEYS);
        row.update(in);
        json host = keys_block(GnssGpuRecordAssemble::ROW_KEYS);
        host["chains"] = {{"gnss0_a", row}};
        BOOST_CHECK_EQUAL(fatal_of([&] { construct<GnssGpuRecordAssemble>(config_with(host)); }),
                          "/gnss0_a_n2assemble: gnss_row and fg_total go together, with tiles_buf");
    }
    // A one-row stage's own block is its row.
    json one = keys_block(GnssGpuRecordAssemble::ROW_KEYS);
    BOOST_CHECK_EQUAL(fatal_of([&] { construct<GnssGpuRecordAssemble>(config_with(one)); }),
                      "/host: set in_buf, or tiles_buf and ctl_buf");
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
    BOOST_CHECK_EQUAL(
        fatal_of([] {
            std::vector<std::string> paths;
            gnss::rest_register_unique("/gnss0_a_n2combine/get_status", noop_get, paths);
            gnss::rest_register_unique("/gnss0_b_n2combine/get_status", noop_get, paths);
        }),
        "");
    BOOST_CHECK_EQUAL(
        fatal_of([] {
            std::vector<std::string> paths;
            gnss::rest_register_unique("/gnss0_a_n2combine/get_status", noop_get, paths);
            gnss::rest_register_unique("/gnss0_a_n2combine/get_status", noop_get, paths);
        }),
        "REST endpoint /gnss0_a_n2combine/get_status is already registered");
    BOOST_CHECK_EQUAL(fatal_of([] {
                          std::vector<std::string> paths;
                          gnss::rest_register_unique("/gnss0_a_n2assemble/x", noop_get, paths);
                          gnss::rest_register_unique("/gnss0_a_n2assemble/x", noop_post, paths);
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

// Deleting a host releases its row names, REST paths and row metrics, so that the same config
// starts again in the same process after a /stop, as kotekan in daemon mode does.
BOOST_AUTO_TEST_CASE(host_restarts_in_one_process) {
    BOOST_CHECK_EQUAL(fatal_of([] {
                          start_stop_start(
                              combiner_config({"rec_buf"}, "gnss0_t_n2combine",
                                              {{"in_bufs", {"rec_buf"}}, {"n_prn", 1}}),
                              {"/gnss0_t_n2combine/get_status"});
                      }),
                      "");
    const json rows = {{"chains",
                        {{"gnss0_a", {{"in_bufs", {"rec_a"}}, {"n_prn", 1}, {"chain", "a"}}},
                         {"gnss0_b", {{"in_bufs", {"rec_b"}}, {"n_prn", 1}, {"chain", "b"}}}}}};
    BOOST_CHECK_EQUAL(fatal_of([&] {
                          start_stop_start(
                              combiner_config({"rec_a", "rec_b"}, "gnss0_combine", rows),
                              {"/gnss0_a_n2combine/get_status", "/gnss0_b_n2combine/get_status",
                               "/gnss0_combine/chains"});
                      }),
                      "");
}

// A record host's workers partition its rows: each row in exactly one worker, each worker with
// rows and cores, and no input buffer read twice by one worker.
BOOST_AUTO_TEST_CASE(record_workers_partition_the_rows) {
    const std::vector<std::string> abc = {"gnss0_a", "gnss0_b", "gnss0_c"};
    auto w = [](const std::vector<std::string>& rows) {
        return json{{"rows", rows}, {"cpu_affinity", {0}}};
    };
    const std::string h = "/gnss0_records: ";
    BOOST_CHECK_EQUAL(start_fatal(record_config(abc, {w({"gnss0_a", "gnss0_c"}), w({"gnss0_b"})})),
                      "");
    BOOST_CHECK_EQUAL(
        start_fatal(record_config(abc, {w({"gnss0_a", "gnss0_c"}), w({"gnss0_b", "gnss0_a"})})),
        h + "row gnss0_a is in workers 0 and 1");
    BOOST_CHECK_EQUAL(start_fatal(record_config(abc, {w({"gnss0_a"}), w({"gnss0_b"})})),
                      h + "row gnss0_c is in no worker");
    // More workers than rows leaves one without rows.
    BOOST_CHECK_EQUAL(
        start_fatal(record_config(abc, {w({"gnss0_a"}), w({"gnss0_b"}), w({"gnss0_c"}), w({})})),
        h + "worker 3 needs rows and cpu_affinity, each a non-empty list");
    BOOST_CHECK_EQUAL(
        start_fatal(record_config(abc, {w({"gnss0_a", "gnss0_b"}), w({"gnss0_c", "gnss0_x"})})),
        h + "worker 1 runs \"gnss0_x\", which is not a row of chains");
    BOOST_CHECK_EQUAL(start_fatal(record_config(abc, json::array({{{"rows", abc}}}))),
                      h + "worker 0 needs rows and cpu_affinity, each a non-empty list");
    BOOST_CHECK_EQUAL(
        start_fatal(record_config(abc, json::array({{{"rows", abc}, {"cpu_affinity", {-1}}}}))),
        h + "worker 0 has cpu_affinity entry -1, not a core");
    BOOST_CHECK_EQUAL(start_fatal(record_config(abc, json::array())),
                      h + "workers must be a non-empty list");
    // Two rows of one worker reading one in_buf.
    json cfg = record_config(abc, json::array({w(abc)}));
    cfg["gnss0_records"]["chains"]["gnss0_b"]["in_buf"] = "epl_gnss0_a";
    BOOST_CHECK_EQUAL(start_fatal(cfg), "/gnss0_b_n2assemble: /gnss0_records/w0 reads epl_gnss0_a "
                                        "for two rows that do not share both tiles_buf and "
                                        "ctl_buf");
    // Workers need a chains block.
    cfg = record_config(abc, json::array({w(abc)}));
    cfg["gnss0_records"].erase("chains");
    cfg["gnss0_records"].update({{"in_buf", "epl_gnss0_a"},
                                 {"out_buf", "rec_gnss0_a"},
                                 {"prns", {1}},
                                 {"channel_ids", {100}}});
    BOOST_CHECK_EQUAL(start_fatal(cfg), h + "workers needs a chains block");
}

// A record host on workers releases its rows and REST paths when deleted, so it starts again in
// the same process.
BOOST_AUTO_TEST_CASE(record_workers_restart_in_one_process) {
    const json workers = {{{"rows", {"gnss0_a", "gnss0_c"}}, {"cpu_affinity", {0}}},
                          {{"rows", {"gnss0_b"}}, {"cpu_affinity", {0}}}};
    BOOST_CHECK_EQUAL(fatal_of([&] {
                          start_stop_start(
                              record_config({"gnss0_a", "gnss0_b", "gnss0_c"}, workers),
                              {"/gnss0_a_n2assemble/get_spectrum",
                               "/gnss0_b_n2assemble/set_sat_geometry",
                               "/gnss0_c_n2assemble/get_spectrum", "/gnss0_records/chains"});
                      }),
                      "");
}

// A record host's workers run as threads of the stage, each named for its row if it has one
// and bound to its own cpu_affinity.
BOOST_AUTO_TEST_CASE(record_workers_are_pinned_threads_of_the_stage) {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    BOOST_REQUIRE(sched_getaffinity(0, sizeof(allowed), &allowed) == 0);
    std::vector<int> cores;
    for (int c = 0; c < CPU_SETSIZE && cores.size() < 2; ++c)
        if (CPU_ISSET(c, &allowed))
            cores.push_back(c);
    if (cores.size() < 2) {
        BOOST_TEST_MESSAGE("fewer than 2 cores allowed: worker pinning not checked");
        return;
    }
    BOOST_CHECK_EQUAL(
        fatal_of([&] {
            const std::vector<std::string> abc = {"gnss0_a", "gnss0_b", "gnss0_c"};
            json cfg = record_config(
                abc, {{{"rows", {"gnss0_a", "gnss0_c"}}, {"cpu_affinity", {cores[0]}}},
                      {{"rows", {"gnss0_b"}}, {"cpu_affinity", {cores[1]}}}});
            cfg["cpu_affinity"] = {cores[1]};
            kotekan::Config config;
            config.update_config(cfg);
            kotekan::bufferContainer bc;
            std::vector<std::unique_ptr<Buffer>> bufs;
            for (const auto& k : abc)
                for (const std::string b : {"epl_", "rec_"}) {
                    bufs.push_back(std::make_unique<Buffer>(2, 1 << 16, nullptr, b + k, "standard",
                                                            0, false, false, std::vector<int>{},
                                                            false));
                    bc.add_buffer(b + k, bufs.back().get());
                }
            GnssGpuRecordAssemble host(config, "/gnss0_records", bc);
            // Each thread's name and its cores, as the kernel has them.
            const std::map<std::string, int> want = {
                {string_tail("/gnss0_records", 15), cores[1]},
                {string_tail("/gnss0_records/w0", 15), cores[0]},
                {string_tail("/gnss0_b_n2assemble", 15), cores[1]}};
            auto seen = [&] {
                std::map<std::string, int> out;
                for (pid_t tid : host.get_tids()) {
                    std::ifstream f("/proc/self/task/" + std::to_string(tid) + "/comm");
                    std::string comm;
                    std::getline(f, comm);
                    cpu_set_t set;
                    CPU_ZERO(&set);
                    if (sched_getaffinity(tid, sizeof(set), &set) != 0 || CPU_COUNT(&set) != 1)
                        return out;
                    for (int c = 0; c < CPU_SETSIZE; ++c)
                        if (CPU_ISSET(c, &set))
                            out[comm] = c;
                }
                return out;
            };
            host.start();
            const auto t0 = std::chrono::steady_clock::now();
            std::map<std::string, int> got;
            while ((got = seen()) != want
                   && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10))
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            host.stop();
            for (auto& b : bufs)
                b->send_shutdown_signal();
            host.join();
            if (got != want) {
                std::string s;
                for (const auto& [n, c] : got)
                    s += " " + n + ":" + std::to_string(c);
                throw FatalError("threads" + s);
            }
        }),
        "");
}

// A combine host's workers partition its rows: each row in exactly one worker, each worker with
// rows and cores, and no in_buf read for two rows by one worker.
BOOST_AUTO_TEST_CASE(combine_workers_partition_the_rows) {
    const std::vector<std::string> abc = {"gnss0_a", "gnss0_b", "gnss0_c"};
    auto w = [](const std::vector<std::string>& rows) {
        return json{{"rows", rows}, {"cpu_affinity", {0}}};
    };
    const std::string h = "/gnss0_combine: ";
    BOOST_CHECK_EQUAL(start_fatal(combine_config(abc, {w({"gnss0_a", "gnss0_c"}), w({"gnss0_b"})})),
                      "");
    BOOST_CHECK_EQUAL(
        start_fatal(combine_config(abc, {w({"gnss0_a", "gnss0_c"}), w({"gnss0_b", "gnss0_a"})})),
        h + "row gnss0_a is in workers 0 and 1");
    BOOST_CHECK_EQUAL(start_fatal(combine_config(abc, {w({"gnss0_a"}), w({"gnss0_b"})})),
                      h + "row gnss0_c is in no worker");
    BOOST_CHECK_EQUAL(
        start_fatal(combine_config(abc, {w({"gnss0_a"}), w({"gnss0_b"}), w({"gnss0_c"}), w({})})),
        h + "worker 3 needs rows and cpu_affinity, each a non-empty list");
    BOOST_CHECK_EQUAL(
        start_fatal(combine_config(abc, {w({"gnss0_a", "gnss0_b"}), w({"gnss0_c", "gnss0_x"})})),
        h + "worker 1 runs \"gnss0_x\", which is not a row of chains");
    BOOST_CHECK_EQUAL(start_fatal(combine_config(abc, json::array({{{"rows", abc}}}))),
                      h + "worker 0 needs rows and cpu_affinity, each a non-empty list");
    BOOST_CHECK_EQUAL(
        start_fatal(combine_config(abc, json::array({{{"rows", abc}, {"cpu_affinity", {-1}}}}))),
        h + "worker 0 has cpu_affinity entry -1, not a core");
    BOOST_CHECK_EQUAL(start_fatal(combine_config(abc, json::array())),
                      h + "workers must be a non-empty list");
    BOOST_CHECK_EQUAL(start_fatal(combine_config(abc, {{"rows", abc}})),
                      h + "workers must be a non-empty list");
    // Two rows of one worker reading one in_buf; of two workers, it is two consumers.
    json cfg = combine_config(abc, json::array({w(abc)}));
    cfg["gnss0_combine"]["chains"]["gnss0_b"]["in_bufs"] = {"rec_gnss0_a"};
    BOOST_CHECK_EQUAL(start_fatal(cfg),
                      "/gnss0_b_n2combine: /gnss0_combine/w0 reads rec_gnss0_a for two rows");
    cfg = combine_config(abc, {w({"gnss0_a", "gnss0_c"}), w({"gnss0_b"})});
    cfg["gnss0_combine"]["chains"]["gnss0_b"]["in_bufs"] = {"rec_gnss0_a"};
    BOOST_CHECK_EQUAL(start_fatal(cfg), "");
    // Workers need a chains block.
    cfg = combine_config(abc, json::array({w(abc)}));
    cfg["gnss0_combine"].erase("chains");
    cfg["gnss0_combine"].update({{"in_bufs", {"rec_gnss0_a"}}, {"n_prn", 1}});
    BOOST_CHECK_EQUAL(start_fatal(cfg), h + "workers needs a chains block");
}

// A combine host on workers releases its rows, REST paths and metrics when deleted, so it starts
// again in the same process.
BOOST_AUTO_TEST_CASE(combine_workers_restart_in_one_process) {
    const json workers = {{{"rows", {"gnss0_a", "gnss0_c"}}, {"cpu_affinity", {0}}},
                          {{"rows", {"gnss0_b"}}, {"cpu_affinity", {0}}}};
    BOOST_CHECK_EQUAL(fatal_of([&] {
                          start_stop_start(
                              combine_config({"gnss0_a", "gnss0_b", "gnss0_c"}, workers),
                              {"/gnss0_a_n2combine/get_status", "/gnss0_b_n2combine/get_elements",
                               "/gnss0_c_n2combine/get_status", "/gnss0_combine/chains"});
                      }),
                      "");
}

// A combine host's workers run as threads of the stage, each named for its row if it has one,
// as is its consumer of the row's in_buf, and bound to its own cpu_affinity.
BOOST_AUTO_TEST_CASE(combine_workers_are_pinned_threads_of_the_stage) {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    BOOST_REQUIRE(sched_getaffinity(0, sizeof(allowed), &allowed) == 0);
    std::vector<int> cores;
    for (int c = 0; c < CPU_SETSIZE && cores.size() < 2; ++c)
        if (CPU_ISSET(c, &allowed))
            cores.push_back(c);
    if (cores.size() < 2) {
        BOOST_TEST_MESSAGE("fewer than 2 cores allowed: worker pinning not checked");
        return;
    }
    BOOST_CHECK_EQUAL(
        fatal_of([&] {
            const std::vector<std::string> abc = {"gnss0_a", "gnss0_b", "gnss0_c"};
            json cfg = combine_config(
                abc, {{{"rows", {"gnss0_a", "gnss0_c"}}, {"cpu_affinity", {cores[0]}}},
                      {{"rows", {"gnss0_b"}}, {"cpu_affinity", {cores[1]}}}});
            cfg["cpu_affinity"] = {cores[1]};
            kotekan::Config config;
            config.update_config(cfg);
            kotekan::bufferContainer bc;
            std::vector<std::unique_ptr<Buffer>> bufs;
            for (const auto& k : abc) {
                bufs.push_back(std::make_unique<Buffer>(2, 1 << 16, nullptr, "rec_" + k, "standard",
                                                        0, false, false, std::vector<int>{},
                                                        false));
                bc.add_buffer("rec_" + k, bufs.back().get());
            }
            GnssCoherentCombiner host(config, "/gnss0_combine", bc);
            // A worker of one row reads its in_buf under its row's name.
            const std::vector<std::vector<std::string>> readers = {
                {"/gnss0_combine/w0"}, {"/gnss0_b_n2combine"}, {"/gnss0_combine/w0"}};
            for (size_t i = 0; i < bufs.size(); ++i) {
                std::string s;
                for (const std::string& n : bufs[i]->get_consumer_names())
                    s += " " + n;
                if (bufs[i]->get_consumer_names() != readers[i])
                    throw FatalError("rec_" + abc[i] + " consumers" + s);
            }
            // Each thread's name and its cores, as the kernel has them.
            const std::map<std::string, int> want = {
                {string_tail("/gnss0_combine", 15), cores[1]},
                {string_tail("/gnss0_combine/w0", 15), cores[0]},
                {string_tail("/gnss0_b_n2combine", 15), cores[1]}};
            auto seen = [&] {
                std::map<std::string, int> out;
                for (pid_t tid : host.get_tids()) {
                    std::ifstream f("/proc/self/task/" + std::to_string(tid) + "/comm");
                    std::string comm;
                    std::getline(f, comm);
                    cpu_set_t set;
                    CPU_ZERO(&set);
                    if (sched_getaffinity(tid, sizeof(set), &set) != 0 || CPU_COUNT(&set) != 1)
                        return out;
                    for (int c = 0; c < CPU_SETSIZE; ++c)
                        if (CPU_ISSET(c, &set))
                            out[comm] = c;
                }
                return out;
            };
            host.start();
            const auto t0 = std::chrono::steady_clock::now();
            std::map<std::string, int> got;
            while ((got = seen()) != want
                   && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10))
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            host.stop();
            for (auto& b : bufs)
                b->send_shutdown_signal();
            host.join();
            if (got != want) {
                std::string s;
                for (const auto& [n, c] : got)
                    s += " " + n + ":" + std::to_string(c);
                throw FatalError("threads" + s);
            }
        }),
        "");
}

/// The combiner metric series, each line without its value, of stages @p names started from
/// @p cfg, once every one of rows @p keys (in_buf rec_<key>) has folded in a record. Stage k has
/// @p rows[k] rows.
std::vector<std::string> combiner_series(const json& cfg, const std::vector<std::string>& names,
                                         const std::vector<size_t>& rows,
                                         const std::vector<std::string>& keys) {
    kotekan::Config config;
    config.update_config(cfg);
    kotekan::bufferContainer bc;
    std::vector<std::unique_ptr<Buffer>> bufs;
    for (const auto& k : keys) {
        bufs.push_back(std::make_unique<Buffer>(2, 1 << 16, nullptr, "rec_" + k, "standard", 0,
                                                false, false, std::vector<int>{}, true, 0));
        bc.add_buffer("rec_" + k, bufs.back().get());
        bufs.back()->register_producer("feeder");
    }
    std::vector<std::unique_ptr<GnssCoherentCombiner>> stages;
    for (const auto& n : names)
        stages.push_back(std::make_unique<GnssCoherentCombiner>(config, n, bc));
    for (auto& st : stages)
        st->start();
    for (auto& b : bufs) {
        b->wait_for_empty_frame("feeder", 0);
        b->mark_frame_full("feeder", 0);
    }
    auto done = [&] {
        size_t n = 0;
        for (size_t k = 0; k < stages.size(); ++k)
            for (size_t r = 0; r < rows[k]; ++r)
                n += stages[k]->records_done(r);
        return n;
    };
    const auto t0 = std::chrono::steady_clock::now();
    while (done() < keys.size()) {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(10))
            throw FatalError("the combiners took no record");
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    std::vector<std::string> out;
    std::istringstream in(kotekan::prometheus::Metrics::instance().serialize());
    for (std::string line; std::getline(in, line);)
        if (line.compare(0, 22, "kotekan_gnss_combiner_") == 0)
            out.push_back(line.substr(0, line.find("} ") + 1));
    for (auto& st : stages)
        st->stop();
    for (auto& b : bufs)
        b->send_shutdown_signal();
    for (auto& st : stages)
        st->join();
    return out;
}

// A combine host's rows keep the metric series of the one-row stages they replace, stage_name
// included, each with a chain label, the row's broker chain, added. Chain a runs on both halves,
// so its label repeats and the series stay distinct by stage_name.
BOOST_AUTO_TEST_CASE(combine_rows_label_their_metrics) {
    BOOST_CHECK_EQUAL(
        fatal_of([] {
            const std::vector<std::string> keys = {"gnss0_a", "gnss0_b", "gnss1_a"};
            json one = combiner_config({}, "gnss0_a_n2combine",
                                       {{"in_bufs", {"rec_gnss0_a"}}, {"n_prn", 1}});
            for (const std::string k : {"gnss0_b", "gnss1_a"}) {
                one[k + "_n2combine"] = one["gnss0_a_n2combine"];
                one[k + "_n2combine"]["in_bufs"] = {"rec_" + k};
            }
            const std::vector<std::string> stages = combiner_series(
                one, {"/gnss0_a_n2combine", "/gnss0_b_n2combine", "/gnss1_a_n2combine"}, {1, 1, 1},
                keys);
            const json workers0 = {{{"rows", {"gnss0_a"}}, {"cpu_affinity", {0}}},
                                   {{"rows", {"gnss0_b"}}, {"cpu_affinity", {0}}}};
            const json workers1 = {{{"rows", {"gnss1_a"}}, {"cpu_affinity", {0}}}};
            json hosts = combine_config({"gnss0_a", "gnss0_b"}, workers0);
            hosts["gnss1_combine"] = combine_config({"gnss1_a"}, workers1).at("gnss1_combine");
            const std::vector<std::string> host =
                combiner_series(hosts, {"/gnss0_combine", "/gnss1_combine"}, {2, 1}, keys);
            std::vector<std::string> want;
            for (std::string l : stages) {
                const size_t a = l.find("stage_name=\"/") + 19;
                const std::string chain = l.substr(a, l.find("_n2combine\"", a) - a);
                want.push_back(l.substr(0, l.size() - 1) + ",chain=\"" + chain + "\"}");
            }
            for (const auto& l : stages)
                std::printf("one-row stage: %s\n", l.c_str());
            for (const auto& l : host)
                std::printf("host row:      %s\n", l.c_str());
            std::fflush(stdout);
            if (stages.size() != 18 || host != want)
                throw FatalError("series differ");
            if (std::set<std::string>(host.begin(), host.end()).size() != host.size())
                throw FatalError("series repeat");
        }),
        "");
}
