#define BOOST_TEST_MODULE "test_gnss_host_rows"

// The row contract of the GNSS host stages (gnss::host_rows) and their REST registration
// (gnss::rest_register_unique), and their release when a host is destroyed. Each case runs in a
// forked child, since FATAL_ERROR raises SIGTERM and the row names and REST endpoints are
// process-wide.

#include "Config.hpp"                // for Config
#include "GnssChanMetadata.hpp"      // for GnssChanMetadata, get_gnss_chan_metadata
#include "GnssCoherentCombiner.hpp"  // for GnssCoherentCombiner
#include "GnssGpuRecordAssemble.hpp" // for GnssGpuRecordAssemble
#include "GnssTelemPack.hpp"         // for GnssTelemPack
#include "GnssTelemSend.hpp"         // for GnssTelemSend
#include "bufferContainer.hpp"       // for bufferContainer
#include "bufferSend.hpp"            // for bufferFrameHeaderNoConfigTracker
#include "gnssGpuChain.hpp"          // for FrameHdr, PrnCtl, off_winstart, off_prnctl
#include "gnssGpuRows.hpp"           // for gpu_rows
#include "gnssHostRows.hpp"          // for host_rows, rest_register_unique
#include "gnssRecRows.hpp"           // for RecDir, RecRow
#include "gnssRecord.hpp"            // for frame_floats, REC_P_ENERGY
#include "gnssTelem.hpp"             // for telem_frame_bytes
#include "kotekanLogging.hpp"        // for FatalError
#include "kotekanMode.hpp"           // for kotekanMode
#include "metadata.hpp"              // for metadataPool
#include "prometheusMetrics.hpp"     // for Metrics
#include "restServer.hpp"            // for restServer, connectionInstance
#include "util.h"                    // for string_tail

#include "fmt.hpp"  // for format
#include "json.hpp" // for json

#include <algorithm>
#include <arpa/inet.h>
#include <boost/test/included/unit_test.hpp>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <netinet/in.h>
#include <sched.h>
#include <set>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <tuple>
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

/// For each of @p row_keys and @p row_extra, a host whose block sets @p host_keys, but whose second
/// row lacks that one of the keys every row sets, FATALs naming it.
template<typename Host>
void check_row_keys(const std::vector<std::string>& row_keys,
                    const std::vector<std::string>& row_extra,
                    const std::vector<std::string>& host_keys) {
    std::vector<std::string> keys = row_keys;
    keys.insert(keys.end(), row_extra.begin(), row_extra.end());
    for (const auto& k : keys) {
        json host = keys_block(host_keys);
        json lacking = keys_block(keys);
        lacking.erase(k);
        host["chains"] = {{"gnss0_a", keys_block(keys)}, {"gnss0_b", lacking}};
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

/// A telemetry host block at /gnss0_telem whose rows @p keys (gnss0_<chain>) read in_buf
/// rec_<key>, carry chain <chain> and write gnss0_telem_buf, a 2-PRN 1-channel frame deep 8, with
/// the buffers they name and the metadata pool their frames take, and @p workers.
json telem_config(const std::vector<std::string>& keys, const json& workers) {
    json cfg = {{"log_level", "warn"},
                {"cpu_affinity", {0}},
                {"telescope", {{"num_polarizations", 2}, {"num_dishes", 2}}},
                {"gnss_pool",
                 {{"kotekan_metadata_pool", "GnssChanMetadata"}, {"num_metadata_objects", 64}}}};
    json host = {{"kotekan_stage", "GnssTelemPack"},
                 {"inst", "node.0"},
                 {"n_prn", 2},
                 {"max_prn", 2},
                 {"n_chan", 1},
                 {"channel_ids", {100}},
                 {"hops_per_record", 2048},
                 {"fft_len", 16384},
                 {"out_buf", "gnss0_telem_buf"},
                 {"workers", workers}};
    for (const auto& k : keys) {
        host["chains"][k] = {{"in_buf", "rec_" + k}, {"chain", k.substr(6)},
                             {"n_prn", 2},           {"max_prn", 2},
                             {"n_chan", 1},          {"channel_ids", {100}}};
        cfg["rec_" + k] = {{"kotekan_buffer", "standard"},
                           {"metadata_pool", "gnss_pool"},
                           {"num_frames", 2},
                           {"frame_size", 1 << 16}};
    }
    cfg["gnss0_telem_buf"] = {{"kotekan_buffer", "standard"},
                              {"metadata_pool", "gnss_pool"},
                              {"num_frames", 8},
                              {"frame_size", gnss::telem_frame_bytes(4, 2, 1)}};
    cfg["gnss0_telem"] = host;
    return cfg;
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

// Every row sets its row keys itself, whatever the host block sets, but for the buffer that a host
// block's setting makes every row's: a GPU half's record buffer, each row then setting its rec_row.
BOOST_AUTO_TEST_CASE(row_missing_key_is_fatal) {
    check_row_keys<GnssGpuRecordAssemble>(GnssGpuRecordAssemble::ROW_KEYS, {},
                                          {"prns", "channel_ids"});
    check_row_keys<GnssGpuRecordAssemble>({"prns", "channel_ids", "rec_row"}, {},
                                          GnssGpuRecordAssemble::ROW_KEYS);
    check_row_keys<GnssCoherentCombiner>(GnssCoherentCombiner::ROW_KEYS, {}, {"n_prn", "chain"});
    check_row_keys<GnssCoherentCombiner>({"n_prn", "chain", "rec_row"}, {},
                                         GnssCoherentCombiner::ROW_KEYS);
    // A telemetry row sets out_buf unless the host block does.
    std::vector<std::string> tel = GnssTelemPack::ROW_KEYS;
    tel.erase(std::find(tel.begin(), tel.end(), "in_buf"));
    check_row_keys<GnssTelemPack>(GnssTelemPack::ROW_KEYS, {"out_buf"}, tel);
    std::vector<std::string> tel_rec = tel;
    tel_rec.push_back("rec_row");
    check_row_keys<GnssTelemPack>(tel_rec, {"out_buf"}, GnssTelemPack::ROW_KEYS);
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
        json host = keys_block({"prns", "channel_ids"});
        host["chains"] = {{"gnss0_a", row}};
        BOOST_CHECK_EQUAL(fatal_of([&] { construct<GnssGpuRecordAssemble>(config_with(host)); }),
                          "/gnss0_a_n2assemble: set in_buf, or tiles_buf and ctl_buf");
    }
    // gnss_row and fg_total come together, and only with tiles_buf.
    for (const json& in : {json{{"tiles_buf", "x"}, {"ctl_buf", "x"}, {"gnss_row", 1}},
                           json{{"in_buf", "x"}, {"gnss_row", 1}, {"fg_total", 1}}}) {
        json row = keys_block(GnssGpuRecordAssemble::ROW_KEYS);
        row.update(in);
        json host = keys_block({"prns", "channel_ids"});
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

// The telemetry rows of one host share the out_buf on its block: each row's frame must fit it,
// no two rows may send one chain and inst, and no row may set an out_buf of its own as well. The
// rows are partitioned among the workers as the other hosts' are.
BOOST_AUTO_TEST_CASE(telem_rows_share_one_out_buf) {
    const std::vector<std::string> abc = {"gnss0_a", "gnss0_b", "gnss0_c"};
    auto w = [](const std::vector<std::string>& rows) {
        return json{{"rows", rows}, {"cpu_affinity", {0}}};
    };
    const std::string h = "/gnss0_telem: ";
    const json one_each = {w({"gnss0_a"}), w({"gnss0_b"}), w({"gnss0_c"})};
    BOOST_CHECK_EQUAL(start_fatal(telem_config(abc, one_each)), "");
    // The widest row's frame does not fit.
    json cfg = telem_config(abc, one_each);
    cfg["gnss0_telem"]["chains"]["gnss0_b"]["max_prn"] = 3;
    BOOST_CHECK_EQUAL(start_fatal(cfg),
                      fmt::format("GnssTelemPack[/gnss0_b_telem_pack]: out_buf frame is {:d} B but "
                                  "the wire format needs at least {:d} B (4 records x 3 PRN x {:d} "
                                  "floats + {:d} B header). The gather's receive buffer must be "
                                  "sized from the SAME two numbers.",
                                  gnss::telem_frame_bytes(4, 2, 1),
                                  gnss::telem_frame_bytes(4, 3, 1), gnss::telem_row_floats(1),
                                  sizeof(gnss::TelemHeader)));
    // A frame larger than every row's is fine on a shared out_buf.
    cfg = telem_config(abc, one_each);
    cfg["gnss0_telem"]["chains"]["gnss0_b"]["max_prn"] = 1;
    BOOST_CHECK_EQUAL(start_fatal(cfg), "");
    // Two rows of one chain and inst.
    cfg = telem_config(abc, one_each);
    cfg["gnss0_telem"]["chains"]["gnss0_c"]["chain"] = "a";
    BOOST_CHECK_EQUAL(start_fatal(cfg),
                      h
                          + "two rows send chain a inst node.0; the gather would merge their "
                            "streams");
    // Another inst is another sender.
    cfg["gnss0_telem"]["chains"]["gnss0_c"]["inst"] = "node.1";
    BOOST_CHECK_EQUAL(start_fatal(cfg), "");
    // A row's own out_buf beside the host's.
    cfg = telem_config(abc, one_each);
    cfg["gnss0_telem"]["chains"]["gnss0_b"]["out_buf"] = "gnss0_telem_buf";
    BOOST_CHECK_EQUAL(start_fatal(cfg),
                      h + "row gnss0_b sets out_buf, which the host block sets for every row");
    // Without the host's out_buf, every row sets its own.
    cfg = telem_config(abc, one_each);
    cfg["gnss0_telem"].erase("out_buf");
    BOOST_CHECK_EQUAL(start_fatal(cfg), h + "row gnss0_a does not set out_buf");
    // The partition.
    BOOST_CHECK_EQUAL(
        start_fatal(telem_config(abc, {w({"gnss0_a", "gnss0_c"}), w({"gnss0_b", "gnss0_a"})})),
        h + "row gnss0_a is in workers 0 and 1");
    BOOST_CHECK_EQUAL(start_fatal(telem_config(abc, {w({"gnss0_a"}), w({"gnss0_b"})})),
                      h + "row gnss0_c is in no worker");
    cfg = telem_config(abc, json::array({w(abc)}));
    cfg["gnss0_telem"]["chains"]["gnss0_b"]["in_buf"] = "rec_gnss0_a";
    BOOST_CHECK_EQUAL(start_fatal(cfg),
                      "/gnss0_b_telem_pack: /gnss0_telem/w0 reads rec_gnss0_a for two rows");
}

/// A GnssTelemSend whose frame_send_bytes the test can call.
struct TelemSendProbe : GnssTelemSend {
    using GnssTelemSend::frame_send_bytes;
    using GnssTelemSend::GnssTelemSend;
};

// GnssTelemSend sends each frame of a buffer sized for a wider row at the length its header
// gives, and a frame whose header gives more than the buffer frame whole.
BOOST_AUTO_TEST_CASE(telem_send_sends_each_frame_at_its_length) {
    BOOST_CHECK_EQUAL(fatal_of([] {
                          json cfg = {{"log_level", "warn"},
                                      {"cpu_affinity", {0}},
                                      {"send", {{"buf", "telem_buf"}, {"server_ip", "127.0.0.1"}}}};
                          kotekan::Config config;
                          config.update_config(cfg);
                          auto pool = metadataPool::create(4, 64, "pool", "GnssChanMetadata");
                          Buffer buf(2, gnss::telem_frame_bytes(4, 32, 6), pool, "telem_buf",
                                     "standard", 0, false, false, std::vector<int>{}, false);
                          kotekan::bufferContainer bc;
                          bc.add_buffer("telem_buf", &buf);
                          TelemSendProbe send(config, "/send", bc);
                          gnss::TelemHeader h;
                          std::memset(&h, 0, sizeof(h));
                          h.n_rec = 4;
                          h.n_prn = 14;
                          h.max_chan = 2;
                          std::memcpy(buf.frames[0], &h, sizeof(h));
                          h.n_prn = 33;
                          h.max_chan = 6;
                          std::memcpy(buf.frames[1], &h, sizeof(h));
                          if (send.frame_send_bytes(0) != gnss::telem_frame_bytes(4, 14, 2)
                              || send.frame_send_bytes(1) != buf.frame_size)
                              throw FatalError("sent " + std::to_string(send.frame_send_bytes(0))
                                               + " and " + std::to_string(send.frame_send_bytes(1))
                                               + " bytes");
                      }),
                      "");
}

// A telemetry host on workers and the GnssTelemSend of its out_buf stop, are deleted and start
// again in one process: the host releases its rows, REST paths and dropped-frame metrics, and the
// send its connection thread.
BOOST_AUTO_TEST_CASE(telem_host_restarts_in_one_process) {
    json cfg =
        telem_config({"gnss0_a", "gnss0_b"}, {{{"rows", {"gnss0_a"}}, {"cpu_affinity", {0}}},
                                              {{"rows", {"gnss0_b"}}, {"cpu_affinity", {0}}}});
    // Port 1 on loopback refuses: the send stays in its connect loop.
    cfg["gnss0_telem_send"] = {{"kotekan_stage", "GnssTelemSend"},
                               {"buf", "gnss0_telem_buf"},
                               {"server_ip", "127.0.0.1"},
                               {"server_port", 1},
                               {"reconnect_time", 30},
                               {"use_config_tracker", false}};
    BOOST_CHECK_EQUAL(fatal_of([&] { start_stop_start(cfg, {"/gnss0_telem/chains"}); }), "");
}

namespace {

/// Records each row of the telemetry stall case is fed, one every TELEM_STALL_DT_S from record
/// TELEM_STALL_BASE, the frames its peer reads before it stops for TELEM_STALL_HOLD_S.
constexpr int TELEM_STALL_RECORDS = 1500, TELEM_STALL_BASE = 1000, TELEM_STALL_AT = 50;
constexpr double TELEM_STALL_DT_S = 0.002, TELEM_STALL_HOLD_S = 2.0;

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/// Reads exactly @p n bytes from @p fd into @p p; false on EOF or error.
bool recv_all(int fd, void* p, size_t n) {
    uint8_t* b = (uint8_t*)p;
    while (n > 0) {
        const ssize_t r = recv(fd, b, n, 0);
        if (r <= 0)
            return false;
        b += r;
        n -= (size_t)r;
    }
    return true;
}

/// The counter of @p metric labelled first with stage_name @p stage in this process's metrics, 0
/// for one not serialized.
double metric_value(const std::string& metric, const std::string& stage) {
    std::istringstream metrics(kotekan::prometheus::Metrics::instance().serialize());
    const std::string head = metric + "{stage_name=\"" + stage + "\"";
    for (std::string line; std::getline(metrics, line);)
        if (line.rfind(head, 0) == 0)
            return std::stod(line.substr(line.rfind(' ') + 1));
    return 0.0;
}

/// Three telemetry rows of a host, of 32 PRNs and 6 channels, 2 and 1, and 14 and 2, on a worker
/// each, write one 8-deep out_buf that a GnssTelemSend with @p drop_frames sends to a loopback
/// peer, which stops reading for TELEM_STALL_HOLD_S. Throws a FatalError naming each check that
/// fails.
void telem_stall(bool drop_frames) {
    const std::vector<std::string> keys = {"gnss0_a", "gnss0_b", "gnss0_c"};
    const std::vector<int> max_prn = {32, 2, 14}, n_chan = {6, 1, 2};
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
        throw FatalError("sched_getaffinity");
    std::vector<int> cpus;
    for (int c = 0; c < CPU_SETSIZE; ++c)
        if (CPU_ISSET(c, &allowed))
            cpus.push_back(c);

    // A small receive buffer, so the peer's kernel holds little once it stops reading.
    const int listener = socket(AF_INET, SOCK_STREAM, 0);
    const int rcvbuf = 4096;
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t len = sizeof(addr);
    if (listener < 0 || setsockopt(listener, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf)) != 0
        || bind(listener, (sockaddr*)&addr, sizeof(addr)) != 0 || listen(listener, 1) != 0
        || getsockname(listener, (sockaddr*)&addr, &len) != 0)
        throw FatalError("cannot listen on loopback");

    json workers = json::array();
    for (const auto& k : keys)
        workers.push_back({{"rows", {k}}, {"cpu_affinity", cpus}});
    json cfg = telem_config(keys, workers);
    cfg["cpu_affinity"] = cpus;
    size_t out_bytes = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
        json& row = cfg["gnss0_telem"]["chains"][keys[i]];
        row["n_prn"] = max_prn[i];
        row["max_prn"] = max_prn[i];
        row["n_chan"] = n_chan[i];
        row["channel_ids"] = std::vector<int>(n_chan[i], 100);
        out_bytes = std::max(out_bytes, gnss::telem_frame_bytes(4, max_prn[i], n_chan[i]));
    }
    cfg["gnss0_telem_send"] = {{"buf", "gnss0_telem_buf"},
                               {"server_ip", "127.0.0.1"},
                               {"server_port", ntohs(addr.sin_port)},
                               {"use_config_tracker", false},
                               {"send_timeout", 30},
                               {"drop_frames", drop_frames}};
    kotekan::Config config;
    config.update_config(cfg);
    auto pool = metadataPool::create(64, sizeof(GnssChanMetadata), "pool", "GnssChanMetadata");
    kotekan::bufferContainer bc;
    std::vector<std::unique_ptr<Buffer>> in;
    for (size_t i = 0; i < keys.size(); ++i) {
        in.push_back(std::make_unique<Buffer>(
            4, gnss::frame_floats(max_prn[i], 0, n_chan[i]) * sizeof(float), pool, "rec_" + keys[i],
            "standard", 0, false, false, std::vector<int>{}, true));
        in.back()->register_producer("feeder");
        bc.add_buffer("rec_" + keys[i], in.back().get());
    }
    // Filled with 0x5A, which no frame may carry past its own length.
    Buffer out(8, out_bytes, pool, "gnss0_telem_buf", "standard", 0, false, false,
               std::vector<int>{}, true, 0x5A);
    bc.add_buffer("gnss0_telem_buf", &out);
    GnssTelemPack host(config, "/gnss0_telem", bc);
    GnssTelemSend send(config, "/gnss0_telem_send", bc);

    // [time, row, seq, win, present, bytes] of each frame the peer reads.
    std::vector<std::vector<double>> taken;
    double hold0 = 0.0, hold1 = 0.0;
    std::thread peer([&] {
        const int fd = accept(listener, nullptr, nullptr);
        if (fd < 0)
            return;
        for (int k = 0;; ++k) {
            if (k == TELEM_STALL_AT) {
                hold0 = now_s();
                std::this_thread::sleep_for(std::chrono::duration<double>(TELEM_STALL_HOLD_S));
                hold1 = now_s();
            }
            bufferFrameHeaderNoConfigTracker bh;
            if (!recv_all(fd, &bh, sizeof(bh)))
                break;
            std::vector<uint8_t> meta(bh.metadata_size), frame(bh.frame_size);
            if (!recv_all(fd, meta.data(), meta.size()) || !recv_all(fd, frame.data(), frame.size())
                || frame.size() < sizeof(gnss::TelemHeader))
                break;
            gnss::TelemHeader th;
            std::memcpy(&th, frame.data(), sizeof(th));
            const std::string chain(th.chain, strnlen(th.chain, gnss::TELEM_NAME));
            double row = -1;
            for (size_t i = 0; i < keys.size(); ++i)
                if (keys[i].substr(6) == chain)
                    row = i;
            taken.push_back({now_s(), row, (double)th.seq, (double)th.win, (double)th.present,
                             (double)frame.size()});
        }
        close(fd);
    });
    host.start();
    send.start();

    const double t0 = now_s() + 0.2;
    std::vector<double> late(keys.size(), 0.0);
    std::vector<std::thread> feeders;
    for (size_t i = 0; i < keys.size(); ++i)
        feeders.emplace_back([&, i] {
            Buffer* b = in[i].get();
            const int64_t rec_samples = 2048 * 16384;
            for (int j = 0; j < TELEM_STALL_RECORDS; ++j) {
                const int id = j % b->num_frames;
                float* f = (float*)b->wait_for_empty_frame("feeder", id);
                if (f == nullptr)
                    return;
                while (now_s() < t0 + j * TELEM_STALL_DT_S)
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                late[i] = std::max(late[i], now_s() - (t0 + j * TELEM_STALL_DT_S));
                f[gnss::REC_P_ENERGY] = 1.0f;
                b->allocate_new_metadata_object(id);
                get_gnss_chan_metadata(b, id)->sample_seq = (TELEM_STALL_BASE + j) * rec_samples;
                b->mark_frame_full("feeder", id);
            }
        });
    for (std::thread& t : feeders)
        t.join();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    host.stop();
    send.stop();
    for (auto& b : in)
        b->send_shutdown_signal();
    out.send_shutdown_signal();
    host.join();
    send.join();
    peer.join();
    close(listener);

    std::string bad;
    if (hold1 - hold0 < TELEM_STALL_HOLD_S)
        bad += " no hold;";
    const int windows = TELEM_STALL_RECORDS / 4, win0 = TELEM_STALL_BASE / 4;
    const double send_dropped =
        metric_value("kotekan_buffer_send_dropped_frame_count", "/gnss0_telem_send");
    double send_share = 0.0;
    for (size_t i = 0; i < keys.size(); ++i) {
        const size_t bytes = gnss::telem_frame_bytes(4, max_prn[i], n_chan[i]);
        int got = 0, after = 0, misfit = 0, disorder = 0;
        int64_t last_seq = -1, last_win = -1, gaps = 0;
        for (const auto& t : taken) {
            if (t[1] != (double)i)
                continue;
            const int64_t seq = (int64_t)t[2], win = (int64_t)t[3];
            ++got;
            after += t[0] > hold1;
            misfit += t[4] != 0xF || t[5] != bytes;
            disorder += seq <= last_seq || win <= last_win;
            gaps += seq - last_seq - 1;
            last_seq = seq;
            last_win = win;
        }
        const double dropped =
            metric_value("kotekan_gnss_telem_dropped_frames_total", "/" + keys[i] + "_telem_pack");
        send_share += gaps - dropped;
        const std::string row = fmt::format(
            " {:s}: {:d} of {:d} windows sent, {:d} after the hold, {:.0f} dropped, {:d} seq "
            "gaps, last seq {:d} win {:d}, {:d} misfit, {:d} out of order, feeder {:.3f} s late;",
            keys[i], got, windows, after, dropped, gaps, last_seq, last_win, misfit, disorder,
            late[i]);
        BOOST_TEST_MESSAGE(row);
        if (late[i] >= 0.1 || dropped <= 0 || after <= 0 || misfit || disorder
            || last_seq != windows - 1 || last_win != win0 + windows - 1 || gaps < dropped
            || (!drop_frames && gaps != dropped))
            bad += row;
    }
    // The send's own drops are the rest of the rows' seq gaps.
    if (send_share != send_dropped || (!drop_frames && send_dropped != 0))
        bad += fmt::format(" the send dropped {:.0f} frames, the seq gaps leave {:.0f};",
                           send_dropped, send_share);
    if (!bad.empty())
        throw FatalError("drop_frames " + std::to_string(drop_frames) + ":" + bad);
}

} // namespace

// A telemetry host's rows sharing one out_buf never wait for its GnssTelemSend: while the send's
// peer stops reading, the feeders standing in for the record assemblers are never held, each row
// drops the windows that find the out_buf full, counts them and skips their seq, and the send
// sends each frame at its row's own length, the rows' frames complete and in order before and
// after the hold.
BOOST_AUTO_TEST_CASE(telem_host_never_waits_for_its_send) {
    for (bool drop_frames : {false, true})
        BOOST_CHECK_EQUAL(fatal_of([&] { telem_stall(drop_frames); }), "");
}

// A telemetry host's workers run as threads of the stage, each named for its row if it has one,
// as is its consumer of the row's in_buf, and bound to its own cpu_affinity; the rows' shared
// out_buf has one producer, the host.
BOOST_AUTO_TEST_CASE(telem_workers_are_pinned_threads_of_the_stage) {
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
            json cfg =
                telem_config(abc, {{{"rows", {"gnss0_a", "gnss0_c"}}, {"cpu_affinity", {cores[0]}}},
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
            Buffer out(8, gnss::telem_frame_bytes(4, 2, 1), nullptr, "gnss0_telem_buf", "standard",
                       0, false, false, std::vector<int>{}, false);
            bc.add_buffer("gnss0_telem_buf", &out);
            GnssTelemPack host(config, "/gnss0_telem", bc);
            if (out.get_producer_names() != std::vector<std::string>{"/gnss0_telem"})
                throw FatalError("gnss0_telem_buf has another producer");
            const std::vector<std::vector<std::string>> readers = {
                {"/gnss0_telem/w0"}, {"/gnss0_b_telem_pack"}, {"/gnss0_telem/w0"}};
            for (size_t i = 0; i < bufs.size(); ++i) {
                std::string s;
                for (const std::string& n : bufs[i]->get_consumer_names())
                    s += " " + n;
                if (bufs[i]->get_consumer_names() != readers[i])
                    throw FatalError("rec_" + abc[i] + " consumers" + s);
            }
            const std::map<std::string, int> want = {
                {string_tail("/gnss0_telem", 15), cores[1]},
                {string_tail("/gnss0_telem/w0", 15), cores[0]},
                {string_tail("/gnss0_b_telem_pack", 15), cores[1]}};
            auto seen = [&] {
                std::map<std::string, int> got;
                for (pid_t tid : host.get_tids()) {
                    std::ifstream f("/proc/self/task/" + std::to_string(tid) + "/comm");
                    std::string comm;
                    std::getline(f, comm);
                    cpu_set_t set;
                    CPU_ZERO(&set);
                    if (sched_getaffinity(tid, sizeof(set), &set) != 0 || CPU_COUNT(&set) != 1)
                        return got;
                    for (int c = 0; c < CPU_SETSIZE; ++c)
                        if (CPU_ISSET(c, &set))
                            got[comm] = c;
                }
                return got;
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
            out.send_shutdown_signal();
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

namespace {

/// The rec_rows of rows of @p n_prn PRNs each, chain ids 1, 2, ..., of one element and @p n_chan
/// comb columns, tiling a record buffer frame in that order.
std::vector<gnss::RecRow> rec_layout(const std::vector<int>& n_prn, int n_chan = 1) {
    std::vector<gnss::RecRow> rows;
    size_t at = gnss::RecDir::bytes(n_prn.size());
    for (size_t i = 0; i < n_prn.size(); ++i) {
        at = (at + gnss::REC_ROW_ALIGN - 1) / gnss::REC_ROW_ALIGN * gnss::REC_ROW_ALIGN;
        const int64_t b = gnss::frame_floats(n_prn[i], 1, n_chan) * sizeof(float);
        rows.push_back({(int64_t)i + 1, n_prn[i], n_chan, (int64_t)at, b});
        at += b;
    }
    return rows;
}

json rec_row_json(const gnss::RecRow& r) {
    return {{"chain_id", r.chain_id},
            {"n_prn", r.n_prn},
            {"n_chan", r.n_chan},
            {"offset", r.offset},
            {"bytes", r.bytes}};
}

/// Rows gnss0_a, gnss0_b and gnss0_c of a GPU half, their PRNs, and their rec_rows.
const std::vector<std::string> ABC = {"gnss0_a", "gnss0_b", "gnss0_c"};
const std::vector<int> ABC_PRN = {3, 1, 2};
const std::vector<gnss::RecRow> ABC_REC = rec_layout(ABC_PRN);

/// A GPU half's record buffer gnss0_rec_buf, @p frames deep, sized for @p rows (in ABC's order),
/// and with @p cmb a combine host gnss0_combine and with @p tel a telemetry host gnss0_telem on it,
/// each running ABC, row k with rec_row @p rows[k], on one worker per row or with @p one_worker
/// one worker for all.
json rec_config(const std::vector<gnss::RecRow>& rows, bool cmb, bool tel, bool one_worker,
                int frames = 4) {
    json cfg = {{"log_level", "warn"},
                {"cpu_affinity", {0}},
                {"telescope", {{"num_polarizations", 2}, {"num_dishes", 2}}},
                {"gnss_pool",
                 {{"kotekan_metadata_pool", "GnssChanMetadata"}, {"num_metadata_objects", 64}}}};
    size_t bytes = 0;
    for (const gnss::RecRow& r : rows)
        bytes = std::max(bytes, (size_t)(r.offset + r.bytes));
    cfg["gnss0_rec_buf"] = {
        {"kotekan_buffer", "standard"}, {"num_frames", frames}, {"frame_size", bytes}};
    json workers = json::array();
    if (one_worker)
        workers.push_back({{"rows", ABC}, {"cpu_affinity", {0}}});
    else
        for (const auto& k : ABC)
            workers.push_back({{"rows", {k}}, {"cpu_affinity", {0}}});
    if (cmb) {
        json& h = cfg["gnss0_combine"];
        h = {{"kotekan_stage", "GnssCoherentCombiner"},
             {"n_elements", 1},
             {"fft_len", 16384},
             {"integration_length", 10},
             {"in_bufs", {"gnss0_rec_buf"}},
             {"workers", workers}};
        for (size_t k = 0; k < ABC.size(); ++k)
            h["chains"][ABC[k]] = {{"n_prn", ABC_PRN[k]},
                                   {"chain", ABC[k].substr(6)},
                                   {"rec_row", rec_row_json(rows[k])}};
    }
    if (tel) {
        cfg["gnss0_telem_buf"] = {{"kotekan_buffer", "standard"},
                                  {"metadata_pool", "gnss_pool"},
                                  {"num_frames", 8},
                                  {"frame_size", gnss::telem_frame_bytes(4, 3, 1)}};
        json& h = cfg["gnss0_telem"];
        h = {{"kotekan_stage", "GnssTelemPack"},
             {"inst", "node.0"},
             {"n_elements", 1},
             {"n_chan", 1},
             {"channel_ids", {100}},
             {"hops_per_record", 2048},
             {"fft_len", 16384},
             {"in_buf", "gnss0_rec_buf"},
             {"out_buf", "gnss0_telem_buf"},
             {"workers", workers}};
        for (size_t k = 0; k < ABC.size(); ++k)
            h["chains"][ABC[k]] = {{"chain", ABC[k].substr(6)}, {"n_prn", ABC_PRN[k]},
                                   {"max_prn", ABC_PRN[k]},     {"n_chan", 1},
                                   {"channel_ids", {100}},      {"rec_row", rec_row_json(rows[k])}};
    }
    return cfg;
}

/// rec_config's combine and telemetry hosts, the combiner's rows on @p cmb_workers, beside a record
/// host writing gnss0_rec_buf, its rows ABC of one PRN and two elements reading in_buf epl_<key> on
/// @p workers.
json rec_hosts_config(const json& workers, const json& cmb_workers) {
    json rec = record_config(ABC, workers);
    json& h = rec["gnss0_records"];
    size_t at = gnss::RecDir::bytes(3);
    for (size_t k = 0; k < ABC.size(); ++k) {
        at = (at + gnss::REC_ROW_ALIGN - 1) / gnss::REC_ROW_ALIGN * gnss::REC_ROW_ALIGN;
        const int64_t b = gnss::frame_floats(1, 2, 0) * sizeof(float);
        h["chains"][ABC[k]].erase("out_buf");
        h["chains"][ABC[k]]["rec_row"] = rec_row_json({(int64_t)k + 1, 1, 0, (int64_t)at, b});
        at += b;
    }
    h["out_buf"] = "gnss0_rec_buf";
    json cfg = rec_config(ABC_REC, true, true, false);
    cfg["gnss0_rec_buf"]["frame_size"] = std::max(at, (size_t)cfg["gnss0_rec_buf"]["frame_size"]);
    for (const auto& [k, v] : rec.items())
        if (!cfg.contains(k))
            cfg[k] = v;
    cfg["gnss0_combine"]["workers"] = cmb_workers;
    return cfg;
}

/// Samples per record of rec_config's hosts, and a record frame's first sample on their grid.
constexpr int64_t REC_SAMPLES = 2048LL * 16384;
constexpr int64_t REC_WSTART = 1000 * REC_SAMPLES;

/// The buffers of @p cfg, every block with kotekan_buffer, in @p bc, those with a metadata_pool
/// on one pool.
std::vector<std::unique_ptr<Buffer>> buffers_of(const json& cfg, kotekan::bufferContainer& bc) {
    auto pool = metadataPool::create(64, 64, "gnss_pool", "GnssChanMetadata");
    std::vector<std::unique_ptr<Buffer>> bufs;
    for (const auto& [name, b] : cfg.items()) {
        if (!b.is_object() || !b.contains("kotekan_buffer"))
            continue;
        bufs.push_back(std::make_unique<Buffer>(
            b.at("num_frames").get<int>(), b.at("frame_size").get<size_t>(),
            b.contains("metadata_pool") ? pool : nullptr, name, "standard", 0, false, false,
            std::vector<int>{}, false));
        bc.add_buffer(name, bufs.back().get());
    }
    return bufs;
}

/// Starts @p cfg's hosts on gnss0_rec_buf and writes @p frames frames of it as a producer whose
/// directory has rows @p dir, frame k's wstarts k records after REC_WSTART, then @p plant applied
/// to the last frame. Returns the first FATAL a host's thread raised, or "" once the hosts have
/// released every frame.
std::string feed_rec_frame(const json& cfg, const std::vector<gnss::RecRow>& dir,
                           const std::function<void(uint8_t*)>& plant = nullptr, int frames = 1) {
    kotekan::Config config;
    config.update_config(cfg);
    kotekan::bufferContainer bc;
    std::vector<std::unique_ptr<Buffer>> bufs = buffers_of(cfg, bc);
    Buffer* buf = bc.get_buffer("gnss0_rec_buf");
    std::vector<std::unique_ptr<kotekan::Stage>> hosts;
    if (cfg.contains("gnss0_combine"))
        hosts.emplace_back(new GnssCoherentCombiner(config, "/gnss0_combine", bc));
    if (cfg.contains("gnss0_telem"))
        hosts.emplace_back(new GnssTelemPack(config, "/gnss0_telem", bc));
    buf->register_producer("feeder");
    const gnss::RecDir rec("feeder", dir, buf);
    for (auto& h : hosts)
        h->start();
    for (int k = 0; k < frames; ++k) {
        uint8_t* f = buf->wait_for_empty_frame("feeder", k);
        std::memset(f, 0, buf->frame_size);
        for (size_t i = 0; i < rec.size(); ++i)
            rec.write(f, i, REC_WSTART + k * REC_SAMPLES);
        if (plant && k == frames - 1)
            plant(f);
        buf->mark_frame_full("feeder", k);
    }
    std::string msg;
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10)) {
        msg = kotekan::kotekanLogging::get_error_message();
        if (msg != "not set" || buf->get_num_full_frames() == 0)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (msg == "not set")
        msg.clear();
    for (auto& h : hosts)
        h->stop();
    for (auto& b : bufs)
        b->send_shutdown_signal();
    for (auto& h : hosts)
        h->join();
    return msg;
}

/// The PRN that read_own_blocks writes in PRN slot @p p of row @p k.
float planted_prn(size_t k, int p) {
    return 10.0f * (float)(k + 1) + (float)p;
}

/// Feeds rec_config's combine and telemetry hosts, each running ABC on one worker, one telemetry
/// window of records, PRN slot p of row k carrying PRN planted_prn(k, p). Throws a FatalError
/// naming each row whose emit or telemetry frame carries another PRN.
void read_own_blocks() {
    const json cfg = rec_config(ABC_REC, true, true, true);
    kotekan::Config config;
    config.update_config(cfg);
    kotekan::bufferContainer bc;
    std::vector<std::unique_ptr<Buffer>> bufs = buffers_of(cfg, bc);
    Buffer* buf = bc.get_buffer("gnss0_rec_buf");
    Buffer* tel = bc.get_buffer("gnss0_telem_buf");
    GnssCoherentCombiner cmb(config, "/gnss0_combine", bc);
    GnssTelemPack pack(config, "/gnss0_telem", bc);
    buf->register_producer("feeder");
    tel->register_consumer("reader");
    const gnss::RecDir rec("feeder", ABC_REC, buf);
    cmb.start();
    pack.start();
    // The telemetry rows' records_per_frame, 4, fills one window, and the buffer.
    const int n_rec = 4;
    const size_t stride = gnss::record_stride(1);
    for (int r = 0; r < n_rec; ++r) {
        uint8_t* f = buf->wait_for_empty_frame("feeder", r);
        std::memset(f, 0, buf->frame_size);
        const double utc = 1.7e9 + r * 0.01;
        for (size_t k = 0; k < ABC.size(); ++k) {
            rec.write(f, k, REC_WSTART + r * REC_SAMPLES);
            for (int p = 0; p < ABC_PRN[k]; ++p) {
                float* x = (float*)(f + ABC_REC[k].offset) + p * stride;
                x[gnss::REC_PRN] = planted_prn(k, p);
                x[gnss::REC_P_ENERGY] = 1.0f;
                std::memcpy(x + gnss::RECORD_UTC_SLOT, &utc, sizeof(utc));
            }
        }
        buf->mark_frame_full("feeder", r);
    }
    timespec until;
    clock_gettime(CLOCK_REALTIME, &until);
    until.tv_sec += 10;
    std::string bad;
    // One telemetry frame per row.
    for (int id = 0; id < (int)ABC.size(); ++id) {
        if (tel->wait_for_full_frame_timeout("reader", id, until) != 0) {
            bad += fmt::format(" no telemetry frame {:d};", id);
            break;
        }
        gnss::TelemHeader th;
        std::memcpy(&th, tel->frames[id], sizeof(th));
        const std::string chain(th.chain, strnlen(th.chain, gnss::TELEM_NAME));
        const auto k = (size_t)(std::find(ABC.begin(), ABC.end(), "gnss0_" + chain) - ABC.begin());
        if (k == ABC.size() || th.present != 0xF || th.n_prn != ABC_PRN[k]) {
            bad += fmt::format(" telemetry frame {:d} of chain {:s}, present {:#x};", id, chain,
                               th.present);
            continue;
        }
        const float* rows = (const float*)(tel->frames[id] + sizeof(th));
        for (int p = 0; p < ABC_PRN[k]; ++p) {
            const float prn = rows[gnss::telem_row_offset(0, p, th.n_prn, th.n_row_total)];
            if (prn != planted_prn(k, p))
                bad += fmt::format(" telemetry {:s} row {:d} PRN {:g};", ABC[k], p, prn);
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    bool done = false;
    while (!done && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10)) {
        done = true;
        for (size_t k = 0; k < ABC.size(); ++k)
            done = done && cmb.records_done(k) == (uint64_t)n_rec;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!done)
        bad += " the combiner did not take every record;";
    for (size_t k = 0; k < ABC.size() && done; ++k) {
        const std::vector<float>& emit = cmb.scratch(k);
        const size_t emit_stride = emit.size() / ABC_PRN[k];
        for (int p = 0; p < ABC_PRN[k]; ++p)
            if (emit[p * emit_stride + gnss::REC_PRN] != planted_prn(k, p))
                bad += fmt::format(" combine {:s} PRN slot {:d} emits PRN {:g};", ABC[k], p,
                                   emit[p * emit_stride + gnss::REC_PRN]);
    }
    cmb.stop();
    pack.stop();
    for (auto& b : bufs)
        b->send_shutdown_signal();
    cmb.join();
    pack.join();
    const std::string msg = kotekan::kotekanLogging::get_error_message();
    if (msg != "not set")
        bad += " " + msg;
    if (!bad.empty())
        throw FatalError("rows on one worker:" + bad);
}

/// Feeds frame @p id of the record host's in_buf @p buf, as producer "feeder", an epl frame
/// (gnssGpuChain.hpp) of @p n_rec records of one PRN slot, PRN 1, run in none of them.
void feed_epl(Buffer* buf, int id, int n_rec) {
    uint8_t* f = buf->wait_for_empty_frame("feeder", id);
    std::memset(f, 0, buf->frame_size);
    gnss_gpu::FrameHdr h{};
    h.n_rec = n_rec;
    h.n_prn = 1;
    h.n_chan = 1;
    h.utc0 = 1.7e9;
    h.n_rows_spec = gnss_gpu::ROWS_PLAIN;
    std::memcpy(f, &h, sizeof(h));
    for (int r = 0; r < n_rec; ++r) {
        const int64_t wstart = REC_WSTART + r * REC_SAMPLES;
        std::memcpy(f + gnss_gpu::off_winstart() + r * sizeof(wstart), &wstart, sizeof(wstart));
        gnss_gpu::PrnCtl c{};
        c.prn = 1;
        std::memcpy(f + gnss_gpu::off_prnctl() + r * sizeof(c), &c, sizeof(c));
    }
    buf->mark_frame_full("feeder", id);
}

/// Starts the record host of rec_hosts_config, its rows ABC on @p workers, and feeds row k an epl
/// frame of @p n_rec[k] records once row k - 1 has written its records. Returns the first FATAL a
/// host's thread raised, or "" once every row has written its records.
std::string feed_epl_rows(const json& workers, const std::vector<int>& n_rec) {
    const json cfg = rec_hosts_config(workers, workers);
    kotekan::Config config;
    config.update_config(cfg);
    kotekan::bufferContainer bc;
    std::vector<std::unique_ptr<Buffer>> bufs = buffers_of(cfg, bc);
    GnssGpuRecordAssemble host(config, "/gnss0_records", bc);
    for (const auto& k : ABC)
        bc.get_buffer("epl_" + k)->register_producer("feeder");
    host.start();
    std::string msg = "not set";
    for (size_t k = 0; k < ABC.size() && msg == "not set"; ++k) {
        feed_epl(bc.get_buffer("epl_" + ABC[k]), 0, n_rec[k]);
        const auto t0 = std::chrono::steady_clock::now();
        while ((msg = kotekan::kotekanLogging::get_error_message()) == "not set"
               && host.records_done(k) != (uint64_t)n_rec[k]) {
            if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(10)) {
                msg = fmt::format("row {:s} wrote {:d} records", ABC[k], host.records_done(k));
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    host.stop();
    for (auto& b : bufs)
        b->send_shutdown_signal();
    host.join();
    return msg == "not set" ? "" : msg;
}

} // namespace

// A GPU half's record buffer frame: the rows tile it after the directory, in offset order, each
// on its own cache lines, inside the frame, with unique chain ids.
BOOST_AUTO_TEST_CASE(rec_rows_tile_the_frame) {
    auto dir_of = [](std::vector<gnss::RecRow> rows, size_t frame) {
        Buffer buf(2, frame, nullptr, "gnss0_rec_buf", "standard", 0, false, false,
                   std::vector<int>{}, false);
        gnss::RecDir rec("/gnss0_records", rows, &buf);
    };
    const size_t frame = ABC_REC.back().offset + ABC_REC.back().bytes;
    BOOST_CHECK_EQUAL(fatal_of([&] { dir_of(ABC_REC, frame); }), "");
    // In any order: the directory lists them by offset.
    BOOST_CHECK_EQUAL(fatal_of([&] { dir_of({ABC_REC[2], ABC_REC[0], ABC_REC[1]}, frame); }), "");
    BOOST_CHECK_EQUAL(fatal_of([&] { dir_of(ABC_REC, frame - 1); }),
                      fmt::format("/gnss0_records: the rec_rows end at byte {:d}, past the {:d} B "
                                  "frame of gnss0_rec_buf",
                                  frame, frame - 1));
    auto rows = ABC_REC;
    rows[0].offset = 0;
    BOOST_CHECK_EQUAL(fatal_of([&] { dir_of(rows, frame); }),
                      fmt::format("/gnss0_records: the rec_row of chain_id 1 holds bytes [0, {:d}) "
                                  "of gnss0_rec_buf; the directory and the rows before it put it "
                                  "at byte {:d}",
                                  rows[0].bytes, ABC_REC[0].offset));
    for (int64_t shift : {-64, -4, 4, 64}) {
        rows = ABC_REC;
        rows[1].offset += shift;
        BOOST_CHECK_MESSAGE(fatal_of([&] { dir_of(rows, frame); }).find("chain_id 2 holds bytes")
                                != std::string::npos,
                            "row 1 moved by " << shift);
    }
    rows = ABC_REC;
    rows[2].chain_id = 1;
    BOOST_CHECK_EQUAL(fatal_of([&] { dir_of(rows, frame); }),
                      "/gnss0_records: chain_id 1 is in two rec_rows of gnss0_rec_buf");
    // A row by its offset, and none at another.
    BOOST_CHECK_EQUAL(
        fatal_of([&] {
            Buffer buf(2, frame, nullptr, "gnss0_rec_buf", "standard", 0, false, false,
                       std::vector<int>{}, false);
            gnss::RecDir rec("/gnss0_records", {ABC_REC[2], ABC_REC[0], ABC_REC[1]}, &buf);
            if (rec.find(ABC_REC[2].offset) != 2)
                throw FatalError("row 2 is not at its offset");
            rec.find(ABC_REC[2].offset + 4);
        }),
        fmt::format("no rec_row of gnss0_rec_buf starts at byte {:d}", ABC_REC[2].offset + 4));
}

// The rows of a combine or telemetry host reading a GPU half's record buffer, an in_buf or
// in_bufs on the host block, each set their rec_row, and only theirs, and it must hold their
// records; a record host's rows writing one, an out_buf on the host block, likewise.
BOOST_AUTO_TEST_CASE(rec_rows_of_the_hosts) {
    BOOST_CHECK_EQUAL(start_fatal(rec_config(ABC_REC, true, true, false)), "");
    for (const std::string h : {"gnss0_combine", "gnss0_telem"}) {
        const std::string in = h == "gnss0_combine" ? "in_bufs" : "in_buf";
        json cfg = rec_config(ABC_REC, true, true, false);
        cfg[h]["chains"]["gnss0_b"].erase("rec_row");
        BOOST_CHECK_EQUAL(start_fatal(cfg), "/" + h + ": row gnss0_b does not set rec_row");
        cfg = rec_config(ABC_REC, true, true, false);
        cfg[h]["rec_row"] = rec_row_json(ABC_REC[0]);
        BOOST_CHECK_EQUAL(start_fatal(cfg),
                          "/" + h + ": rec_row is set per row in chains, not on the host");
        cfg = rec_config(ABC_REC, true, true, false);
        cfg[h]["chains"]["gnss0_b"][in] = cfg[h][in];
        BOOST_CHECK_EQUAL(start_fatal(cfg), "/" + h + ": row gnss0_b sets " + in
                                                + ", which the host block sets for every row");
        // Without the host's buffer, a row's rec_row is refused.
        cfg = rec_config(ABC_REC, true, true, false);
        for (const auto& k : ABC)
            cfg[h]["chains"][k][in] = cfg[h][in];
        cfg[h].erase(in);
        const std::string pack = h == "gnss0_combine" ? "_n2combine" : "_telem_pack";
        BOOST_CHECK_EQUAL(start_fatal(cfg), "/gnss0_a" + pack
                                                + ": rec_row is set, but the host "
                                                  "block sets no "
                                                + in + " for its rows");
    }
    // A rec_row that does not hold its row's record.
    auto rows = ABC_REC;
    rows[1].n_prn = 2;
    json cfg = rec_config(rows, true, false, false);
    const size_t b1 = gnss::frame_floats(1, 1, 1) * sizeof(float);
    BOOST_CHECK_EQUAL(start_fatal(cfg),
                      fmt::format("/gnss0_b_n2combine: rec_row holds 2 PRN x 1 comb columns in "
                                  "{:d} B, this row's record is 1 PRN of 1 elements x 1 comb "
                                  "columns in {:d} B",
                                  b1, b1));
    rows = ABC_REC;
    rows[1].bytes += 4;
    BOOST_CHECK(start_fatal(rec_config(rows, true, false, false)).find("rec_row holds")
                != std::string::npos);
    rows = ABC_REC;
    rows[1].n_chan = 2;
    BOOST_CHECK(
        start_fatal(rec_config(rows, false, true, false)).find("/gnss0_b_telem_pack: rec_row")
        != std::string::npos);
    // The combiner takes a row's comb columns from its rec_row, and its bytes must agree.
    rows = rec_layout(ABC_PRN, 2);
    BOOST_CHECK_EQUAL(start_fatal(rec_config(rows, true, false, false)), "");
    BOOST_CHECK(start_fatal(rec_config(rows, false, true, false)).find("rec_row holds")
                != std::string::npos);
    // The rows must tile the frame.
    rows = ABC_REC;
    std::swap(rows[0].offset, rows[2].offset);
    BOOST_CHECK(start_fatal(rec_config(rows, true, false, false)).find("the directory and the rows")
                != std::string::npos);

    // A record host's rows writing the host block's out_buf.
    json rec = record_config(ABC, json::array({{{"rows", ABC}, {"cpu_affinity", {0}}}}));
    json& h = rec["gnss0_records"];
    const auto rr = rec_layout({1, 1, 1}, 0);
    for (size_t k = 0; k < ABC.size(); ++k) {
        h["chains"][ABC[k]].erase("out_buf");
        h["chains"][ABC[k]]["rec_row"] = rec_row_json(rr[k]);
    }
    h["out_buf"] = "rec_gnss0_a";
    // record_config's rows have 2 elements.
    BOOST_CHECK_EQUAL(start_fatal(rec),
                      fmt::format("/gnss0_a_n2assemble: rec_row holds 1 PRN x 0 comb columns in "
                                  "{:d} B, this row's record is 1 PRN of 2 elements x 0 comb "
                                  "columns in {:d} B",
                                  rr[0].bytes, gnss::frame_floats(1, 2, 0) * sizeof(float)));
    size_t at = gnss::RecDir::bytes(3);
    for (size_t k = 0; k < ABC.size(); ++k) {
        at = (at + gnss::REC_ROW_ALIGN - 1) / gnss::REC_ROW_ALIGN * gnss::REC_ROW_ALIGN;
        const int64_t b = gnss::frame_floats(1, 2, 0) * sizeof(float);
        h["chains"][ABC[k]]["rec_row"] = rec_row_json({(int64_t)k + 7, 1, 0, (int64_t)at, b});
        at += b;
    }
    BOOST_CHECK_EQUAL(start_fatal(rec), "");
    json bad = rec;
    bad["gnss0_records"]["chains"]["gnss0_b"]["out_buf"] = "rec_gnss0_b";
    BOOST_CHECK_EQUAL(start_fatal(bad), "/gnss0_records: row gnss0_b sets out_buf, which the host "
                                        "block sets for every row");
    bad = rec;
    bad["gnss0_records"]["chains"]["gnss0_b"].erase("rec_row");
    BOOST_CHECK_EQUAL(start_fatal(bad), "/gnss0_records: row gnss0_b does not set rec_row");
    bad = record_config(ABC, json::array({{{"rows", ABC}, {"cpu_affinity", {0}}}}));
    bad["gnss0_records"]["chains"]["gnss0_c"]["rec_row"] =
        rec["gnss0_records"]["chains"]["gnss0_c"]["rec_row"];
    BOOST_CHECK_EQUAL(start_fatal(bad), "/gnss0_c_n2assemble: rec_row is set, but the host block "
                                        "sets no out_buf for its rows");
}

// Each worker of a record host is a producer of the GPU half's record buffer, and each worker of
// a combine and of a telemetry host a consumer of it, named as the hosts' other buffers' are.
BOOST_AUTO_TEST_CASE(rec_buf_producers_and_consumers) {
    BOOST_CHECK_EQUAL(
        fatal_of([] {
            const json cfg =
                rec_hosts_config({{{"rows", {"gnss0_a", "gnss0_c"}}, {"cpu_affinity", {0}}},
                                  {{"rows", {"gnss0_b"}}, {"cpu_affinity", {0}}}},
                                 {{{"rows", {"gnss0_a", "gnss0_b"}}, {"cpu_affinity", {0}}},
                                  {{"rows", {"gnss0_c"}}, {"cpu_affinity", {0}}}});
            kotekan::Config config;
            config.update_config(cfg);
            kotekan::bufferContainer bc;
            std::vector<std::unique_ptr<Buffer>> bufs = buffers_of(cfg, bc);
            Buffer* buf = bc.get_buffer("gnss0_rec_buf");
            GnssGpuRecordAssemble rec_host(config, "/gnss0_records", bc);
            GnssCoherentCombiner cmb_host(config, "/gnss0_combine", bc);
            GnssTelemPack tel_host(config, "/gnss0_telem", bc);
            const std::vector<std::string> producers = {"/gnss0_records/w0", "/gnss0_records/w1"};
            const std::vector<std::string> consumers = {"/gnss0_a_telem_pack",
                                                        "/gnss0_b_telem_pack", "/gnss0_c_n2combine",
                                                        "/gnss0_c_telem_pack", "/gnss0_combine/w0"};
            std::vector<std::string> p = buf->get_producer_names(), c = buf->get_consumer_names();
            std::sort(p.begin(), p.end());
            std::sort(c.begin(), c.end());
            std::string s;
            for (const auto& n : p)
                s += " p:" + n;
            for (const auto& n : c)
                s += " c:" + n;
            if (p != producers || c != consumers)
                throw FatalError("gnss0_rec_buf" + s);
        }),
        "");
}

// The record, combine and telemetry hosts on a GPU half's record buffer stop, are deleted and
// start again in one process, their rows, REST paths and metrics released.
BOOST_AUTO_TEST_CASE(rec_buf_hosts_restart_in_one_process) {
    const json workers = {{{"rows", {"gnss0_a", "gnss0_c"}}, {"cpu_affinity", {0}}},
                          {{"rows", {"gnss0_b"}}, {"cpu_affinity", {0}}}};
    BOOST_CHECK_EQUAL(fatal_of([&] {
                          start_stop_start(rec_hosts_config(workers, workers),
                                           {"/gnss0_a_n2assemble/get_spectrum",
                                            "/gnss0_c_n2assemble/set_sat_geometry",
                                            "/gnss0_b_n2combine/get_status",
                                            "/gnss0_records/chains", "/gnss0_combine/chains",
                                            "/gnss0_telem/chains"});
                      }),
                      "");
}

// A combine or telemetry worker checks each frame's directory against its host's rec_rows before
// it reads a row, and FATALs on any field that differs, a row missing or a row more.
BOOST_AUTO_TEST_CASE(rec_dir_mismatch_is_fatal) {
    const std::string who = "/gnss0_combine/w0: frame 0 of gnss0_rec_buf";
    BOOST_CHECK_EQUAL(fatal_of([] {
                          const std::string m =
                              feed_rec_frame(rec_config(ABC_REC, true, true, true), ABC_REC);
                          if (!m.empty())
                              throw FatalError(m);
                      }),
                      "");
    // Each field of row b in the producer's directory: the field, the value planted, and the
    // host's own.
    const gnss::RecRow& b = ABC_REC[1];
    const std::vector<std::tuple<std::string, int64_t gnss::RecRow::*, int64_t, int64_t>> fields = {
        {"chain_id", &gnss::RecRow::chain_id, 9, b.chain_id},
        {"n_prn", &gnss::RecRow::n_prn, 2, b.n_prn},
        {"n_chan", &gnss::RecRow::n_chan, 0, b.n_chan},
        {"offset", &gnss::RecRow::offset, b.offset + 64, b.offset},
        {"bytes", &gnss::RecRow::bytes, b.bytes - 4, b.bytes}};
    for (const auto& [field, member, planted, own] : fields) {
        const std::string msg = fatal_of([&, member = member, planted = planted] {
            throw FatalError(
                feed_rec_frame(rec_config(ABC_REC, true, false, true), ABC_REC, [&](uint8_t* f) {
                    gnss::RecRow r;
                    uint8_t* at = f + 16 + sizeof(r);
                    std::memcpy(&r, at, sizeof(r));
                    r.*member = planted;
                    std::memcpy(at, &r, sizeof(r));
                }));
        });
        BOOST_CHECK_EQUAL(msg, fmt::format("{:s}: directory row 1 has {:s} {:d}, this host's row 1 "
                                           "(chain_id 2) has {:d}",
                                           who, field, planted, own));
    }
    // A row missing, a row more, and no directory at all.
    const std::vector<gnss::RecRow> two = rec_layout({3, 1});
    const std::vector<gnss::RecRow> four = rec_layout({3, 1, 2, 5});
    BOOST_CHECK_EQUAL(
        fatal_of(
            [&] { throw FatalError(feed_rec_frame(rec_config(ABC_REC, true, false, true), two)); }),
        who + " lists 2 rows (chain_id 1, 2), this host reads 3 (chain_id 1, 2, 3)");
    BOOST_CHECK_EQUAL(fatal_of([&] {
                          json cfg = rec_config(ABC_REC, true, false, true);
                          cfg["gnss0_rec_buf"]["frame_size"] =
                              four.back().offset + four.back().bytes;
                          throw FatalError(feed_rec_frame(cfg, four));
                      }),
                      who
                          + " lists 4 rows (chain_id 1, 2, 3, 4), this host reads 3 (chain_id 1, "
                            "2, 3)");
    BOOST_CHECK_EQUAL(fatal_of([&] {
                          throw FatalError(
                              feed_rec_frame(rec_config(ABC_REC, true, false, true), ABC_REC,
                                             [](uint8_t* f) { std::memset(f, 0, 8); }));
                      }),
                      who + " starts with 0x0, not a row directory");
    // The telemetry host checks it too.
    BOOST_CHECK_EQUAL(
        fatal_of(
            [&] { throw FatalError(feed_rec_frame(rec_config(ABC_REC, false, true, true), two)); }),
        "/gnss0_telem/w0: frame 0 of gnss0_rec_buf lists 2 rows (chain_id 1, 2), "
        "this host reads 3 (chain_id 1, 2, 3)");
    // Both check every frame, not only the first.
    for (const std::string h : {"/gnss0_combine", "/gnss0_telem"}) {
        const bool cmb = h == "/gnss0_combine";
        BOOST_CHECK_EQUAL(fatal_of([&] {
                              throw FatalError(feed_rec_frame(
                                  rec_config(ABC_REC, cmb, !cmb, true), ABC_REC,
                                  [](uint8_t* f) {
                                      const int64_t chain_id = 9;
                                      std::memcpy(f + 16 + sizeof(gnss::RecRow), &chain_id,
                                                  sizeof(chain_id));
                                  },
                                  2));
                          }),
                          h
                              + "/w0: frame 1 of gnss0_rec_buf: directory row 1 has chain_id 9, "
                                "this host's row 1 (chain_id 2) has 2");
    }
}

// A combine or telemetry worker of several rows reads each row's records from the row's own block
// of each frame.
BOOST_AUTO_TEST_CASE(rec_rows_read_their_own_blocks) {
    BOOST_CHECK_EQUAL(fatal_of(read_own_blocks), "");
}

// Every row of a record host writing a GPU half's record buffer writes the same records per frame,
// whatever its worker, and a worker of several rows fewer than the buffer's frames.
BOOST_AUTO_TEST_CASE(rec_buf_rows_write_the_same_records_per_frame) {
    json one_each = json::array();
    for (const auto& k : ABC)
        one_each.push_back({{"rows", {k}}, {"cpu_affinity", {0}}});
    const json one = json::array({{{"rows", ABC}, {"cpu_affinity", {0}}}});
    for (const json& w : {one_each, one}) {
        BOOST_CHECK_EQUAL(fatal_of([&] { throw FatalError(feed_epl_rows(w, {2, 2, 2})); }), "");
        BOOST_CHECK_EQUAL(fatal_of([&] { throw FatalError(feed_epl_rows(w, {2, 1, 2})); }),
                          "/gnss0_b_n2assemble: 1 records this frame, another row of "
                          "gnss0_rec_buf 2; every row of a GPU half's record buffer must write "
                          "the same records per frame");
    }
    // rec_config's buffer has 4 frames.
    BOOST_CHECK_EQUAL(fatal_of([&] { throw FatalError(feed_epl_rows(one_each, {4, 4, 4})); }), "");
    BOOST_CHECK_EQUAL(fatal_of([&] { throw FatalError(feed_epl_rows(one, {4, 4, 4})); }),
                      "/gnss0_a_n2assemble: 4 records this frame; a worker of several rows must "
                      "write fewer records per frame than the 4 frames of gnss0_rec_buf");
}

// A frame of a GPU half's record buffer reaches its consumers only once every producer, each a
// record host's worker, has marked it full, with peek_hold as the generator sets it. A worker may
// run ahead of another, but never into a frame of its own it has marked and that is not yet
// full and taken.
BOOST_AUTO_TEST_CASE(rec_buf_frame_is_full_once_every_worker_has_marked_it) {
    Buffer buf(4, 64, nullptr, "gnss0_rec_buf", "standard", 0, false, false, std::vector<int>{},
               false);
    buf.enable_peek_hold();
    buf.register_producer("w0");
    buf.register_producer("w1");
    buf.register_consumer("c");
    auto soon = [] {
        timespec t;
        clock_gettime(CLOCK_REALTIME, &t);
        t.tv_nsec += 50000000;
        if (t.tv_nsec >= 1000000000) {
            t.tv_nsec -= 1000000000;
            t.tv_sec += 1;
        }
        return t;
    };
    // w0 marks frames 0-3, a whole lap ahead of w1; none is full.
    for (int id = 0; id < 4; ++id) {
        BOOST_REQUIRE(buf.wait_for_empty_frame("w0", id) != nullptr);
        buf.mark_frame_full("w0", id);
        BOOST_CHECK_EQUAL(buf.wait_for_full_frame_timeout("c", id, soon()), 1);
    }
    BOOST_CHECK_EQUAL(buf.get_num_full_frames(), 0);
    // w0 cannot take frame 0 again until w1 has marked it and the consumer released it.
    std::atomic<bool> lapped{false};
    std::thread w0([&] {
        if (buf.wait_for_empty_frame("w0", 0) != nullptr)
            lapped = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    BOOST_CHECK(!lapped);
    BOOST_REQUIRE(buf.wait_for_empty_frame("w1", 0) != nullptr);
    buf.mark_frame_full("w1", 0);
    BOOST_CHECK_EQUAL(buf.wait_for_full_frame_timeout("c", 0, soon()), 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    BOOST_CHECK(!lapped);
    buf.mark_frame_empty("c", 0);
    // peek_hold keeps frame 0, the newest full one, until frame 1 is full.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    BOOST_CHECK(!lapped);
    BOOST_REQUIRE(buf.wait_for_empty_frame("w1", 1) != nullptr);
    buf.mark_frame_full("w1", 1);
    w0.join();
    BOOST_CHECK(lapped);
    BOOST_CHECK_EQUAL(buf.wait_for_full_frame_timeout("c", 1, soon()), 0);
}

// ---- The hosts' verbs by chain (gnss::rest_register_rows) ----

namespace {

/// An HTTP reply: its status and its body.
struct Reply {
    int status = 0;
    std::string body;
    bool operator==(const Reply& o) const {
        return status == o.status && body == o.body;
    }
};

/// GET @p path from the REST server on 127.0.0.1:@p port, or with @p body POST that.
Reply http(unsigned short port, const std::string& path, const json& body = json()) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (fd < 0 || connect(fd, (sockaddr*)&addr, sizeof(addr)) != 0)
        throw FatalError("cannot connect to 127.0.0.1:" + std::to_string(port));
    const std::string data = body.is_null() ? "" : body.dump();
    std::string req = (body.is_null() ? "GET " : "POST ") + path + " HTTP/1.0\r\n";
    if (!body.is_null())
        req += "Content-Type: application/json\r\nContent-Length: " + std::to_string(data.size())
               + "\r\n";
    req += "\r\n" + data;
    if (send(fd, req.data(), req.size(), 0) != (ssize_t)req.size())
        throw FatalError("cannot send " + path);
    std::string all;
    char b[4096];
    ssize_t n;
    while ((n = recv(fd, b, sizeof(b), 0)) > 0)
        all.append(b, (size_t)n);
    close(fd);
    const size_t head = all.find("\r\n\r\n");
    if (all.compare(0, 5, "HTTP/") != 0 || head == std::string::npos)
        throw FatalError("no HTTP reply to " + path + ": " + all);
    return {std::stoi(all.substr(all.find(' ') + 1, 3)), all.substr(head + 4)};
}

/// kotekan_gnss_rest_legacy_hits_total of @p path, a path of row @p row, in this process's
/// metrics; -1 for one not serialized.
double hits(const std::string& row, const std::string& path) {
    std::istringstream metrics(kotekan::prometheus::Metrics::instance().serialize());
    const std::string head =
        "kotekan_gnss_rest_legacy_hits_total{stage_name=\"" + row + "\",path=\"" + path + "\"} ";
    for (std::string line; std::getline(metrics, line);)
        if (line.rfind(head, 0) == 0)
            return std::stod(line.substr(head.size()));
    return -1.0;
}

/// The paths of a GET /endpoints reply, as "GET <path>" and "POST <path>".
std::set<std::string> endpoints(unsigned short port) {
    const json ep = json::parse(http(port, "/endpoints").body);
    std::set<std::string> out;
    for (const char* verb : {"GET", "POST"})
        for (const auto& p : ep.at(verb))
            out.insert(std::string(verb) + " " + p.get<std::string>());
    return out;
}

double fixed_clock_s() {
    return 1000.0;
}

/// record_config's host on one worker per row, each row with its chain (a, b, c); row a also with
/// elem_sum and lane transfers on two channels, so that it alone has every verb.
json rest_record_config() {
    json workers = json::array();
    for (const auto& k : ABC)
        workers.push_back({{"rows", {k}}, {"cpu_affinity", {0}}});
    json cfg = record_config(ABC, workers);
    for (const auto& k : ABC)
        cfg["gnss0_records"]["chains"][k]["chain"] = k.substr(6);
    cfg["gnss0_records"]["chains"]["gnss0_a"].update({{"elem_sum", true},
                                                      {"channel_ids", {100, 116}},
                                                      {"lane_transfer_freq_ids", {84}},
                                                      {"elem_dir_publish", "all"}});
    return cfg;
}

/// Every GET verb of each row of rest_record_config's host, as {row key: [verbs]}.
const std::map<std::string, std::vector<std::string>> RECORD_GETS = {
    {"gnss0_a", {"get_spectrum", "get_elem_cal", "get_lane_inst"}},
    {"gnss0_b", {"get_spectrum"}},
    {"gnss0_c", {"get_spectrum"}}};

/// The replies to every GET of every row of rest_record_config's host at its own path.
std::vector<Reply> record_gets(unsigned short port) {
    std::vector<Reply> out;
    for (const auto& [k, verbs] : RECORD_GETS)
        for (const auto& v : verbs)
            out.push_back(http(port, "/" + k + "_n2assemble/" + v));
    return out;
}

/// One POST of the record transcript: its row, verb and request.
struct Post {
    std::string key, verb;
    json body;
};

const std::vector<Post> RECORD_POSTS = {
    {"gnss0_a",
     "set_sat_geometry",
     {{"1", {10.0, 20.0, 0.0, 0.0, 999.0}}, {"_bore", {5.0, 999.0}}}},
    {"gnss0_a", "set_reference_element", {{"element", 1}}},
    {"gnss0_b", "set_reference_element", {{"element", 1}}},
    {"gnss0_a", "set_elem_sum_adapt", {{"adapt", true}}},
    {"gnss0_a", "set_elem_sum_shared", {{"shared", false}}},
    {"gnss0_a", "set_elem_gain", {{"gain", {{1.0, 0.0}, {0.5, 0.5}}}}},
    {"gnss0_a", "set_elem_sum_shared_ref", {{"ref", {{1.0, 0.0}, {0.0, 1.0}}}, {"mode", "live"}}},
    {"gnss0_a", "set_elem_proj", {{"mode", "shadow"}}},
    {"gnss0_c", "set_sat_geometry", {{"1", {30.0, 40.0}}}},
    {"gnss0_a", "set_elem_sum_adapt", {{"adapt", "yes"}}},
    {"gnss0_c", "set_reference_element", {{"element", 99}}}};

/// Starts the REST server and rest_record_config's host (without its threads), then posts
/// RECORD_POSTS, each at its row's own path or, with @p by_chain, at the host's, and after each
/// reads record_gets(). Returns every reply, in order, and the hits of each POST's own path.
std::pair<std::vector<Reply>, std::vector<double>> record_transcript(bool by_chain) {
    kotekan::restServer::instance().start("127.0.0.1", 0);
    const unsigned short port = kotekan::restServer::instance().port();
    const json cfg = rest_record_config();
    kotekan::Config config;
    config.update_config(cfg);
    kotekan::bufferContainer bc;
    std::vector<std::unique_ptr<Buffer>> bufs = buffers_of(cfg, bc);
    GnssGpuRecordAssemble host(config, "/gnss0_records", bc, &fixed_clock_s);
    std::vector<Reply> out = record_gets(port);
    for (const Post& p : RECORD_POSTS) {
        if (by_chain)
            out.push_back(http(port, "/gnss0_records/" + p.verb,
                               {{"chain", p.key.substr(6)}, {"body", p.body}}));
        else
            out.push_back(http(port, "/" + p.key + "_n2assemble/" + p.verb, p.body));
        for (const Reply& r : record_gets(port))
            out.push_back(r);
    }
    std::vector<double> h;
    for (const Post& p : RECORD_POSTS)
        h.push_back(hits("/" + p.key + "_n2assemble", "/" + p.key + "_n2assemble/" + p.verb));
    return {out, h};
}

/// Runs @p fn in a forked child and returns what it wrote to the file it is given, or a line
/// starting "FATAL " with the FatalError it raised.
std::string child_text(const std::function<std::string()>& fn) {
    char path[] = "/tmp/gnss_host_rows_XXXXXX";
    const int fd = mkstemp(path);
    BOOST_REQUIRE(fd >= 0);
    close(fd);
    const std::string msg = fatal_of([&] {
        const std::string text = fn();
        std::ofstream(path) << text;
    });
    std::ifstream in(path);
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    unlink(path);
    return msg.empty() ? text : "FATAL " + msg;
}

json replies_json(const std::vector<Reply>& rs) {
    json out = json::array();
    for (const Reply& r : rs)
        out.push_back({r.status, r.body});
    return out;
}

} // namespace

// Each POST of a record row through the host, {"chain": <c>, "body": <request>}, answers and acts
// as the same POST at the row's own path: the replies, refusals included, and every GET of every
// row after each are byte for byte those of the run through the rows' own paths. Only the own
// paths count hits: once per request there, and none for the host's.
BOOST_AUTO_TEST_CASE(record_posts_by_chain_are_the_rows_own) {
    const std::string own = child_text([] {
        const auto [r, h] = record_transcript(false);
        return json{{"replies", replies_json(r)}, {"hits", h}}.dump();
    });
    const std::string by_chain = child_text([] {
        const auto [r, h] = record_transcript(true);
        return json{{"replies", replies_json(r)}, {"hits", h}}.dump();
    });
    BOOST_REQUIRE_MESSAGE(own.rfind("FATAL", 0) != 0, own);
    BOOST_REQUIRE_MESSAGE(by_chain.rfind("FATAL", 0) != 0, by_chain);
    const json a = json::parse(own), b = json::parse(by_chain);
    BOOST_REQUIRE_EQUAL(a["replies"].size(), b["replies"].size());
    int refused = 0;
    for (size_t i = 0; i < a["replies"].size(); ++i) {
        BOOST_CHECK_MESSAGE(a["replies"][i] == b["replies"][i],
                            "reply " << i << ": " << a["replies"][i] << " by chain "
                                     << b["replies"][i]);
        refused += a["replies"][i][0].get<int>() == 400;
    }
    // The two last posts are refused by the rows themselves.
    BOOST_CHECK_EQUAL(refused, 2);
    // Each own POST path: one hit per post there (a's set_elem_sum_adapt twice), none by chain.
    const std::vector<double> want = {1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1};
    BOOST_CHECK(a["hits"].get<std::vector<double>>() == want);
    BOOST_CHECK(b["hits"].get<std::vector<double>>() == std::vector<double>(want.size(), 0.0));
}

// Every GET of every record row answers through the host, ?chain=<c>, byte for byte as at the
// row's own path, its other arguments passed on; without chain with each row's reply under its
// chain. A request with no chain or another chain is refused, naming the rows' chains, and
// changes nothing. The host's paths are all the rows' own ones plus one per verb and /chains,
// which lists the rows with their chains. The host's GETs count no hits, and the own paths count
// every request, refused ones included; every own path reads 0 before its first request. Deleting
// the host releases every path and counter.
BOOST_AUTO_TEST_CASE(record_gets_by_chain_are_the_rows_own) {
    const std::string msg = fatal_of([] {
        kotekan::restServer::instance().start("127.0.0.1", 0);
        const unsigned short port = kotekan::restServer::instance().port();
        // The same rows without their chains: their own paths only.
        std::set<std::string> plain;
        {
            json cfg = rest_record_config();
            for (const auto& k : ABC)
                cfg["gnss0_records"]["chains"][k].erase("chain");
            kotekan::Config config;
            config.update_config(cfg);
            kotekan::bufferContainer bc;
            std::vector<std::unique_ptr<Buffer>> bufs = buffers_of(cfg, bc);
            GnssGpuRecordAssemble host(config, "/gnss0_records", bc, &fixed_clock_s);
            plain = endpoints(port);
            if (hits("/gnss0_a_n2assemble", "/gnss0_a_n2assemble/get_spectrum") != -1.0)
                throw FatalError("rows without chains count hits");
        }
        const json cfg = rest_record_config();
        kotekan::Config config;
        config.update_config(cfg);
        kotekan::bufferContainer bc;
        std::vector<std::unique_ptr<Buffer>> bufs = buffers_of(cfg, bc);
        auto* host = new GnssGpuRecordAssemble(config, "/gnss0_records", bc, &fixed_clock_s);
        const std::set<std::string> all = endpoints(port);
        std::set<std::string> added;
        for (const auto& p : all)
            if (!plain.count(p))
                added.insert(p);
        for (const auto& p : plain)
            if (!all.count(p))
                throw FatalError("lost " + p);
        const std::set<std::string> want_added = {"GET /gnss0_records/get_elem_cal",
                                                  "GET /gnss0_records/get_lane_inst",
                                                  "GET /gnss0_records/get_spectrum",
                                                  "POST /gnss0_records/set_elem_gain",
                                                  "POST /gnss0_records/set_elem_proj",
                                                  "POST /gnss0_records/set_elem_sum_adapt",
                                                  "POST /gnss0_records/set_elem_sum_shared",
                                                  "POST /gnss0_records/set_elem_sum_shared_ref",
                                                  "POST /gnss0_records/set_reference_element",
                                                  "POST /gnss0_records/set_sat_geometry"};
        if (added != want_added)
            throw FatalError("added " + json(added).dump());
        const json chains = json::parse(http(port, "/gnss0_records/chains").body);
        json want_chains = json::array();
        for (const auto& k : ABC)
            want_chains.push_back(
                {{"name", k}, {"stage", "/" + k + "_n2assemble"}, {"chain", k.substr(6)}});
        if (chains != want_chains)
            throw FatalError("chains " + chains.dump());
        // Every own path reads 0 before its first request.
        int own = 0;
        for (const auto& p : plain) {
            const std::string path = p.substr(p.find(' ') + 1);
            const std::string row = path.substr(0, path.rfind('/'));
            if (row.size() < 11 || row.compare(row.size() - 11, 11, "_n2assemble") != 0)
                continue;
            ++own;
            if (hits(row, path) != 0.0)
                throw FatalError(path + " reads " + std::to_string(hits(row, path)));
        }
        if (own != 16)
            throw FatalError(std::to_string(own) + " own paths");
        std::string bad;
        // Byte for byte, with and without a further argument, and a refused one.
        for (const std::string arg : {"", "&window=3", "&window=x"})
            for (const auto& [k, verbs] : RECORD_GETS)
                for (const auto& v : verbs) {
                    const std::string q = arg.empty() ? "" : "?" + arg.substr(1);
                    const Reply r = http(port, "/" + k + "_n2assemble/" + v + q);
                    const Reply c =
                        http(port, "/gnss0_records/" + v + "?chain=" + k.substr(6) + arg);
                    if (!(r == c))
                        bad += " " + k + "/" + v + arg + ": " + std::to_string(r.status) + " "
                               + r.body + " by chain " + std::to_string(c.status) + " " + c.body;
                }
        // Without chain: every row of the verb, under its chain.
        for (const std::string v : {"get_spectrum", "get_elem_cal", "get_lane_inst"}) {
            json want = json::object();
            for (const auto& [k, verbs] : RECORD_GETS)
                if (std::find(verbs.begin(), verbs.end(), v) != verbs.end())
                    want[k.substr(6)] = json::parse(http(port, "/" + k + "_n2assemble/" + v).body);
            const Reply r = http(port, "/gnss0_records/" + v);
            if (r.status != 200 || json::parse(r.body) != want)
                bad += " " + v + " without chain: " + r.body;
        }
        const Reply w = http(port, "/gnss0_records/get_spectrum?window=x");
        if (w.status != 400
            || json::parse(w.body).at("message")
                   != "a: get_spectrum: 'window' must be an integer window index")
            bad += " a refused row without chain: " + w.body;
        // Refusals, each naming the chains that have the verb, and changing nothing.
        const std::vector<Reply> before = record_gets(port);
        const std::vector<std::tuple<std::string, json, std::string>> refusals = {
            {"get_lane_inst?chain=b", json(),
             "/gnss0_records/get_lane_inst has no chain b; its "
             "chains are a"},
            {"get_spectrum?chain=x", json(),
             "/gnss0_records/get_spectrum has no chain x; its "
             "chains are a, b, c"},
            {"set_reference_element", json{{"element", 1}}, "its chains are a, b, c"},
            {"set_reference_element", json{{"chain", "a"}}, "its chains are a, b, c"},
            {"set_reference_element", json{{"chain", "a"}, {"body", {{"element", 1}}}, {"x", 1}},
             "its chains are a, b, c"},
            {"set_reference_element", json{{"chain", 1}, {"body", {{"element", 1}}}},
             "its chains are a, b, c"},
            {"set_reference_element", json{{"chain", "x"}, {"body", {{"element", 1}}}},
             "/gnss0_records/set_reference_element has no chain x; its chains are a, b, c"},
            {"set_elem_gain", json{{"chain", "b"}, {"body", {{"gain", json::array()}}}},
             "/gnss0_records/set_elem_gain has no chain b; its chains are a"},
            {"set_reference_element", json{{"chains", {{"a", {{"element", 1}}}}}},
             "its chains are a, b, c"},
            {"set_reference_element", json::array({1}), "its chains are a, b, c"}};
        for (const auto& [path, body, want] : refusals) {
            const Reply r = http(port, "/gnss0_records/" + path, body);
            if (r.status != 400 || r.body.find(want) == std::string::npos)
                bad +=
                    " " + path + " " + body.dump() + ": " + std::to_string(r.status) + " " + r.body;
        }
        if (record_gets(port) != before)
            bad += " a refusal changed a row";
        // a's get_spectrum: 3 byte for byte, 1 without chain and 2 in record_gets.
        if (hits("/gnss0_a_n2assemble", "/gnss0_a_n2assemble/get_spectrum") != 6.0)
            bad +=
                " a's get_spectrum hits "
                + std::to_string(hits("/gnss0_a_n2assemble", "/gnss0_a_n2assemble/get_spectrum"));
        if (hits("/gnss0_a_n2assemble", "/gnss0_a_n2assemble/set_reference_element") != 0.0)
            bad += " a POST by chain counted";
        delete host;
        for (const auto& p : all)
            if (p.find(" /gnss0_") != std::string::npos
                && kotekan::restServer::instance().has_callback(p.substr(p.find(' ') + 1)))
                bad += " " + p + " left";
        if (hits("/gnss0_a_n2assemble", "/gnss0_a_n2assemble/get_spectrum") != -1.0)
            bad += " the hits outlive the host";
        if (!bad.empty())
            throw FatalError(bad);
    });
    BOOST_CHECK_EQUAL(msg, "");
}

// Every GET of every combine row answers through the host, ?chain=<c>, byte for byte as at the
// row's own path once the rows have folded records in, and without chain with each row's reply
// under its chain; a request for another chain is refused, naming the rows' chains. Only the own
// paths count hits. Deleting the host releases every path and counter.
BOOST_AUTO_TEST_CASE(combine_gets_by_chain_are_the_rows_own) {
    const std::string msg = fatal_of([] {
        kotekan::restServer::instance().start("127.0.0.1", 0);
        const unsigned short port = kotekan::restServer::instance().port();
        json cfg = rec_config(ABC_REC, true, false, false);
        cfg["gnss0_combine"]["record_export"] = 8;
        kotekan::Config config;
        config.update_config(cfg);
        kotekan::bufferContainer bc;
        std::vector<std::unique_ptr<Buffer>> bufs = buffers_of(cfg, bc);
        Buffer* buf = bc.get_buffer("gnss0_rec_buf");
        auto* cmb = new GnssCoherentCombiner(config, "/gnss0_combine", bc);
        buf->register_producer("feeder");
        const gnss::RecDir rec("feeder", ABC_REC, buf);
        cmb->start();
        const int n_rec = 3;
        const size_t stride = gnss::record_stride(1);
        for (int r = 0; r < n_rec; ++r) {
            uint8_t* f = buf->wait_for_empty_frame("feeder", r);
            std::memset(f, 0, buf->frame_size);
            const double utc = 1.7e9 + r * 0.01;
            for (size_t k = 0; k < ABC.size(); ++k) {
                rec.write(f, k, REC_WSTART + r * REC_SAMPLES);
                for (int p = 0; p < ABC_PRN[k]; ++p) {
                    float* x = (float*)(f + ABC_REC[k].offset) + p * stride;
                    x[gnss::REC_PRN] = planted_prn(k, p);
                    x[gnss::REC_P_RE] = 0.5f + (float)(k + p + r);
                    x[gnss::REC_P_ENERGY] = 1.0f;
                    std::memcpy(x + gnss::RECORD_UTC_SLOT, &utc, sizeof(utc));
                }
            }
            buf->mark_frame_full("feeder", r);
        }
        const auto t0 = std::chrono::steady_clock::now();
        bool done = false;
        while (!done && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10)) {
            done = kotekan::restServer::instance().has_callback("/gnss0_combine/chains");
            for (size_t k = 0; k < ABC.size(); ++k)
                done = done && cmb->records_done(k) == (uint64_t)n_rec;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        std::string bad = done ? "" : " the rows did not take their records";
        const std::vector<std::string> verbs = {"get_status", "get_records", "get_elements"};
        for (const auto& v : verbs) {
            json want = json::object();
            for (const auto& k : ABC) {
                const Reply r = http(port, "/" + k + "_n2combine/" + v);
                const Reply c = http(port, "/gnss0_combine/" + v + "?chain=" + k.substr(6));
                if (!(r == c) || r.status != 200 || r.body.size() < 3)
                    bad += " " + k + "/" + v + ": " + r.body + " by chain " + c.body;
                want[k.substr(6)] = json::parse(r.body);
            }
            const Reply all = http(port, "/gnss0_combine/" + v);
            if (all.status != 200 || json::parse(all.body) != want)
                bad += " " + v + " without chain: " + all.body;
        }
        const Reply x = http(port, "/gnss0_combine/get_status?chain=gnss0_a");
        if (x.status != 400
            || json::parse(x.body).at("message")
                   != "/gnss0_combine/get_status has no chain gnss0_a; its chains are a, b, c")
            bad += " a row key as chain: " + x.body;
        const json chains = json::parse(http(port, "/gnss0_combine/chains").body);
        for (size_t k = 0; k < ABC.size(); ++k)
            if (chains[k].value("chain", "") != ABC[k].substr(6))
                bad += " chains " + chains.dump();
        // One own GET of each verb of each row.
        for (const auto& k : ABC)
            for (const auto& v : verbs)
                if (hits("/" + k + "_n2combine", "/" + k + "_n2combine/" + v) != 1.0)
                    bad +=
                        " " + k + "/" + v + " hits "
                        + std::to_string(hits("/" + k + "_n2combine", "/" + k + "_n2combine/" + v));
        cmb->stop();
        for (auto& b : bufs)
            b->send_shutdown_signal();
        cmb->join();
        delete cmb;
        for (const std::string p : {"/gnss0_combine/get_status", "/gnss0_combine/get_records",
                                    "/gnss0_combine/get_elements", "/gnss0_combine/chains",
                                    "/gnss0_a_n2combine/get_status"})
            if (kotekan::restServer::instance().has_callback(p))
                bad += " " + p + " left";
        if (hits("/gnss0_a_n2combine", "/gnss0_a_n2combine/get_status") != -1.0)
            bad += " the hits outlive the host";
        if (!bad.empty())
            throw FatalError(bad);
    });
    BOOST_CHECK_EQUAL(msg, "");
}

// Two clients at once, one at a record row's own paths and one at the host's by chain, POST and
// GET while the row's records run: every request is answered 200, and the own paths count
// exactly the first client's requests.
BOOST_AUTO_TEST_CASE(rest_by_chain_while_the_rows_run) {
    const std::string msg = fatal_of([] {
        kotekan::restServer::instance().start("127.0.0.1", 0);
        const unsigned short port = kotekan::restServer::instance().port();
        json workers = json::array();
        for (const auto& k : ABC)
            workers.push_back({{"rows", {k}}, {"cpu_affinity", {0}}});
        json cfg = record_config(ABC, workers);
        for (const auto& k : ABC)
            cfg["gnss0_records"]["chains"][k]["chain"] = k.substr(6);
        kotekan::Config config;
        config.update_config(cfg);
        kotekan::bufferContainer bc;
        std::vector<std::unique_ptr<Buffer>> bufs = buffers_of(cfg, bc);
        GnssGpuRecordAssemble host(config, "/gnss0_records", bc);
        for (const auto& k : ABC) {
            bc.get_buffer("epl_" + k)->register_producer("feeder");
            bc.get_buffer("rec_" + k)->register_consumer("drain");
        }
        host.start();
        std::vector<std::thread> drains;
        for (const auto& k : ABC)
            drains.emplace_back([b = bc.get_buffer("rec_" + k)] {
                for (int id = 0; b->wait_for_full_frame("drain", id) != nullptr;
                     id = (id + 1) % b->num_frames)
                    b->mark_frame_empty("drain", id);
            });
        const int frames = 100, n_rec = 4, requests = 100;
        std::thread feeder([&] {
            for (int i = 0; i < frames; ++i)
                for (const auto& k : ABC)
                    feed_epl(bc.get_buffer("epl_" + k), i % 2, n_rec);
        });
        const json geom = {{"1", {10.0, 20.0, 0.0, 0.0, 1.7e9}}, {"_bore", {5.0, 1.7e9}}};
        std::atomic<int> refused{0};
        auto client = [&](bool own) {
            const std::string at = own ? "/gnss0_a_n2assemble/" : "/gnss0_records/";
            auto body = [&](const json& b) { return own ? b : json{{"chain", "a"}, {"body", b}}; };
            for (int i = 0; i < requests; ++i) {
                refused += http(port, at + "set_sat_geometry", body(geom)).status != 200;
                refused +=
                    http(port, at + "set_reference_element", body({{"element", i % 2}})).status
                    != 200;
                refused += http(port, at + "get_spectrum").status != 200;
                if (!own)
                    refused += http(port, at + "get_spectrum?chain=a").status != 200;
            }
        };
        std::thread own(client, true), by_chain(client, false);
        own.join();
        by_chain.join();
        feeder.join();
        const auto t0 = std::chrono::steady_clock::now();
        bool done = false;
        while (!done && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10)) {
            done = true;
            for (size_t k = 0; k < ABC.size(); ++k)
                done = done && host.records_done(k) == (uint64_t)(frames * n_rec);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        host.stop();
        for (auto& b : bufs)
            b->send_shutdown_signal();
        host.join();
        for (auto& t : drains)
            t.join();
        std::string bad = done ? "" : " the rows did not take every record";
        if (refused.load() != 0)
            bad += " " + std::to_string(refused.load()) + " requests refused";
        for (const std::string v : {"set_sat_geometry", "set_reference_element", "get_spectrum"})
            if (hits("/gnss0_a_n2assemble", "/gnss0_a_n2assemble/" + v) != requests)
                bad += " " + v + " hits "
                       + std::to_string(hits("/gnss0_a_n2assemble", "/gnss0_a_n2assemble/" + v));
        const std::string err = kotekan::kotekanLogging::get_error_message();
        if (err != "not set")
            bad += " " + err;
        if (!bad.empty())
            throw FatalError(bad);
    });
    BOOST_CHECK_EQUAL(msg, "");
}
