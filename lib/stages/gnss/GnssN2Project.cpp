#include "GnssN2Project.hpp"

#include "N2Util.hpp"            // for frameID
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "Telescope.hpp"         // for Telescope
#include "chordMetadata.hpp"     // for chordMetadata
#include "kotekanLogging.hpp"    // for INFO, WARN, FATAL_ERROR
#include "prometheusMetrics.hpp" // for Metrics

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;
using N2::frameID;

REGISTER_KOTEKAN_STAGE(GnssN2Project);

GnssN2Project::GnssN2Project(Config& config, const std::string& unique_name,
                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&GnssN2Project::main_thread, this)),
    num_elements(config.get<int>(unique_name, "num_elements")),
    num_local_freq(config.get<int>(unique_name, "num_local_freq")),
    samples_per_data_set(config.get<int>(unique_name, "samples_per_data_set")),
    sub_integration_ntime(config.get<int>(unique_name, "sub_integration_ntime")),
    n_integrations(samples_per_data_set / sub_integration_ntime),
    stations(config.get_default<std::vector<int>>(unique_name, "stations", std::vector<int>{})),
    mode(config.get_default<std::string>(unique_name, "mode", "shadow")), live(mode == "live"),
    tau_s(config.get_default<double>(unique_name, "tau_s", 0.5)),
    k_max(config.get_default<int>(unique_name, "k_max", 3)),
    frac_first_min(config.get_default<double>(unique_name, "frac_first_min", 0.4)),
    lambda_min_rel(config.get_default<double>(unique_name, "lambda_min_rel", 3.0)),
    rel_min(config.get_default<double>(unique_name, "rel_min", 0.2)),
    frac_next_min(config.get_default<double>(unique_name, "frac_next_min", 0.3)),
    solve_every(std::max(1, config.get_default<int>(unique_name, "solve_every", 4))),
    metric_period_s(config.get_default<double>(unique_name, "metric_period_s", 1.0)),
    archive_path(config.get_default<std::string>(unique_name, "archive_path", "")),
    archive_period_s(config.get_default<double>(unique_name, "archive_period_s", 10.0)) {

    if (mode != "shadow" && mode != "live")
        FATAL_ERROR("mode must be shadow or live, not {:s}", mode);
    if (num_elements % 16 != 0 || n_integrations < 1)
        FATAL_ERROR("num_elements {:d} must be a multiple of 16 and samples_per_data_set / "
                    "sub_integration_ntime ({:d}) >= 1",
                    num_elements, n_integrations);
    layout.n_elem = num_elements;
    if (stations.empty())
        for (int i = 0; i < num_elements; ++i)
            stations.push_back(i);
    std::sort(stations.begin(), stations.end());
    stations.erase(std::unique(stations.begin(), stations.end()), stations.end());
    for (const int s : stations)
        if (s < 0 || s >= num_elements)
            FATAL_ERROR("stations: {:d} is outside [0, {:d})", s, num_elements);
    if (k_max < 1 || k_max > (int)stations.size())
        FATAL_ERROR("k_max {:d} must be in [1, {:d}]", k_max, (int)stations.size());

    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    const std::string out_name = config.get_default<std::string>(unique_name, "out_buf", "");
    if (!out_name.empty()) {
        out_buf = buffer_container.get_buffer(out_name);
        out_buf->register_producer(unique_name);
    }
    if (live && !out_buf)
        FATAL_ERROR("mode live needs an out_buf (N2Accumulate reads the projected copy)");
    const auto desc = kotekan::GenericNDArray::describe(
        kotekan::int32, "n2k_correlation",
        {n_integrations, num_local_freq, layout.num_blocks(), layout.bs, layout.bs, 2},
        {"Tc", "F", "DPhi", "DPlo1", "DPlo2", "C"}, {sub_integration_ntime, 1, 16, 1, 1, 1});
    in_buf->require_frame_desc(desc);
    if (out_buf)
        out_buf->require_frame_desc(desc);

    const int n = (int)stations.size();
    sub = gnss::ProjSubspace(n, num_local_freq, tau_s, k_max);
    sub.rel_min = rel_min;
    sub.frac_next_min = frac_next_min;
    nul = gnss::ProjSubspace(n, num_local_freq, tau_s, 1);
    k_cur.assign(num_local_freq, 0);
    on.assign(num_local_freq, 0);
    lam0_rel.assign(num_local_freq, 0.0);
    frac0.assign(num_local_freq, 0.0);
    null_db.assign(num_local_freq, 0.0);
    mean_auto.assign(num_local_freq, 0.0);
    if (!archive_path.empty()) {
        archive.open(archive_path, std::ios::app);
        if (!archive)
            WARN("GnssN2Project: cannot open archive {:s}; archiving off", archive_path);
    }
    INFO("GnssN2Project {:s}: {:d} live stations of {:d}, k_max {:d}, tau {:.2f} s, trigger "
         "frac0 >= {:.2f} and lambda0 >= {:.1f} x mean auto, solve every {:d} frames{:s}",
         mode, n, num_elements, k_max, tau_s, frac_first_min, lambda_min_rel, solve_every,
         out_buf ? " -> " + out_name : "");
}

void GnssN2Project::main_thread() {
    frameID in_id(in_buf);
    frameID out_id(out_buf ? out_buf : in_buf);
    const int n = (int)stations.size();
    const size_t per_freq = layout.per_freq();
    const size_t per_int = per_freq * (size_t)num_local_freq; // words per integration
    std::vector<cd> V((size_t)n * n), W, M;
    std::vector<cd> q((size_t)k_max * n);
    std::vector<int> freq_ids(num_local_freq);
    for (int f = 0; f < num_local_freq; ++f)
        freq_ids[f] = f;
    bool labelled = false;

    auto& g_k = Metrics::instance().add_gauge("kotekan_gnss_n2project_k", unique_name, {"freq_id"});
    auto& g_lam = Metrics::instance().add_gauge("kotekan_gnss_n2project_lambda0_rel", unique_name,
                                                {"freq_id"});
    auto& g_frac =
        Metrics::instance().add_gauge("kotekan_gnss_n2project_frac0", unique_name, {"freq_id"});
    auto& g_null =
        Metrics::instance().add_gauge("kotekan_gnss_n2project_null_db", unique_name, {"freq_id"});
    auto& c_frames =
        Metrics::instance().add_counter("kotekan_gnss_n2project_frames_total", unique_name);
    auto& c_proj = Metrics::instance().add_counter(
        "kotekan_gnss_n2project_projected_channel_frames_total", unique_name);

    const double tick_s = Telescope::instance().seq_length_nsec() * 1e-9;
    int64_t prev_seq = -1;
    double since_metric = 1e9, since_archive = 1e9;
    uint64_t frame_count = 0;

    while (!stop_thread) {
        const int32_t* in = (const int32_t*)in_buf->wait_for_full_frame(unique_name, in_id);
        if (in == nullptr)
            break;
        int32_t* out = nullptr;
        if (out_buf) {
            out = (int32_t*)out_buf->wait_for_empty_frame(unique_name, out_id);
            if (out == nullptr)
                break;
            std::memcpy(out, in, in_buf->frame_size);
        }
        // Frame spacing from the sequence numbers (robust to dropped frames); the labels once.
        double dt = samples_per_data_set * tick_s;
        const auto meta = std::dynamic_pointer_cast<chordMetadata>(in_buf->get_metadata(in_id));
        if (meta) {
            const int64_t seq = meta->get_fpga_seq_num();
            if (prev_seq >= 0 && seq > prev_seq)
                dt = (double)(seq - prev_seq) * tick_s;
            prev_seq = seq;
            if (!labelled && meta->has_coarse_freq()) {
                const std::vector<int> cf = meta->get_coarse_freq();
                if ((int)cf.size() == num_local_freq)
                    freq_ids = cf;
                labelled = true;
            }
        }
        const double dt_int = dt / n_integrations;
        const bool solve_now = (frame_count % (uint64_t)solve_every) == 0;

        for (int t = 0; t < n_integrations; ++t) {
            for (int f = 0; f < num_local_freq; ++f) {
                const int32_t* fin = in + (size_t)t * per_int + (size_t)f * per_freq;
                gnss_n2proj::extract(layout, fin, stations, V.data());
                mean_auto[f] = gnss_n2proj::mean_auto(n, V.data());
                sub.push_cov(f, V.data(), dt_int);
                if (solve_now) {
                    const int ks = sub.solve(f, 2);
                    frac0[f] = sub.frac(f, 0);
                    const double lam0 = sub.lambda(f, 0);
                    lam0_rel[f] = mean_auto[f] > 0.0 ? lam0 / mean_auto[f] : 0.0;
                    // Hysteresis on the trigger so a threshold-grazing emitter does not
                    // flicker the rank frame by frame.
                    const double fmin = on[f] ? 0.8 * frac_first_min : frac_first_min;
                    on[f] =
                        (sub.warm(f) && frac0[f] >= fmin && lam0_rel[f] >= lambda_min_rel) ? 1 : 0;
                    k_cur[f] = on[f] ? std::min(ks, k_max) : 0;
                }
                const int k = k_cur[f];
                if (k > 0) {
                    for (int j = 0; j < k; ++j)
                        std::copy(sub.q(f, j), sub.q(f, j) + n, q.begin() + (size_t)j * n);
                    gnss_n2proj::project(n, k, q.data(), V.data(), W, M);
                    nul.push_cov(f, V.data(), dt_int);
                    if (solve_now) {
                        nul.solve(f, 2);
                        const double l0 = sub.lambda(f, 0), l1 = nul.lambda(f, 0);
                        null_db[f] =
                            (l0 > 0.0) ? 10.0 * std::log10(std::max(l1, 1e-12 * l0) / l0) : 0.0;
                    }
                    if (live)
                        gnss_n2proj::writeback(layout,
                                               out + (size_t)t * per_int + (size_t)f * per_freq,
                                               stations, V.data());
                    c_proj.inc();
                } else {
                    null_db[f] = 0.0;
                }
            }
        }
        ++frame_count;
        c_frames.inc();
        since_metric += dt;
        since_archive += dt;
        if (since_metric >= metric_period_s) {
            since_metric = 0.0;
            for (int f = 0; f < num_local_freq; ++f) {
                const std::string fid = std::to_string(freq_ids[f]);
                g_k.labels({fid}).set(k_cur[f]);
                g_lam.labels({fid}).set(lam0_rel[f]);
                g_frac.labels({fid}).set(frac0[f]);
                g_null.labels({fid}).set(null_db[f]);
            }
        }
        if (archive && since_archive >= archive_period_s) {
            since_archive = 0.0;
            const int64_t t_ns =
                (meta && prev_seq >= 0) ? Telescope::instance().to_time_ns(prev_seq) : 0;
            archive << "{\"t_ns\":" << t_ns << ",\"mode\":\"" << mode << "\",\"freq_id\":[";
            for (int f = 0; f < num_local_freq; ++f)
                archive << (f ? "," : "") << freq_ids[f];
            archive << "],\"k\":[";
            for (int f = 0; f < num_local_freq; ++f)
                archive << (f ? "," : "") << k_cur[f];
            archive << "],\"lam0_rel\":[";
            for (int f = 0; f < num_local_freq; ++f)
                archive << (f ? "," : "") << (float)lam0_rel[f];
            archive << "],\"frac0\":[";
            for (int f = 0; f < num_local_freq; ++f)
                archive << (f ? "," : "") << (float)frac0[f];
            archive << "],\"null_db\":[";
            for (int f = 0; f < num_local_freq; ++f)
                archive << (f ? "," : "") << (float)null_db[f];
            archive << "]}\n";
            archive.flush();
        }
        if (out_buf) {
            out_buf->allocate_new_metadata_object(out_id);
            in_buf->copy_metadata(in_id, out_buf, out_id);
            out_buf->mark_frame_full(unique_name, out_id++);
        }
        in_buf->mark_frame_empty(unique_name, in_id++);
    }
}
