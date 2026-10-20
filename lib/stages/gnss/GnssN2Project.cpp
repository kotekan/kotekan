#include "GnssN2Project.hpp"

#include "N2Util.hpp"            // for frameID
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "Telescope.hpp"         // for Telescope
#include "chordMetadata.hpp"     // for chordMetadata
#include "kotekanLogging.hpp"    // for INFO, WARN, FATAL_ERROR
#include "prometheusMetrics.hpp" // for Metrics

#include <algorithm>
#include <chrono>
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
    stations(config.get<std::vector<int>>(unique_name, "stations")),
    live_freq_ids(config.get<std::vector<int>>(unique_name, "live_freq_ids")),
    tau_s(config.get_default<double>(unique_name, "tau_s", 0.5)),
    k_max(config.get_default<int>(unique_name, "k_max", 3)),
    frac_first_min(config.get_default<double>(unique_name, "frac_first_min", 0.4)),
    lambda_min_rel(config.get_default<double>(unique_name, "lambda_min_rel", 3.0)),
    pr_min(config.get_default<double>(unique_name, "pr_min", 6.0)),
    solve_every(std::max(1, config.get_default<int>(unique_name, "solve_every", 4))),
    metric_period_s(config.get_default<double>(unique_name, "metric_period_s", 1.0)),
    source(config.get_default<std::string>(unique_name, "source", "eigen")),
    use_lanes(source != "eigen"),
    use_residual(source != "lanes"),
    board_group(config.get_default<std::string>(unique_name, "board_group", "")),
    tracker_stations(config.get_default<std::vector<int>>(unique_name, "tracker_stations", {})),
    k_lanes_max(config.get_default<int>(unique_name, "k_lanes_max", 8)),
    lane_max_age_s(config.get_default<double>(unique_name, "lane_max_age_s", 1.0)),
    lane_coh_min(config.get_default<double>(unique_name, "lane_coh_min", 0.08)),
    lane_min_new_frac(config.get_default<double>(unique_name, "lane_min_new_frac", 0.1)),
    lane_conj(config.get_default<bool>(unique_name, "lane_conj", true)) {
    if (source != "eigen" && source != "lanes" && source != "lanes+residual")
        FATAL_ERROR("GnssN2Project: source '{:s}' must be eigen, lanes or lanes+residual", source);

    // The stage always writes the projected copy; mode is read only to refuse another value.
    const std::string mode = config.get_default<std::string>(unique_name, "mode", "live");
    if (mode != "live")
        FATAL_ERROR("GnssN2Project: mode '{:s}' is not supported; it must be 'live'", mode);
    // archive_period_s has no effect; it is read only so a generated config that still carries
    // it does not report it unused.
    config.get_default<double>(unique_name, "archive_period_s", 10.0);
    if (live_freq_ids.empty())
        FATAL_ERROR("GnssN2Project: live_freq_ids is empty; it must list the freq_ids to write");
    if (num_elements % 16 != 0 || n_integrations < 1)
        FATAL_ERROR("num_elements {:d} must be a multiple of 16 and samples_per_data_set / "
                    "sub_integration_ntime ({:d}) >= 1",
                    num_elements, n_integrations);
    layout.n_elem = num_elements;
    std::sort(stations.begin(), stations.end());
    stations.erase(std::unique(stations.begin(), stations.end()), stations.end());
    for (const int s : stations)
        if (s < 0 || s >= num_elements)
            FATAL_ERROR("stations: {:d} is outside [0, {:d})", s, num_elements);
    if (k_max < 1 || k_max > (int)stations.size())
        FATAL_ERROR("k_max {:d} must be in [1, {:d}]", k_max, (int)stations.size());

    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);
    const auto desc = kotekan::GenericNDArray::describe(
        kotekan::int32, "n2k_correlation",
        {n_integrations, num_local_freq, layout.num_blocks(), layout.bs, layout.bs, 2},
        {"Tc", "F", "DPhi", "DPlo1", "DPlo2", "C"}, {sub_integration_ntime, 1, 16, 1, 1, 1});
    in_buf->require_frame_desc(desc);
    out_buf->require_frame_desc(desc);

    const int n = (int)stations.size();
    sub = gnss::ProjSubspace(n, num_local_freq, tau_s, k_max);
    // The projected block's tracker keeps all k_max components (no rank gates) and null_db
    // takes its first spread one: a lone pair the PR gate skipped survives the projection and
    // would otherwise floor the metric.
    nul = gnss::ProjSubspace(n, num_local_freq, tau_s, k_max);
    nul.rel_min = 0.0;
    nul.frac_next_min = 0.0;
    live_ch.assign(num_local_freq, 0);
    k_cur.assign(num_local_freq, 0);
    on.assign(num_local_freq, 0);
    lam0_rel.assign(num_local_freq, 0.0);
    frac0.assign(num_local_freq, 0.0);
    null_db.assign(num_local_freq, 0.0);
    mean_auto.assign(num_local_freq, 0.0);
    pr0.assign(num_local_freq, 0.0);
    q_use.assign((size_t)num_local_freq * k_max, -1);
    lane_k.assign(num_local_freq, 0);
    lane_rq_conj_db.assign(num_local_freq, 0.0);
    lane_rq_raw_db.assign(num_local_freq, 0.0);
    if (use_lanes) {
        if (board_group.empty() || tracker_stations.empty())
            FATAL_ERROR("GnssN2Project: source {:s} needs board_group and tracker_stations", source);
        if (k_lanes_max < 1 || k_lanes_max > n)
            FATAL_ERROR("k_lanes_max {:d} must be in [1, {:d}]", k_lanes_max, n);
        e2a.assign(tracker_stations.size(), -1);
        int n_mapped = 0;
        for (size_t e = 0; e < tracker_stations.size(); ++e) {
            const int st = tracker_stations[e];
            if (st < 0 || st >= num_elements)
                FATAL_ERROR("tracker_stations: {:d} is outside [0, {:d})", st, num_elements);
            const auto it = std::lower_bound(stations.begin(), stations.end(), st);
            if (it != stations.end() && *it == st) {
                e2a[e] = (int)(it - stations.begin());
                ++n_mapped;
            }
        }
        lanes.init(n, k_lanes_max);
        if (n_mapped < n)
            WARN("GnssN2Project: {:d} of the {:d} stations have no tracker element: a lane "
                 "direction is zero there and the projection cannot null those inputs",
                 n - n_mapped, n);
        if (n_mapped < (int)tracker_stations.size())
            WARN("GnssN2Project: {:d} of the {:d} tracker elements are outside the stations block "
                 "and are dropped from every lane direction",
                 (int)tracker_stations.size() - n_mapped, (int)tracker_stations.size());
        INFO("GnssN2Project: source {:s}, board {:s}, {:d} tracker elements mapped onto {:d} "
             "stations, up to {:d} lanes per channel, max age {:.1f} s, coherence >= {:.2f}, "
             "conj {:d}",
             source, board_group, n_mapped, n, k_lanes_max, lane_max_age_s, lane_coh_min,
             (int)lane_conj);
    }
    INFO("GnssN2Project {:s}: {:d} live stations of {:d}, k_max {:d}, tau {:.2f} s, trigger "
         "frac0 >= {:.2f}, lambda0 >= {:.1f} x mean auto, spread >= {:.1f} inputs, solve every "
         "{:d} frames -> {:s}",
         mode, n, num_elements, k_max, tau_s, frac_first_min, lambda_min_rel, pr_min, solve_every,
         out_buf->buffer_name);
}

int GnssN2Project::build_lanes(int f, int freq_id, const cd* V, double now_s, bool diag) {
    const int n = (int)stations.size();
    lanes.clear();
    cand.clear();
    // Copy the fresh, coherent lane entries out under the board's lock, mapped onto the
    // stations axis; everything else happens on the copies. The not-owner argument must
    // be a name no publisher has: an empty one matches every owner's prefix and skips all.
    gnss::ProjBoard::instance().for_each(
        board_group, freq_id, unique_name, now_s, lane_max_age_s, [&](const gnss::ProjEntry& e) {
            if (e.src != 2 || e.k < 1 || e.coh < lane_coh_min || (size_t)e.n != e2a.size()
                || e.q.size() < (size_t)e.n)
                return;
            cand.emplace_back();
            LaneCand& c = cand.back();
            c.power = e.power;
            c.coh = e.coh;
            c.v.assign((size_t)n, cd(0.0, 0.0));
            for (int el = 0; el < e.n; ++el) {
                const int a = e2a[(size_t)el];
                if (a < 0)
                    continue;
                const cd x((double)e.q[(size_t)el].real(), (double)e.q[(size_t)el].imag());
                c.v[(size_t)a] = lane_conj ? std::conj(x) : x;
            }
        });
    if (cand.empty()) {
        if (diag)
            lane_rq_conj_db[f] = lane_rq_raw_db[f] = 0.0;
        return 0;
    }
    std::sort(cand.begin(), cand.end(),
              [](const LaneCand& a, const LaneCand& b) { return a.power > b.power; });
    if (diag) {
        // The strongest lane's q^H V q / mean auto both ways: the right conjugation reads the
        // emitter's INR, the wrong one ~1/n of it.
        const double ma = mean_auto[f] > 0.0 ? mean_auto[f] : 1.0;
        double e = 0.0;
        for (const cd& x : cand[0].v)
            e += std::norm(x);
        if (e > 0.0) {
            double rq[2] = {0.0, 0.0};
            for (int way = 0; way < 2; ++way) {
                cd acc(0.0, 0.0);
                for (int i = 0; i < n; ++i) {
                    const cd qi = way ? std::conj(cand[0].v[(size_t)i]) : cand[0].v[(size_t)i];
                    cd s(0.0, 0.0);
                    for (int c = 0; c < n; ++c) {
                        const cd qc = way ? std::conj(cand[0].v[(size_t)c]) : cand[0].v[(size_t)c];
                        s += V[(size_t)i * n + c] * qc;
                    }
                    acc += std::conj(qi) * s;
                }
                rq[way] = acc.real() / (e * ma);
            }
            // way 0 is the configured convention; the gauges are named by what the vector IS
            const double as_conf = 10.0 * std::log10(std::max(rq[0], 1e-6));
            const double as_other = 10.0 * std::log10(std::max(rq[1], 1e-6));
            lane_rq_conj_db[f] = lane_conj ? as_conf : as_other;
            lane_rq_raw_db[f] = lane_conj ? as_other : as_conf;
        }
    }
    for (const LaneCand& c : cand) {
        if (lanes.k >= k_lanes_max)
            break;
        if (lanes.add(c.v.data(), lane_min_new_frac))
            ++lane_used;
        else
            ++lane_dup;
    }
    return lanes.k;
}

void GnssN2Project::main_thread() {
    frameID in_id(in_buf);
    frameID out_id(out_buf);
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
    auto& g_pr =
        Metrics::instance().add_gauge("kotekan_gnss_n2project_pr0", unique_name, {"freq_id"});
    auto& c_frames =
        Metrics::instance().add_counter("kotekan_gnss_n2project_frames_total", unique_name);
    auto& c_proj = Metrics::instance().add_counter(
        "kotekan_gnss_n2project_projected_channel_frames_total", unique_name);
    auto& g_lane_k =
        Metrics::instance().add_gauge("kotekan_gnss_n2project_lane_k", unique_name, {"freq_id"});
    auto& g_lane_rqc = Metrics::instance().add_gauge("kotekan_gnss_n2project_lane_rq_conj_db",
                                                     unique_name, {"freq_id"});
    auto& g_lane_rqr = Metrics::instance().add_gauge("kotekan_gnss_n2project_lane_rq_raw_db",
                                                     unique_name, {"freq_id"});
    auto& g_lane_used =
        Metrics::instance().add_gauge("kotekan_gnss_n2project_lane_directions_used", unique_name);
    auto& g_lane_dup = Metrics::instance().add_gauge(
        "kotekan_gnss_n2project_lane_directions_duplicate", unique_name);

    const double tick_s = Telescope::instance().seq_length_nsec() * 1e-9;
    int64_t prev_seq = -1;
    double since_metric = 1e9;
    uint64_t frame_count = 0;

    while (!stop_thread) {
        const int32_t* in = (const int32_t*)in_buf->wait_for_full_frame(unique_name, in_id);
        if (in == nullptr)
            break;
        int32_t* out = (int32_t*)out_buf->wait_for_empty_frame(unique_name, out_id);
        if (out == nullptr)
            break;
        std::memcpy(out, in, in_buf->frame_size);
        // Frame spacing from the sequence numbers, so a dropped frame lengthens dt; freq_id
        // labels from the first frame that carries them.
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
        if (!live_ch_resolved) {
            // Absolute ids need the labels, so the first frame must carry them.
            if (!labelled)
                FATAL_ERROR("live_freq_ids needs the coarse frequencies in {:s}'s metadata",
                            in_buf->buffer_name);
            std::vector<int> wanted(live_freq_ids);
            std::sort(wanted.begin(), wanted.end());
            int n_live_ch = 0;
            for (int f = 0; f < num_local_freq; ++f) {
                live_ch[f] = std::binary_search(wanted.begin(), wanted.end(), freq_ids[f]) ? 1 : 0;
                n_live_ch += live_ch[f];
            }
            live_ch_resolved = true;
            INFO("GnssN2Project: live writes restricted to {:d} of {:d} channels", n_live_ch,
                 num_local_freq);
            if (n_live_ch == 0)
                WARN("GnssN2Project: none of the {:d} live_freq_ids is on this GPU: it writes "
                     "nothing here",
                     (int)live_freq_ids.size());
        }
        const double dt_int = dt / n_integrations;
        const bool solve_now = (frame_count % (uint64_t)solve_every) == 0;
        const double now_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
                .count();

        for (int t = 0; t < n_integrations; ++t) {
            for (int f = 0; f < num_local_freq; ++f) {
                const int32_t* fin = in + (size_t)t * per_int + (size_t)f * per_freq;
                gnss_n2proj::extract(layout, fin, stations, V.data());
                mean_auto[f] = gnss_n2proj::mean_auto(n, V.data());
                // #190: the known directions first, on the live channels (the only ones a
                // chain publishes for); what follows sees the residual.
                int kl = 0;
                if (use_lanes && live_ch[f]) {
                    kl = build_lanes(f, freq_ids[f], V.data(), now_s, solve_now);
                    if (kl > 0)
                        gnss_n2proj::project(n, kl, lanes.q.data(), V.data(), W, M);
                }
                lane_k[f] = kl;
                if (!use_residual) {
                    k_cur[f] = 0;
                    frac0[f] = lam0_rel[f] = pr0[f] = 0.0;
                }
                if (use_residual)
                    sub.push_cov(f, V.data(), dt_int);
                if (use_residual && solve_now) {
                    const int ks = sub.solve(f, 2);
                    // Components spread over the array, in energy order; a lone correlated
                    // pair (PR 2) is skipped and the next component judged.
                    int* use = &q_use[(size_t)f * k_max];
                    int nuse = 0;
                    int first = -1;
                    for (int j = 0; j < ks; ++j) {
                        const cd* qj = sub.q(f, j);
                        double s4 = 0.0;
                        for (int i = 0; i < n; ++i)
                            s4 += std::norm(qj[i]) * std::norm(qj[i]);
                        const double pr = s4 > 0.0 ? 1.0 / s4 : 0.0;
                        if (j == 0)
                            pr0[f] = pr;
                        if (pr >= pr_min) {
                            if (first < 0)
                                first = j;
                            use[nuse++] = j;
                        }
                    }
                    for (int j = nuse; j < k_max; ++j)
                        use[j] = -1;
                    // The trigger is judged on the first spread component: its share of the
                    // off-diagonal energy left after the ones before it, and its eigenvalue.
                    frac0[f] = first >= 0 ? sub.frac(f, first) : 0.0;
                    const double lam0 = first >= 0 ? sub.lambda(f, first) : 0.0;
                    lam0_rel[f] = mean_auto[f] > 0.0 ? lam0 / mean_auto[f] : 0.0;
                    // Hysteresis on the trigger so a threshold-grazing emitter does not
                    // flicker the rank frame by frame.
                    const double fmin = on[f] ? 0.8 * frac_first_min : frac_first_min;
                    on[f] = (first >= 0 && sub.warm(f) && frac0[f] >= fmin
                             && lam0_rel[f] >= lambda_min_rel)
                                ? 1
                                : 0;
                    k_cur[f] = on[f] ? std::min(nuse, k_max) : 0;
                }
                const int k = k_cur[f];
                if (k > 0) {
                    const int* use = &q_use[(size_t)f * k_max];
                    for (int j = 0; j < k; ++j)
                        std::copy(sub.q(f, use[j]), sub.q(f, use[j]) + n,
                                  q.begin() + (size_t)j * n);
                    gnss_n2proj::project(n, k, q.data(), V.data(), W, M);
                }
                if (k > 0 || kl > 0) {
                    nul.push_cov(f, V.data(), dt_int);
                    if (solve_now) {
                        const int kn = nul.solve(f, 2);
                        double l1 = 0.0;
                        for (int j = 0; j < kn; ++j) {
                            const cd* qj = nul.q(f, j);
                            double s4 = 0.0;
                            for (int i = 0; i < n; ++i)
                                s4 += std::norm(qj[i]) * std::norm(qj[i]);
                            if (s4 > 0.0 && 1.0 / s4 >= pr_min) {
                                l1 = std::fabs(nul.lambda(f, j));
                                break;
                            }
                        }
                        // Reference: the first eigen component projected, else the strongest
                        // lane's Rayleigh quotient on the raw block (lane_rq, in mean autos).
                        const double l0 =
                            k > 0 ? sub.lambda(f, q_use[(size_t)f * k_max])
                                  : mean_auto[f]
                                        * std::pow(10.0, (lane_conj ? lane_rq_conj_db[f]
                                                                    : lane_rq_raw_db[f])
                                                             / 10.0);
                        null_db[f] =
                            (l0 > 0.0) ? 10.0 * std::log10(std::max(l1, 1e-12 * l0) / l0) : 0.0;
                    }
                    if (live_ch[f])
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
        if (since_metric >= metric_period_s) {
            since_metric = 0.0;
            for (int f = 0; f < num_local_freq; ++f) {
                const std::string fid = std::to_string(freq_ids[f]);
                g_k.labels({fid}).set(k_cur[f]);
                g_lam.labels({fid}).set(lam0_rel[f]);
                g_frac.labels({fid}).set(frac0[f]);
                g_null.labels({fid}).set(null_db[f]);
                g_pr.labels({fid}).set(pr0[f]);
                if (use_lanes && live_ch[f]) {
                    g_lane_k.labels({fid}).set(lane_k[f]);
                    g_lane_rqc.labels({fid}).set(lane_rq_conj_db[f]);
                    g_lane_rqr.labels({fid}).set(lane_rq_raw_db[f]);
                }
            }
            if (use_lanes) {
                g_lane_used.set((double)lane_used);
                g_lane_dup.set((double)lane_dup);
            }
        }
        out_buf->allocate_new_metadata_object(out_id);
        in_buf->copy_metadata(in_id, out_buf, out_id);
        out_buf->mark_frame_full(unique_name, out_id++);
        in_buf->mark_frame_empty(unique_name, in_id++);
    }
    if (use_lanes)
        INFO("GnssN2Project {:s}: {:d} frames; lane directions used {:d}, duplicates {:d}",
             source, (int)frame_count, (int)lane_used, (int)lane_dup);
}
