#include "cudaGnssChordTrack.hpp"

#include "cudaGnssChordDespread.hpp"
#include "gnssBandPlan.hpp"
#include "gnssGpuChain.hpp"
#include "gnssSignal.hpp"
#include "GnssChanMetadata.hpp"
#include "cudaUtils.hpp"
#include "kotekanLogging.hpp"
#include "pfbPrototype.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>

using kotekan::Config;
using kotekan::bufferContainer;

REGISTER_CUDA_COMMAND_WITH_STATE(cudaGnssChordTrack, cudaGnssChordTrackState);

// ---------------------------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------------------------
cudaGnssChordTrackState::cudaGnssChordTrackState(Config& config, const std::string& unique_name,
                                                 bufferContainer& host_buffers,
                                                 cudaDeviceInterface& device) :
    cudaCommandState(config, unique_name, host_buffers, device) {
    using namespace std::placeholders;

    prns = config.get<std::vector<int>>(unique_name, "prns");
    n_prn = (int)prns.size();
    n_chan = config.get<int>(unique_name, "n_channels");
    n_elem = config.get<int>(unique_name, "n_elements");
    elem_stride = config.get_default<int>(unique_name, "elem_stride", n_elem);
    frame_chan_stride = config.get_default<int>(unique_name, "frame_chan_stride", n_chan);
    n_hops_frame = config.get<int>(unique_name, "samples_per_data_set");
    hops_per_record = config.get_default<int>(unique_name, "hops_per_record", 2048);
    fft_len = config.get_default<int>(unique_name, "fft_length", 16384);
    sample_rate = config.get_default<double>(unique_name, "sample_rate", 3.2e9);
    f_offset_hz = config.get_default<double>(unique_name, "f_offset_hz", 0.0);
    // F-engine conjugation, measured on sky 2026-07-30 -- see DespreadParams::conj_data.
    // Without it the tracker despreads the conjugate of the sky and never locks.
    _conjugate = config.get_default<bool>(unique_name, "conjugate", false);
    dll_spacing = config.get_default<double>(unique_name, "dll_spacing", 0.5);
    doppler_margin_hz = config.get_default<double>(unique_name, "doppler_margin_hz", 5000.0);
    trim_enable = config.get_default<bool>(unique_name, "code_trim", false);
    trim_gain = config.get_default<double>(unique_name, "trim_gain", 0.15);
    trim_leak = config.get_default<double>(unique_name, "trim_leak", 0.002);
    trim_clamp = config.get_default<double>(unique_name, "trim_clamp", 3.0);
    trim_quality_min = config.get_default<double>(unique_name, "trim_quality_min", 1.8);
    trim_ref_elem = config.get_default<int>(unique_name, "trim_ref_elem", 0);

    if (hops_per_record <= 0 || n_hops_frame % hops_per_record != 0)
        FATAL_ERROR("cudaGnssChordTrack: hops_per_record {:d} must divide samples_per_data_set "
                    "{:d} exactly -- a partial trailing record would silently drop data.",
                    hops_per_record, n_hops_frame);
    const int n_rec = n_hops_frame / hops_per_record;
    if (n_rec > gnss_gpu::MAX_REC)
        FATAL_ERROR("cudaGnssChordTrack: {:d} records/frame exceeds MAX_REC {:d}; raise "
                    "hops_per_record.",
                    n_rec, gnss_gpu::MAX_REC);

    const std::string signame = config.get<std::string>(unique_name, "signal");
    const gnss::SignalDescriptor* sig = gnss::signal_by_name(signame);
    if (!sig)
        FATAL_ERROR("cudaGnssChordTrack: unknown signal '{:s}'", signame);

    // The tap already selected the covering channels, so locally they are 0..n_chan-1 and every
    // one of them is covered. (The band plan decided WHICH sky channels those are; see
    // config/chord_band_plan.py, which mirrors gnssBandPlan.cpp.)
    covering.resize((size_t)n_chan);
    for (int c = 0; c < n_chan; ++c)
        covering[(size_t)c] = c;

    // The replica bank must be built with the F-ENGINE's exact PFB, or the channelized replica
    // does not match the data: CHORD is a 4-tap Hamming critically-sampled bank over 8192
    // positive-frequency bins (arXiv:2607.01625 s2.2.2, recorded in chord_gnss_node.yaml).
    const int N = fft_len / 2;
    replica = std::make_unique<gnss::ChannelizedReplicaBank>(
        *sig, sample_rate, f_offset_hz, N, /*num_taps=*/4, dsp::window_from_string("hamming"),
        prns);
    despread = std::make_unique<GnssCudaDespread>(*replica, n_prn, n_chan, /*chan_offset=*/0,
                                                 hops_per_record, sample_rate, f_offset_hz);

    seeds.assign((size_t)n_prn, Seed{});
    trim.assign((size_t)n_prn, 0.0);
    trim_disc.assign((size_t)n_prn, 0.0);
    trim_q.assign((size_t)n_prn, 0.0);
    trim_n.assign((size_t)n_prn, 0);

    const std::string ep =
        config.get_default<std::string>(unique_name, "seed_endpoint", "/chord_track/set_seeds");
    kotekan::restServer::instance().register_post_callback(
        ep, std::bind(&cudaGnssChordTrackState::set_seeds_callback, this, _1, _2));
    const std::string tep =
        config.get_default<std::string>(unique_name, "trim_endpoint", "/chord_track/get_trim");
    kotekan::restServer::instance().register_get_callback(
        tep, std::bind(&cudaGnssChordTrackState::get_trim_callback, this, _1));
    INFO_NON_OO("cudaGnssChordTrack: {:d} PRN x {:d} chan x {:d} elem, {:d} hops/record "
                "({:d} rec/frame), seeds on {:s}",
                n_prn, n_chan, n_elem, hops_per_record, n_rec, ep);
}

void cudaGnssChordTrackState::set_seeds_callback(kotekan::connectionInstance& conn,
                                                 nlohmann::json& request) {
    // Parse first, lock last -- the execute path takes seed_mtx every GPU frame, so holding it
    // across a parse would stall the tracker directly (the same lesson cudaGnssTrack learned).
    std::vector<std::pair<int, Seed>> upd;
    try {
        upd.reserve(request.size());
        for (const auto& s : request) {
            const int prn = s.at("prn").get<int>();
            for (int i = 0; i < n_prn; ++i)
                if (prns[i] == prn) {
                    Seed sd;
                    sd.have = true;
                    sd.doppler_hz = s.at("doppler_hz").get<double>();
                    sd.cp_chips = s.at("code_phase_chips").get<double>();
                    sd.cp_rate = s.value("code_phase_rate", 0.0);
                    sd.dop_rate = s.value("doppler_rate_hz_s", 0.0);
                    sd.ctrim_hz = s.value("carrier_trim_hz", 0.0);
                    sd.ref_hop = s.value("ref_hop", (long long)0);
                    upd.emplace_back(i, sd);
                    break;
                }
        }
    } catch (const std::exception& e) {
        conn.send_error(std::string("bad seed payload: ") + e.what(),
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    {
        std::lock_guard<std::mutex> lk(seed_mtx);
        for (auto& u : upd)
            seeds[(size_t)u.first] = u.second;
    }
    conn.send_empty_reply(kotekan::HTTP_RESPONSE::OK);
}

cudaGnssChordTrackState::~cudaGnssChordTrackState() {
    for (auto& s : trim_slots) {
        if (s.ev)
            cudaEventDestroy(s.ev);
        if (s.host)
            cudaFreeHost(s.host);
    }
}

void cudaGnssChordTrackState::get_trim_callback(kotekan::connectionInstance& conn) {
    nlohmann::json out = nlohmann::json::array();
    {
        std::lock_guard<std::mutex> lk(trim_mtx);
        for (int p = 0; p < n_prn; ++p)
            out.push_back({{"prn", prns[(size_t)p]},
                           {"trim_chips", trim[(size_t)p]},
                           {"disc", trim_disc[(size_t)p]},
                           {"quality", trim_q[(size_t)p]},
                           {"updates", trim_n[(size_t)p]}});
    }
    nlohmann::json reply = {{"enabled", trim_enable}, {"trims", out}};
    conn.send_json_reply(reply);
}

// ---------------------------------------------------------------------------------------------
// Command
// ---------------------------------------------------------------------------------------------
cudaGnssChordTrack::cudaGnssChordTrack(Config& config, const std::string& unique_name,
                                       bufferContainer& host_buffers, cudaDeviceInterface& device,
                                       int instance_num,
                                       std::shared_ptr<cudaCommandState> state) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, state,
                "cudaGnssChordTrack", "cudaGnssChordTrack") {
    _gpu_mem_input = config.get<std::string>(unique_name, "gpu_mem_input");
    _gpu_mem_output = config.get<std::string>(unique_name, "gpu_mem_output");
    _mem_jobs = unique_name + "_jobs";
    _mem_wave = unique_name + "_wave";
    _mem_scale = unique_name + "_scale";
    _mem_chanids = unique_name + "_chanids";

    cudaGnssChordTrackState& S = *st();
    _in_frame_len = (size_t)S.n_hops_frame * S.frame_chan_stride * S.elem_stride;
    _out_frame_len = gnss_gpu::frame_bytes(S.n_prn, S.n_chan, gnss_gpu::ROWS_PLAIN, S.n_elem);
    _ctl_stage.resize(gnss_gpu::off_corr(S.n_prn));

    if (S.trim_ref_elem < 0 || S.trim_ref_elem >= S.n_elem)
        FATAL_ERROR("cudaGnssChordTrack: trim_ref_elem {:d} outside [0, {:d})", S.trim_ref_elem,
                    S.n_elem);

    // Trim readback slots: one per GPU frame slot, sized for the worst case (every PRN active
    // in every record). Lazily created by whichever command instance gets here first.
    if (S.trim_enable) {
        std::lock_guard<std::mutex> lk(S.trim_slot_mtx);
        if (S.trim_slots.empty()) {
            const int n_rec = S.n_hops_frame / S.hops_per_record;
            const size_t max_rows = (size_t)gnss_gpu::ROWS_PLAIN * S.n_prn * n_rec;
            S.trim_slots.resize((size_t)_gpu_buffer_depth);
            for (auto& sl : S.trim_slots) {
                CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&sl.ev, cudaEventDisableTiming));
                CHECK_CUDA_ERROR(cudaHostAlloc(
                    &sl.host, max_rows * S.n_chan * 2 * sizeof(double), cudaHostAllocDefault));
                sl.job0.assign((size_t)n_rec * S.n_prn, -1);
            }
            INFO("code trim ON: gain {:.3f} leak {:.4f} clamp {:.1f} q_min {:.2f} ref_elem {:d}",
                 S.trim_gain, S.trim_leak, S.trim_clamp, S.trim_quality_min, S.trim_ref_elem);
        }
    }

    set_command_type(gpuCommandType::KERNEL);
}

cudaGnssChordTrackState* cudaGnssChordTrack::st() {
    return static_cast<cudaGnssChordTrackState*>(command_state.get());
}

cudaEvent_t cudaGnssChordTrack::execute(cudaPipelineState& pipestate,
                                        const std::vector<cudaEvent_t>& pre_events) {
    (void)pre_events;
    pre_execute();
    cudaGnssChordTrackState& S = *st();
    const cudaStream_t stream = device.getStream(cuda_stream_id);
    const int n_rec = S.n_hops_frame / S.hops_per_record;

    const char* d_frame = (const char*)device.get_gpu_memory_array(
        _gpu_mem_input, pipestate.gpu_frame_id, _gpu_buffer_depth, _in_frame_len);
    char* d_out = (char*)device.get_gpu_memory_array(_gpu_mem_output, pipestate.gpu_frame_id,
                                                     _gpu_buffer_depth, _out_frame_len);
    auto* d_jobs = (gnss_cuda::DespreadJob*)device.get_gpu_memory_array(
        _mem_jobs, pipestate.gpu_frame_id, _gpu_buffer_depth,
        (size_t)gnss_gpu::max_specs(S.n_prn) * sizeof(gnss_cuda::DespreadJob));
    // Replica scratch: [3*specs][n_chan][hops_per_record]. Never leaves the GPU. This is the
    // memory the split buys its reuse with -- generated once per record, read n_elem times.
    auto* d_wave = (float2*)device.get_gpu_memory(
        _mem_wave, (size_t)3 * gnss_gpu::max_specs(S.n_prn) * S.n_chan * S.hops_per_record
                       * sizeof(float2));
    auto* d_scale = (float*)device.get_gpu_memory(_mem_scale, (size_t)S.n_chan * sizeof(float));
    auto* d_chanids = (int*)device.get_gpu_memory(_mem_chanids, (size_t)S.n_chan * sizeof(int));

    // Static per-channel tables. The tap already selected the covering channels, so within the
    // frame they are dense 0..n_chan-1. The scale is unity: CHORD's per-bin digital gain is
    // applied in the F-engine and is part of the sky-to-bits chain we solve for end to end, so
    // there is nothing to undo here (see chord_gnss_node.yaml `gains`).
    if (!_uploaded_static) {
        std::vector<float> h_scale((size_t)S.n_chan, 1.0f);
        std::vector<int> h_ids((size_t)S.n_chan);
        for (int c = 0; c < S.n_chan; ++c)
            h_ids[(size_t)c] = c;
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_scale, h_scale.data(), h_scale.size() * sizeof(float),
                                         cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_chanids, h_ids.data(), h_ids.size() * sizeof(int),
                                         cudaMemcpyHostToDevice, stream));
        _uploaded_static = true;
    }

    // Absolute sample of this frame's first hop, stamped by GnssChordVoltageTap.
    auto meta = std::dynamic_pointer_cast<GnssChanMetadata>(
        device.get_gpu_memory_array_metadata(_gpu_mem_input, pipestate.gpu_frame_id));
    const long long seq0 = (meta && meta->sample_seq >= 0) ? meta->sample_seq : -1;

    std::memset(_ctl_stage.data(), 0, _ctl_stage.size());
    auto* hdr = (gnss_gpu::FrameHdr*)_ctl_stage.data();
    auto* winstart = (int64_t*)(_ctl_stage.data() + gnss_gpu::off_winstart());
    auto* pctl = (gnss_gpu::PrnCtl*)(_ctl_stage.data() + gnss_gpu::off_prnctl());

    hdr->n_rec = n_rec;
    hdr->n_prn = S.n_prn;
    hdr->n_chan = S.n_chan;
    hdr->n_rows_spec = gnss_gpu::ROWS_PLAIN;
    hdr->seq0 = seq0;
    hdr->utc0 = 0.0;

    // DLL trim, step 1: fold in every completed E/P/L readback (enqueued by earlier frames'
    // executes; cudaEventQuery keeps this non-blocking, so an unfinished frame just waits its
    // turn). Coherent channel sum at the reference element -- exactly the assembler's header
    // observable -- then the broker's 3c math per PRN: disc = (E^2-L^2)/(E^2+L^2) ~ -4*tau at
    // 0.5-chip spacing, leaky integrator, clamp. Gated on q = 2P^2/(E^2+L^2) (~1 noise, ~4 on
    // the peak) so an unlocked PRN's noise cannot walk its trim.
    if (S.trim_enable) {
        std::lock_guard<std::mutex> slk(S.trim_slot_mtx);
        const int n_rec_s = S.n_hops_frame / S.hops_per_record;
        for (auto& sl : S.trim_slots) {
            if (!sl.pending || cudaEventQuery(sl.ev) != cudaSuccess)
                continue;
            sl.pending = false;
            std::lock_guard<std::mutex> tlk(S.trim_mtx);
            for (int p = 0; p < S.n_prn; ++p) {
                double e2 = 0.0, p2 = 0.0, l2 = 0.0;
                int n_meas = 0;
                for (int r = 0; r < n_rec_s; ++r) {
                    const int j0 = sl.job0[(size_t)r * S.n_prn + p];
                    if (j0 < 0)
                        continue;
                    double pow3[3];
                    for (int t = 0; t < 3; ++t) { // rows j0+0..2 = E, P, L
                        double sr = 0.0, si = 0.0;
                        const double* row = sl.host + (size_t)(j0 + t) * S.n_chan * 2;
                        for (int ch = 0; ch < S.n_chan; ++ch) {
                            sr += row[2 * ch];
                            si += row[2 * ch + 1];
                        }
                        pow3[t] = sr * sr + si * si;
                    }
                    e2 += pow3[0];
                    p2 += pow3[1];
                    l2 += pow3[2];
                    ++n_meas;
                }
                if (n_meas == 0 || e2 + l2 <= 0.0)
                    continue;
                const double q = 2.0 * p2 / (e2 + l2);
                S.trim_q[(size_t)p] = q;
                if (q < S.trim_quality_min)
                    continue;
                const double disc = (e2 - l2) / (e2 + l2);
                const double tau =
                    -std::max(-1.0, std::min(1.0, disc)) / 4.0 * (S.dll_spacing / 0.5);
                const double t_new = (1.0 - S.trim_leak) * S.trim[(size_t)p] + S.trim_gain * tau;
                S.trim[(size_t)p] = std::max(-S.trim_clamp, std::min(S.trim_clamp, t_new));
                S.trim_disc[(size_t)p] = disc;
                ++S.trim_n[(size_t)p];
            }
        }
        // Heartbeat: one line every ~256 frames (~10 s) showing whatever is actively trimming.
        if ((++S.trim_frames & 0xFF) == 0) {
            std::string line;
            std::lock_guard<std::mutex> tlk(S.trim_mtx);
            for (int p = 0; p < S.n_prn; ++p)
                if (S.trim_n[(size_t)p] > 0)
                    line += fmt::format(" P{:d} {:+.2f}c (d {:+.3f} q {:.1f})", S.prns[(size_t)p],
                                        S.trim[(size_t)p], S.trim_disc[(size_t)p],
                                        S.trim_q[(size_t)p]);
            if (!line.empty())
                INFO("code trim:{:s}", line);
        }
    }

    // Snapshot the seeds (and trims) once for the whole frame; the broker updates
    // asynchronously, and the trim update above runs on other frames' executes.
    std::vector<cudaGnssChordTrackState::Seed> seeds;
    std::vector<double> trim_now;
    {
        std::lock_guard<std::mutex> lk(S.seed_mtx);
        seeds = S.seeds;
    }
    if (S.trim_enable) {
        std::lock_guard<std::mutex> lk(S.trim_mtx);
        trim_now = S.trim;
    } else
        trim_now.assign((size_t)S.n_prn, 0.0);

    int n_out_rows = 0;
    for (int r = 0; r < n_rec; ++r) {
        const long long hop0 = (seq0 < 0) ? 0 : (seq0 / S.fft_len) + (long long)r * S.hops_per_record;
        const long long wstart = hop0 * (long long)S.fft_len; // absolute SAMPLE of the window
        winstart[r] = wstart;

        std::vector<GnssCudaDespread::Spec> specs;
        specs.reserve((size_t)S.n_prn);
        for (int p = 0; p < S.n_prn; ++p) {
            gnss_gpu::PrnCtl& c = pctl[(size_t)r * S.n_prn + p];
            const auto& sd = seeds[(size_t)p];
            c.job0 = -1;
            if (!sd.have || seq0 < 0)
                continue;

            // MODEL-PRIMARY EXTRAPOLATION. The broker refreshes doppler/cp every cycle from the
            // BRDC model, so all this does is carry the seed forward to THIS window: the code
            // phase advances at the broker's measured cp_rate (chips per hop, which absorbs the
            // residual LO-vs-ADC offset l-a), and the Doppler along its own rate. No frozen
            // anchor, no fence -- if the model moves, the seed moves with it.
            const double dh = (double)(hop0 - sd.ref_hop);
            const double dt = dh * (double)S.fft_len / S.sample_rate;
            const double dop = sd.doppler_hz + sd.dop_rate * dt;
            // The DLL trim rides ON TOP of the model cp: the broker keeps owning the seed
            // (fit/coast state stays pure), the trim owns the sub-chip residual -- including
            // the clock chain's +-1 chip / ~20 s breathing that no external loop can follow.
            const double cp = sd.cp_chips + sd.cp_rate * dh + trim_now[(size_t)p];

            GnssCudaDespread::Spec sp;
            sp.p = p;
            sp.doppler_hz = dop;
            sp.cp_seed = cp;
            sp.spacing_chips = S.dll_spacing;
            sp.covering = S.covering;

            c.run = 1;
            c.reanchored = 0;
            c.job0 = (int)specs.size() * gnss_gpu::ROWS_PLAIN + n_out_rows;
            c.fcar_report = (float)dop;
            c.n_owned = (float)S.n_chan;
            c.cp_seed = cp;
            c.f_nco = sd.ctrim_hz;
            c.chan_mask = (S.n_chan >= 64) ? ~0ULL : ((1ULL << S.n_chan) - 1ULL);
            c.energy_scale = 1.0;
            c.fcar = S.f_offset_hz + dop;
            specs.push_back(sp);
        }
        if (specs.empty())
            continue;

        // corr carries the element axis; energy does not (one replica, every antenna).
        auto* d_corr = (double2*)(d_out + gnss_gpu::off_corr(S.n_prn))
                       + (size_t)n_out_rows * S.n_chan * S.n_elem;
        auto* d_energy =
            (double*)(d_out
                      + gnss_gpu::off_energy(S.n_prn, S.n_chan, gnss_gpu::ROWS_PLAIN, S.n_elem))
            + (size_t)n_out_rows * S.n_chan;
        const char* d_window =
            d_frame
            + (size_t)r * S.hops_per_record * S.frame_chan_stride * S.elem_stride;

        n_out_rows += S.despread->enqueue_batch_nm(d_window, d_scale, d_chanids, d_wave, S.n_elem,
                                                   S.elem_stride, S.frame_chan_stride, wstart,
                                                   specs, d_jobs + (size_t)n_out_rows, d_corr,
                                                   d_energy, (void*)stream, S._conjugate);
    }
    hdr->n_jobs = n_out_rows;

    // DLL trim, step 2: enqueue THIS frame's E/P/L readback -- the reference element's
    // double2 per (row, chan), strided out of the [row][chan][elem] corr block. It lands
    // after the despread kernels on the same stream; the recorded event is what a later
    // execute() polls before believing the host buffer.
    if (S.trim_enable && n_out_rows > 0) {
        std::lock_guard<std::mutex> lk(S.trim_slot_mtx);
        auto& sl = S.trim_slots[(size_t)pipestate.gpu_frame_id % S.trim_slots.size()];
        if (!sl.pending) { // an unconsumed slot is one dropped measurement, never a stall
            for (int r = 0; r < n_rec; ++r)
                for (int p = 0; p < S.n_prn; ++p) {
                    const gnss_gpu::PrnCtl& c = pctl[(size_t)r * S.n_prn + p];
                    sl.job0[(size_t)r * S.n_prn + p] = c.run ? c.job0 : -1;
                }
            sl.n_rows = n_out_rows;
            const char* src = d_out + gnss_gpu::off_corr(S.n_prn)
                              + (size_t)S.trim_ref_elem * sizeof(double2);
            CHECK_CUDA_ERROR(cudaMemcpy2DAsync(
                sl.host, sizeof(double2), src, (size_t)S.n_elem * sizeof(double2),
                sizeof(double2), (size_t)n_out_rows * S.n_chan, cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA_ERROR(cudaEventRecord(sl.ev, stream));
            sl.pending = true;
        }
    }

    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_out, _ctl_stage.data(), _ctl_stage.size(),
                                     cudaMemcpyHostToDevice, stream));
    return record_end_event();
}
