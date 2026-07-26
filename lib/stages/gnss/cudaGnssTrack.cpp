#include "cudaGnssTrack.hpp"

#include "GnssChanMetadata.hpp"
#include "cudaGnssDespreadKernel.hpp"
#include "cudaUtils.hpp" // for CHECK_CUDA_ERROR
#include "gnssGpuChain.hpp"
#include "gnssRecord.hpp"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <limits>
#include <set>

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND_WITH_STATE(cudaGnssTrack, cudaGnssTrackState);

/// Split a PRN's summed-gain statistics into (signal power, per-record noise power), the two
/// numbers both the peel gate and the depth floor are built from. Kept in one place because
/// they MUST agree: a gate stricter than the floor it publishes silently rejects satellites the
/// health line then reports as peelable. False while the difference estimator is unseeded.
static bool peel_gain_split(const cudaGnssTrackState& S, int p, double& sig2, double& noise) {
    if (S.gain_dvar[p] < 0.0)
        return false;
    noise = S.gain_dvar[p];
    sig2 = std::max(0.0, S.gain_sum_p2[p] - noise);
    return true;
}

cudaGnssTrackState::cudaGnssTrackState(Config& config, const std::string& unique_name,
                                       bufferContainer& host_buffers,
                                       cudaDeviceInterface& device) :
    cudaCommandState(config, unique_name, host_buffers, device) {
    const std::string signal = config.get_default<std::string>(unique_name, "signal", "GPS_L1CA");
    const gnss::SignalDescriptor* sig = gnss::signal_by_name(signal);
    if (sig == nullptr)
        throw std::runtime_error("cudaGnssTrack: unknown signal " + signal);

    sample_rate = config.get_default<double>(unique_name, "sample_rate", 5e6);
    const double f_offset = config.get_default<double>(unique_name, "f_offset", 0.0);
    capture_utc0 = config.get_default<double>(unique_name, "capture_utc0", 0.0);
    doppler_margin_hz = config.get_default<double>(unique_name, "doppler_margin_hz", 5000.0);
    f_offset_hz = f_offset;
    max_cover_bins = config.get_default<int>(unique_name, "max_cover_bins", 0);
    const int N = config.get<int>(unique_name, "spectrum_length");
    fft_len = 2 * N;
    chan_offset = config.get_default<int>(unique_name, "channel_offset", 0);
    n_chan = config.get<int>(unique_name, "n_channels");
    const int num_taps = config.get_default<int>(unique_name, "num_taps", 4);
    const std::string win = config.get_default<std::string>(unique_name, "pfb_window", "hamming");
    prns = config.get<std::vector<int>>(unique_name, "prns");
    n_prn = (int)prns.size();
    if (n_prn == 0)
        throw std::runtime_error("cudaGnssTrack: 'prns' is empty");
    // Carrier model: the GPU chain supports the shared-carrier NCO path only (the per-record
    // FLL needs amplitude feedback into pass-1, which the assembler-side NCO doesn't provide;
    // every live config runs fll_gain 0 + carrier_shared).
    if (config.get_default<double>(unique_name, "fll_gain", 0.0) != 0.0)
        throw std::runtime_error("cudaGnssTrack: fll_gain must be 0 (carrier_shared only)");

    replica = std::make_unique<gnss::ChannelizedReplicaBank>(
        *sig, sample_rate, f_offset, N, num_taps, dsp::window_from_string(win), prns);
    replica->code_doppler_sign =
        config.get_default<double>(unique_name, "code_doppler_sign", 1.0);
    hops_per_record =
        config.get_default<int>(unique_name, "hops_per_record", replica->repl_period_hops());
    // dll_spacing_chips is in EFFECTIVE (post-comb) chips: /comb_mult converts to the component
    // chips the cp API uses. This keeps Early/Late on the LINEAR flanks of the true correlation
    // peak for every modulation. At +-0.5 COMPONENT chips on BOC(1,1), E/L sit at the ACF's
    // NEGATIVE minima and the power discriminator's slope is sign-INVERTED at the peak -> the
    // broker DLL walks to a stable false lock +-0.25 chips off-peak, prompt = R(0.25) = 0.25
    // (-12 dB: the 2026-07-12 E1C/B1C deficit vs direct raw-voltage correlation). Same fix
    // revives L2C's dead discriminator (E/L at the zero-stuffed ACF's feet).
    dll_spacing = config.get_default<double>(unique_name, "dll_spacing_chips", 0.5)
                  / (double)replica->comb_mult();
    // f_ref fence, DERIVED from the record period (0.1 cycle/record), not a constant: the
    // replica carrier is FIXED between re-anchors, so a stale f_ref decoheres the despread
    // WITHIN one record, and the tolerable staleness scales with that record. 100 Hz at GPS's
    // 1 ms, 25 Hz at E1C's 4 ms, 10 Hz at B1C's 10 ms. The old fixed 200 Hz was 0.2 cycle for
    // GPS but TWO WHOLE CYCLES at a 10 ms B1C record. (The tri-constellation config already
    // carries the right per-chain numbers; a derived DEFAULT means the next band cannot get
    // this wrong by omission.) Same 0.1-cycle rule as the broker's hold_max_dop_hz.
    {
        const double t_rec_s =
            (double)hops_per_record * (double)fft_len / sample_rate;
        fll_reacq_hz = config.get_default<double>(unique_name, "fll_reacq_hz",
                                                  (t_rec_s > 0.0) ? 0.1 / t_rec_s : 200.0);
    }
    // Cap the carrier-anchor AGE regardless of the Doppler fence: the NCO's Doppler-rate
    // feed-forward is LINEAR from the anchor, but the true rate drifts (~1.7e-4 Hz/s^2 for
    // MEO), so the accumulated quadratic carrier error is ~0.5*rate_dot*age^2 -- ~14 cycles
    // at the 400 s ages the loose GPS fence allowed (the trim loop absorbs most, leaving
    // ~0.1-0.3 cycles across a 1 s deep window: the ladder retreated to 125 ms on ~half of
    // strong GPS emits while E1C/B1C -- whose tight fences re-anchor every 20-40 s -- held
    // full windows on the SAME LO, which is what localized this). 30 s caps it at ~0.08
    // cycles; the fence's phase glitch costs one deep window (~3% duty at 1 s windows).
    max_anchor_age_s = config.get_default<double>(unique_name, "max_anchor_age_s", 30.0);

    const int ring_records = config.get_default<int>(unique_name, "ring_records", 50);
    ring_hops = (long long)ring_records * hops_per_record;

    // CHORD 4+4b input: frames arrive as one byte per (hop, chan) sample, already quantized by
    // GnssQuantize44 (the RFSoC's job, done upstream on the CPU); the ring stays bytes and the
    // despread decodes with the per-channel scales riding the frame metadata.
    quantized = config.get_default<bool>(unique_name, "quantized_input", false);

    despread = std::make_unique<GnssCudaDespread>(*replica, n_prn, n_chan, chan_offset,
                                                  hops_per_record, sample_rate,
                                                  replica->f_offset());

    dop.assign(n_prn, 0.0);
    cp.assign(n_prn, 0.0);
    cp_rate.assign(n_prn, 0.0);
    dop_rate.assign(n_prn, 0.0);
    ctrim.assign(n_prn, 0.0);
    ref_hop.assign(n_prn, 0);
    active.assign(n_prn, 0);
    f_ref.assign(n_prn, std::nan(""));
    reacq_hop.assign(n_prn, 0);

    // ---- voltage peel (docs/gnss_voltage_peel_live.md) ----
    peel = config.get_default<bool>(unique_name, "peel", false);
    // alpha sets the gain-averaging window in RECORDS. The peel depth is limited by the gain
    // estimate's noise, which falls as ~sqrt(1/alpha): a per-record gain is ~18 dB SNR at L1, so
    // alpha 0.01 (~100 records, 0.1 s) buys ~20 dB and lands near the prototype's ~35 dB ceiling.
    // Averaging much longer runs into the receiver's carrier coherence instead.
    peel_alpha = config.get_default<double>(unique_name, "peel_gain_alpha", 0.01);
    peel_min_amp = config.get_default<double>(unique_name, "peel_min_amp", 0.0);
    peel_min_gain_snr = config.get_default<double>(unique_name, "peel_min_gain_snr", 3.0);
    peel_warmup = config.get_default<int>(unique_name, "peel_warmup_records", 100);
    peel_require_bits = config.get_default<bool>(unique_name, "peel_require_bits", true);
    peel_fll_gain = config.get_default<double>(unique_name, "peel_fll_gain", 0.05);
    peel_fll_max_hz = config.get_default<double>(unique_name, "peel_fll_max_hz", 10.0);
    peel_fll_max_slew_hz =
        config.get_default<double>(unique_name, "peel_fll_max_slew_hz", 2.0);
    peel_tag = signal + "@" + unique_name;
    peel_component =
        config.get_default<std::string>(unique_name, "peel_component", "P");
    if (peel) {
        gain.assign((size_t)n_prn * n_chan, std::complex<float>(0.f, 0.f));
        gain_n.assign(n_prn, 0);
        gain_p2.assign(n_prn, 0.0);
        gain_sum_p2.assign(n_prn, 0.0);
        gain_sum_prev.assign(n_prn, std::complex<double>(0.0, 0.0));
        gain_dvar.assign(n_prn, -1.0);
        gain_sign.assign(n_prn, 1.0);
        nav_bits.assign(n_prn, {});
        peel_f_track.assign(n_prn, 0.0);
        peel_phi_track.assign(n_prn, 0.0);
        peel_a_prev.assign(n_prn, std::complex<double>(0.0, 0.0));
        peel_prev_ok.assign(n_prn, 0);
        peel_prev_wstart.assign(n_prn, 0);
        peel_ratio.assign(n_prn, -1.0);
        peel_ratio_n.assign(n_prn, 0);
        peel_ratio_ic.assign(n_prn, -1.0);
        pres2.assign(n_prn, -1.0);
        pres_d2.assign(n_prn, 0.0);
        pres_prev.assign(n_prn, 0.0);
        pres_ok.assign(n_prn, 0);
        pres_c1.assign(n_prn, 0.0);
        pres_c2.assign(n_prn, 0.0);
        pres_same.assign(n_prn, 0.0);
        pres_diff.assign(n_prn, 0.0);
        pres_fdiff.assign(n_prn, 0.0);
        pres_pos.assign(n_prn, 0.0);
        pres_neg.assign(n_prn, 0.0);
        pres_fzero.assign(n_prn, 0.0);
        pres_blo.assign(n_prn, 0.0);
        pres_bhi.assign(n_prn, 0.0);
        peel_skip.assign(n_prn, PK_WARMUP);
        phi_nco.assign(n_prn, 0.0);
        fcar_prev.assign(n_prn, 0.0);
        nco_ok.assign(n_prn, 0);
        wstart_prev_p.assign(n_prn, 0);
        used.assign((size_t)gnss_gpu::MAX_REC * n_prn, PeelUsed{});
        used_prn.assign((size_t)gnss_gpu::MAX_REC * n_prn, -1);
        const size_t rows = (size_t)gnss_gpu::max_jobs(n_prn, gnss_gpu::ROWS_PEEL);
        mir_corr.assign(rows * n_chan, double2{0.0, 0.0});
        mir_energy.assign(rows * n_chan, 0.0);
        INFO("cudaGnssTrack: VOLTAGE PEEL enabled (alpha {:.4f} ~ {:.0f} records, warmup {:d}, "
             "min_amp {:.3g}) -- records carry the FULL correlation via the analytic add-back, "
             "plus the residual in slots 20-23",
             peel_alpha, (peel_alpha > 0) ? 1.0 / peel_alpha : 0.0, peel_warmup, peel_min_amp);
    }

    // Broker seed contract == GnssChannelizedTracker::set_seeds_callback.
    const std::string ep =
        config.get_default<std::string>(unique_name, "seed_endpoint", "/track/set_seeds");
    using namespace std::placeholders;
    kotekan::restServer::instance().register_post_callback(
        ep, std::bind(&cudaGnssTrackState::set_seeds_callback, this, _1, _2));
}

cudaGnssTrackState::~cudaGnssTrackState() {
    if (mir_evt)
        cudaEventDestroy(mir_evt);
}

void cudaGnssTrackState::set_seeds_callback(kotekan::connectionInstance& conn,
                                            nlohmann::json& request) {
    // PARSE FIRST, LOCK LAST. The old version held seed_mtx across the whole parse -- including
    // the per-bit loop over every PRN's table -- while pass 1 takes that same mutex EVERY GPU
    // frame to snapshot the seeds. A large payload therefore stalled the tracker directly.
    // Nothing here needs the lock: build the update locally, then swap it in.
    struct Upd {
        int i;
        double dop, cp, cp_rate, dop_rate, ctrim;
        long long ref_hop;
        bool has_bits;
        NavBits bits;
    };
    std::vector<Upd> upd;
    int bad_bits = 0;
    try {
        upd.reserve(request.size());
        for (const auto& s : request) {
            const int prn = s.at("prn").get<int>();
            for (int i = 0; i < n_prn; ++i)
                if (prns[i] == prn) {
                    Upd u{};
                    u.i = i;
                    u.dop = s.at("doppler_hz").get<double>();
                    u.cp = s.at("code_phase_chips").get<double>();
                    u.cp_rate = s.value("code_phase_rate", 0.0);
                    u.dop_rate = s.value("doppler_rate_hz_s", 0.0);
                    u.ctrim = s.value("carrier_trim_hz", 0.0);
                    u.ref_hop = s.value("ref_hop", (long long)0);
                    // P7a predicted nav bits (optional): +-1 per 20 ms bit, 0 = unknown.
                    if (peel && s.contains("nav_bits")) {
                        try {
                            // Component-keyed ({"P": {...}, "D": {...}}) or bare legacy table
                            // (= "P"). A dict without our component is NOT an error -- another
                            // consumer's component may ride the same row -- it is simply no
                            // bits for us.
                            const auto& outer = s.at("nav_bits");
                            const nlohmann::json* nbp = nullptr;
                            if (outer.contains("bits"))
                                nbp = &outer;                      // legacy bare table
                            else if (outer.contains(peel_component))
                                nbp = &outer.at(peel_component);
                            if (nbp == nullptr)
                                throw std::runtime_error("no component " + peel_component);
                            const auto& nb = *nbp;
                            NavBits t;
                            t.utc0 = nb.at("utc0").get<double>();
                            t.bit_s = nb.value("bit_s", 0.02);
                            const auto& arr = nb.at("bits");
                            t.bits.reserve(arr.size());
                            for (const auto& b : arr)
                                t.bits.push_back((int8_t)b.get<int>());
                            if (t.bit_s > 0.0 && !t.bits.empty()) {
                                u.bits = std::move(t);
                                u.has_bits = true;
                            } else
                                ++bad_bits;
                        } catch (const std::exception&) {
                            // COUNTED, not swallowed -- see nav_bits_bad in the header. The
                            // previous table is still kept (it ages out via the lookup's
                            // freshness guard), but the health line now says it happened.
                            ++bad_bits;
                        }
                    }
                    upd.push_back(std::move(u));
                }
        }
        {
            std::lock_guard<std::mutex> lk(seed_mtx);
            std::fill(active.begin(), active.end(), (uint8_t)0); // list is authoritative
            for (auto& u : upd) {
                dop[u.i] = u.dop;
                cp[u.i] = u.cp;
                cp_rate[u.i] = u.cp_rate;
                dop_rate[u.i] = u.dop_rate;
                ctrim[u.i] = u.ctrim;
                ref_hop[u.i] = u.ref_hop;
                if (u.has_bits)
                    nav_bits[u.i] = std::move(u.bits);
                active[u.i] = 1;
            }
        }
        if (bad_bits)
            nav_bits_bad.fetch_add(bad_bits, std::memory_order_relaxed);
    } catch (const std::exception& e) {
        conn.send_error(e.what(), kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    conn.send_empty_reply(kotekan::HTTP_RESPONSE::OK);
}

cudaGnssTrack::cudaGnssTrack(Config& config, const std::string& unique_name,
                             bufferContainer& host_buffers, cudaDeviceInterface& device,
                             int instance_num, std::shared_ptr<cudaCommandState> state) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, state) {
    _gpu_mem_input = config.get<std::string>(unique_name, "gpu_mem_input");
    _gpu_mem_output = config.get<std::string>(unique_name, "gpu_mem_output");
    set_command_type(gpuCommandType::KERNEL);
    set_name("cudaGnssTrack");

    cudaGnssTrackState* s = st();
    _in_frame_len = config.get<size_t>(unique_name, "in_frame_len");
    // Sample size follows the input format: 1 byte (4+4b) vs 8 (complex float).
    _n_hops_frame =
        (int)(_in_frame_len / ((size_t)s->n_chan * (s->quantized ? 1 : 2 * sizeof(float))));
    _rows_spec = s->peel ? gnss_gpu::ROWS_PEEL : gnss_gpu::ROWS_PLAIN;
    _out_frame_len = gnss_gpu::frame_bytes(s->n_prn, s->n_chan, _rows_spec);
    _ctl_stage.resize(gnss_gpu::off_corr(s->n_prn));

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_input, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_output, true, false, true));

    // Persistent shared allocations: the ring (survives across frames) + code/Phi live in the
    // despread driver. Jobs arena + output frames are per-frame arrays. NAMES ARE NAMESPACED
    // BY STAGE: the cudaDeviceInterface memory registry is shared across every cudaProcess on
    // the GPU, so a bare "gnss_ring" COLLIDES when multiple constellation chains run (first
    // allocation wins, the rest error and trample each other's ring geometry -- observed as
    // the BDS chain silently degrading on the first tri-constellation night).
    _mem_ring = unique_name + "/gnss_ring";
    _mem_jobs = unique_name + "/gnss_jobs_arena";
    _mem_scale = unique_name + "/gnss_chan_scale";
    const size_t samp = s->quantized ? 1 : sizeof(float2);
    device.get_gpu_memory(_mem_ring, (size_t)s->n_chan * s->ring_hops * samp);
    if (s->quantized) {
        // The dequantization scales, zeroed until the quantizer's freeze arrives on metadata.
        // Zero (not garbage) matters: warmup frames are offset-zero bytes, which decode to
        // 0 volts under any FINITE scale -- but an uninitialized float can be NaN, and
        // NaN * 0 = NaN would poison the correlations.
        void* d_scale = device.get_gpu_memory(_mem_scale, (size_t)s->n_chan * sizeof(float));
        CHECK_CUDA_ERROR(cudaMemset(d_scale, 0, (size_t)s->n_chan * sizeof(float)));
    }
    if (s->peel) {
        // Peel scratch. The residual is ONE record window, FLOAT2 -- shared across the frame's
        // records because the peel and the despread that consumes it are enqueued back-to-back on
        // the same (ordered) stream, so record k+1 cannot overwrite what record k is still
        // reading. Names are stage-namespaced for the same reason the ring is: the device memory
        // registry is shared across every cudaProcess on the GPU.
        _mem_resid = unique_name + "/gnss_peel_resid";
        _mem_pjobs = unique_name + "/gnss_peel_jobs";
        _mem_gain = unique_name + "/gnss_peel_gain";
        _mem_xcorr = unique_name + "/gnss_peel_xcorr";
        device.get_gpu_memory(_mem_resid, (size_t)s->n_chan * s->hops_per_record * sizeof(float2));
        device.get_gpu_memory(_mem_pjobs,
                              (size_t)gnss_gpu::max_specs(s->n_prn) * sizeof(gnss_cuda::PeelJob));
        device.get_gpu_memory(_mem_gain,
                              (size_t)2 * gnss_gpu::max_specs(s->n_prn) * s->n_chan
                                  * sizeof(float2));
        device.get_gpu_memory(_mem_xcorr,
                              (size_t)4 * gnss_gpu::max_specs(s->n_prn) * s->n_chan
                                  * sizeof(double2));
    }
}

cudaGnssTrack::~cudaGnssTrack() = default;

cudaGnssTrackState* cudaGnssTrack::st() {
    return static_cast<cudaGnssTrackState*>(command_state.get());
}

cudaEvent_t cudaGnssTrack::execute(cudaPipelineState& pipestate,
                                   const std::vector<cudaEvent_t>& pre_events) {
    pre_execute();
    cudaGnssTrackState& S = *st();
    const cudaStream_t stream = device.getStream(cuda_stream_id);

    const size_t samp = S.quantized ? 1 : sizeof(float2); // ring/frame sample size (bytes)
    char* d_ring =
        (char*)device.get_gpu_memory(_mem_ring, (size_t)S.n_chan * S.ring_hops * samp);
    const char* d_frame = (const char*)device.get_gpu_memory_array(
        _gpu_mem_input, pipestate.gpu_frame_id, _gpu_buffer_depth, _in_frame_len);
    char* d_out = (char*)device.get_gpu_memory_array(_gpu_mem_output, pipestate.gpu_frame_id,
                                                     _gpu_buffer_depth, _out_frame_len);
    gnss_cuda::DespreadJob* d_jobs = (gnss_cuda::DespreadJob*)device.get_gpu_memory_array(
        _mem_jobs, pipestate.gpu_frame_id, _gpu_buffer_depth,
        (size_t)gnss_gpu::max_specs(S.n_prn) * sizeof(gnss_cuda::DespreadJob));

    // ---- PEEL: fold the PREVIOUS frame's measurements into the feed-forward gain.
    // The mirror D2H was enqueued at the end of that frame, so by now it is long done -- the
    // sync is a formality that costs nothing and makes the ordering explicit rather than assumed.
    if (S.peel && S.mir_pending) {
        CHECK_CUDA_ERROR(cudaEventSynchronize(S.mir_evt));
        S.mir_pending = false;
        for (int r = 0; r < S.used_n_rec; ++r)
            for (int i = 0; i < S.n_prn; ++i) {
                const size_t u = (size_t)r * S.n_prn + i;
                const cudaGnssTrackState::PeelUsed& pu = S.used[u];
                const int p = S.used_prn[u];
                if (!pu.run || p < 0)
                    continue;
                // Rows are already ADDED BACK on the device, so row P holds the FULL correlation
                // -- the gain loop sees the same quantity it would without any peel, which is
                // what keeps it from chasing its own subtraction.
                const size_t rowP = (size_t)(pu.job0 + gnss_gpu::ROW_P) * S.n_chan;
                const size_t rowH = (size_t)(pu.job0 + gnss_gpu::ROW_PH) * S.n_chan;
                if (S.mir_rows_spec > gnss_gpu::ROW_RES_P) { // ground-truth cancellation ratio
                    const size_t rowR = (size_t)(pu.job0 + gnss_gpu::ROW_RES_P) * S.n_chan;
                    std::complex<double> fu(0.0, 0.0), re(0.0, 0.0);
                    // ...and the same two quantities as sums of per-channel MAGNITUDES. The
                    // coherent sums above divide by |sum_c full_c|, which SELF-CANCELS on a
                    // phase-diverse satellite and drives the ratio to 1 -- identical to "nothing
                    // was subtracted". These cannot self-cancel, so they say whether the peel
                    // worked even when the coherent pair cannot. See peel_ratio_ic in the hpp.
                    double fu2 = 0.0, re2 = 0.0;
                    for (int c = 0; c < S.n_chan; ++c) {
                        const std::complex<double> fc(S.mir_corr[rowP + c].x,
                                                      S.mir_corr[rowP + c].y);
                        const std::complex<double> rc(S.mir_corr[rowR + c].x,
                                                      S.mir_corr[rowR + c].y);
                        fu += fc;
                        re += rc;
                        fu2 += std::norm(fc);
                        re2 += std::norm(rc);
                    }
                    if (std::abs(fu) > 0.0) {
                        const double r = std::abs(re) / std::abs(fu);
                        S.peel_ratio[p] = (S.peel_ratio_n[p] == 0)
                                              ? r
                                              : 0.98 * S.peel_ratio[p] + 0.02 * r;
                        if (S.peel_ratio_n[p] < 1000000)
                            ++S.peel_ratio_n[p];
                    }
                    if (fu2 > 0.0) {
                        const double ric = std::sqrt(re2 / fu2);
                        S.peel_ratio_ic[p] = (S.peel_ratio_ic[p] < 0.0)
                                                 ? ric
                                                 : 0.98 * S.peel_ratio_ic[p] + 0.02 * ric;
                    }
                }
                const std::complex<double> derot = std::polar(1.0, -pu.phi);



                // OVERLAY SIGN. The correlation carries the record's nav symbol / secondary chip,
                // and averaging across a flip cancels it. Two things follow:
                //  * de-bit before averaging (decision-directed against the running gain), and
                //  * SKIP records whose head and tail disagree -- the whole-record prompt then
                //    partially cancels to |2f-1| and its gain is not a measurement of anything.
                //    That is ~1 record in 20 on GPS nav data and never on a pilot.
                // PEEL FLL (see the hpp note): the NCO mirror carries only the COMMANDED carrier;
                // the residual (true - commanded) rotates the derotated gain, and at Hz-level the
                // EMA would average a rotating phasor into a lagged, shrunken gain that subtracts
                // at the wrong angle. Advance phi_track by the tracked residual over the gap, fold
                // it into the derotation, and drive f_track with the bit-robust squared
                // discriminator on consecutive cross-channel gain measurements.
                const double dt_fll =
                    S.peel_prev_ok[p]
                        ? (double)(pu.wstart - S.peel_prev_wstart[p]) / S.sample_rate
                        : 0.0;
                if (S.peel_prev_ok[p] && dt_fll > 0.0)
                    S.peel_phi_track[p] = std::remainder(
                        S.peel_phi_track[p] + 2.0 * M_PI * S.peel_f_track[p] * dt_fll,
                        2.0 * M_PI);
                // Total derotation for this record: NCO mirror + FLL.
                const std::complex<double> derot_t =
                    derot * std::polar(1.0, -S.peel_phi_track[p]);

                std::complex<double> sum_h(0.0, 0.0), sum_t(0.0, 0.0), sum_ref(0.0, 0.0);
                for (int c = 0; c < S.n_chan; ++c) {
                    const double eH = S.mir_energy[rowH + c];
                    const double eT = S.mir_energy[rowP + c] - eH;
                    if (eH > 0.0)
                        sum_h += std::complex<double>(S.mir_corr[rowH + c].x,
                                                      S.mir_corr[rowH + c].y);
                    if (eT > 0.0)
                        sum_t += std::complex<double>(S.mir_corr[rowP + c].x - S.mir_corr[rowH + c].x,
                                                      S.mir_corr[rowP + c].y - S.mir_corr[rowH + c].y);
                    sum_ref += std::complex<double>(S.gain[(size_t)p * S.n_chan + c]);
                }
                // FLL discriminator on the FLL-frame cross-channel measurement (sign-blind: the
                // product is squared, so the +-1 nav flips between consecutive records cancel).
                const std::complex<double> a_fll =
                    (pu.s_head != 0 && pu.s_tail != 0)
                        ? (sum_h * (double)pu.s_head + sum_t * (double)pu.s_tail) * derot_t
                        : (sum_h + sum_t) * derot_t;
                if (S.peel_prev_ok[p] && dt_fll > 0.0 && dt_fll < 0.02 && std::abs(a_fll) > 0.0
                    && std::abs(S.peel_a_prev[p]) > 0.0) {
                    const std::complex<double> prod = a_fll * std::conj(S.peel_a_prev[p]);
                    // ALIAS CAPTURE. The squared discriminator only determines f modulo
                    // 1/(2*dt) -- every multiple of that is an equally good solution, so an
                    // unbounded integrator random-walks out along the ladder and parks on one.
                    // Measured 2026-07-25 over an 8h45m soak: 99% of |f_track|>1 Hz values sit
                    // within 5% of a multiple of this chain's 1/(2*T_code) (GPS 500, GAL 125,
                    // BDS/L1C 50 Hz), reaching 19 kHz. EVEN multiples are whole cycles per
                    // record and turn out to be harmless -- the record-boundary phase advances
                    // by exactly 2pi -- which is why the damage is modest on GPS/GAL/BDS
                    // (~100% even) and real on L1C (19% ODD -> pi per record -> the EMA
                    // averages +g,-g,+g and collapses: ratio 0.42 vs 0.15, floor 0.37 vs 0.08).
                    // The residual this loop exists to track is ~0.01 Hz with ctrim closed and
                    // "Hz-level" during convergence, so bound it well inside the first rung:
                    // a solution beyond the clamp is an alias, never physics.
                    const double step =
                        S.peel_fll_gain * std::arg(prod * prod) / 2.0 / (2.0 * M_PI * dt_fll);
                    const double slew = S.peel_fll_max_slew_hz;
                    S.peel_f_track[p] += std::max(-slew, std::min(slew, step));
                    S.peel_f_track[p] = std::max(-S.peel_fll_max_hz,
                                                 std::min(S.peel_fll_max_hz, S.peel_f_track[p]));
                }
                S.peel_a_prev[p] = a_fll;
                S.peel_prev_ok[p] = 1;
                S.peel_prev_wstart[p] = pu.wstart;

                // De-bit for the EMA. With PREDICTED signs (pu.s_head/s_tail): per-segment,
                // per-channel -- a record straddling a KNOWN flip is a fully usable
                // measurement, so the old straddle skip does not apply, and the gain's
                // polarity locks to the prediction's frame. Without them: the
                // decision-directed fallback (cross-channel sign vs the running gain,
                // straddles skipped), exactly the old behaviour.
                const bool have_bits = (pu.s_head != 0 && pu.s_tail != 0);
                double sgn_obs = 1.0;
                if (!have_bits) {
                    // Signs relative to the running gain (cross-channel: best available SNR).
                    const std::complex<double> ref = sum_ref / derot_t; // gain -> replica frame
                    const double dh = std::real(sum_h * std::conj(ref));
                    const double dt_ = std::real(sum_t * std::conj(ref));
                    const bool have_ref = (S.gain_n[p] > 0) && std::abs(ref) > 0.0;
                    const bool split = have_ref && std::abs(sum_h) > 0.0
                                       && std::abs(sum_t) > 0.0 && ((dh > 0.0) != (dt_ > 0.0));
                    if (split)
                        continue; // straddles an UNKNOWN flip: not a usable measurement
                    sgn_obs = (!have_ref || (dh + dt_) >= 0.0) ? 1.0 : -1.0;
                }

                // Did this record have a predicted sign at all? (see pres_fzero in the hpp)
                S.pres_fzero[p] = (1.0 - S.peel_alpha) * S.pres_fzero[p]
                                  + S.peel_alpha * (have_bits ? 0.0 : 1.0);
                double rec_p2 = 0.0; // this record's total gain power (signal + noise)
                std::complex<double> csum(0.0, 0.0); // C = sum_c a_c*E_c, the ratio's own space
                // Phase residual of THIS record against the gain built so far (see pres2 in
                // the hpp): z = sum_c a_c*conj(g_c), accumulated with g still PRE-update so
                // the record is scored against history rather than against itself.
                std::complex<double> zsum(0.0, 0.0);
                for (int c = 0; c < S.n_chan; ++c) {
                    const double e = S.mir_energy[rowP + c];
                    if (!(e > 0.0))
                        continue; // channel not in this PRN's covering set this record
                    // Average in the DEROTATED (NCO + FLL) frame, where the gain is slowly
                    // varying, and DE-BITTED, so only the slow dish gain is averaged. The
                    // carrier and the bit signs are re-applied at subtraction time.
                    std::complex<double> a_num;
                    if (have_bits) {
                        const std::complex<double> hd(S.mir_corr[rowH + c].x,
                                                      S.mir_corr[rowH + c].y);
                        const std::complex<double> pp(S.mir_corr[rowP + c].x,
                                                      S.mir_corr[rowP + c].y);
                        a_num = hd * (double)pu.s_head + (pp - hd) * (double)pu.s_tail;
                    } else {
                        a_num = std::complex<double>(S.mir_corr[rowP + c].x,
                                                     S.mir_corr[rowP + c].y)
                                * sgn_obs;
                    }
                    const std::complex<double> a_d = (a_num / e) * derot_t;
                    rec_p2 += std::norm(a_d);
                    csum += a_d * e; // undo the per-channel normalisation: raw, summed
                    std::complex<float>& g = S.gain[(size_t)p * S.n_chan + c];
                    zsum += a_d * std::conj(std::complex<double>(g)); // g still PRE-update
                    g = (S.gain_n[p] == 0)
                            ? (std::complex<float>)a_d
                            : (std::complex<float>)((1.0 - S.peel_alpha)
                                                        * std::complex<double>(g)
                                                    + S.peel_alpha * a_d);
                }
                S.gain_p2[p] = (S.gain_n[p] == 0)
                                   ? rec_p2
                                   : (1.0 - S.peel_alpha) * S.gain_p2[p] + S.peel_alpha * rec_p2;
                // Phase residual level and record-to-record increment (see pres2 in the hpp).
                // Only meaningful once a gain exists to score against.
                if (S.gain_n[p] > 0 && std::abs(zsum) > 0.0) {
                    const double d = std::arg(zsum);
                    S.pres2[p] = (S.pres2[p] < 0.0)
                                     ? d * d
                                     : (1.0 - S.peel_alpha) * S.pres2[p] + S.peel_alpha * d * d;
                    if (S.pres_ok[p]) {
                        const double dd = std::remainder(d - S.pres_prev[p], 2.0 * M_PI);
                        S.pres_d2[p] =
                            (1.0 - S.peel_alpha) * S.pres_d2[p] + S.peel_alpha * dd * dd;
                    }
                    // SIGN vs PHASE (see pres_c1/pres_c2 in the hpp): cos(2d) is blind to a
                    // pi flip, so <cos 2d> ~ 1 means the BITS are wrong and <cos 2d> ~ 0 means
                    // the PHASE is genuinely scattered. pres2 alone conflates the two.
                    S.pres_c1[p] = (1.0 - S.peel_alpha) * S.pres_c1[p]
                                   + S.peel_alpha * std::cos(d);
                    S.pres_c2[p] = (1.0 - S.peel_alpha) * S.pres_c2[p]
                                   + S.peel_alpha * std::cos(2.0 * d);
                    // OFF-BY-ONE TEST (see pres_same/pres_diff in the hpp): condition the same
                    // residual on whether the APPLIED head/tail chips straddle a flip.
                    const bool straddle = (pu.s_head != pu.s_tail);
                    S.pres_fdiff[p] = (1.0 - S.peel_alpha) * S.pres_fdiff[p]
                                      + S.peel_alpha * (straddle ? 1.0 : 0.0);
                    if (straddle)
                        S.pres_diff[p] = (1.0 - S.peel_alpha) * S.pres_diff[p]
                                         + S.peel_alpha * std::cos(d);
                    else
                        S.pres_same[p] = (1.0 - S.peel_alpha) * S.pres_same[p]
                                         + S.peel_alpha * std::cos(d);
                    // TABLE-SEMANTICS split: the tracker's OWN bf at this record (see
                    // pres_blo in the hpp). Record-grid table read as time-domain predicts
                    // the low-bf bin inverted/decohered while the high-bf bin stays clean.
                    if (pu.bf >= 0.f) {
                        if (pu.bf < 0.5f)
                            S.pres_blo[p] = (1.0 - S.peel_alpha) * S.pres_blo[p]
                                            + S.peel_alpha * std::cos(d);
                        else
                            S.pres_bhi[p] = (1.0 - S.peel_alpha) * S.pres_bhi[p]
                                            + S.peel_alpha * std::cos(d);
                    }
                    // ORTHOGONAL to the straddle test: split by the chip's VALUE, which is what
                    // an even/odd overlay application would key on (see pres_pos in the hpp).
                    if (pu.s_head > 0)
                        S.pres_pos[p] = (1.0 - S.peel_alpha) * S.pres_pos[p]
                                        + S.peel_alpha * std::cos(d);
                    else if (pu.s_head < 0)
                        S.pres_neg[p] = (1.0 - S.peel_alpha) * S.pres_neg[p]
                                        + S.peel_alpha * std::cos(d);
                    S.pres_prev[p] = d;
                    S.pres_ok[p] = 1;
                }
                if (S.gain_n[p] == 0) {
                    S.gain_sum_p2[p] = std::norm(csum);
                    S.gain_dvar[p] = -1.0; // no previous record yet -- seeded on the next one
                } else {
                    S.gain_sum_p2[p] = (1.0 - S.peel_alpha) * S.gain_sum_p2[p]
                                       + S.peel_alpha * std::norm(csum);
                    // sigma^2 = <|dC|^2>/2 (see the hpp note): phase-drift-immune, unlike
                    // <|C|^2> - |<C>|^2, which double-counts the drift.
                    const double d2 = 0.5 * std::norm(csum - S.gain_sum_prev[p]);
                    S.gain_dvar[p] = (S.gain_dvar[p] < 0.0)
                                         ? d2
                                         : (1.0 - S.peel_alpha) * S.gain_dvar[p]
                                               + S.peel_alpha * d2;
                }
                S.gain_sum_prev[p] = csum;
                if (!have_bits)
                    S.gain_sign[p] = sgn_obs; // fallback prediction for the next records
                if (S.gain_n[p] < 1000000)
                    ++S.gain_n[p];
            }
        S.used_n_rec = 0;
    }
    if (S.peel) {
        std::fill(S.used.begin(), S.used.end(), cudaGnssTrackState::PeelUsed{});
        std::fill(S.used_prn.begin(), S.used_prn.end(), -1);
    }

    // Absolute hop of this frame from the claimed metadata (contiguous fallback without it).
    long long frame_hop = S.next_hop;
    auto meta = std::dynamic_pointer_cast<GnssChanMetadata>(
        device.get_gpu_memory_array_metadata(_gpu_mem_input, pipestate.gpu_frame_id));
    if (meta && meta->sample_seq >= 0)
        frame_hop = meta->sample_seq / S.fft_len;

    // 4+4b dequantization scales: the frame carries the gains its bytes were ENCODED with
    // (GnssChanMetadata::chan_scale; empty during the quantizer's bandpass-measurement window,
    // when frames are offset-zero bytes = 0 volts under any scale). Upload on change only --
    // they freeze once at startup, so this fires ~once per run.
    float* d_scale = nullptr;
    if (S.quantized) {
        d_scale = (float*)device.get_gpu_memory(_mem_scale, (size_t)S.n_chan * sizeof(float));
        if (meta && !meta->chan_scale.empty() && meta->chan_scale != S.chan_scale_host) {
            if ((int)meta->chan_scale.size() != S.n_chan) {
                WARN("cudaGnssTrack: metadata carries {:d} chan scales but the frame has {:d} "
                     "channels -- ignoring (subband-sliced scales are not supported)",
                     (int)meta->chan_scale.size(), S.n_chan);
            } else {
                S.chan_scale_host = meta->chan_scale; // pageable: async H2D is host-sync on return
                CHECK_CUDA_ERROR(cudaMemcpyAsync(d_scale, S.chan_scale_host.data(),
                                                 (size_t)S.n_chan * sizeof(float),
                                                 cudaMemcpyHostToDevice, stream));
                if (!S.chan_scale_valid)
                    INFO("cudaGnssTrack: 4+4b dequantization scales arrived (quantizer froze "
                         "its bandpass) -- despread live from this frame");
                S.chan_scale_valid = true;
            }
        }
    }

    // The despread work must follow the frame's H2D copy (COPY_IN stream).
    record_start_event();
    if (pre_events[0])
        CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream, pre_events[0], 0));

    // ---- Ring ingest: init / gap-fill / reset, then transpose the frame in. Ring position is
    // identically (absolute hop - hop0); gaps are zero-filled so a window overlapping one
    // despreads against clean zeros (SNR loss, never misalignment) -- the search-snapshot rule.
    const long long ring_capacity = S.ring_hops;
    bool reset = !S.ring_init;
    long long gap = S.ring_init ? frame_hop - S.next_hop : 0;
    if (S.ring_init && (gap < 0 || gap > ring_capacity - _n_hops_frame))
        reset = true; // out-of-order or a pathological stall: rebase the tiling anchor
    if (reset) {
        // 4+4b zero is the OFFSET code 0x88, not 0x00 (which decodes to a full-scale (-8,-8)
        // DC spike). cudaMemset writes a byte pattern, so 0x88 fills exactly.
        CHECK_CUDA_ERROR(cudaMemsetAsync(d_ring, S.quantized ? 0x88 : 0,
                                         (size_t)S.n_chan * S.ring_hops * samp, stream));
        if (S.ring_init)
            WARN("cudaGnssTrack: ring rebase (gap {:d} hops) -- window tiling re-anchored",
                 gap);
        S.hop0 = frame_hop;
        S.next_hop = frame_hop;
        S.next_rec = 0;
        S.ring_init = true;
        gap = 0;
    } else if (gap > 0) {
        if (S.quantized)
            CHECK_CUDA_ERROR(gnss_cuda::launch_ring_zero_q((unsigned char*)d_ring, S.n_chan,
                                                           S.ring_hops, S.next_hop - S.hop0,
                                                           gap, stream));
        else
            CHECK_CUDA_ERROR(gnss_cuda::launch_ring_zero((float2*)d_ring, S.n_chan, S.ring_hops,
                                                         S.next_hop - S.hop0, gap, stream));
    }
    if (S.quantized)
        CHECK_CUDA_ERROR(gnss_cuda::launch_chan_ingest_q(
            (const unsigned char*)d_frame, (unsigned char*)d_ring, _n_hops_frame, S.n_chan,
            S.ring_hops, frame_hop - S.hop0, stream));
    else
        CHECK_CUDA_ERROR(gnss_cuda::launch_chan_ingest((const float2*)d_frame, (float2*)d_ring,
                                                       _n_hops_frame, S.n_chan, S.ring_hops,
                                                       frame_hop - S.hop0, stream));
    S.next_hop = frame_hop + _n_hops_frame;

    // Overflow guard: if the record backlog would exceed the ring, fast-forward (drop the
    // stale backlog, keep alignment: next_rec stays on the hop0 tiling).
    const long long written = S.next_hop - S.hop0;
    const int hpr = S.hops_per_record;
    if (written - S.next_rec * hpr > ring_capacity) {
        const long long skip_to = (written - ring_capacity) / hpr + 1;
        WARN("cudaGnssTrack: backlog overflow, skipping records {:d}..{:d}", S.next_rec,
             skip_to - 1);
        S.next_rec = skip_to;
    }

    // ---- Pass 1 + batched despread per complete record window.
    using namespace gnss_gpu;
    FrameHdr hdr{};
    hdr.n_prn = S.n_prn;
    hdr.n_chan = S.n_chan;
    hdr.seq0 = frame_hop * S.fft_len;
    hdr.utc0 = S.capture_utc0;
    int64_t* winstart = (int64_t*)(_ctl_stage.data() + off_winstart());
    PrnCtl* pctl = (PrnCtl*)(_ctl_stage.data() + off_prnctl());
    std::memset(_ctl_stage.data(), 0, _ctl_stage.size());

    // Snapshot the REST-updated seeds once per frame (the tracker snapshots per window; the
    // broker cadence is 0.2 s, a frame is ~10 ms -- same behavior).
    std::vector<double> dop, cp, cp_rate, dop_rate, ctrim;
    std::vector<long long> ref_hop;
    std::vector<uint8_t> active;
    std::vector<cudaGnssTrackState::NavBits> navb;
    {
        std::lock_guard<std::mutex> lk(S.seed_mtx);
        dop = S.dop;
        cp = S.cp;
        cp_rate = S.cp_rate;
        dop_rate = S.dop_rate;
        ctrim = S.ctrim;
        ref_hop = S.ref_hop;
        active = S.active;
        if (S.peel)
            navb = S.nav_bits;
    }

    const double L = (double)S.replica->code_length();
    const double chip_component = S.replica->eff_chip_rate() / (double)S.replica->comb_mult();
    const double sgn = S.replica->code_doppler_sign;
    const double fc = S.replica->carrier_hz();
    // Jobs and output rows are counted SEPARATELY: the despread kernel takes one job per active
    // PRN and emits four rows from it (E, P, L, P_HEAD). n_spec indexes d_jobs; n_out indexes
    // corr/energy and is what PrnCtl::job0 / FrameHdr::n_jobs carry downstream.
    int n_rec = 0, n_spec = 0, n_out = 0;
    while ((S.next_rec + 1) * hpr <= written && n_rec < MAX_REC) {
        const long long whop = S.hop0 + S.next_rec * hpr;
        const long long wstart = whop * (long long)S.fft_len;
        winstart[n_rec] = wstart;
        std::vector<GnssCudaDespread::Spec> specs;
        std::vector<int> spec_prn;
        std::vector<GnssCudaDespread::PeelSpec> pspecs;
        for (int p = 0; p < S.n_prn; ++p) {
            PrnCtl& c = pctl[(size_t)n_rec * S.n_prn + p];
            c.job0 = -1;
            if (!active[p]) {
                S.f_ref[p] = std::nan(""); // forget the loop; re-acquire when it returns
                if (S.peel)
                    S.nco_ok[p] = 0; // gap: break the mirror's arc (see below)
                continue;
            }
            // Covering channels at the seed Doppler (GLOBAL ids -> this subband's local set).
            auto cover = S.replica->covering_bins(dop[p], S.doppler_margin_hz);
            // DIAGNOSTIC channel-width trim (max_cover_bins, 2026-07-21): keep only the N
            // covering channels nearest the carrier IF -- the narrow-despread A/B for the
            // L5-band ADR wander (wide 10.23 Mcps chains wander 20-50x vs the narrow chains
            // at the SAME 1 ms record rate; this tests the cross-channel combining width).
            if (S.max_cover_bins > 0 && (int)cover.size() > S.max_cover_bins) {
                const double bin_w = S.sample_rate / (double)S.fft_len;
                std::sort(cover.begin(), cover.end(), [&](int a, int b) {
                    return std::fabs(a * bin_w - S.f_offset_hz)
                           < std::fabs(b * bin_w - S.f_offset_hz);
                });
                cover.resize((size_t)S.max_cover_bins);
                std::sort(cover.begin(), cover.end()); // restore natural channel order
            }
            std::vector<int> local;
            uint64_t mask = 0;
            for (int ch : cover)
                if (ch >= S.chan_offset && ch < S.chan_offset + S.n_chan) {
                    local.push_back(ch - S.chan_offset);
                    mask |= (1ULL << (ch - S.chan_offset));
                }
            if (local.empty()) {
                // Carrier not in this subband. THIS IS A GAP, and the NCO mirror below must
                // break its arc here exactly as GnssGpuRecordAssemble does -- see the fix note
                // at the mirror. Falling through with nco_ok still set made the mirror
                // integrate ACROSS the gap while the assembler skips it, and the two then run
                // permanently out of step.
                // ⚠️ GUARDED: every peel vector (nco_ok included) is allocated only under
                // `if (peel)`, so on a non-peeling chain it is EMPTY and an unguarded write
                // here is out-of-bounds -- it segfaulted the whole 3-band node at launch on
                // 2026-07-25, and the launcher still printed "node running".
                if (S.peel)
                    S.nco_ok[p] = 0;
                continue;
            }

            // Pass-1 control -- verbatim port of GnssChannelizedTracker (see its comments for
            // the full physics): linear cp extrapolation + QUADRATIC code-Doppler FF, the
            // f_ref fence, the CODE-CURRENCY translation into the pinned f_ref, and the NCO
            // carrier feed-forward ramp (integrated downstream by the assembler).
            const double dt_anchor =
                (double)(whop - ref_hop[p]) * (double)S.fft_len / S.sample_rate;
            double cp_seed = cp[p] + cp_rate[p] * (double)(whop - ref_hop[p])
                             + 0.5 * sgn * dop_rate[p] * chip_component / fc * dt_anchor
                                   * dt_anchor;
            uint8_t reanchored = 0;
            const double anchor_age =
                (double)(whop - S.reacq_hop[p]) * (double)S.fft_len / S.sample_rate;
            const bool fresh = std::isnan(S.f_ref[p]);
            const bool fence = !fresh && std::fabs(S.f_ref[p] - dop[p]) > S.fll_reacq_hz;
            if (fresh || fence || anchor_age > S.max_anchor_age_s) {
                if (fresh) {
                    S.f_ref[p] = dop[p]; // genuine (re)acquisition: adopt the seed
                    reanchored = 1;      // no phase history worth keeping
                } else {
                    if (fence)
                        S.f_ref[p] = dop[p]; // the seed moved out of the fence: adopt it
                    else // AGE re-pin: fold the FF ramp into f_ref -- frequency-continuous
                         // (see GnssChannelizedTracker for the full story)
                        S.f_ref[p] +=
                            dop_rate[p] * anchor_age; // PHYSICAL frame: no NCO-side negation
                    // Either way the ABSOLUTELY-ANCHORED replica phase just stepped by
                    // df*t_abs. That step is FOLDABLE (an NCO absorbs any constant phase), so
                    // the assembler folds it and the despread output stays phase-continuous.
                    reanchored = 2;
                }
                S.reacq_hop[p] = whop;
            }
            const double fcar = S.f_ref[p];
            cp_seed += (double)whop * (double)S.fft_len / S.sample_rate * chip_component * sgn
                       * (dop[p] - fcar) / fc;
            cp_seed = std::fmod(cp_seed, L);
            if (cp_seed < 0.0)
                cp_seed += L;
            const double ff_hz = -dop_rate[p] * (double)(whop - S.reacq_hop[p])
                                 * (double)S.fft_len / S.sample_rate;

            c.run = 1;
            c.reanchored = reanchored;
            // Row block for the spec pushed below. Stride is 6 when this chain peels (the two
            // extra rows are the residual prompt + its head), else 4.
            c.job0 = n_out + _rows_spec * (int)specs.size();
            c.fcar_report = (float)(fcar - ff_hz + ctrim[p]);
            c.n_owned = (float)local.size();
            c.cp_seed = cp_seed;
            c.f_nco = ctrim[p] + ff_hz;
            c.chan_mask = mask;
            c.energy_scale = 1.0;
            c.fcar = fcar; // replica f_ref -> the assembler's commanded-carrier-phase export
            // ---- PEEL: mirror the NCO phase, then hand the peel this record's gain.
            if (S.peel) {
                // NCO MIRROR -- the same integration GnssGpuRecordAssemble runs, on the same
                // inputs. The gain is AVERAGED derotated (where it is slow) but must be
                // SUBTRACTED in the replica frame, so the peel needs this record's phase, and it
                // needs it now rather than a frame late. ⚠️ Keep in step with the assembler.
                //
                // ⚠️⚠️ AND THAT INCLUDES THE GAPS (fixed 2026-07-25). On a record where the PRN
                // does not run, the assembler clears _a_prev_ok/_fcar_prev_ok and `continue`s
                // WITHOUT advancing _wstart_prev, so when the PRN returns its dt spans the whole
                // gap and the integration is SKIPPED -- _phi carries across unchanged. This
                // mirror used to be reached only on running records, so nco_ok stayed 1 and it
                // INTEGRATED that same gap: every gap injected 2*pi*f_nco*t_gap of error, and
                // the two never resynced (both keep integrating from now-different values).
                // Because gaps recur, each one contributes a DIFFERENT arbitrary angle, so the
                // ~100-record gain EMA averaged phasors that no longer share a frame and
                // collapsed toward zero -- measured as gain coherence 0.00-0.02 on the failing
                // satellites vs 0.33-0.95 on the working ones, with |resid|/|full| = 1 -
                // sqrt(gcoh) to a mean error of 0.10 across 22 sats. Tracking never noticed,
                // because the assembler is the authority and is self-consistent; only the
                // mirror was wrong, which is why deep_snr sat at 200+ on sats peeling 0.0 dB.
                // The gaps are now cleared at both `continue`s above.
                double& phi = S.phi_nco[p];
                if (reanchored == 1 || (reanchored == 2 && !S.nco_ok[p])) {
                    phi = 0.0; // fresh acquisition: no phase history worth keeping
                    S.nco_ok[p] = 0;
                    // ...and RESET the gain. The EMA'd gain is coherent only relative to a
                    // continuous phi: its stored phase is (absolute signal phase - phi) at the
                    // records that built it. A fresh re-acq restarts phi from 0 against a NEW
                    // absolute reference (new f_ref pin, new t_reacq), so the old gain's phase no
                    // longer matches this arc -- subtracting it removes the right MAGNITUDE at the
                    // wrong PHASE, i.e. nothing cancels (measured 2026-07-24: |g| converged fine
                    // yet depth stuck at ~0 dB, because the bench re-seeds churned re-acqs faster
                    // than the ~100-record EMA could re-cohere). Relearn from this arc.
                    S.gain_n[p] = 0;
                    S.gain_p2[p] = 0.0;
                    S.gain_sum_p2[p] = 0.0;
                    S.gain_dvar[p] = -1.0; // the difference estimator restarts with the arc too
                    for (int ci = 0; ci < S.n_chan; ++ci)
                        S.gain[(size_t)p * S.n_chan + ci] = std::complex<float>(0.f, 0.f);
                    S.peel_f_track[p] = 0.0; // the FLL restarts with the arc
                    S.peel_phi_track[p] = 0.0;
                    S.peel_prev_ok[p] = 0;
                } else if (reanchored == 2) {
                    // Phase-continuous re-pin: the absolutely-anchored replica just stepped by
                    // df*t_abs, and an NCO absorbs a constant phase. Fold it, exactly as the
                    // assembler does, or the peel and the despread disagree about the frame.
                    const double t_pin = (double)wstart / S.sample_rate;
                    phi = std::remainder(phi
                                             + 2.0 * M_PI * (fcar - S.fcar_prev[p]) * t_pin,
                                         2.0 * M_PI);
                }
                const double dts = (double)(wstart - S.wstart_prev_p[p]) / S.sample_rate;
                if (S.nco_ok[p] && dts > 0.0)
                    phi = std::remainder(phi + 2.0 * M_PI * c.f_nco * dts, 2.0 * M_PI);
                S.nco_ok[p] = 1;
                S.wstart_prev_p[p] = wstart;
                S.fcar_prev[p] = fcar;

                // MEASURE this PRN's gain every record, ALWAYS -- that is how it converges past
                // the warmup gate below. (The first cut gated the measurement on the SUBTRACT
                // decision, which gated on gain_n >= warmup: a deadlock where the gain never
                // climbed because it was never measured until it had already converged.) The
                // measurement reads the FULL correlation the assembler adds back, so it does not
                // depend on whether we peel this record.
                // P7a predicted bit signs for this window's head/tail code periods. The
                // boundary sits (L - cp_seed)/L of a record after the window start (the same
                // convention as the peel's m_head); sample the bit table half a period either
                // side of it so edge rounding can never pick the wrong bit. 0 = unknown ->
                // the decision-directed fallback below (and in the next frame's measurement).
                int8_t s_head = 0, s_tail = 0;
                if (p < (int)navb.size() && !navb[p].bits.empty() && S.capture_utc0 > 0.0) {
                    const cudaGnssTrackState::NavBits& nb = navb[p];
                    const double t_rec =
                        (double)S.hops_per_record * (double)S.fft_len / S.sample_rate;
                    const double utc_w = S.capture_utc0 + (double)wstart / S.sample_rate;
                    const double t_b = utc_w + (L - cp_seed) / L * t_rec;
                    auto bit_at = [&](double t) -> int8_t {
                        const double x = (t - nb.utc0) / nb.bit_s;
                        if (x < 0.0)
                            return 0;
                        const size_t k = (size_t)x;
                        return (k < nb.bits.size()) ? nb.bits[k] : 0; // past horizon = unknown
                    };
                    s_head = bit_at(t_b - 0.5 * t_rec);
                    s_tail = bit_at(t_b + 0.5 * t_rec);
                }

                const size_t u = (size_t)n_rec * S.n_prn + p;
                S.used[u].run = 1;
                S.used[u].s_head = s_head;
                S.used[u].s_tail = s_tail;
                S.used[u].job0 = c.job0;
                S.used[u].phi = phi;
                S.used[u].wstart = wstart;
                S.used[u].bf = (float)((L - cp_seed) / L);
                S.used_prn[u] = p;

                // SUBTRACT only once the gain has actually converged: an unconverged gain adds
                // noise where it should remove signal, and it would corrupt the residual that is
                // supposed to MEASURE the peel.
                S.peel_skip[p] = cudaGnssTrackState::PK_WARMUP;
                if (S.gain_n[p] >= S.peel_warmup) {
                    // derotated -> replica frame: the NCO mirror + the FLL's tracked residual,
                    // extrapolated from its last update to THIS record (the measurements trail by
                    // one GPU frame; at Hz-level residuals that is a few percent of a cycle).
                    const double dt_x =
                        S.peel_prev_ok[p]
                            ? (double)(wstart - S.peel_prev_wstart[p]) / S.sample_rate
                            : 0.0;
                    const double phi_fll =
                        S.peel_phi_track[p] + 2.0 * M_PI * S.peel_f_track[p] * dt_x;
                    const std::complex<double> rot = std::polar(1.0, phi + phi_fll);
                    // Segment signs: the PREDICTED bits, when both are known. There is no usable
                    // fallback -- the decision-directed gain_sign is one GPU FRAME stale (not one
                    // record), so on 20 ms nav data it is worse than a coin flip and the
                    // subtraction ADDS power. See peel_require_bits. Pilots always predict.
                    const bool have_bits = (s_head != 0 && s_tail != 0);
                    if (have_bits || !S.peel_require_bits) {
                        const double sh = have_bits ? (double)s_head : S.gain_sign[p];
                        const double stl = have_bits ? (double)s_tail : S.gain_sign[p];
                        std::vector<std::complex<float>> ah(S.n_chan, {0.f, 0.f});
                        std::vector<std::complex<float>> at(S.n_chan, {0.f, 0.f});
                        double amp2 = 0.0;
                        for (int ci = 0; ci < S.n_chan; ++ci) {
                            if (!((mask >> ci) & 1ULL))
                                continue;
                            const std::complex<double> g(S.gain[(size_t)p * S.n_chan + ci]);
                            amp2 += std::norm(g);
                            ah[(size_t)ci] = (std::complex<float>)(g * rot * sh);
                            at[(size_t)ci] = (std::complex<float>)(g * rot * stl);
                        }
                        // GAIN-QUALITY GATE. Peeling with gain a_hat = a + e leaves e*R, so
                        // |resid|/|full| = |e|/|a| = 1/SNR(a_hat): at SNR 1 the peel is a no-op
                        // and BELOW 1 it makes the band dirtier. Estimate that SNR from what the
                        // EMA already knows -- coherent power |<a>|^2 vs total per-record power
                        // <|a|^2>; their difference is the per-record noise, and the EMA
                        // suppresses it by alpha/(2-alpha). Only peel with real margin (default
                        // SNR 3 = the residual is 1/3 of the signal, ~9.5 dB). NOTE this gate
                        // sees RANDOM gain error only; systematic sign error is the gate above.
                        // Measured in the SUMMED space (gain_sum), not per channel: the residual
                        // that matters is the coherent channel sum, so a per-channel SNR
                        // understates it by ~sqrt(n_chan) and made this gate ~3x too strict.
                        const double a_eff = S.peel_alpha / (2.0 - S.peel_alpha);
                        double sig2 = 0.0, noise_rec = 0.0;
                        const bool split = peel_gain_split(S, p, sig2, noise_rec);
                        // FAIL CLOSED. This used to default to INFINITY when the noise estimate
                        // was unavailable -- i.e. a safety gate that waved the satellite through
                        // precisely when it knew nothing. gain_dvar is set to -1 by every
                        // re-anchor reset, so the window is not rare: after each re-acquisition
                        // the peel subtracted using a gain no one had validated. Found
                        // 2026-07-26 while asking why the gate failed to reject BELOW-HORIZON
                        // noise probes (el -84..-52 deg, no signal at all) that had just been
                        // handed constructed nav bits; they sailed past a gate whose whole job
                        // is to answer "is this gain good enough to subtract with".
                        // Not peeling is always safe -- the peel is an optimisation, and the
                        // add-back keeps the tracking products exact either way. So: no noise
                        // estimate, no subtraction.
                        const double snr_gain = (split && noise_rec * a_eff > 0.0)
                                                    ? std::sqrt(sig2 / (noise_rec * a_eff))
                                                    : 0.0;
                        if (!(std::sqrt(amp2) > S.peel_min_amp))
                            S.peel_skip[p] = cudaGnssTrackState::PK_AMP;
                        else if (!(snr_gain > S.peel_min_gain_snr))
                            S.peel_skip[p] = cudaGnssTrackState::PK_SNR;
                        else {
                            S.peel_skip[p] = cudaGnssTrackState::PK_PEELING;
                            pspecs.push_back(
                                GnssCudaDespread::PeelSpec{p, cp_seed, fcar, local, ah, at});
                        }
                    } else
                        S.peel_skip[p] = cudaGnssTrackState::PK_NOBITS;
                }
            }
            specs.push_back(GnssCudaDespread::Spec{p, cp_seed, S.dll_spacing, fcar,
                                                   std::move(local)});
            spec_prn.push_back(p);
        }
        if (!specs.empty()) {
            // Window base: an ELEMENT offset into the ring row, scaled by the sample size
            // (1 byte 4+4b / 8 bytes fp32). data_stride stays in SAMPLES either way.
            const char* d_window = d_ring + (size_t)((S.next_rec * hpr) % S.ring_hops) * samp;
            // off_energy MUST carry _rows_spec: the energy array sits after the corr array, whose
            // size grows with rows/spec. Passing the default (4) here while the frame is sized and
            // read at 6 rows misaligns every energy the assembler reads -> the deep integration
            // (which normalizes by the prompt energy) silently reads garbage and never locks.
            double2* d_corr = (double2*)(d_out + off_corr(S.n_prn)) + (size_t)n_out * S.n_chan;
            double* d_energy = (double*)(d_out + off_energy(S.n_prn, S.n_chan, _rows_spec))
                               + (size_t)n_out * S.n_chan;
            // Returns ROWS (rows_spec/spec); the job arena advances by SPECS (1/spec). d_scale
            // non-null selects the 4+4b read path (null on the fp32 ring).
            if (!S.peel) {
                n_out += S.despread->enqueue_batch_device(d_window, (int)S.ring_hops, wstart,
                                                          specs, d_jobs + n_spec, d_corr,
                                                          d_energy, (void*)stream, d_scale);
            } else {
                // PEEL -> DESPREAD THE RESIDUAL -> ADD BACK, all on this stream, in order.
                // The residual is a single shared window buffer, which is safe precisely BECAUSE
                // the three are stream-ordered: record k+1's peel cannot start before record k's
                // despread has finished reading it.
                float2* d_resid = (float2*)device.get_gpu_memory(
                    _mem_resid, (size_t)S.n_chan * S.hops_per_record * sizeof(float2));
                gnss_cuda::PeelJob* d_pjobs = (gnss_cuda::PeelJob*)device.get_gpu_memory(
                    _mem_pjobs, (size_t)max_specs(S.n_prn) * sizeof(gnss_cuda::PeelJob));
                float2* d_gain = (float2*)device.get_gpu_memory(
                    _mem_gain, (size_t)2 * max_specs(S.n_prn) * S.n_chan * sizeof(float2));
                double2* d_xcorr = (double2*)device.get_gpu_memory(
                    _mem_xcorr, (size_t)4 * max_specs(S.n_prn) * S.n_chan * sizeof(double2));

                // pspecs is a SUBSET of specs (only PRNs whose gain has converged), but the peel
                // gains are indexed by PEEL job, not by spec -- so the add-back needs the gains
                // laid out per SPEC. Re-expand: unconverged PRNs get a zero gain, which the
                // add-back adds back as zero, i.e. exactly the un-peeled correlation.
                std::vector<GnssCudaDespread::PeelSpec> full;
                full.reserve(specs.size());
                for (size_t i = 0; i < specs.size(); ++i) {
                    const int pp = spec_prn[i];
                    auto it = std::find_if(pspecs.begin(), pspecs.end(),
                                           [&](const GnssCudaDespread::PeelSpec& q) {
                                               return q.p == pp;
                                           });
                    if (it != pspecs.end())
                        full.push_back(*it);
                    else
                        full.push_back(GnssCudaDespread::PeelSpec{
                            pp, specs[i].cp_seed, specs[i].doppler_hz, specs[i].covering,
                            std::vector<std::complex<float>>((size_t)S.n_chan, {0.f, 0.f}),
                            std::vector<std::complex<float>>((size_t)S.n_chan, {0.f, 0.f})});
                }
                S.despread->enqueue_peel_device(d_window, (int)S.ring_hops, wstart, full, d_pjobs,
                                                d_gain, d_resid, (void*)stream, d_scale);
                // The residual is fp32 and PACKED -- stride n_hops, no chan_scale.
                n_out += S.despread->enqueue_batch_device(
                    d_resid, S.hops_per_record, wstart, specs, d_jobs + n_spec, d_corr, d_energy,
                    (void*)stream, nullptr, d_xcorr, gnss_gpu::ROWS_PEEL);
                // Rows 0-3 leave holding the FULL un-peeled correlation; 4-5 hold the residual.
                CHECK_CUDA_ERROR(gnss_cuda::launch_peel_addback(
                    d_corr, d_energy, d_xcorr, d_gain, (int)specs.size(), S.n_chan,
                    gnss_gpu::ROWS_PEEL, stream));
            }
            n_spec += (int)specs.size();
        }
        ++S.next_rec;
        ++n_rec;
    }
    hdr.n_rec = n_rec;
    hdr.n_jobs = n_out;
    hdr.n_rows_spec = _rows_spec;
    std::memcpy(_ctl_stage.data(), &hdr, sizeof(hdr));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_out, _ctl_stage.data(), _ctl_stage.size(),
                                     cudaMemcpyHostToDevice, stream));

    // ---- PEEL: mirror this frame's (added-back) correlations back to the host, so the NEXT
    // frame can fold them into the feed-forward gain. Self-contained on purpose: everything the
    // gain loop needs is computed in this stage, so there is no cross-stage plumbing and no REST
    // hop, and the staleness is one frame (~10 ms at L1) of a quasi-static quantity.
    if (S.peel && n_out > 0) {
        if (!S.mir_evt)
            CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&S.mir_evt, cudaEventDisableTiming));
        const double* d_energy_base = (const double*)(d_out + off_energy(S.n_prn, S.n_chan,
                                                                        _rows_spec));
        const double2* d_corr_base = (const double2*)(d_out + off_corr(S.n_prn));
        const size_t n = (size_t)n_out * S.n_chan;
        CHECK_CUDA_ERROR(cudaMemcpyAsync(S.mir_corr.data(), d_corr_base, n * sizeof(double2),
                                         cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(S.mir_energy.data(), d_energy_base, n * sizeof(double),
                                         cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaEventRecord(S.mir_evt, stream));
        S.mir_rows = n_out;
        S.mir_rows_spec = _rows_spec;
        S.used_n_rec = n_rec;
        S.mir_pending = true;
    }

    // Throttled peel health line (~every 10 s at L1 frame rates): PRNs past warmup, the
    // strongest converged gain, and its FLL-tracked residual carrier. The residual carrier is
    // the live tell -- near zero when the broker's ctrim is closed, Hz-level otherwise (where
    // only the peel FLL keeps the gain coherent; see the hpp note).
    if (S.peel && ++_peel_dbg_ctr % 1000 == 0) {
        int npeel = 0, nact = 0, gmax_n = 0, gmax_p = -1;
        double gmax = 0.0;
        for (int p = 0; p < S.n_prn; ++p) {
            if (!active[p])
                continue;
            ++nact;
            double a2 = 0.0;
            for (int c = 0; c < S.n_chan; ++c)
                a2 += std::norm(S.gain[(size_t)p * S.n_chan + c]);
            if (std::sqrt(a2) > gmax) {
                gmax = std::sqrt(a2);
                gmax_n = S.gain_n[p];
                gmax_p = S.prns[p];
            }
        }
        for (int p = 0; p < S.n_prn; ++p)
            if (active[p] && S.peel_skip[p] == cudaGnssTrackState::PK_PEELING)
                ++npeel;
        int gi = -1;
        for (int p = 0; p < S.n_prn; ++p)
            if (S.prns[p] == gmax_p)
                gi = p;
        // Which active PRNs are NOT yet subtracting, and how far along their gain is -- the
        // direct answer to "why is this PRN's depth 0 dB". WARN so it clears the node's
        // log_level: warn (debug aid while the peel beds in; demote to INFO later).
        static const char* const skip_name[] = {"", "warmup", "snr", "nobits", "amp"};
        std::string lag;
        for (int p = 0; p < S.n_prn; ++p) {
            if (!active[p] || S.peel_skip[p] == cudaGnssTrackState::PK_PEELING)
                continue;
            lag += fmt::format(" {:d}:{:s}", S.prns[p], skip_name[S.peel_skip[p]]);
            if (S.peel_skip[p] == cudaGnssTrackState::PK_WARMUP)
                lag += fmt::format("(n{:d})", S.gain_n[p]);
        }
        // |resid|/|full| per PRN, from the records themselves: the number the depth SHOULD
        // reflect. ~0 = peeling hard, ~1 = subtracting nothing.
        // ...paired with the FLOOR that ratio could possibly reach. With a perfect gain the
        // residual is just this record's thermal noise, so |resid|/|full| -> 1/SNR_rec, and
        // SNR_rec is exactly what the gain EMA already separates: |<a>|^2 vs <|a|^2>. Without
        // this, a small ratio is unreadable -- "0.10" is 20 dB of peeling if the floor is 0.01
        // and NOTHING if the floor is 0.10. (Every depth error on 2026-07-24 was an unmeasured
        // floor reported as a measurement; this is that discipline made per-PRN and continuous.)
        // NOTE this is a BEST-CASE floor: it assumes the un-cancelled part is pure thermal
        // noise, so any systematic (code/carrier model error, quantizer, PFB cross-terms)
        // raises the true achievable floor above it. Printed ratio/floor, "*" = at floor.
        // Computed in the SUMMED space (gain_sum), the same space peel_ratio lives in -- a
        // per-channel estimate understates the SNR by ~sqrt(n_chan) and printed floors ABOVE
        // the ratios they were meant to bound (impossible, which is how it was caught).
        // THE INVARIANT TO CHECK THIS AGAINST: a thermal-limited residual cannot land below
        // 0.886*floor (Rayleigh mean of complex noise), so a population of ratio/floor with
        // real weight under 0.886 means the FLOOR is wrong, not the peel. That is what caught
        // the coherent-mean estimator on 2026-07-25 (min 0.35, GPS median 0.90).
        std::string ratios;
        int n_hurt = 0, n_floor = 0;
        for (int p = 0; p < S.n_prn; ++p)
            if (active[p] && S.peel_ratio[p] >= 0.0) {
                double sig2 = 0.0, nz = 0.0;
                const bool split = peel_gain_split(S, p, sig2, nz);
                const double flr = (split && sig2 > 0.0) ? std::sqrt(nz / sig2) : 0.0;
                const bool at_floor = (flr > 0.0 && S.peel_ratio[p] <= 1.3 * flr);
                if (at_floor)
                    ++n_floor;
                // ...and the LOOP STATE behind that ratio: the FLL's tracked residual and how
                // many records the gain EMA has averaged. WHY PER-PRN (2026-07-25): the pilot
                // peel is bimodal -- a sat sits at EXACTLY 0.0 dB (resid/full 1.00, floor ~1)
                // for minutes while its deep integration is perfectly healthy, then steps to
                // 26-38 dB and stays. Sequence and phase were both cleared by measurement
                // (bit_pred cross-correlates at +1.000 and matches BRDC geometry to <0.4 chips
                // on the sats that fail), so what is left is this loop -- and the line printed
                // only the STRONGEST PRN's fll, which is by construction a sat that is working.
                // A squaring discriminator (arg(prod^2)/2) has an absorbing state; that is
                // exactly the class of fault [[carrier-loop-absorbing-state]] catalogues one
                // layer down, and it is invisible until the loop state is per-PRN visible.
                // TEST A: the phase-diversity-immune ratio beside the coherent one (i...).
                // TEST B: per-channel GAIN COHERENCE g = sum_c|g_c|^2 / <sum_c|a_c|^2>. BOTH
                // terms are sums of MAGNITUDES, so cross-channel phase cannot touch either --
                // unlike sig2/floor, which live in the coherent sum. ->1 means the gain EMA is
                // clean (the per-record phasors agree); <<1 means it is genuinely decohering.
                // Together they split the remaining suspects, which the 2026-07-25 elimination
                // narrowed to exactly two (the replica itself is SHARED with the despread, and
                // the bit/overlay signs are cleared by measurement):
                //   i~1, g~1      -> gain and cancellation both fine on their own terms; the
                //                    fault is the PHASE applied at subtraction (measurement and
                //                    application disagreeing -- a CONSTANT error cannot do it,
                //                    since the gain is measured in the derotated frame and
                //                    re-applied with the same rotation, so it cancels)
                //   i~1, g<<1     -> the GAIN is decohering; the subtraction has nothing right
                //                    to apply
                //   i<<1          -> the peel WORKS; the coherent ratio and the floor were both
                //                    measuring channel geometry, not the peel (open item 0a)
                double a2g = 0.0;
                for (int c = 0; c < S.n_chan; ++c)
                    a2g += std::norm(S.gain[(size_t)p * S.n_chan + c]);
                const double gcoh = (S.gain_p2[p] > 0.0) ? a2g / S.gain_p2[p] : -1.0;
                // p = RMS phase residual (deg), d = RMS record-to-record INCREMENT (deg).
                // Fully random phase reads p=104 d=147; a slow drift reads p large, d small.
                const double DEG = 180.0 / M_PI;
                const double prms = (S.pres2[p] >= 0.0) ? DEG * std::sqrt(S.pres2[p]) : -1.0;
                const double drms = S.pres_ok[p] ? DEG * std::sqrt(S.pres_d2[p]) : -1.0;
                ratios += fmt::format(
                    " {:d}:{:.2f}/{:.2f}{:s}i{:.2f}g{:.2f}p{:.0f}d{:.0f}c{:+.2f}/{:+.2f}"
                    "s{:+.2f}/{:+.2f}x{:.2f}v{:+.2f}/{:+.2f}z{:.2f}b{:+.2f}/{:+.2f}f{:+.2f}n{:d}",
                    S.prns[p], S.peel_ratio[p], flr, at_floor ? "*" : "", S.peel_ratio_ic[p],
                    gcoh, prms, drms, S.pres_c1[p], S.pres_c2[p], S.pres_same[p],
                    S.pres_diff[p], S.pres_fdiff[p], S.pres_pos[p], S.pres_neg[p],
                    S.pres_fzero[p], S.pres_blo[p], S.pres_bhi[p],
                    S.peel_f_track[p], S.gain_n[p]);
                if (S.peel_ratio[p] > 1.05)
                    ++n_hurt;
            }
        WARN("peel[{:s}]: {:d}/{:d} past warmup; strongest PRN{:d} |g|={:.3g} n={:d} "
             "fll={:+.2f}Hz; HURTING={:d}; AT-FLOOR={:d}; NAVBITS-BAD={:d}; "
             "NOT-PEELING:{:s}; resid/full/floor:{:s}",
             S.peel_tag, npeel, nact, gmax_p, gmax, gmax_n,
             (gi >= 0) ? S.peel_f_track[gi] : 0.0, n_hurt, n_floor,
             S.nav_bits_bad.exchange(0, std::memory_order_relaxed),
             lag.empty() ? " none" : lag, ratios.empty() ? " -" : ratios);
    }

    return record_end_event();
}
