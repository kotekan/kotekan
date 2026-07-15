#include "cudaGnssTrack.hpp"

#include "GnssChanMetadata.hpp"
#include "cudaGnssDespreadKernel.hpp"
#include "cudaUtils.hpp" // for CHECK_CUDA_ERROR
#include "gnssGpuChain.hpp"
#include "gnssRecord.hpp"

#include <cmath>
#include <cstring>
#include <set>

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND_WITH_STATE(cudaGnssTrack, cudaGnssTrackState);

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

    // Broker seed contract == GnssChannelizedTracker::set_seeds_callback.
    const std::string ep =
        config.get_default<std::string>(unique_name, "seed_endpoint", "/track/set_seeds");
    using namespace std::placeholders;
    kotekan::restServer::instance().register_post_callback(
        ep, std::bind(&cudaGnssTrackState::set_seeds_callback, this, _1, _2));
}

cudaGnssTrackState::~cudaGnssTrackState() = default;

void cudaGnssTrackState::set_seeds_callback(kotekan::connectionInstance& conn,
                                            nlohmann::json& request) {
    try {
        std::lock_guard<std::mutex> lk(seed_mtx);
        std::fill(active.begin(), active.end(), (uint8_t)0); // list is authoritative
        for (const auto& s : request) {
            const int prn = s.at("prn").get<int>();
            for (int i = 0; i < n_prn; ++i)
                if (prns[i] == prn) {
                    dop[i] = s.at("doppler_hz").get<double>();
                    cp[i] = s.at("code_phase_chips").get<double>();
                    cp_rate[i] = s.value("code_phase_rate", 0.0);
                    dop_rate[i] = s.value("doppler_rate_hz_s", 0.0);
                    ctrim[i] = s.value("carrier_trim_hz", 0.0);
                    ref_hop[i] = s.value("ref_hop", (long long)0);
                    active[i] = 1;
                }
        }
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
    _n_hops_frame = (int)(_in_frame_len / ((size_t)s->n_chan * 2 * sizeof(float)));
    _out_frame_len = gnss_gpu::frame_bytes(s->n_prn, s->n_chan);
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
    device.get_gpu_memory(_mem_ring, (size_t)s->n_chan * s->ring_hops * sizeof(float2));
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

    float2* d_ring = (float2*)device.get_gpu_memory(
        _mem_ring, (size_t)S.n_chan * S.ring_hops * sizeof(float2));
    const float2* d_frame = (const float2*)device.get_gpu_memory_array(
        _gpu_mem_input, pipestate.gpu_frame_id, _gpu_buffer_depth, _in_frame_len);
    char* d_out = (char*)device.get_gpu_memory_array(_gpu_mem_output, pipestate.gpu_frame_id,
                                                     _gpu_buffer_depth, _out_frame_len);
    gnss_cuda::DespreadJob* d_jobs = (gnss_cuda::DespreadJob*)device.get_gpu_memory_array(
        _mem_jobs, pipestate.gpu_frame_id, _gpu_buffer_depth,
        (size_t)gnss_gpu::max_jobs(S.n_prn) * sizeof(gnss_cuda::DespreadJob));

    // Absolute hop of this frame from the claimed metadata (contiguous fallback without it).
    long long frame_hop = S.next_hop;
    auto meta = std::dynamic_pointer_cast<GnssChanMetadata>(
        device.get_gpu_memory_array_metadata(_gpu_mem_input, pipestate.gpu_frame_id));
    if (meta && meta->sample_seq >= 0)
        frame_hop = meta->sample_seq / S.fft_len;

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
        CHECK_CUDA_ERROR(cudaMemsetAsync(d_ring, 0,
                                         (size_t)S.n_chan * S.ring_hops * sizeof(float2),
                                         stream));
        if (S.ring_init)
            WARN("cudaGnssTrack: ring rebase (gap {:d} hops) -- window tiling re-anchored",
                 gap);
        S.hop0 = frame_hop;
        S.next_hop = frame_hop;
        S.next_rec = 0;
        S.ring_init = true;
        gap = 0;
    } else if (gap > 0) {
        CHECK_CUDA_ERROR(gnss_cuda::launch_ring_zero(d_ring, S.n_chan, S.ring_hops,
                                                     S.next_hop - S.hop0, gap, stream));
    }
    CHECK_CUDA_ERROR(gnss_cuda::launch_chan_ingest(d_frame, d_ring, _n_hops_frame, S.n_chan,
                                                   S.ring_hops, frame_hop - S.hop0, stream));
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
    {
        std::lock_guard<std::mutex> lk(S.seed_mtx);
        dop = S.dop;
        cp = S.cp;
        cp_rate = S.cp_rate;
        dop_rate = S.dop_rate;
        ctrim = S.ctrim;
        ref_hop = S.ref_hop;
        active = S.active;
    }

    const double L = (double)S.replica->code_length();
    const double chip_component = S.replica->eff_chip_rate() / (double)S.replica->comb_mult();
    const double sgn = S.replica->code_doppler_sign;
    const double fc = S.replica->carrier_hz();
    int n_rec = 0, n_jobs = 0;
    while ((S.next_rec + 1) * hpr <= written && n_rec < MAX_REC) {
        const long long whop = S.hop0 + S.next_rec * hpr;
        const long long wstart = whop * (long long)S.fft_len;
        winstart[n_rec] = wstart;
        std::vector<GnssCudaDespread::Spec> specs;
        std::vector<int> spec_prn;
        for (int p = 0; p < S.n_prn; ++p) {
            PrnCtl& c = pctl[(size_t)n_rec * S.n_prn + p];
            c.job0 = -1;
            if (!active[p]) {
                S.f_ref[p] = std::nan(""); // forget the loop; re-acquire when it returns
                continue;
            }
            // Covering channels at the seed Doppler (GLOBAL ids -> this subband's local set).
            const auto cover = S.replica->covering_bins(dop[p], S.doppler_margin_hz);
            std::vector<int> local;
            uint64_t mask = 0;
            for (int ch : cover)
                if (ch >= S.chan_offset && ch < S.chan_offset + S.n_chan) {
                    local.push_back(ch - S.chan_offset);
                    mask |= (1ULL << (ch - S.chan_offset));
                }
            if (local.empty())
                continue; // carrier not in this subband

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
            c.job0 = n_jobs + 4 * (int)specs.size(); // E, P, L, P_HEAD per spec
            c.fcar_report = (float)(fcar - ff_hz + ctrim[p]);
            c.n_owned = (float)local.size();
            c.cp_seed = cp_seed;
            c.f_nco = ctrim[p] + ff_hz;
            c.chan_mask = mask;
            c.energy_scale = 1.0;
            c.fcar = fcar; // replica f_ref -> the assembler's commanded-carrier-phase export
            specs.push_back(GnssCudaDespread::Spec{p, cp_seed, S.dll_spacing, fcar,
                                                   std::move(local)});
            spec_prn.push_back(p);
        }
        if (!specs.empty()) {
            const float2* d_window = d_ring + (size_t)((S.next_rec * hpr) % S.ring_hops);
            double2* d_corr =
                (double2*)(d_out + off_corr(S.n_prn)) + (size_t)n_jobs * S.n_chan;
            double* d_energy =
                (double*)(d_out + off_energy(S.n_prn, S.n_chan)) + (size_t)n_jobs * S.n_chan;
            n_jobs += S.despread->enqueue_batch_device(d_window, (int)S.ring_hops, wstart,
                                                       specs, d_jobs + n_jobs, d_corr,
                                                       d_energy, (void*)stream);
        }
        ++S.next_rec;
        ++n_rec;
    }
    hdr.n_rec = n_rec;
    hdr.n_jobs = n_jobs;
    std::memcpy(_ctl_stage.data(), &hdr, sizeof(hdr));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_out, _ctl_stage.data(), _ctl_stage.size(),
                                     cudaMemcpyHostToDevice, stream));

    return record_end_event();
}
