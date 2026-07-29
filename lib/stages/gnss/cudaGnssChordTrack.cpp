#include "cudaGnssChordTrack.hpp"

#include "cudaGnssChordDespread.hpp"
#include "gnssBandPlan.hpp"
#include "gnssGpuChain.hpp"
#include "gnssSignal.hpp"
#include "GnssChanMetadata.hpp"
#include "cudaUtils.hpp"
#include "kotekanLogging.hpp"
#include "pfbPrototype.hpp"

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
    dll_spacing = config.get_default<double>(unique_name, "dll_spacing", 0.5);
    doppler_margin_hz = config.get_default<double>(unique_name, "doppler_margin_hz", 5000.0);

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

    const std::string ep =
        config.get_default<std::string>(unique_name, "seed_endpoint", "/chord_track/set_seeds");
    kotekan::restServer::instance().register_post_callback(
        ep, std::bind(&cudaGnssChordTrackState::set_seeds_callback, this, _1, _2));
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

    // Snapshot the seeds once for the whole frame; the broker updates asynchronously.
    std::vector<cudaGnssChordTrackState::Seed> seeds;
    {
        std::lock_guard<std::mutex> lk(S.seed_mtx);
        seeds = S.seeds;
    }

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
            const double cp = sd.cp_chips + sd.cp_rate * dh;

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
                                                   d_energy, (void*)stream);
    }
    hdr->n_jobs = n_out_rows;

    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_out, _ctl_stage.data(), _ctl_stage.size(),
                                     cudaMemcpyHostToDevice, stream));
    return record_end_event();
}
