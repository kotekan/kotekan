#include "cudaGnssBeamDump.hpp"

#include "Telescope.hpp"     // for Telescope
#include "cudaUtils.hpp"     // for CHECK_CUDA_ERROR
#include "gnssFleetLane.hpp" // for gnss::FleetLane, gnss::fleet_lane_direction
#include "kotekanLogging.hpp"

#include <algorithm>
#include <cassert>
#include <chordMetadata.hpp>
#include <chrono>
#include <cstdlib>
#include <cstring>

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaGnssBeamDump);

namespace {
constexpr double CHAN_HZ = 0.1953125e6; // freq_id to RF, as the assemblers steer

double steady_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
} // namespace

cudaGnssBeamDump::cudaGnssBeamDump(Config& config, const std::string& unique_name,
                                   bufferContainer& host_buffers, cudaDeviceInterface& device,
                                   int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num),
    _buffer_depth(config.get<int>(unique_name, "buffer_depth")),
    _num_times(config.get<int>(unique_name, "num_times")),
    _num_elements(config.get<int>(unique_name, "num_elements")),
    _num_local_freq(config.get<int>(unique_name, "num_local_freq")),
    _voltage_name(config.get<std::string>(unique_name, "voltage_name")),
    _beam_name(config.get<std::string>(unique_name, "beam_name")),
    _local_channels(config.get<std::vector<std::int32_t>>(unique_name, "beam_local_channels")),
    _freq_ids(config.get<std::vector<std::int32_t>>(unique_name, "beam_freq_ids")),
    _ref_input(config.get<int>(unique_name, "beam_ref_input")),
    _eig_iters(config.get_default<int>(unique_name, "beam_eig_iters", 4)),
    _lane_owner(config.get_default<std::string>(unique_name, "beam_lane_owner", "")),
    _n_targets(_lane_owner.empty() ? 0 : config.get_default<int>(unique_name, "beam_targets", 2)),
    _layout(gnss_cuda::beam_dump_layout((int)_local_channels.size(), _num_times, _n_targets)),
    voltage(_voltage_name, "E",
            std::array<std::ptrdiff_t, 4>{_buffer_depth * _num_times, _num_local_freq, 2,
                                          _num_elements / 2},
            std::array<std::string, 4>{"T", "F", "P", "D"},
            std::array<std::ptrdiff_t, 4>{1, 1, 1, 1}, *this),
    beam(_beam_name, "gnss_beam", std::array<std::ptrdiff_t, 2>{1, (std::ptrdiff_t)_layout.bytes},
         std::array<std::string, 2>{"B", "X"}, std::array<std::ptrdiff_t, 2>{1, 1}, *this) {
    if (_num_elements != gnss_cuda::BEAM_INPUTS)
        FATAL_ERROR("cudaGnssBeamDump: num_elements {:d} != {:d}; the kernels are written for "
                    "128 inputs",
                    _num_elements, gnss_cuda::BEAM_INPUTS);
    if (_num_times % 256)
        FATAL_ERROR("cudaGnssBeamDump: num_times {:d} must be a multiple of 256", _num_times);
    if (_local_channels.empty() || _local_channels.size() != _freq_ids.size())
        FATAL_ERROR("cudaGnssBeamDump: beam_local_channels ({:d}) and beam_freq_ids ({:d}) must "
                    "be non-empty and the same length",
                    (int)_local_channels.size(), (int)_freq_ids.size());
    for (std::int32_t f : _local_channels)
        if (f < 0 || f >= _num_local_freq)
            FATAL_ERROR("cudaGnssBeamDump: beam_local_channels entry {:d} outside [0, {:d}) -- "
                        "LOCAL indices, not global freq_ids",
                        f, _num_local_freq);
    if (_ref_input < 0 || _ref_input >= _num_elements)
        FATAL_ERROR("cudaGnssBeamDump: beam_ref_input {:d} outside [0, {:d})", _ref_input,
                    _num_elements);
    if (_eig_iters < 1 || _eig_iters > 64)
        FATAL_ERROR("cudaGnssBeamDump: beam_eig_iters {:d} outside [1, 64]", _eig_iters);
    if (!_lane_owner.empty() && (_n_targets < 1 || _n_targets > 16))
        FATAL_ERROR("cudaGnssBeamDump: beam_targets {:d} outside [1, 16]", _n_targets);

    voltage.register_consumer();
    gpu_buffers_used.push_back(std::make_tuple(_beam_name, true, false, true));

    const int n = (int)_local_channels.size();
    // Shared scratch and the warm-start state, named here so --dry-run sees them. The state
    // is zeroed once (a cold start); every instance's constructor repeats the idempotent fill.
    const size_t cov_bytes =
        (size_t)n * gnss_cuda::BEAM_INPUTS * gnss_cuda::BEAM_INPUTS * sizeof(float2);
    const size_t q_bytes = (size_t)n * gnss_cuda::BEAM_INPUTS * sizeof(float2);
    device.get_gpu_memory(unique_name + "_cov", cov_bytes);
    void* d_q1 = device.get_gpu_memory(unique_name + "_q1", q_bytes);
    void* d_q2 = device.get_gpu_memory(unique_name + "_q2", q_bytes);
    void* d_sel = device.get_gpu_memory(unique_name + "_sel", (size_t)n * sizeof(int));
    if (instance_num == 0) {
        CHECK_CUDA_ERROR(cudaMemset(d_q1, 0, q_bytes));
        CHECK_CUDA_ERROR(cudaMemset(d_q2, 0, q_bytes));
        CHECK_CUDA_ERROR(cudaMemcpy(d_sel, _local_channels.data(), (size_t)n * sizeof(int),
                                    cudaMemcpyHostToDevice));
    }

    _hdr_stage.assign(_layout.off_freq + sizeof(std::int32_t) * n, 0);
    auto* hdr = reinterpret_cast<gnss_cuda::BeamDumpHdr*>(_hdr_stage.data());
    hdr->magic = gnss_cuda::BEAM_DUMP_MAGIC;
    hdr->version = gnss_cuda::BEAM_DUMP_VERSION;
    hdr->n_chan = n;
    hdr->n_times = _num_times;
    hdr->n_inputs = _num_elements;
    hdr->ref_input = _ref_input;
    hdr->conj = 0;
    hdr->eig_iters = _eig_iters;
    hdr->n_targets = _n_targets;
    std::memcpy(_hdr_stage.data() + _layout.off_freq, _freq_ids.data(), sizeof(std::int32_t) * n);

    if (_n_targets > 0) {
        _lane_prefix = _lane_owner + "/lane/";
        _lane_group = _lane_owner.substr(0, _lane_owner.find('_'));
        _lane_max_age_s = config.get_default<double>(unique_name, "beam_lane_max_age_s", 2.0);
        _target_hold_s = config.get_default<double>(unique_name, "beam_target_hold_s", 2.0);
        _target_gain = config.get_default<double>(unique_name, "beam_target_gain", 1.5);
        _target_min_coh = config.get_default<double>(unique_name, "beam_target_min_coh", 0.15);
        _lane_conj = config.get_default<bool>(unique_name, "beam_lane_conj", false);
        auto pos = config.get<std::vector<double>>(unique_name, "elem_positions_enu");
        if ((int)pos.size() != 3 * gnss_cuda::BEAM_INPUTS)
            FATAL_ERROR("cudaGnssBeamDump: elem_positions_enu has {:d} values, need {:d} (3 per "
                        "input)",
                        (int)pos.size(), 3 * gnss_cuda::BEAM_INPUTS);
        const double sign = config.get_default<double>(unique_name, "elem_steer_sign", 1.0);
        std::vector<double> f_mhz;
        for (std::int32_t fid : _freq_ids)
            f_mhz.push_back(fid * CHAN_HZ * 1e-6);
        _geo = gnss::ElemSteer(std::move(pos), std::move(f_mhz), 0, sign, 1.0);
        // The frame times that place the re-steer: the assembler stamps its rows with
        // cudaGnssChordTrack's record epoch (config frame0_utc, else the telescope's time0).
        const double cfg_frame0 = config.get_default<double>(unique_name, "frame0_utc", 0.0);
        const auto& tel = Telescope::instance();
        if (cfg_frame0 > 0.0)
            _frame0_offset_s = cfg_frame0 - (double)tel.to_time_ns(0) * 1e-9;
        _time_ok = cfg_frame0 > 0.0 || tel.gps_time_enabled();
        if (!_time_ok)
            WARN("cudaGnssBeamDump: no record epoch (no telescope GPS time, no frame0_utc): "
                 "the target beams cannot be placed in time; the first frame will stop");
        _target_prn.assign((size_t)_n_targets, 0);
        _tgt_stage.assign(_layout.off_tbeam - _layout.off_tgt, 0);
        _dir.assign((size_t)gnss_cuda::BEAM_INPUTS, std::complex<double>(0.0, 0.0));
        INFO("cudaGnssBeamDump: {:d} target beams from the lanes of '{:s}' (board group '{:s}'), "
             "hold {:.1f} s at gain {:.2f}, coherence >= {:.2f}, conj {}, +{:.2f} MB/frame",
             _n_targets, _lane_owner, _lane_group, _target_hold_s, _target_gain, _target_min_coh,
             _lane_conj, (_layout.bytes - _layout.off_tgt) / 1.0e6);
    }

    set_command_type(gpuCommandType::KERNEL);
    INFO("cudaGnssBeamDump: {:d} channels (freq_ids {:d}..{:d}) x {:d} samples into '{:s}' "
         "({:.2f} MB/frame), reference input {:d}, {:d} power iterations",
         n, _freq_ids.front(), _freq_ids.back(), _num_times, _beam_name, _layout.bytes / 1.0e6,
         _ref_input, _eig_iters);
}

cudaGnssBeamDump::~cudaGnssBeamDump() {}

int cudaGnssBeamDump::wait_on_precondition() {
    const int errcode = voltage.wait_and_claim_readable([&](const std::ptrdiff_t available) {
        if (available < _num_times)
            return read_descriptor_t{.claimed = 0, .read = 0};
        else
            return read_descriptor_t{.claimed = _num_times, .read = _num_times};
    });
    return errcode < 0 ? errcode : 0;
}

double cudaGnssBeamDump::frame_utc(long long seq) const {
    return (double)Telescope::instance().to_time_ns((uint64_t)seq) * 1e-9 + _frame0_offset_s;
}

int cudaGnssBeamDump::strongest_outsider(const std::map<int, double>& score) const {
    int best = 0;
    double best_p = 0.0;
    for (const auto& [prn, p] : score) {
        if (std::find(_target_prn.begin(), _target_prn.end(), prn) != _target_prn.end())
            continue;
        if (best == 0 || p > best_p) {
            best = prn;
            best_p = p;
        }
    }
    return best;
}

void cudaGnssBeamDump::choose_targets(const std::map<int, double>& score, double now_s) {
    // A target whose lane left the board (the slot stopped or changed PRN) or lost its
    // coherence frees its slot.
    for (int& t : _target_prn)
        if (t != 0 && !score.count(t)) {
            INFO("cudaGnssBeamDump: target PRN {:d} dropped: its lane left the board or lost "
                 "coherence",
                 t);
            t = 0;
        }
    // Empty slots take the strongest lanes not yet targeted.
    for (int& t : _target_prn) {
        if (t != 0)
            continue;
        const int best = strongest_outsider(score);
        if (best == 0)
            break;
        INFO("cudaGnssBeamDump: target PRN {:d} (coherent power {:.3g})", best, score.at(best));
        t = best;
    }
    // The strongest outsider replaces the weakest target once it has out-scored it by
    // beam_target_gain for beam_target_hold_s: the top-2 by coherent power, with hysteresis
    // so a handover does not flap frame to frame.
    const int best = strongest_outsider(score);
    int weak = -1;
    for (int k = 0; k < _n_targets; k++) {
        const int t = _target_prn[(size_t)k];
        if (t != 0 && (weak < 0 || score.at(t) < score.at(_target_prn[(size_t)weak])))
            weak = k;
    }
    if (best == 0 || weak < 0
        || score.at(best) <= _target_gain * score.at(_target_prn[(size_t)weak])) {
        _challenger_prn = 0;
        return;
    }
    if (_challenger_prn != best) {
        _challenger_prn = best;
        _challenger_since = now_s;
        return;
    }
    if (now_s - _challenger_since < _target_hold_s)
        return;
    INFO("cudaGnssBeamDump: target PRN {:d} -> {:d} (coherent power {:.3g} -> {:.3g})",
         _target_prn[(size_t)weak], best, score.at(_target_prn[(size_t)weak]), score.at(best));
    _target_prn[(size_t)weak] = best;
    _challenger_prn = 0;
}

void cudaGnssBeamDump::build_targets(long long seq0) {
    const int n = (int)_local_channels.size();
    const int ni = gnss_cuda::BEAM_INPUTS;
    const double now_s = steady_s();
    if (!_time_ok)
        FATAL_ERROR("cudaGnssBeamDump: no record epoch for the target beams (telescope GPS time "
                    "off and no frame0_utc)");
    const double t_mid = frame_utc(seq0 + _num_times / 2);
    // This frame's lanes of the owner chain, per PRN and selected channel. The entries are
    // copied out under the board's lock: a few hundred bytes each.
    _lanes.clear();
    const gnss::ProjBoard& board = gnss::ProjBoard::instance();
    for (int c = 0; c < n; c++) {
        board.for_each(_lane_group, _freq_ids[(size_t)c], unique_name, now_s, _lane_max_age_s,
                       [&](const gnss::ProjEntry& e) {
                           if (e.src != 2 || e.xfer != 0
                               || e.owner.compare(0, _lane_prefix.size(), _lane_prefix) != 0)
                               return;
                           const int prn = std::atoi(e.owner.c_str() + _lane_prefix.size());
                           if (prn <= 0)
                               return;
                           std::vector<gnss::ProjEntry>& rows = _lanes[prn];
                           if (rows.empty())
                               rows.resize((size_t)n);
                           rows[(size_t)c] = e;
                       });
    }
    // Rank the lanes by coherent power, coherence x row power, over the channels. The row
    // power alone is noise-dominated and nearly equal for every lane, so a held lane of a
    // satellite that never locked would outrank a locked one. A lane below
    // beam_target_min_coh on every channel is not a target.
    std::map<int, double> score;
    for (const auto& [prn, rows] : _lanes) {
        double s = -1.0;
        for (const gnss::ProjEntry& e : rows)
            if (e.k > 0 && e.coh >= _target_min_coh)
                s = std::max(s, e.coh * e.power);
        if (s >= 0.0)
            score[prn] = s;
    }
    choose_targets(score, now_s);

    // Each target's direction on each channel, re-steered to the frame's mid-time along the
    // lane's track (the entry is on the channel itself, so no instrumental-delay transfer).
    std::fill(_tgt_stage.begin(), _tgt_stage.end(), 0);
    auto* tgt = reinterpret_cast<gnss_cuda::BeamDumpTarget*>(_tgt_stage.data());
    auto* tw = reinterpret_cast<float2*>(_tgt_stage.data() + (_layout.off_tw - _layout.off_tgt));
    gnss::FleetLane L;
    for (int t = 0; t < _n_targets; t++) {
        const int prn = _target_prn[(size_t)t];
        if (prn == 0)
            continue;
        const std::vector<gnss::ProjEntry>& rows = _lanes[prn];
        for (int c = 0; c < n; c++) {
            const gnss::ProjEntry& e = rows[(size_t)c];
            if (e.k <= 0)
                continue;
            if (e.n != ni) {
                if (!_warned_n_elem)
                    WARN("cudaGnssBeamDump: lane '{:s}' has {:d} elements, the ring {:d}: the "
                         "assembler's element axis must be the ring's inputs; no target beams",
                         e.owner, e.n, ni);
                _warned_n_elem = true;
                continue;
            }
            L.f_src_hz = _freq_ids[(size_t)c] * CHAN_HZ;
            L.t_utc = e.t_utc;
            std::copy(e.los, e.los + 3, L.los);
            std::copy(e.los_rate, e.los_rate + 3, L.los_rate);
            L.q.assign(e.q.begin(), e.q.begin() + ni);
            if (!gnss::fleet_lane_direction(L, _geo, L.f_src_hz, t_mid, nullptr, _sh, _dir.data()))
                continue;
            float2* w = tw + ((size_t)t * n + c) * ni;
            for (int i = 0; i < ni; i++) {
                const std::complex<double> d =
                    _lane_conj ? std::conj(_dir[(size_t)i]) : _dir[(size_t)i];
                w[i] = make_float2((float)d.real(), (float)d.imag());
            }
            gnss_cuda::BeamDumpTarget& T = tgt[(size_t)t * n + c];
            T.prn = prn;
            T.held = e.held_s > 0.0 ? 1 : 0;
            T.coh = (float)e.coh;
            T.power = (float)e.power;
            T.held_s = (float)e.held_s;
            T.age_s = e.t_utc > 0.0 ? (float)(t_mid - e.t_utc) : 0.0f;
        }
    }
}

cudaEvent_t cudaGnssBeamDump::execute(cudaPipelineState& pipestate,
                                      const std::vector<cudaEvent_t>&) {
    (void)pipestate;
    pre_execute();
    record_start_event();
    const cudaStream_t stream = device.getStream(cuda_stream_id);
    const int n = (int)_local_channels.size();

    voltage.check_metadata();
    const std::shared_ptr<const chordMetadata> meta = voltage.get_metadata();
    if (!_checked_freqs && !meta->has_coarse_freq()) {
        WARN("cudaGnssBeamDump: the voltage ring carries no coarse frequencies (test data?); "
             "beam_freq_ids are taken on trust");
        _checked_freqs = true;
    }
    if (!_checked_freqs) {
        // The config's freq_ids were resolved at generation time; the ring's labels are the
        // authority. A mismatch means a rewired comb, and a beam on the wrong channel is worse
        // than none.
        const std::vector<int> coarse = meta->get_coarse_freq();
        for (int c = 0; c < n; c++) {
            const int f = _local_channels[(size_t)c];
            if (f >= (int)coarse.size() || coarse[(size_t)f] != _freq_ids[(size_t)c])
                FATAL_ERROR("cudaGnssBeamDump: local channel {:d} carries freq_id {:d}, config "
                            "expects {:d}",
                            f, f < (int)coarse.size() ? coarse[(size_t)f] : -1,
                            _freq_ids[(size_t)c]);
        }
        _checked_freqs = true;
    }
    const long long seq0 = meta->get_fpga_seq_num()
                           + voltage.get_read_valid().begin() * meta->get_time_downsampling_fpga();
    beam.set_metadata(meta);
    beam.get_metadata()->set_fpga_seq_num(seq0);

    const std::ptrdiff_t time_offset =
        voltage.get_read_valid().begin() % voltage.get_ndarray().extent(0);
    assert((voltage.get_read_valid().end() - 1) % voltage.get_ndarray().extent(0) >= time_offset);
    const auto* d_in =
        reinterpret_cast<const unsigned char*>(&voltage.get_ndarray()(time_offset, 0, 0, 0));
    unsigned char* d_out = beam.get_ndarray().data();

    const size_t cov_bytes =
        (size_t)n * gnss_cuda::BEAM_INPUTS * gnss_cuda::BEAM_INPUTS * sizeof(float2);
    const size_t q_bytes = (size_t)n * gnss_cuda::BEAM_INPUTS * sizeof(float2);
    auto* d_cov = (float2*)device.get_gpu_memory(unique_name + "_cov", cov_bytes);
    auto* d_q1 = (float2*)device.get_gpu_memory(unique_name + "_q1", q_bytes);
    auto* d_q2 = (float2*)device.get_gpu_memory(unique_name + "_q2", q_bytes);
    auto* d_sel = (const int*)device.get_gpu_memory(unique_name + "_sel", (size_t)n * sizeof(int));

    // Header + freq_ids. The staging is pageable, so this copy returns once the bytes are
    // taken, and the vector is reusable next frame.
    auto* hdr = reinterpret_cast<gnss_cuda::BeamDumpHdr*>(_hdr_stage.data());
    hdr->seq0 = seq0;
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_out, _hdr_stage.data(), _hdr_stage.size(),
                                     cudaMemcpyHostToDevice, stream));

    gnss_cuda::launch_beam_cov(d_in, d_sel, n, _num_local_freq, _num_times, d_cov, stream);
    gnss_cuda::launch_beam_eig(d_cov, n, _num_times, _ref_input, _eig_iters, d_q1, d_q2,
                               (float2*)(d_out + _layout.off_w), (float*)(d_out + _layout.off_pw),
                               (float*)(d_out + _layout.off_lam), stream);
    gnss_cuda::launch_beam_form(d_in, d_sel, n, _num_local_freq, _num_times,
                                (const float2*)(d_out + _layout.off_w), d_out + _layout.off_beam,
                                stream);
    if (_n_targets > 0) {
        build_targets(seq0);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_out + _layout.off_tgt, _tgt_stage.data(),
                                         _tgt_stage.size(), cudaMemcpyHostToDevice, stream));
        const size_t w_per = (size_t)n * gnss_cuda::BEAM_INPUTS * sizeof(float2);
        const size_t b_per = (size_t)n * _num_times * 4;
        for (int t = 0; t < _n_targets; t++)
            gnss_cuda::launch_beam_form(d_in, d_sel, n, _num_local_freq, _num_times,
                                        (const float2*)(d_out + _layout.off_tw + t * w_per),
                                        d_out + _layout.off_tbeam + t * b_per, stream);
    }
    if ((++_frames & 0x3FF) == 1) {
        std::string targets;
        for (int t : _target_prn)
            targets += " " + std::to_string(t);
        INFO("cudaGnssBeamDump: frame seq0 {:d}, {:d} channels{:s}{:s}", seq0, n,
             _n_targets > 0 ? ", target PRNs" : "", targets);
    }
    return record_end_event();
}

void cudaGnssBeamDump::finalize_frame() {
    voltage.finish_read();
    cudaCommand::finalize_frame();
}
