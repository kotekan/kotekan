#include "cudaGnssInject.hpp"

#include "cudaGnssChordDespread.hpp"  // for launch_pack44
#include "cudaGnssDespreadKernel.hpp" // for DespreadJob
#include "cudaGnssPsiSynth.hpp"       // for PsiJob (the Psi path's per-record slots)
#include "cudaUtils.hpp"              // for CHECK_CUDA_ERROR
#include "gnssGpuChain.hpp"           // for max_specs
#include "gnssSeedTransport.hpp"      // for propagate_seed
#include "kotekanLogging.hpp"
#include "kotekanTrackers.hpp" // for KotekanTrackers

#include <algorithm> // for find
#include <array>
#include <chordMetadata.hpp>
#include <cmath>
#include <cstring>
#include <memory> // for shared_ptr

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND_WITH_STATE(cudaGnssInject, cudaGnssChordTrackState);

cudaGnssInject::cudaGnssInject(Config& config, const std::string& unique_name,
                               bufferContainer& host_buffers, cudaDeviceInterface& device,
                               int instance_num, std::shared_ptr<cudaCommandState> state) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, state, "cudaGnssInject",
                "cudaGnssInject"),
    _hosted(st()->hosted()), _buffer_depth(config.get<int>(unique_name, "buffer_depth")),
    _num_times(config.get<int>(unique_name, "num_times")),
    _num_elements(config.get<int>(unique_name, "num_elements")),
    _num_local_freq(config.get<int>(unique_name, "num_local_freq")),
    _voltage_name(config.get<std::string>(unique_name, "voltage_name")),
    _gnss_synth_name(config.get<std::string>(unique_name, "gnss_synth_name")),
    _synth_lane_pitch(config.get<int>(unique_name, "synth_lane_pitch")),
    _gnss_synth_channels(config.get<std::vector<std::int32_t>>(unique_name, "gnss_synth_channels")),
    voltage(_voltage_name, "E",
            std::array<std::ptrdiff_t, 4>{_buffer_depth * _num_times, _num_local_freq, 2,
                                          _num_elements / 2},
            std::array<std::string, 4>{"T", "F", "P", "D"},
            std::array<std::ptrdiff_t, 4>{1, 1, 1, 1}, *this) {
    // The synth array's channel axis is the correlator's comb (gnss_synth_compact).
    _synth_n_chan = (int)_gnss_synth_channels.size();

    size_t slot2spec_ints = 0;
    for (cudaGnssChordTrackState::Row& sr : st()->rows) {
        const gnss::TrackChain& S = *sr.chain;
        const std::string& path = sr.host.cfg_path;
        const gnss_gpu::GpuRow& g = sr.gpu;
        Row row;
        row.chain = sr.chain.get();
        row.gpu = g;
        row.name = sr.host.name;
        row.label = _hosted ? sr.host.key + ": " : "";
        row.num_synth = g.name.empty() ? config.get<int>(path, "num_synth") : g.lanes;
        row.lanes_per_slot = config.get_default<int>(path, "lanes_per_slot", gnss_gpu::LANES_FULL);
        row.local_channels =
            g.name.empty() ? config.get<std::vector<std::int32_t>>(path, "gnss_local_channels")
                           : g.channels;
        row.lane_base = g.name.empty() ? config.get<int>(path, "synth_lane_base") : g.lane_base;

        if (row.lanes_per_slot != gnss_gpu::LANES_FULL
            && row.lanes_per_slot != gnss_gpu::LANES_PROMPT)
            FATAL_ERROR("cudaGnssInject: {:s}lanes_per_slot {:d} must be {:d} (E/P/L/P_HEAD "
                        "lanes) or {:d} (P/P_HEAD lanes, E and L rotated from the prompt on the "
                        "host)",
                        row.label, row.lanes_per_slot, gnss_gpu::LANES_FULL,
                        gnss_gpu::LANES_PROMPT);
        if (row.lanes_per_slot == gnss_gpu::LANES_PROMPT && !S.epl_from_prompt)
            FATAL_ERROR("cudaGnssInject: {:s}lanes_per_slot {:d} needs epl_from_prompt: the E "
                        "and L rows are the prompt rotated by what the kernel applied, which only "
                        "that path computes",
                        row.label, row.lanes_per_slot);
        if (row.lanes_per_slot == gnss_gpu::LANES_PROMPT && S.n_chan > gnss_gpu::MAX_ROT_CHAN)
            FATAL_ERROR("cudaGnssInject: {:s}{:d} covering channels exceed the {:d} PrnCtl::rot "
                        "carries; this chain must pack {:d} lanes per slot",
                        row.label, S.n_chan, gnss_gpu::MAX_ROT_CHAN, gnss_gpu::LANES_FULL);
        if (row.lanes_per_slot * S.n_prn > row.num_synth)
            FATAL_ERROR("cudaGnssInject: {:s}{:d}*n_prn ({:d}) exceeds num_synth ({:d}) -- "
                        "shrink the PRN list or widen the synth axis.",
                        row.label, row.lanes_per_slot, row.lanes_per_slot * S.n_prn, row.num_synth);
        if ((int)row.local_channels.size() != S.n_chan)
            FATAL_ERROR("cudaGnssInject: {:s}gnss_local_channels has {:d} entries but "
                        "n_channels is {:d} -- one LOCAL frame index per covering channel, in "
                        "local order.",
                        row.label, (int)row.local_channels.size(), S.n_chan);
        for (std::int32_t f : row.local_channels)
            if (f < 0 || f >= _num_local_freq)
                FATAL_ERROR("cudaGnssInject: {:s}gnss_local_channels entry {:d} outside "
                            "[0, num_local_freq {:d}) -- LOCAL indices, not global freq_ids.",
                            row.label, f, _num_local_freq);
        if (S.n_hops_frame != _num_times)
            FATAL_ERROR("cudaGnssInject: {:s}state samples_per_data_set {:d} != num_times {:d}",
                        row.label, S.n_hops_frame, _num_times);
        // The lane range must be whole tile rows (16 lanes) inside the shared axis, or the
        // correlator's tile arithmetic and this pack disagree about which lane is whose.
        if (row.lane_base < 0 || row.lane_base % 16 || _synth_lane_pitch % 16
            || row.lane_base + row.num_synth > _synth_lane_pitch)
            FATAL_ERROR("cudaGnssInject: {:s}lanes [{:d}, {:d}) do not fit a synth lane axis of "
                        "{:d} on 16-lane tile rows (synth_lane_base / synth_lane_pitch / "
                        "num_synth)",
                        row.label, row.lane_base, row.lane_base + row.num_synth, _synth_lane_pitch);
        // Each covering channel maps to its index on the synth array's channel axis.
        for (std::int32_t f : row.local_channels) {
            const auto it = std::find(_gnss_synth_channels.begin(), _gnss_synth_channels.end(), f);
            if (it == _gnss_synth_channels.end())
                FATAL_ERROR("cudaGnssInject: {:s}covering channel {:d} is not on the synth "
                            "array's channel axis (gnss_synth_channels) -- the correlator would "
                            "never read this chain's replica there",
                            row.label, f);
            row.chan_map.push_back((int)(it - _gnss_synth_channels.begin()));
        }

        // Control block: the epl layout without corr (see the header), sized for every PRN
        // active in every record (gnss_gpu::max_jobs).
        const int n_rec = S.n_hops_frame / S.hops_per_record;
        row.ctl_off_energy =
            gnss_gpu::off_prnctl() + sizeof(gnss_gpu::PrnCtl) * gnss_gpu::MAX_REC * S.n_prn;
        row.ctl_bytes =
            row.ctl_off_energy
            + sizeof(double) * gnss_gpu::max_jobs(S.n_prn, gnss_gpu::ROWS_PLAIN) * S.n_chan;
        row.ctl_offset = 0;
        if (g.name.empty()) {
            _mem_ctl = config.get<std::string>(path, "gnss_ctl_name");
            _ctl_array_bytes = row.ctl_bytes;
        } else {
            if (g.n_prn != S.n_prn || g.ctl_bytes != row.ctl_bytes)
                FATAL_ERROR("cudaGnssInject: gnss_rows {:s} has {:d} PRN slots and {:d} control "
                            "bytes, this chain {:d} and {:d}",
                            g.name, g.n_prn, g.ctl_bytes, S.n_prn, row.ctl_bytes);
            const gnss_gpu::GpuRow last = gnss_gpu::gpu_rows(config, path).back();
            _mem_ctl = config.get<std::string>(path, "gnss_rows_ctl_name");
            row.ctl_offset = g.ctl_offset;
            _ctl_array_bytes = last.ctl_offset + last.ctl_bytes;
        }
        slot2spec_ints += (size_t)n_rec * S.n_prn;

        // Taken in execute(); attribute them to this stage now (see gpuMemoryClaims.hpp).
        std::vector<std::string> arrays = {"_jobs", "_slot2spec", "_energy", "_energy0",
                                           "_chan_map"};
        if (S.psi_synth)
            arrays.insert(arrays.end(), {"_pjobs", "_theta"});
        else
            arrays.push_back("_wave");
        for (const std::string& a : arrays)
            device.register_gpu_memory_name(row.name + a);
        if (_hosted) {
            row.execute_time = kotekan::KotekanTrackers::instance().add_tracker(
                row.name, get_name() + "_execute_time", "seconds");
            row.utilization =
                kotekan::KotekanTrackers::instance().add_tracker(row.name, get_name() + "_u", "");
        }

        INFO("cudaGnssInject: {:s}{:d} PRN slots x {:d} lanes into '{:s}' [{:d}][{:d}][{:d}] at "
             "lanes [{:d}, {:d}), {:d} records/frame, channels {:d}; synthesis: {:s}",
             row.label, S.n_prn, row.lanes_per_slot, _gnss_synth_name, _num_times, _synth_n_chan,
             _synth_lane_pitch, row.lane_base, row.lane_base + row.num_synth, n_rec, S.n_chan,
             S.psi_synth ? "Psi code-phase tables, fused with the pack (launch_psi_pack)"
                         : "Phi chip walk (launch_waveform + launch_pack44)");
        _rows.push_back(row);
    }
    device.register_gpu_memory_name(_mem_ctl);

    voltage.register_consumer();

    // The rows' blocks are one span of the control array: cudaGnssChordTrackState keeps hosted
    // rows in gnss_rows order, which gpu_rows checks is the array's.
    _ctl_begin = _rows.front().ctl_offset;
    _ctl_span = _rows.back().ctl_offset + _rows.back().ctl_bytes - _ctl_begin;
    CHECK_CUDA_ERROR(cudaHostAlloc((void**)&_stage, _ctl_span + slot2spec_ints * sizeof(int),
                                   cudaHostAllocDefault));
    int* slot2spec = (int*)(_stage + _ctl_span);
    for (Row& row : _rows) {
        row.slot2spec = slot2spec;
        slot2spec +=
            (size_t)row.chain->n_hops_frame / row.chain->hops_per_record * row.chain->n_prn;
    }

    set_command_type(gpuCommandType::KERNEL);
}

cudaGnssInject::~cudaGnssInject() {
    cudaFreeHost(_stage);
}

cudaGnssChordTrackState* cudaGnssInject::st() {
    return static_cast<cudaGnssChordTrackState*>(command_state.get());
}

int cudaGnssInject::wait_on_precondition() {
    // The claim is the time reference: this command produces the synth window for exactly the
    // voltage window the correlator (a sibling consumer of the same ring) will process.
    const int errcode = voltage.wait_and_claim_readable([&](const std::ptrdiff_t available) {
        if (available < _num_times)
            return read_descriptor_t{.claimed = 0, .read = 0};
        else
            return read_descriptor_t{.claimed = _num_times, .read = _num_times};
    });
    return errcode < 0 ? errcode : 0;
}

cudaEvent_t cudaGnssInject::execute(cudaPipelineState& pipestate, const std::vector<cudaEvent_t>&) {
    pre_execute();
    record_start_event();
    const cudaStream_t stream = device.getStream(cuda_stream_id);

    voltage.check_metadata();
    const std::shared_ptr<const chordMetadata> meta = voltage.get_metadata();

    // The frame's absolute hop. The F-engine frame counter counts hops (as in
    // GnssChordVoltageTap), i.e. time_downsampling_fpga == 1 on this ring. Anything else breaks
    // the seq -> sample conversion below, and one hop is 52.4 chips at 10.23 Mcps.
    if (meta->get_time_downsampling_fpga() != 1)
        FATAL_ERROR("cudaGnssInject: voltage ring time_downsampling_fpga = {:d}, expected 1 "
                    "(hop-counted seq). The wstart arithmetic must be revisited for this "
                    "convention -- refusing to inject at a wrong code phase.",
                    (int)meta->get_time_downsampling_fpga());
    const long long hop0_frame = meta->get_fpga_seq_num() + voltage.get_read_valid().begin();

    auto* d_ctl = (char*)device.get_gpu_memory_array(_mem_ctl, pipestate.gpu_frame_id,
                                                     _gpu_buffer_depth, _ctl_array_bytes);

    const size_t synth_len = (size_t)_num_times * _synth_n_chan * _synth_lane_pitch;
    auto* d_synth = (unsigned char*)device.get_gpu_memory_array(
        _gnss_synth_name, pipestate.gpu_frame_id, _gpu_buffer_depth, synth_len);

    // Every row's control block for this frame, zeroed each frame: a PRN that just set must read
    // as run=0, never as last frame's contents.
    std::memset(_stage, 0, _ctl_span);
    for (Row& row : _rows)
        enqueue_row(row, pipestate, hop0_frame, d_synth, stream);

    // Every row's block in one copy, then each row's energy rows over the zeros the copy left
    // in them. The consumer needs the true replica energy to undo the pack's per-lane scale:
    // the quantizer holds every lane near a constant M^2, so V/M^2 alone would weight the
    // channel combine by the PFB response instead of by SNR.
    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(d_ctl + _ctl_begin, _stage, _ctl_span, cudaMemcpyHostToDevice, stream));
    for (const Row& row : _rows)
        if (row.n_jobs > 0)
            CHECK_CUDA_ERROR(cudaMemcpyAsync(
                d_ctl + row.ctl_offset + row.ctl_off_energy, row.d_energy,
                (size_t)row.n_jobs * gnss_gpu::ROWS_PLAIN * row.chain->n_chan * sizeof(double),
                cudaMemcpyDeviceToDevice, stream));
    // The last row's time carries the shared copies, so the rows' times sum to the command's.
    if (_hosted && profiling) {
        CHECK_CUDA_ERROR(cudaEventCreate(&_rows.back().end_event));
        CHECK_CUDA_ERROR(cudaEventRecord(_rows.back().end_event, stream));
    }

    return record_end_event();
}

void cudaGnssInject::enqueue_row(Row& row, const cudaPipelineState& pipestate, long long hop0_frame,
                                 unsigned char* d_synth, cudaStream_t stream) {
    if (_hosted && profiling) {
        CHECK_CUDA_ERROR(cudaEventCreate(&row.start_event));
        CHECK_CUDA_ERROR(cudaEventRecord(row.start_event, stream));
    }
    gnss::TrackChain& S = *row.chain;
    const int n_rec = S.n_hops_frame / S.hops_per_record;

    // Device buffers. jobs, the Psi path's pjobs and theta, and slot2spec are per frame slot
    // (frames pipeline); wave, energy and energy0 are shared scratch, which is safe because
    // every frame's work is enqueued on the same stream. The Psi path writes the lanes straight
    // from the synthesis, so it has no wave buffer. The energy rows of every record are kept
    // until the frame's control blocks are uploaded.
    const size_t max_jobs = (size_t)gnss_gpu::max_specs(S.n_prn);
    auto* d_jobs = (gnss_cuda::DespreadJob*)device.get_gpu_memory_array(
        row.name + "_jobs", pipestate.gpu_frame_id, _gpu_buffer_depth,
        max_jobs * sizeof(gnss_cuda::DespreadJob));
    auto* d_slot2spec =
        (int*)device.get_gpu_memory_array(row.name + "_slot2spec", pipestate.gpu_frame_id,
                                          _gpu_buffer_depth, (size_t)n_rec * S.n_prn * sizeof(int));
    gnss_cuda::PsiJob* d_pjobs = nullptr;
    float* d_theta = nullptr;
    float2* d_wave = nullptr;
    if (S.psi_synth) {
        d_pjobs = (gnss_cuda::PsiJob*)device.get_gpu_memory_array(
            row.name + "_pjobs", pipestate.gpu_frame_id, _gpu_buffer_depth,
            max_jobs * sizeof(gnss_cuda::PsiJob));
        d_theta = (float*)device.get_gpu_memory_array(row.name + "_theta", pipestate.gpu_frame_id,
                                                      _gpu_buffer_depth,
                                                      max_jobs * S.n_chan * sizeof(float));
    } else {
        d_wave = (float2*)device.get_gpu_memory(row.name + "_wave", (size_t)3 * S.n_prn * S.n_chan
                                                                        * S.hops_per_record
                                                                        * sizeof(float2));
    }
    row.d_energy = (double*)device.get_gpu_memory(row.name + "_energy",
                                                  (size_t)gnss_gpu::ROWS_PLAIN * n_rec * S.n_prn
                                                      * S.n_chan * sizeof(double));
    // The quantizer scale rows per slot, [n_prn][4][n_chan]: a slot's pack scales every record
    // of the frame by the energy of its first running record, as GnssN2RecordAssemble undoes it.
    auto* d_energy0 = (double*)device.get_gpu_memory(
        row.name + "_energy0", (size_t)4 * S.n_prn * S.n_chan * sizeof(double));
    auto* d_chan_map =
        (int*)device.get_gpu_memory(row.name + "_chan_map", (size_t)S.n_chan * sizeof(int));
    if (!row.uploaded_static) {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_chan_map, row.chan_map.data(),
                                         (size_t)S.n_chan * sizeof(int), cudaMemcpyHostToDevice,
                                         stream));
        row.uploaded_static = true;
    }

    // Slot swaps (docs/CHORD_LIVE_PRN_RECONFIG.md) are applied here, at the frame boundary,
    // before any job is built. note_frame_hop must come first: it is the clock the scheduled
    // deadline is tested on. apply_prn_swaps resets the state's per-slot seeds and trims; the
    // fold history of a swapped slot is reset below, through slot_gen, so the new satellite's
    // arc does not continue the old one's.
    S.note_frame_hop(hop0_frame);
    auto& FH = S.fold; // the re-pin fold history, shared by this command's instances
    S.apply_prn_swaps((void*)stream);
    {
        std::lock_guard<std::mutex> lk(S.prn_mtx);
        for (int p = 0; p < S.n_prn; ++p)
            if (FH.slot_gen_seen[(size_t)p] != S.slot_gen[(size_t)p]) {
                FH.slot_gen_seen[(size_t)p] = S.slot_gen[(size_t)p];
                FH.dop_prev[(size_t)p] = 0.0;
                FH.ok[(size_t)p] = 0;
            }
    }
    // This row's control block in the stage, zeroed by execute().
    char* ctl_stage = _stage + (row.ctl_offset - _ctl_begin);
    auto* hdr = (gnss_gpu::FrameHdr*)ctl_stage;
    auto* winstart = (int64_t*)(ctl_stage + gnss_gpu::off_winstart());
    auto* pctl = (gnss_gpu::PrnCtl*)(ctl_stage + gnss_gpu::off_prnctl());
    hdr->n_rec = n_rec;
    hdr->n_prn = S.n_prn;
    hdr->n_chan = S.n_chan;
    hdr->n_rows_spec = gnss_gpu::ROWS_PLAIN;
    hdr->lanes_per_slot = row.lanes_per_slot;
    hdr->seq0 = hop0_frame * (long long)S.fft_len;
    hdr->utc0 = S.frame0_utc;
    if (!row.gpu.name.empty())
        hdr->_pad1 = row.gpu.tag();

    std::vector<int> expired;
    const std::vector<gnss::TrackChain::Seed> seeds = S.snapshot_seeds(expired);
    for (int prn : expired)
        INFO("cudaGnssInject: {:s}PRN {:d} seed expired -- lane group goes dark", row.label, prn);

    // The fleet code trim (/set_trim), applied in propagate_seed below.
    std::vector<int> trim_gone;
    const std::vector<double> trim_now = S.snapshot_trims(trim_gone);
    for (int prn : trim_gone)
        WARN("cudaGnssInject: {:s}PRN {:d} trim EXPIRED with no /set_trim -- zeroed; the code "
             "phase has stepped back to the broker's bare model.",
             row.label, prn);

    // Unset (NaN) until the pack sets them.
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(d_energy0, 0xFF, (size_t)4 * S.n_prn * S.n_chan * sizeof(double), stream));
    int n_jobs_frame = 0, n_active = 0, n_psi_dark0 = 0;
    for (int r = 0; r < n_rec; ++r) {
        const long long hop0 = hop0_frame + (long long)r * S.hops_per_record;
        const long long wstart = hop0 * (long long)S.fft_len;

        std::vector<GnssCudaDespread::Spec> specs;
        specs.reserve((size_t)S.n_prn);
        int n_psi_dark = 0;
        int* slot2spec = row.slot2spec + (size_t)r * S.n_prn;
        for (int p = 0; p < S.n_prn; ++p) {
            slot2spec[p] = -1;
            // Identity first, before any `continue`: a slot with no seed still holds a
            // satellite, and the assembler must see a PRN change even while the slot is dark,
            // so it drops the old satellite's element cal rather than carry it to the new one.
            pctl[(size_t)r * S.n_prn + p].prn = (uint16_t)S.prns[(size_t)p];
            const auto& sd = seeds[(size_t)p];
            if (!sd.have) {
                // Seed gone (set, expired, never acquired): the replica phase history is
                // meaningless across the gap, so drop it. The next record this PRN produces
                // emits reanchored = 1 and the assembler breaks the arc.
                FH.ok[(size_t)p] = 0;
                continue;
            }
            // FDMA: a slot whose lobe has no channel on this GPU is dark here, as unseeded.
            const bool fdma = !S.slot_covering.empty();
            if (fdma && S.slot_covering[(size_t)p].empty()) {
                FH.ok[(size_t)p] = 0;
                continue;
            }

            // Propagate the seed to this record's window, with the fleet trim.
            gnss::SeedState ss;
            ss.cp_chips = sd.cp_chips;
            ss.phase_ref_chips = sd.phase_ref_chips;
            ss.doppler_hz = sd.doppler_hz;
            ss.dop_rate = sd.dop_rate;
            ss.cp_rate = sd.cp_rate;
            ss.ref_hop = sd.ref_hop;
            const gnss::SeedPropagation pr = gnss::propagate_seed(
                *S.replica, ss, hop0, S.sample_rate, S.f_offset_hz, trim_now[(size_t)p]);

            // Psi tables are built on the despread's builder thread and uploaded one per
            // record: a slot whose tables are not on the device yet (first seed, a swap, a
            // bucket past its grace) is dark this record exactly as a slot without a seed,
            // and joins when they land. Asked here, before anything is written for the slot:
            // no run, no job row, no fold history, slot2spec stays -1.
            if (S.psi_synth && !S.despread->psi_ready(p, pr.doppler_hz, (void*)stream)) {
                FH.ok[(size_t)p] = 0;
                ++n_psi_dark;
                continue;
            }

            GnssCudaDespread::Spec sp;
            sp.p = p;
            sp.doppler_hz = pr.doppler_hz;
            sp.cp_seed = pr.cp;
            sp.spacing_chips = S.dll_spacing;
            sp.ctrim_hz = sd.ctrim_hz;
            sp.covering = fdma ? S.slot_covering[(size_t)p] : S.covering;
            // GuoWang data-sign wipe: the frame at this record's reference sample is the
            // seed's frame at ref_hop plus the whole periods the phase advanced since (the
            // phase at ref_hop is in [0, L)).
            if (sd.nav_pages && sd.nav_frame_at_ref >= 0) {
                sp.nav_pages = sd.nav_pages.get();
                sp.nav_frame =
                    sd.nav_frame_at_ref
                    + (long long)std::floor(pr.phase_now / (double)S.replica->eff_code_length());
            }

            // PrnCtl, the gnssGpuChain.hpp contract the assembler reads. job0 indexes the 4 rows
            // (E/P/L/P_HEAD) this spec occupies in the consumer's corr/energy arrays.
            //
            // Re-pin phase step: the replica's carrier phase is anchored absolutely,
            // 2*pi*(f_offset + applied)*t_abs, so a change in the applied carrier since the
            // previous record steps it by (applied - dop_prev)*t_abs cycles, many cycles at days
            // of uptime. It is taken here, in the Doppler domain, where the difference keeps full
            // precision (see PrnCtl::dcyc). Both terms use this record's t_abs: the step is the
            // phase difference at this instant between the old and the new replica.
            const double t_abs = (double)wstart / S.sample_rate;
            const bool have_hist = FH.ok[(size_t)p] != 0;
            // The applied carrier includes the broker's carrier trim: ctrim is part of the
            // despread's reference frequency (ang0 and wc both take it), so a ctrim change steps
            // the replica phase by dctrim*t_abs just as a Doppler change does.
            const double applied = pr.doppler_hz + sd.ctrim_hz;
            const double dcyc = have_hist ? (applied - FH.dop_prev[(size_t)p]) * t_abs : 0.0;
            FH.dop_prev[(size_t)p] = applied;
            FH.ok[(size_t)p] = 1;

            gnss_gpu::PrnCtl& c = pctl[(size_t)r * S.n_prn + p];
            c.run = 1;
            c.reanchored = have_hist ? 3 : 1;
            c.dcyc = dcyc;
            c.job0 = (n_jobs_frame + (int)specs.size()) * gnss_gpu::ROWS_PLAIN;
            c.fcar_report = (float)pr.doppler_hz;
            c.n_owned = fdma ? (float)S.slot_covering[(size_t)p].size() : (float)S.n_chan;
            c.cp_seed = pr.cp;
            c.f_nco = sd.ctrim_hz;
            c.chan_mask = fdma               ? S.slot_chan_mask[(size_t)p]
                          : (S.n_chan >= 64) ? ~0ULL
                                             : ((1ULL << S.n_chan) - 1ULL);
            c.ctrim_hz = sd.ctrim_hz;
            // carrier_offset: f_offset plus the slot's FDMA offset (GLONASS L2OF), the carrier
            // the despread rotates to.
            c.fcar = S.replica->carrier_offset(p) + pr.doppler_hz;

            slot2spec[p] = (int)specs.size();
            specs.push_back(sp);
        }
        if (r == 0) {
            n_active = (int)specs.size();
            n_psi_dark0 = n_psi_dark;
        }

        // Upload slot2spec even when empty: the pack must still refresh this record's rows (the
        // frame slot's contents are a buffer depth of frames old, and a PRN that just set must
        // go dark, not replay).
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_slot2spec + (size_t)r * S.n_prn, slot2spec,
                                         (size_t)S.n_prn * sizeof(int), cudaMemcpyHostToDevice,
                                         stream));

        winstart[r] = wstart;

        gnss_cuda::DespreadJob* d_jobs_r = d_jobs + (size_t)r * S.n_prn;
        unsigned char* d_synth_rec =
            d_synth + (size_t)r * S.hops_per_record * _synth_n_chan * _synth_lane_pitch;
        // This record's energy rows, after the records before it.
        double* d_energy_r = row.d_energy + (size_t)n_jobs_frame * gnss_gpu::ROWS_PLAIN * S.n_chan;
        if (S.psi_synth)
            // Synthesis and pack in one launch, every record (an empty record still refreshes
            // its lanes).
            S.despread->enqueue_psi_pack(wstart, specs, d_jobs_r, d_pjobs + (size_t)r * S.n_prn,
                                         d_theta + (size_t)r * S.n_prn * S.n_chan,
                                         d_slot2spec + (size_t)r * S.n_prn, S.n_prn,
                                         row.lanes_per_slot, d_chan_map, _synth_n_chan,
                                         row.num_synth, row.lane_base, _synth_lane_pitch, d_energy0,
                                         d_energy_r, d_synth_rec, (void*)stream);
        else if (!specs.empty())
            S.despread->enqueue_waveform(wstart, specs, d_jobs_r, d_wave, d_energy_r,
                                         (void*)stream);
        // ang0 exists only once the despread has built the jobs, so this is a second pass
        // over pctl. It is read from the despread's record of what it handed the kernel,
        // not re-derived here.
        for (const auto& sp2 : specs) {
            gnss_gpu::PrnCtl& cc = pctl[(size_t)r * S.n_prn + sp2.p];
            cc.ang0 = S.despread->last_ang0(sp2.p);
            cc.phi_ddop = S.despread->last_phi_ddop(sp2.p);
            // The rotation the kernel applied to its E trial this record, for the
            // assembler to form the E and L rows under LANES_PROMPT (PrnCtl::rot). The
            // stage is zeroed per frame, so a chain without the rotated path leaves 0.
            if (const float* rot = S.despread->last_rot(sp2.p))
                for (int c = 0; c < S.n_chan && c < gnss_gpu::MAX_ROT_CHAN; ++c)
                    cc.rot[c] = rot[c];
        }
        n_jobs_frame += (int)specs.size();

        if (!S.psi_synth)
            CHECK_CUDA_ERROR(gnss_cuda::launch_pack44(
                d_wave, d_energy_r, d_energy0, d_jobs_r, d_slot2spec + (size_t)r * S.n_prn, S.n_prn,
                row.lanes_per_slot, S.n_chan, S.hops_per_record, d_chan_map, _synth_n_chan,
                row.num_synth, row.lane_base, _synth_lane_pitch, d_synth_rec, stream));
    }

    hdr->n_jobs = n_jobs_frame * gnss_gpu::ROWS_PLAIN;
    row.n_jobs = n_jobs_frame;

    // Startup coverage: seeded slots waiting for their Psi tables, by record 0 of each frame.
    // Logged at the first frame with a dark slot and at the first with none again; the state
    // holds it because this command's instances (one per in-flight frame) each see every
    // n-th frame.
    if (S.psi_synth) {
        if (n_psi_dark0 > 0 && S.psi_dark_hop0 < 0) {
            S.psi_dark_hop0 = hop0_frame;
            INFO("cudaGnssInject: {:s}Psi tables pending for {:d} of {:d} seeded slots: dark "
                 "until built (frame hop0 {:d})",
                 row.label, n_psi_dark0, n_psi_dark0 + n_active, hop0_frame);
        } else if (n_psi_dark0 == 0 && S.psi_dark_hop0 >= 0) {
            const double frames = (double)(hop0_frame - S.psi_dark_hop0) / S.n_hops_frame;
            S.psi_dark_hop0 = -1;
            INFO("cudaGnssInject: {:s}Psi coverage complete: {:d} seeded slots, {:.0f} frames "
                 "({:.0f} ms) after the first dark one",
                 row.label, n_active, frames,
                 frames * S.n_hops_frame * S.fft_len / S.sample_rate * 1e3);
        }
    }

    if ((++row.frames & 0xFF) == 1)
        INFO("cudaGnssInject: {:s}frame hop0 {:d}, {:d} active PRNs, {:d} jobs across {:d} "
             "records",
             row.label, hop0_frame, n_active, n_jobs_frame, n_rec);

    if (_hosted && profiling && &row != &_rows.back()) {
        CHECK_CUDA_ERROR(cudaEventCreate(&row.end_event));
        CHECK_CUDA_ERROR(cudaEventRecord(row.end_event, stream));
    }
}

void cudaGnssInject::finalize_frame() {
    // As cudaCommand::finalize_frame, per row.
    for (Row& row : _rows) {
        if (!_hosted)
            continue;
        if (profiling && row.start_event != nullptr && row.end_event != nullptr) {
            float exec_time;
            CHECK_CUDA_ERROR(cudaEventElapsedTime(&exec_time, row.start_event, row.end_event));
            row.execute_time->add_sample(exec_time * 1e-3);
            row.utilization->add_sample(exec_time * 1e-3 / frame_arrival_period);
        } else {
            row.execute_time->add_sample(0.);
            row.utilization->add_sample(0.);
        }
        for (cudaEvent_t* e : {&row.start_event, &row.end_event})
            if (*e != nullptr) {
                CHECK_CUDA_ERROR(cudaEventDestroy(*e));
                *e = nullptr;
            }
    }
    voltage.finish_read();
    cudaCommand::finalize_frame();
}

std::vector<std::pair<std::string, std::shared_ptr<StatTracker>>> cudaGnssInject::profile_rows() {
    if (!_hosted)
        return cudaCommand::profile_rows();
    std::vector<std::pair<std::string, std::shared_ptr<StatTracker>>> out;
    for (const Row& row : _rows)
        out.emplace_back(get_name(), row.execute_time);
    return out;
}
