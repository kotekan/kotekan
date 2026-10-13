#include "cudaGnssInject.hpp"

#include "cudaGnssChordDespread.hpp"  // for launch_pack44
#include "cudaGnssDespreadKernel.hpp" // for DespreadJob
#include "cudaUtils.hpp"              // for CHECK_CUDA_ERROR
#include "gnssGpuChain.hpp"           // for max_specs
#include "gnssSeedTransport.hpp"      // for propagate_seed
#include "kotekanLogging.hpp"

#include <algorithm> // for find
#include <array>
#include <chordMetadata.hpp>
#include <cstring>

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND_WITH_STATE(cudaGnssInject, cudaGnssInjectState);

cudaGnssInject::cudaGnssInject(Config& config, const std::string& unique_name,
                               bufferContainer& host_buffers, cudaDeviceInterface& device,
                               int instance_num, std::shared_ptr<cudaCommandState> state) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, state, "cudaGnssInject",
                "cudaGnssInject"),
    _buffer_depth(config.get<int>(unique_name, "buffer_depth")),
    _num_times(config.get<int>(unique_name, "num_times")),
    _num_elements(config.get<int>(unique_name, "num_elements")),
    _num_local_freq(config.get<int>(unique_name, "num_local_freq")),
    _num_synth(config.get_default<int>(unique_name, "num_synth", 128)),
    _voltage_name(config.get<std::string>(unique_name, "voltage_name")),
    _gnss_synth_name(config.get_default<std::string>(unique_name, "gnss_synth_name", "gnss_synth")),
    _gnss_local_channels(config.get<std::vector<std::int32_t>>(unique_name, "gnss_local_channels")),
    _synth_lane_base(config.get_default<int>(unique_name, "synth_lane_base", 0)),
    _synth_lane_pitch(config.get_default<int>(unique_name, "synth_lane_pitch", _num_synth)),
    _gnss_synth_channels(
        config.get_default<std::vector<std::int32_t>>(unique_name, "gnss_synth_channels", {})),
    voltage(_voltage_name, "E",
            std::array<std::ptrdiff_t, 4>{_buffer_depth * _num_times, _num_local_freq, 2,
                                          _num_elements / 2},
            std::array<std::string, 4>{"T", "F", "P", "D"},
            std::array<std::ptrdiff_t, 4>{1, 1, 1, 1}, *this) {
    cudaGnssChordTrackState& S = *st();

    if (4 * S.n_prn > _num_synth)
        FATAL_ERROR("cudaGnssInject: 4*n_prn ({:d}) exceeds num_synth ({:d}) -- lanes are "
                    "4/PRN slot (E/P/L/P_HEAD), shrink the PRN list or widen the synth axis.",
                    4 * S.n_prn, _num_synth);
    if ((int)_gnss_local_channels.size() != S.n_chan)
        FATAL_ERROR("cudaGnssInject: gnss_local_channels has {:d} entries but n_channels is "
                    "{:d} -- one LOCAL frame index per covering channel, in local order.",
                    (int)_gnss_local_channels.size(), S.n_chan);
    for (std::int32_t f : _gnss_local_channels)
        if (f < 0 || f >= _num_local_freq)
            FATAL_ERROR("cudaGnssInject: gnss_local_channels entry {:d} outside "
                        "[0, num_local_freq {:d}) -- LOCAL indices, not global freq_ids.",
                        f, _num_local_freq);
    if (S.n_hops_frame != _num_times)
        FATAL_ERROR("cudaGnssInject: state samples_per_data_set {:d} != num_times {:d}",
                    S.n_hops_frame, _num_times);
    // The lane range must be whole tile rows (16 lanes) inside the shared axis, or the
    // correlator's tile arithmetic and this pack disagree about which lane is whose.
    if (_synth_lane_base < 0 || _synth_lane_base % 16 || _synth_lane_pitch % 16
        || _synth_lane_base + _num_synth > _synth_lane_pitch)
        FATAL_ERROR("cudaGnssInject: lanes [{:d}, {:d}) do not fit a synth lane axis of {:d} "
                    "on 16-lane tile rows (synth_lane_base / synth_lane_pitch / num_synth)",
                    _synth_lane_base, _synth_lane_base + _num_synth, _synth_lane_pitch);
    // The synth array's channel axis: every local channel (the one-chain layout), or the
    // correlator's comb (compact). Each covering channel maps to its index on that axis.
    if (_gnss_synth_channels.empty()) {
        _synth_n_chan = _num_local_freq;
        _h_chan_map.assign(_gnss_local_channels.begin(), _gnss_local_channels.end());
    } else {
        _synth_n_chan = (int)_gnss_synth_channels.size();
        for (std::int32_t f : _gnss_local_channels) {
            const auto it = std::find(_gnss_synth_channels.begin(), _gnss_synth_channels.end(), f);
            if (it == _gnss_synth_channels.end())
                FATAL_ERROR("cudaGnssInject: covering channel {:d} is not on the synth array's "
                            "channel axis (gnss_synth_channels) -- the correlator would never "
                            "read this chain's replica there",
                            f);
            _h_chan_map.push_back((int)(it - _gnss_synth_channels.begin()));
        }
    }

    voltage.register_consumer();

    const int n_rec = S.n_hops_frame / S.hops_per_record;
    _h_slot2spec.assign((size_t)n_rec * S.n_prn, -1);

    // Control block: the epl layout without corr (see the header), sized for every PRN active
    // in every record (gnss_gpu::max_jobs).
    _mem_ctl = config.get_default<std::string>(unique_name, "gnss_ctl_name", "gnss_n2ctl");
    _ctl_off_energy =
        gnss_gpu::off_prnctl() + sizeof(gnss_gpu::PrnCtl) * gnss_gpu::MAX_REC * S.n_prn;
    _ctl_bytes = _ctl_off_energy
                 + sizeof(double) * gnss_gpu::max_jobs(S.n_prn, gnss_gpu::ROWS_PLAIN) * S.n_chan;
    _ctl_stage.resize(_ctl_bytes);

    set_command_type(gpuCommandType::KERNEL);
    // The F-engine conjugation is absorbed by conjugating the replica (launch_pack44): the N^2
    // kernel has no conj_data flag and its antenna input is production's.
    if (!S._conjugate)
        WARN("cudaGnssInject: conjugate=false. On CHORD the F-engine output IS conjugated "
             "relative to the decode (measured on sky 2026-07-30); with this false the "
             "synthetic lanes despread the conjugate of the sky and every tile reads NOISE.");
    INFO("cudaGnssInject: {:d} PRN slots x 4 lanes into '{:s}' [{:d}][{:d}][{:d}] at lanes "
         "[{:d}, {:d}), {:d} records/frame, channels {:d}{:s}, conjugate {:s}",
         S.n_prn, _gnss_synth_name, _num_times, _synth_n_chan, _synth_lane_pitch, _synth_lane_base,
         _synth_lane_base + _num_synth, n_rec, S.n_chan,
         _gnss_synth_channels.empty() ? "" : " (compact channel axis)",
         S._conjugate ? "ON" : "off");
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
    cudaGnssChordTrackState& S = *st();
    const cudaStream_t stream = device.getStream(cuda_stream_id);
    const int n_rec = S.n_hops_frame / S.hops_per_record;

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

    // Device buffers. jobs and slot2spec are per frame slot (frames pipeline); wave, energy and
    // energy0 are shared scratch, which is safe because every frame's work is enqueued on the
    // same stream.
    const size_t max_jobs = (size_t)gnss_gpu::max_specs(S.n_prn);
    auto* d_jobs = (gnss_cuda::DespreadJob*)device.get_gpu_memory_array(
        unique_name + "_jobs", pipestate.gpu_frame_id, _gpu_buffer_depth,
        max_jobs * sizeof(gnss_cuda::DespreadJob));
    auto* d_slot2spec =
        (int*)device.get_gpu_memory_array(unique_name + "_slot2spec", pipestate.gpu_frame_id,
                                          _gpu_buffer_depth, (size_t)n_rec * S.n_prn * sizeof(int));
    auto* d_wave = (float2*)device.get_gpu_memory(
        unique_name + "_wave", (size_t)3 * S.n_prn * S.n_chan * S.hops_per_record * sizeof(float2));
    auto* d_energy = (double*)device.get_gpu_memory(
        unique_name + "_energy", (size_t)4 * S.n_prn * S.n_chan * sizeof(double));
    // The quantizer scale is fixed for the frame: record 0's energy, handed to every record's
    // pack. GnssN2RecordAssemble recomputes the scale from record 0's energy rows to undo it.
    auto* d_energy0 = (double*)device.get_gpu_memory(
        unique_name + "_energy0", (size_t)4 * S.n_prn * S.n_chan * sizeof(double));
    auto* d_chan_map =
        (int*)device.get_gpu_memory(unique_name + "_chan_map", (size_t)S.n_chan * sizeof(int));
    if (!_uploaded_static) {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_chan_map, _h_chan_map.data(),
                                         (size_t)S.n_chan * sizeof(int), cudaMemcpyHostToDevice,
                                         stream));
        _uploaded_static = true;
    }

    auto* d_ctl = (char*)device.get_gpu_memory_array(_mem_ctl, pipestate.gpu_frame_id,
                                                     _gpu_buffer_depth, _ctl_bytes);

    const size_t synth_len = (size_t)_num_times * _synth_n_chan * _synth_lane_pitch;
    auto* d_synth = (unsigned char*)device.get_gpu_memory_array(
        _gnss_synth_name, pipestate.gpu_frame_id, _gpu_buffer_depth, synth_len);

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
    // Control block for this frame, zeroed each frame: a PRN that just set must read as run=0,
    // never as last frame's contents.
    std::memset(_ctl_stage.data(), 0, _ctl_stage.size());
    auto* hdr = (gnss_gpu::FrameHdr*)_ctl_stage.data();
    auto* winstart = (int64_t*)(_ctl_stage.data() + gnss_gpu::off_winstart());
    auto* pctl = (gnss_gpu::PrnCtl*)(_ctl_stage.data() + gnss_gpu::off_prnctl());
    hdr->n_rec = n_rec;
    hdr->n_prn = S.n_prn;
    hdr->n_chan = S.n_chan;
    hdr->n_rows_spec = gnss_gpu::ROWS_PLAIN;
    hdr->seq0 = hop0_frame * (long long)S.fft_len;
    hdr->utc0 = S.frame0_utc;

    std::vector<int> expired;
    const std::vector<cudaGnssChordTrackState::Seed> seeds = S.snapshot_seeds(expired);
    for (int prn : expired)
        INFO("cudaGnssInject: PRN {:d} seed expired -- lane group goes dark", prn);

    // The fleet code trim (/set_trim), applied in propagate_seed below.
    std::vector<int> trim_gone;
    const std::vector<double> trim_now = S.snapshot_trims(trim_gone);
    for (int prn : trim_gone)
        WARN("cudaGnssInject: PRN {:d} trim EXPIRED with no /set_trim -- zeroed; the code "
             "phase has stepped back to the broker's bare model.",
             prn);

    int n_jobs_frame = 0, n_active = 0;
    for (int r = 0; r < n_rec; ++r) {
        const long long hop0 = hop0_frame + (long long)r * S.hops_per_record;
        const long long wstart = hop0 * (long long)S.fft_len;

        std::vector<GnssCudaDespread::Spec> specs;
        specs.reserve((size_t)S.n_prn);
        int* slot2spec = _h_slot2spec.data() + (size_t)r * S.n_prn;
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

            GnssCudaDespread::Spec sp;
            sp.p = p;
            sp.doppler_hz = pr.doppler_hz;
            sp.cp_seed = pr.cp;
            sp.spacing_chips = S.dll_spacing;
            sp.ctrim_hz = sd.ctrim_hz;
            sp.covering = S.covering;

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
            c.n_owned = (float)S.n_chan;
            c.cp_seed = pr.cp;
            c.f_nco = sd.ctrim_hz;
            c.chan_mask = (S.n_chan >= 64) ? ~0ULL : ((1ULL << S.n_chan) - 1ULL);
            c.ctrim_hz = sd.ctrim_hz;
            c.fcar = S.f_offset_hz + pr.doppler_hz;

            slot2spec[p] = (int)specs.size();
            specs.push_back(sp);
        }
        if (r == 0)
            n_active = (int)specs.size();

        // Upload slot2spec even when empty: the pack must still refresh this record's rows (the
        // frame slot's contents are a buffer depth of frames old, and a PRN that just set must
        // go dark, not replay).
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_slot2spec + (size_t)r * S.n_prn, slot2spec,
                                         (size_t)S.n_prn * sizeof(int), cudaMemcpyHostToDevice,
                                         stream));

        winstart[r] = wstart;

        gnss_cuda::DespreadJob* d_jobs_r = d_jobs + (size_t)r * S.n_prn;
        if (!specs.empty()) {
            S.despread->enqueue_waveform(wstart, specs, d_jobs_r, d_wave, d_energy, (void*)stream);
            // ang0 exists only once the despread has built the jobs, so this is a second pass
            // over pctl. It is read from the despread's record of what it handed the kernel,
            // not re-derived here.
            for (const auto& sp2 : specs) {
                gnss_gpu::PrnCtl& cc = pctl[(size_t)r * S.n_prn + sp2.p];
                cc.ang0 = S.despread->last_ang0(sp2.p);
                cc.phi_ddop = S.despread->last_phi_ddop(sp2.p);
            }
            // The consumer needs the true replica energy to undo the pack's per-lane scale: the
            // quantizer holds every lane near a constant M^2, so V/M^2 alone would weight the
            // channel combine by the PFB response instead of by SNR. Copied D2D on the stream.
            // Record 0's energy is also kept as the frame's quantizer reference (d_energy0).
            if (r == 0)
                CHECK_CUDA_ERROR(cudaMemcpyAsync(d_energy0, d_energy,
                                                 (size_t)gnss_gpu::ROWS_PLAIN * specs.size()
                                                     * S.n_chan * sizeof(double),
                                                 cudaMemcpyDeviceToDevice, stream));
            CHECK_CUDA_ERROR(cudaMemcpyAsync(
                d_ctl + _ctl_off_energy
                    + (size_t)n_jobs_frame * gnss_gpu::ROWS_PLAIN * S.n_chan * sizeof(double),
                d_energy, (size_t)gnss_gpu::ROWS_PLAIN * specs.size() * S.n_chan * sizeof(double),
                cudaMemcpyDeviceToDevice, stream));
        }
        n_jobs_frame += (int)specs.size();

        unsigned char* d_synth_rec =
            d_synth + (size_t)r * S.hops_per_record * _synth_n_chan * _synth_lane_pitch;
        CHECK_CUDA_ERROR(gnss_cuda::launch_pack44(
            d_wave, d_energy0, d_jobs_r, d_slot2spec + (size_t)r * S.n_prn, S.n_prn, S.n_chan,
            S.hops_per_record, d_chan_map, _synth_n_chan, _num_synth, _synth_lane_base,
            _synth_lane_pitch, S._conjugate, d_synth_rec, stream));
    }

    hdr->n_jobs = n_jobs_frame * gnss_gpu::ROWS_PLAIN;
    // Header, winstart and PrnCtl only: the energy rows were already written D2D above.
    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(d_ctl, _ctl_stage.data(), _ctl_off_energy, cudaMemcpyHostToDevice, stream));

    if ((++_frames & 0xFF) == 1)
        INFO("cudaGnssInject: frame hop0 {:d}, {:d} active PRNs, {:d} jobs across {:d} records",
             hop0_frame, n_active, n_jobs_frame, n_rec);

    return record_end_event();
}

void cudaGnssInject::finalize_frame() {
    voltage.finish_read();
    cudaCommand::finalize_frame();
}
