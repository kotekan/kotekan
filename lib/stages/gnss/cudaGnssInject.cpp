#include "cudaGnssInject.hpp"

#include "cudaGnssChordDespread.hpp" // for launch_pack44
#include "cudaGnssDespreadKernel.hpp" // for DespreadJob
#include "cudaUtils.hpp"              // for CHECK_CUDA_ERROR
#include "gnssGpuChain.hpp"           // for max_specs
#include "gnssSeedTransport.hpp"      // for propagate_seed
#include "kotekanLogging.hpp"

#include <array>
#include <cassert>
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
    _gnss_synth_name(
        config.get_default<std::string>(unique_name, "gnss_synth_name", "gnss_synth")),
    _gnss_local_channels(
        config.get<std::vector<std::int32_t>>(unique_name, "gnss_local_channels")),
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

    voltage.register_consumer();

    const int n_rec = S.n_hops_frame / S.hops_per_record;
    _h_slot2spec.assign((size_t)n_rec * S.n_prn, -1);

    set_command_type(gpuCommandType::KERNEL);
    INFO("cudaGnssInject: {:d} PRN slots x 4 lanes into '{:s}' [{:d}][{:d}][{:d}], {:d} "
         "records/frame, channels {:d}",
         S.n_prn, _gnss_synth_name, _num_times, _num_local_freq, _num_synth, n_rec, S.n_chan);
}

cudaGnssChordTrackState* cudaGnssInject::st() {
    return static_cast<cudaGnssChordTrackState*>(command_state.get());
}

int cudaGnssInject::wait_on_precondition() {
    // The claim is the TIME REFERENCE: this command produces the synth window for exactly the
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

    // The frame's absolute HOP. The tap's convention (GnssChordVoltageTap): the F-engine
    // frame counter counts hops, i.e. time_downsampling_fpga == 1 on this ring. Anything else
    // means the seq->sample conversion below is wrong -- fail loudly rather than despread at
    // the wrong code phase (1 hop of error is 52.4 chips).
    if (meta->get_time_downsampling_fpga() != 1)
        FATAL_ERROR("cudaGnssInject: voltage ring time_downsampling_fpga = {:d}, expected 1 "
                    "(hop-counted seq). The wstart arithmetic must be revisited for this "
                    "convention -- refusing to inject at a wrong code phase.",
                    (int)meta->get_time_downsampling_fpga());
    const long long hop0_frame = meta->get_fpga_seq_num() + voltage.get_read_valid().begin();

    // Device buffers. jobs/slot2spec are per frame slot (frames pipeline); wave/energy are
    // shared scratch -- safe because every frame's work enqueues on the SAME stream, so uses
    // are serialized device-side (the tracker's d_wave relies on the same property).
    const size_t max_jobs = (size_t)gnss_gpu::max_specs(S.n_prn);
    auto* d_jobs = (gnss_cuda::DespreadJob*)device.get_gpu_memory_array(
        unique_name + "_jobs", pipestate.gpu_frame_id, _gpu_buffer_depth,
        max_jobs * sizeof(gnss_cuda::DespreadJob));
    auto* d_slot2spec =
        (int*)device.get_gpu_memory_array(unique_name + "_slot2spec", pipestate.gpu_frame_id,
                                          _gpu_buffer_depth, (size_t)n_rec * S.n_prn * sizeof(int));
    auto* d_wave = (float2*)device.get_gpu_memory(
        unique_name + "_wave",
        (size_t)3 * S.n_prn * S.n_chan * S.hops_per_record * sizeof(float2));
    auto* d_energy = (double*)device.get_gpu_memory(
        unique_name + "_energy", (size_t)4 * S.n_prn * S.n_chan * sizeof(double));
    auto* d_chan_map =
        (int*)device.get_gpu_memory(unique_name + "_chan_map", (size_t)S.n_chan * sizeof(int));
    if (!_uploaded_static) {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_chan_map, _gnss_local_channels.data(),
                                         (size_t)S.n_chan * sizeof(int), cudaMemcpyHostToDevice,
                                         stream));
        _uploaded_static = true;
    }

    const size_t synth_len = (size_t)_num_times * _num_local_freq * _num_synth;
    auto* d_synth = (unsigned char*)device.get_gpu_memory_array(
        _gnss_synth_name, pipestate.gpu_frame_id, _gpu_buffer_depth, synth_len);

    std::vector<int> expired;
    const std::vector<cudaGnssChordTrackState::Seed> seeds = S.snapshot_seeds(expired);
    for (int prn : expired)
        INFO("cudaGnssInject: PRN {:d} seed expired -- lane group goes dark", prn);

    int n_jobs_frame = 0, n_active = 0;
    for (int r = 0; r < n_rec; ++r) {
        const long long hop0 = hop0_frame + (long long)r * S.hops_per_record;
        const long long wstart = hop0 * (long long)S.fft_len;

        std::vector<GnssCudaDespread::Spec> specs;
        specs.reserve((size_t)S.n_prn);
        int* slot2spec = _h_slot2spec.data() + (size_t)r * S.n_prn;
        for (int p = 0; p < S.n_prn; ++p) {
            slot2spec[p] = -1;
            const auto& sd = seeds[(size_t)p];
            if (!sd.have)
                continue;

            // Same propagation as the tracker (gnssSeedTransport) -- replicas cannot fork.
            // No DLL trim here (the injector has no loop); A/B against code_trim: false.
            gnss::SeedState ss;
            ss.cp_chips = sd.cp_chips;
            ss.phase_ref_chips = sd.phase_ref_chips;
            ss.doppler_hz = sd.doppler_hz;
            ss.dop_rate = sd.dop_rate;
            ss.cp_rate = sd.cp_rate;
            ss.ref_hop = sd.ref_hop;
            const gnss::SeedPropagation pr = gnss::propagate_seed(
                *S.replica, ss, hop0, S.sample_rate, S.f_offset_hz, /*trim_chips=*/0.0);

            GnssCudaDespread::Spec sp;
            sp.p = p;
            sp.doppler_hz = pr.doppler_hz;
            sp.cp_seed = pr.cp;
            sp.spacing_chips = S.dll_spacing;
            sp.ctrim_hz = sd.ctrim_hz;
            sp.covering = S.covering;

            slot2spec[p] = (int)specs.size();
            specs.push_back(sp);
        }
        if (r == 0)
            n_active = (int)specs.size();

        // slot2spec upload even when empty: the pack must still refresh this record's rows
        // (previous frame-slot contents are four frames stale, and a PRN that just set must
        // go dark, not replay).
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_slot2spec + (size_t)r * S.n_prn, slot2spec,
                                         (size_t)S.n_prn * sizeof(int), cudaMemcpyHostToDevice,
                                         stream));

        gnss_cuda::DespreadJob* d_jobs_r = d_jobs + (size_t)r * S.n_prn;
        if (!specs.empty())
            S.despread->enqueue_waveform(wstart, specs, d_jobs_r, d_wave, d_energy,
                                         (void*)stream);
        n_jobs_frame += (int)specs.size();

        unsigned char* d_synth_rec =
            d_synth + (size_t)r * S.hops_per_record * _num_local_freq * _num_synth;
        CHECK_CUDA_ERROR(gnss_cuda::launch_pack44(
            d_wave, d_energy, d_jobs_r, d_slot2spec + (size_t)r * S.n_prn, S.n_prn, S.n_chan,
            S.hops_per_record, d_chan_map, _num_local_freq, _num_synth, d_synth_rec, stream));
    }

    if ((++_frames & 0xFF) == 1)
        INFO("cudaGnssInject: frame hop0 {:d}, {:d} active PRNs, {:d} jobs across {:d} records",
             hop0_frame, n_active, n_jobs_frame, n_rec);

    return record_end_event();
}

void cudaGnssInject::finalize_frame() {
    voltage.finish_read();
    cudaCommand::finalize_frame();
}
