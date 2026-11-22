#include "cudaGnssTrack.hpp"

#include "GnssChanMetadata.hpp"       // for GnssChanMetadata
#include "cudaGnssDespreadKernel.hpp" // for DespreadJob
#include "cudaUtils.hpp"              // for CHECK_CUDA_ERROR
#include "gnssGpuChain.hpp"           // for FrameHdr, PrnCtl, frame_bytes
#include "kotekanLogging.hpp"         // for FATAL_ERROR, INFO, WARN

#include <cstring> // for memset
#include <memory>  // for dynamic_pointer_cast, shared_ptr
#include <mutex>   // for lock_guard
#include <tuple>   // for make_tuple

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND_WITH_STATE(cudaGnssTrack, cudaGnssTrackState);

cudaGnssTrackState::cudaGnssTrackState(Config& config, const std::string& unique_name,
                                       bufferContainer& host_buffers, cudaDeviceInterface& device) :
    cudaGnssChordTrackState(config, unique_name, host_buffers, device) {
    if (config.exists(unique_name, "data"))
        data = std::make_unique<gnss::TrackChain>(config, unique_name + "/data");
}

cudaGnssTrack::cudaGnssTrack(Config& config, const std::string& unique_name,
                             bufferContainer& host_buffers, cudaDeviceInterface& device,
                             int instance_num, std::shared_ptr<cudaCommandState> state) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, state, "cudaGnssTrack",
                "cudaGnssTrack"),
    _gpu_mem_input(config.get<std::string>(unique_name, "gpu_mem_input")),
    _gpu_mem_output(config.get<std::string>(unique_name, "gpu_mem_output")),
    _num_local_freq(config.get<int>(unique_name, "num_local_freq")),
    _local_channels(config.get<std::vector<std::int32_t>>(unique_name, "gnss_local_channels")) {
    const auto* st = static_cast<cudaGnssTrackState*>(command_state.get());
    if (st->hosted())
        FATAL_ERROR("cudaGnssTrack: {:s} has a chains block; it runs one chain", unique_name);
    const gnss::TrackChain& S = chain();
    if (S.n_elem != 1)
        FATAL_ERROR("cudaGnssTrack: n_elements is {:d}; the input frame holds one element",
                    S.n_elem);
    // enqueue_batch_nm reads the Phi tables; it honours epl_from_prompt.
    if (S.psi_synth)
        FATAL_ERROR("cudaGnssTrack: psi_synth is cudaGnssInject's");
    if ((int)_local_channels.size() != S.n_chan)
        FATAL_ERROR("cudaGnssTrack: gnss_local_channels has {:d} entries but n_channels is {:d}",
                    (int)_local_channels.size(), S.n_chan);
    for (std::int32_t f : _local_channels)
        if (f < 0 || f >= _num_local_freq)
            FATAL_ERROR("cudaGnssTrack: gnss_local_channels entry {:d} outside [0, "
                        "num_local_freq {:d})",
                        f, _num_local_freq);

    for (const char* a : {"_jobs", "_wave", "_chan_scale", "_chan_ids"})
        device.register_gpu_memory_name(unique_name + a);
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_input, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_output, true, false, true));

    CHECK_CUDA_ERROR(
        cudaHostAlloc((void**)&_stage, gnss_gpu::off_corr(S.n_prn), cudaHostAllocDefault));

    if (st->data) {
        const gnss::TrackChain& D = *st->data;
        // Inherited from the command, gpu_mem_output would name this chain's own frame.
        if (!config.exists(unique_name + "/data", "gpu_mem_output"))
            FATAL_ERROR("cudaGnssTrack: {:s}/data must set its own gpu_mem_output", unique_name);
        _gpu_mem_output_data = config.get<std::string>(unique_name + "/data", "gpu_mem_output");
        if (D.psi_synth)
            FATAL_ERROR("cudaGnssTrack: psi_synth is cudaGnssInject's");
        // The pilot's seeds, trims and slots apply to it unchanged only on the same slots,
        // channels, records and carrier, with a code whose period divides the pilot's.
        if (D.prns != S.prns || D.n_chan != S.n_chan || D.hops_per_record != S.hops_per_record
            || D.sig->carrier_hz != S.sig->carrier_hz
            || S.replica->eff_code_length() % D.replica->eff_code_length() != 0)
            FATAL_ERROR("cudaGnssTrack: {:s}/data ({:s}) needs the PRN slots, channels, records "
                        "and carrier of {:s}, and a code length dividing its {:d} chips",
                        unique_name, D.sig->name, S.sig->name, S.replica->eff_code_length());
        device.register_gpu_memory_name(unique_name + "_data_jobs");
        device.register_gpu_memory_name(unique_name + "_wave_data");
        gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_output_data, true, false, true));
        CHECK_CUDA_ERROR(
            cudaHostAlloc((void**)&_stage_data, gnss_gpu::off_corr(D.n_prn), cudaHostAllocDefault));
    }

    set_command_type(gpuCommandType::KERNEL);
}

cudaGnssTrack::~cudaGnssTrack() {
    cudaFreeHost(_stage);
    if (_stage_data)
        cudaFreeHost(_stage_data);
}

gnss::TrackChain& cudaGnssTrack::chain() {
    return *static_cast<cudaGnssChordTrackState*>(command_state.get())->rows[0].chain;
}

void cudaGnssTrack::enqueue_frame(ChainFrame& P, ChainFrame* D, const std::vector<double>& trim_now,
                                  long long hop0_frame, const unsigned char* d_frame,
                                  const float* d_chan_scale, const int* d_chan_ids,
                                  cudaStream_t stream) {
    ChainFrame* frames[2] = {&P, D};
    const int n_rec = P.C->n_hops_frame / P.C->hops_per_record;

    // The control block is zeroed each frame: a PRN that just set must read as run = 0.
    for (ChainFrame* f : frames) {
        if (!f)
            continue;
        gnss::TrackChain& C = *f->C;
        std::memset(f->stage, 0, gnss_gpu::off_corr(C.n_prn));
        auto* hdr = (gnss_gpu::FrameHdr*)f->stage;
        hdr->n_rec = n_rec;
        hdr->n_prn = C.n_prn;
        hdr->n_chan = C.n_chan;
        hdr->n_rows_spec = gnss_gpu::ROWS_PLAIN;
        hdr->lanes_per_slot = gnss_gpu::LANES_FULL;
        hdr->seq0 = hop0_frame * (long long)C.fft_len;
        hdr->utc0 = C.frame0_utc;
        f->n_jobs = 0;
        f->n_active = 0;
    }

    for (int r = 0; r < n_rec; ++r) {
        const long long hop0 = hop0_frame + (long long)r * P.C->hops_per_record;
        const long long wstart = hop0 * (long long)P.C->fft_len;
        const unsigned char* d_rec = d_frame + (size_t)r * P.C->hops_per_record * _num_local_freq;
        std::vector<GnssCudaDespread::Spec> specs[2];
        // The kernels write each record's rows straight into the output frame.
        char* d_corr[2] = {nullptr, nullptr};
        char* d_energy[2] = {nullptr, nullptr};
        void* d_jobs[2] = {nullptr, nullptr};
        for (int k = 0; k < 2; ++k) {
            ChainFrame* f = frames[k];
            if (!f)
                continue;
            gnss::TrackChain& C = *f->C;
            auto* pctl = (gnss_gpu::PrnCtl*)(f->stage + gnss_gpu::off_prnctl());
            int n_psi_dark = 0;
            specs[k] =
                C.record_specs(hop0, *f->seeds, trim_now, f->n_jobs, pctl + (size_t)r * C.n_prn,
                               nullptr, (void*)stream, n_psi_dark);
            if (r == 0)
                f->n_active = (int)specs[k].size();
            ((int64_t*)(f->stage + gnss_gpu::off_winstart()))[r] = wstart;
            const size_t row0 = (size_t)f->n_jobs * gnss_gpu::ROWS_PLAIN * C.n_chan;
            d_corr[k] = f->d_out + gnss_gpu::off_corr(C.n_prn) + row0 * 2 * sizeof(double);
            d_energy[k] = f->d_out
                          + gnss_gpu::off_energy(C.n_prn, C.n_chan, gnss_gpu::ROWS_PLAIN, 1)
                          + row0 * sizeof(double);
            d_jobs[k] = (gnss_cuda::DespreadJob*)f->d_jobs + (size_t)r * C.n_prn;
        }
        const bool paired =
            D
            && P.C->despread->enqueue_batch_nm_pair(
                *D->C->despread, d_rec, d_chan_scale, d_chan_ids, 1, 1, _num_local_freq, wstart,
                specs[0], specs[1], P.d_wave, d_jobs[0], d_corr[0], d_energy[0], D->d_wave,
                d_jobs[1], d_corr[1], d_energy[1], (void*)stream, /*conjugate=*/true);
        for (int k = 0; k < 2; ++k) {
            ChainFrame* f = frames[k];
            if (!f)
                continue;
            if (!paired)
                f->C->despread->enqueue_batch_nm(d_rec, d_chan_scale, d_chan_ids, f->d_wave, 1, 1,
                                                 _num_local_freq, wstart, specs[k], d_jobs[k],
                                                 d_corr[k], d_energy[k], (void*)stream,
                                                 /*conjugate=*/true);
            auto* pctl = (gnss_gpu::PrnCtl*)(f->stage + gnss_gpu::off_prnctl());
            f->C->record_built(specs[k], pctl + (size_t)r * f->C->n_prn);
            f->n_jobs += (int)specs[k].size();
        }
    }

    for (ChainFrame* f : frames) {
        if (!f)
            continue;
        ((gnss_gpu::FrameHdr*)f->stage)->n_jobs = f->n_jobs * gnss_gpu::ROWS_PLAIN;
        CHECK_CUDA_ERROR(cudaMemcpyAsync(f->d_out, f->stage, gnss_gpu::off_corr(f->C->n_prn),
                                         cudaMemcpyHostToDevice, stream));
    }
}

cudaEvent_t cudaGnssTrack::execute(cudaPipelineState& pipestate, const std::vector<cudaEvent_t>&) {
    pre_execute();
    record_start_event();
    const cudaStream_t stream = device.getStream(cuda_stream_id);
    gnss::TrackChain& S = chain();
    gnss::TrackChain* D = static_cast<cudaGnssTrackState*>(command_state.get())->data.get();
    const int n_rec = S.n_hops_frame / S.hops_per_record;

    // The frame's first sample, stamped by the F-engine: every code phase is referred to it.
    const auto meta = std::dynamic_pointer_cast<GnssChanMetadata>(
        device.get_gpu_memory_array_metadata(_gpu_mem_input, pipestate.gpu_frame_id));
    if (!meta || meta->sample_seq < 0 || meta->sample_seq % S.fft_len != 0)
        FATAL_ERROR("cudaGnssTrack: the frame in '{:s}' has no sample_seq on a hop boundary",
                    _gpu_mem_input);
    const long long hop0_frame = meta->sample_seq / S.fft_len;

    const auto* d_frame = (const unsigned char*)device.get_gpu_memory_array(
        _gpu_mem_input, pipestate.gpu_frame_id, _gpu_buffer_depth,
        (size_t)S.n_hops_frame * _num_local_freq);
    const size_t out_bytes = gnss_gpu::frame_bytes(S.n_prn, S.n_chan, gnss_gpu::ROWS_PLAIN, 1);
    auto* d_out = (char*)device.get_gpu_memory_array(_gpu_mem_output, pipestate.gpu_frame_id,
                                                     _gpu_buffer_depth, out_bytes);
    // The jobs are per frame slot, since frames pipeline. The replicas are scratch shared by
    // every frame, whose work is all on this command's stream.
    const size_t jobs_bytes = (size_t)gnss_gpu::max_specs(S.n_prn) * sizeof(gnss_cuda::DespreadJob);
    void* d_jobs = device.get_gpu_memory_array(unique_name + "_jobs", pipestate.gpu_frame_id,
                                               _gpu_buffer_depth, jobs_bytes);
    // The data component's replicas have their own scratch, as a paired launch writes both.
    const size_t wave_bytes =
        (size_t)3 * S.n_prn * S.n_chan * S.hops_per_record * 2 * sizeof(float);
    void* d_wave = device.get_gpu_memory(unique_name + "_wave", wave_bytes);
    auto* d_chan_scale =
        (float*)device.get_gpu_memory(unique_name + "_chan_scale", S.n_chan * sizeof(float));
    auto* d_chan_ids =
        (int*)device.get_gpu_memory(unique_name + "_chan_ids", S.n_chan * sizeof(int));
    if (!_uploaded_static) {
        // The records are in lsb, as the N^2 correlator's are.
        const std::vector<float> ones((size_t)S.n_chan, 1.0f);
        CHECK_CUDA_ERROR(cudaMemcpy(d_chan_scale, ones.data(), S.n_chan * sizeof(float),
                                    cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_chan_ids, _local_channels.data(), S.n_chan * sizeof(int),
                                    cudaMemcpyHostToDevice));
        _uploaded_static = true;
    }

    S.begin_frame(hop0_frame, (void*)stream);

    std::vector<int> expired;
    std::vector<gnss::TrackChain::Seed> seeds = S.snapshot_seeds(expired);
    for (int prn : expired)
        INFO("cudaGnssTrack: PRN {:d} seed expired", prn);
    std::vector<int> trim_gone;
    const std::vector<double> trim_now = S.snapshot_trims(trim_gone);
    for (int prn : trim_gone)
        WARN("cudaGnssTrack: PRN {:d} trim expired with no /set_trim and is zeroed; the code "
             "phase has stepped back to the broker's model.",
             prn);

    ChainFrame pilot{&S, _stage, d_out, d_jobs, &seeds, d_wave};
    ChainFrame data{nullptr, _stage_data, nullptr, nullptr, nullptr, nullptr};
    std::vector<gnss::TrackChain::Seed> seeds_data;
    if (D) {
        // The data component follows the pilot's PRN map, staged here and applied at its
        // begin_frame. A slot the map could not move (no code for the PRN) gets no seed.
        {
            std::lock_guard<std::mutex> lk(D->prn_mtx);
            if (D->prns != S.prns)
                D->pending_prns = S.prns;
        }
        D->begin_frame(hop0_frame, (void*)stream);
        seeds_data = seeds;
        for (int p = 0; p < S.n_prn; ++p)
            if (D->prns[(size_t)p] != S.prns[(size_t)p])
                seeds_data[(size_t)p].have = false;
        data.C = D;
        data.d_out = (char*)device.get_gpu_memory_array(
            _gpu_mem_output_data, pipestate.gpu_frame_id, _gpu_buffer_depth, out_bytes);
        data.d_jobs = device.get_gpu_memory_array(
            unique_name + "_data_jobs", pipestate.gpu_frame_id, _gpu_buffer_depth, jobs_bytes);
        data.seeds = &seeds_data;
        data.d_wave = device.get_gpu_memory(unique_name + "_wave_data", wave_bytes);
    }
    enqueue_frame(pilot, D ? &data : nullptr, trim_now, hop0_frame, d_frame, d_chan_scale,
                  d_chan_ids, stream);

    if ((++_frames & 0xFF) == 1)
        INFO("cudaGnssTrack: frame hop0 {:d}, {:d} active PRNs, {:d} jobs across {:d} records",
             hop0_frame, pilot.n_active, pilot.n_jobs, n_rec);

    return record_end_event();
}
