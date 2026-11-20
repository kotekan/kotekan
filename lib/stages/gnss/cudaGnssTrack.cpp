#include "cudaGnssTrack.hpp"

#include "GnssChanMetadata.hpp"       // for GnssChanMetadata
#include "cudaGnssDespreadKernel.hpp" // for DespreadJob
#include "cudaUtils.hpp"              // for CHECK_CUDA_ERROR
#include "gnssGpuChain.hpp"           // for FrameHdr, PrnCtl, frame_bytes
#include "kotekanLogging.hpp"         // for FATAL_ERROR, INFO, WARN

#include <cstring> // for memset
#include <memory>  // for dynamic_pointer_cast, shared_ptr
#include <tuple>   // for make_tuple

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND_WITH_STATE(cudaGnssTrack, cudaGnssTrackState);

cudaGnssTrack::cudaGnssTrack(Config& config, const std::string& unique_name,
                             bufferContainer& host_buffers, cudaDeviceInterface& device,
                             int instance_num, std::shared_ptr<cudaCommandState> state) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, state, "cudaGnssTrack",
                "cudaGnssTrack"),
    _gpu_mem_input(config.get<std::string>(unique_name, "gpu_mem_input")),
    _gpu_mem_output(config.get<std::string>(unique_name, "gpu_mem_output")),
    _num_local_freq(config.get<int>(unique_name, "num_local_freq")),
    _local_channels(config.get<std::vector<std::int32_t>>(unique_name, "gnss_local_channels")) {
    const auto* st = static_cast<cudaGnssChordTrackState*>(command_state.get());
    if (st->hosted())
        FATAL_ERROR("cudaGnssTrack: {:s} has a chains block; it runs one chain", unique_name);
    const gnss::TrackChain& S = chain();
    if (S.n_elem != 1)
        FATAL_ERROR("cudaGnssTrack: n_elements is {:d}; the input frame holds one element",
                    S.n_elem);
    // enqueue_batch_nm synthesises three trials from the Phi tables.
    if (S.psi_synth || S.epl_from_prompt)
        FATAL_ERROR("cudaGnssTrack: psi_synth and epl_from_prompt are cudaGnssInject's");
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

    set_command_type(gpuCommandType::KERNEL);
}

cudaGnssTrack::~cudaGnssTrack() {
    cudaFreeHost(_stage);
}

gnss::TrackChain& cudaGnssTrack::chain() {
    return *static_cast<cudaGnssChordTrackState*>(command_state.get())->rows[0].chain;
}

cudaEvent_t cudaGnssTrack::execute(cudaPipelineState& pipestate, const std::vector<cudaEvent_t>&) {
    pre_execute();
    record_start_event();
    const cudaStream_t stream = device.getStream(cuda_stream_id);
    gnss::TrackChain& S = chain();
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
    auto* d_out = (char*)device.get_gpu_memory_array(
        _gpu_mem_output, pipestate.gpu_frame_id, _gpu_buffer_depth,
        gnss_gpu::frame_bytes(S.n_prn, S.n_chan, gnss_gpu::ROWS_PLAIN, 1));
    // The jobs are per frame slot, since frames pipeline. The replicas are scratch shared by
    // every frame, whose work is all on this command's stream.
    auto* d_jobs = (gnss_cuda::DespreadJob*)device.get_gpu_memory_array(
        unique_name + "_jobs", pipestate.gpu_frame_id, _gpu_buffer_depth,
        (size_t)gnss_gpu::max_specs(S.n_prn) * sizeof(gnss_cuda::DespreadJob));
    void* d_wave =
        device.get_gpu_memory(unique_name + "_wave", (size_t)3 * S.n_prn * S.n_chan
                                                         * S.hops_per_record * 2 * sizeof(float));
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

    // The control block is zeroed each frame: a PRN that just set must read as run = 0.
    std::memset(_stage, 0, gnss_gpu::off_corr(S.n_prn));
    auto* hdr = (gnss_gpu::FrameHdr*)_stage;
    auto* winstart = (int64_t*)(_stage + gnss_gpu::off_winstart());
    auto* pctl = (gnss_gpu::PrnCtl*)(_stage + gnss_gpu::off_prnctl());
    hdr->n_rec = n_rec;
    hdr->n_prn = S.n_prn;
    hdr->n_chan = S.n_chan;
    hdr->n_rows_spec = gnss_gpu::ROWS_PLAIN;
    hdr->lanes_per_slot = gnss_gpu::LANES_FULL;
    hdr->seq0 = hop0_frame * (long long)S.fft_len;
    hdr->utc0 = S.frame0_utc;

    std::vector<int> expired;
    const std::vector<gnss::TrackChain::Seed> seeds = S.snapshot_seeds(expired);
    for (int prn : expired)
        INFO("cudaGnssTrack: PRN {:d} seed expired", prn);
    std::vector<int> trim_gone;
    const std::vector<double> trim_now = S.snapshot_trims(trim_gone);
    for (int prn : trim_gone)
        WARN("cudaGnssTrack: PRN {:d} trim expired with no /set_trim and is zeroed; the code "
             "phase has stepped back to the broker's model.",
             prn);

    // The kernels write each record's rows straight into the output frame.
    char* d_corr = d_out + gnss_gpu::off_corr(S.n_prn);
    char* d_energy = d_out + gnss_gpu::off_energy(S.n_prn, S.n_chan, gnss_gpu::ROWS_PLAIN, 1);
    int n_jobs_frame = 0, n_active = 0;
    for (int r = 0; r < n_rec; ++r) {
        const long long hop0 = hop0_frame + (long long)r * S.hops_per_record;
        const long long wstart = hop0 * (long long)S.fft_len;
        int n_psi_dark = 0;
        const std::vector<GnssCudaDespread::Spec> specs =
            S.record_specs(hop0, seeds, trim_now, n_jobs_frame, pctl + (size_t)r * S.n_prn, nullptr,
                           (void*)stream, n_psi_dark);
        if (r == 0)
            n_active = (int)specs.size();
        winstart[r] = wstart;

        const size_t row0 = (size_t)n_jobs_frame * gnss_gpu::ROWS_PLAIN * S.n_chan;
        S.despread->enqueue_batch_nm(
            d_frame + (size_t)r * S.hops_per_record * _num_local_freq, d_chan_scale, d_chan_ids,
            d_wave, 1, 1, _num_local_freq, wstart, specs, d_jobs + (size_t)r * S.n_prn,
            d_corr + row0 * 2 * sizeof(double), d_energy + row0 * sizeof(double), (void*)stream,
            /*conjugate=*/true);
        S.record_built(specs, pctl + (size_t)r * S.n_prn);
        n_jobs_frame += (int)specs.size();
    }
    hdr->n_jobs = n_jobs_frame * gnss_gpu::ROWS_PLAIN;

    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_out, _stage, gnss_gpu::off_corr(S.n_prn),
                                     cudaMemcpyHostToDevice, stream));

    if ((++_frames & 0xFF) == 1)
        INFO("cudaGnssTrack: frame hop0 {:d}, {:d} active PRNs, {:d} jobs across {:d} records",
             hop0_frame, n_active, n_jobs_frame, n_rec);

    return record_end_event();
}
