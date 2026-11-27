#include "cudaGnssN2Project.hpp"

#include "chordMetadata.hpp"           // for chordMetadata
#include "cudaGnssN2ProjectKernel.hpp" // for launch_n2proj_*, n2proj_wz, N2P_*
#include "cudaUtils.hpp"               // for CHECK_CUDA_ERROR
#include "gnssN2ProjectMath.hpp"       // for the record and live-block layouts
#include "gpuCommand.hpp"              // for gpuCommandType
#include "kotekanLogging.hpp"          // for INFO, FATAL_ERROR

#include <algorithm>
#include <cstring>
#include <ctime>
#include <tuple>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::prometheus::Metrics;

REGISTER_CUDA_COMMAND_WITH_STATE(cudaGnssN2Project, cudaGnssN2ProjectState);

cudaGnssN2ProjectState::cudaGnssN2ProjectState(Config& config, const std::string& unique_name,
                                               bufferContainer& host_buffers,
                                               cudaDeviceInterface& device) :
    cudaCommandState(config, unique_name, host_buffers, device),
    lag(Metrics::instance().add_gauge("kotekan_gnss_n2project_gpu_basis_lag_frames", unique_name)),
    dropped(Metrics::instance().add_counter("kotekan_gnss_n2project_gpu_live_dropped_total",
                                            unique_name)) {
    lag.set(-1.0);
    dropped.inc(0); // served from the start
}

cudaGnssN2ProjectState::~cudaGnssN2ProjectState() {
    if (blas)
        cublasDestroy(blas);
}

cudaGnssN2Project::cudaGnssN2Project(Config& config, const std::string& unique_name,
                                     bufferContainer& host_buffers, cudaDeviceInterface& device,
                                     int instance_num,
                                     const std::shared_ptr<cudaCommandState>& state_) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, state_,
                "cudaGnssN2Project"),
    num_elements(config.get<int>(unique_name, "num_elements")),
    num_local_freq(config.get<int>(unique_name, "num_local_freq")),
    buffer_depth(config.get<int>(unique_name, "buffer_depth")),
    samples_per_data_set(config.get<int>(unique_name, "samples_per_data_set")),
    stations(config.get<std::vector<int>>(unique_name, "stations")),
    live_freq_ids(config.get<std::vector<int>>(unique_name, "live_freq_ids")),
    k_slot(config.get<int>(unique_name, "k_lanes_max")),
    corr_mem(config.get<std::string>(unique_name, "n2k_correlation_name") + "_buffer"),
    basis_mem(config.get<std::string>(unique_name, "gpu_mem_basis")),
    live_mem(unique_name + "_live"),
    in_basis(host_buffers.get_buffer(config.get<std::string>(unique_name, "in_basis_buf"))),
    live(host_buffers.get_buffer(config.get<std::string>(unique_name, "live_buf"))),
    corr_bytes((size_t)num_local_freq * gnss_cuda::N2P_WORDS * sizeof(int32_t)),
    rec_bytes((size_t)in_basis->frame_size) {
    const int sub = config.get<int>(unique_name, "sub_integration_ntime");
    if (num_elements != gnss_cuda::N2P_INPUTS)
        FATAL_ERROR("cudaGnssN2Project: num_elements {:d} != {:d}; the kernels are written for "
                    "128 inputs",
                    num_elements, gnss_cuda::N2P_INPUTS);
    if (sub != samples_per_data_set)
        FATAL_ERROR("cudaGnssN2Project: one integration per frame only (samples_per_data_set "
                    "{:d}, sub_integration_ntime {:d})",
                    samples_per_data_set, sub);
    if (stations.empty() || !std::is_sorted(stations.begin(), stations.end())
        || std::adjacent_find(stations.begin(), stations.end()) != stations.end()
        || stations.front() < 0 || stations.back() >= num_elements)
        FATAL_ERROR("cudaGnssN2Project: stations must be ascending, unique and in [0, {:d})",
                    num_elements);
    if (live_freq_ids.empty())
        FATAL_ERROR("cudaGnssN2Project: live_freq_ids is empty");
    if (k_slot < 1 || k_slot > (int)stations.size() || k_slot > 96)
        FATAL_ERROR("cudaGnssN2Project: k_lanes_max {:d} must be in [1, min(96, {:d})]", k_slot,
                    (int)stations.size());
    if (instance_num == 0) {
        in_basis->register_consumer(unique_name);
        live->register_producer(unique_name);
    }
    register_host_buffer(live);
    // Taken in execute(); attributed to this stage now, so --dry-run sees a conflict.
    for (const std::string& m : {corr_mem, basis_mem, live_mem})
        device.register_gpu_memory_name(m);
    for (const char* m : {"_chan", "_v", "_q", "_w", "_m"})
        device.register_gpu_memory_name(unique_name + m);
    gpu_buffers_used.push_back(std::make_tuple(corr_mem, true, true, true));
    gpu_buffers_used.push_back(std::make_tuple(basis_mem, true, false, true));
    gpu_buffers_used.push_back(std::make_tuple(live_mem, true, false, true));
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&staging, rec_bytes));
    set_command_type(gpuCommandType::KERNEL);
    set_name("cudaGnssN2Project");
}

cudaGnssN2Project::~cudaGnssN2Project() {
    unregister_host_buffer(live);
    if (staging)
        cudaFreeHost(staging);
}

cudaGnssN2ProjectState& cudaGnssN2Project::state() {
    return *std::static_pointer_cast<cudaGnssN2ProjectState>(command_state);
}

void cudaGnssN2Project::resolve(const chordMetadata& meta) {
    cudaGnssN2ProjectState& st = state();
    if (!meta.has_coarse_freq())
        FATAL_ERROR("cudaGnssN2Project: {:s} carries no coarse frequencies; live_freq_ids needs "
                    "them",
                    corr_mem);
    const std::vector<int> coarse = meta.get_coarse_freq();
    if ((int)coarse.size() != num_local_freq)
        FATAL_ERROR("cudaGnssN2Project: {:d} coarse frequencies for {:d} channels",
                    (int)coarse.size(), num_local_freq);
    std::vector<int> wanted(live_freq_ids);
    std::sort(wanted.begin(), wanted.end());
    for (int f = 0; f < num_local_freq; ++f)
        if (std::binary_search(wanted.begin(), wanted.end(), coarse[f])) {
            st.slot_chan.push_back(f);
            st.slot_fid.push_back(coarse[f]);
        }
    const int n = (int)stations.size();
    const int S = (int)st.slot_chan.size();
    if (S == 0)
        FATAL_ERROR("cudaGnssN2Project: none of the {:d} live_freq_ids is on this GPU",
                    (int)live_freq_ids.size());
    const size_t need =
        gnss_n2proj::basis_head_bytes(n) + S * gnss_n2proj::basis_slot_bytes(k_slot, n);
    if (rec_bytes < need)
        FATAL_ERROR("cudaGnssN2Project: {:s} frames hold {:d} bytes; the record of {:d} slots "
                    "needs {:d}",
                    in_basis->buffer_name, (int)rec_bytes, S, (int)need);
    const size_t live_need =
        gnss_n2proj::live_head_bytes(S) + 2 * (size_t)S * gnss_cuda::N2P_WORDS * sizeof(int32_t);
    if ((size_t)live->frame_size < live_need)
        FATAL_ERROR("cudaGnssN2Project: {:s} frames hold {:d} bytes; {:d} slots need {:d}",
                    live->buffer_name, (int)live->frame_size, S, (int)live_need);
    const size_t nn = (size_t)gnss_cuda::N2P_INPUTS * gnss_cuda::N2P_INPUTS;
    const size_t nk = (size_t)gnss_cuda::N2P_INPUTS * k_slot;
    st.d_chan = (int*)device.get_gpu_memory(unique_name + "_chan", S * sizeof(int));
    st.d_v = (double2*)device.get_gpu_memory(unique_name + "_v", S * nn * sizeof(double2));
    st.d_q = (double2*)device.get_gpu_memory(unique_name + "_q", S * nk * sizeof(double2));
    st.d_w = (double2*)device.get_gpu_memory(unique_name + "_w", S * nk * sizeof(double2));
    st.d_m = (double2*)device.get_gpu_memory(unique_name + "_m",
                                             S * (size_t)k_slot * k_slot * sizeof(double2));
    CHECK_CUDA_ERROR(
        cudaMemcpy(st.d_chan, st.slot_chan.data(), S * sizeof(int), cudaMemcpyHostToDevice));
    const cublasStatus_t bs = cublasCreate(&st.blas);
    if (bs != CUBLAS_STATUS_SUCCESS)
        FATAL_ERROR("cudaGnssN2Project: cublasCreate: {:s}", cublasGetStatusString(bs));
    st.newest.assign(rec_bytes, 0);
    st.resolved = true;
    INFO("cudaGnssN2Project: {:d} live channels (freq_ids {:d}..{:d}) of {:d}, {:d} stations, up "
         "to {:d} directions, FP64; records from {:s}, live blocks to {:s} ({:.2f} MB/frame)",
         S, st.slot_fid.front(), st.slot_fid.back(), num_local_freq, n, k_slot,
         in_basis->buffer_name, live->buffer_name, live_need / 1e6);
}

void cudaGnssN2Project::check_record(const uint8_t* rec) {
    const cudaGnssN2ProjectState& st = state();
    if (!st.resolved)
        FATAL_ERROR("cudaGnssN2Project: a record in {:s} before the first frame",
                    in_basis->buffer_name);
    const int n = (int)stations.size(), S = (int)st.slot_chan.size();
    int32_t h[7];
    std::memcpy(h, rec + 8, sizeof(h));
    if (h[0] != gnss_n2proj::BASIS_VERSION || h[6] != gnss_n2proj::BASIS_MAGIC || h[1] != n
        || h[2] != 1 || h[3] != S || h[4] != k_slot
        || h[5] != (int32_t)gnss_n2proj::basis_slot_bytes(k_slot, n))
        FATAL_ERROR("cudaGnssN2Project: {:s} holds a record of version {:d}, {:d} stations, {:d} "
                    "integrations, {:d} slots of {:d}; expected version {:d}, {:d}, 1, {:d} of "
                    "{:d}",
                    in_basis->buffer_name, h[0], h[1], h[2], h[3], h[4], gnss_n2proj::BASIS_VERSION,
                    n, S, k_slot);
    if (std::memcmp(rec + 64, stations.data(), 4 * (size_t)n) != 0)
        FATAL_ERROR("cudaGnssN2Project: the record's stations are not this command's");
    for (int s = 0; s < S; ++s) {
        int32_t sh[2];
        std::memcpy(sh,
                    rec + gnss_n2proj::basis_head_bytes(n)
                        + s * gnss_n2proj::basis_slot_bytes(k_slot, n),
                    sizeof(sh));
        if (sh[0] != st.slot_fid[s] || sh[1] < 0 || sh[1] > k_slot)
            FATAL_ERROR("cudaGnssN2Project: record slot {:d} holds freq_id {:d} with k {:d}; "
                        "expected freq_id {:d}, k <= {:d}",
                        s, sh[0], sh[1], st.slot_fid[s], k_slot);
    }
}

void cudaGnssN2Project::write_empty() {
    const cudaGnssN2ProjectState& st = state();
    const int n = (int)stations.size(), S = (int)st.slot_chan.size();
    std::memset(staging, 0, rec_bytes);
    const int32_t h[7] = {gnss_n2proj::BASIS_VERSION,
                          n,
                          1,
                          S,
                          k_slot,
                          (int32_t)gnss_n2proj::basis_slot_bytes(k_slot, n),
                          gnss_n2proj::BASIS_MAGIC};
    std::memcpy(staging + 8, h, sizeof(h));
    const int64_t none = -1;
    std::memcpy(staging + gnss_n2proj::BASIS_SRC_SEQ, &none, sizeof(none));
    std::memcpy(staging + 64, stations.data(), 4 * (size_t)n);
    for (int s = 0; s < S; ++s)
        std::memcpy(staging + gnss_n2proj::basis_head_bytes(n)
                        + s * gnss_n2proj::basis_slot_bytes(k_slot, n),
                    &st.slot_fid[s], sizeof(int32_t));
}

int cudaGnssN2Project::wait_on_precondition() {
    cudaGnssN2ProjectState& st = state();
    // Every record GnssN2Project has finished since the last frame; the newest is applied.
    const timespec now0 = {0, 0};
    for (;;) {
        const int r = in_basis->wait_for_full_frame_timeout(unique_name, st.basis_in_id, now0);
        if (r < 0)
            return -1;
        if (r == 1)
            break;
        const uint8_t* rec = in_basis->frames[st.basis_in_id];
        check_record(rec);
        std::memcpy(st.newest.data(), rec, rec_bytes);
        in_basis->mark_frame_empty(unique_name, st.basis_in_id);
        st.basis_in_id = (st.basis_in_id + 1) % in_basis->num_frames;
        ++st.records;
    }
    if (st.records > 0)
        std::memcpy(staging, st.newest.data(), rec_bytes);
    // A live-block frame if GnssN2Project has one free; otherwise this frame's are dropped.
    live_frame = -1;
    if (live->is_frame_empty(st.live_id)) {
        if (live->wait_for_empty_frame(unique_name, st.live_id) == nullptr)
            return -1;
        live_frame = st.live_id;
        st.live_id = (st.live_id + 1) % live->num_frames;
    } else {
        st.dropped.inc();
        ++st.drops;
    }
    return 0;
}

cudaEvent_t cudaGnssN2Project::execute(cudaPipelineState&, const std::vector<cudaEvent_t>&) {
    pre_execute();
    cudaGnssN2ProjectState& st = state();
    const cudaStream_t stream = device.getStream(cuda_stream_id);
    const auto meta = std::dynamic_pointer_cast<chordMetadata>(
        device.get_gpu_memory_array_metadata(corr_mem, gpu_frame_id));
    if (!meta)
        FATAL_ERROR("cudaGnssN2Project: {:s}[{:d}] has no metadata", corr_mem, gpu_frame_id);
    if (!st.resolved)
        resolve(*meta);
    const int n = (int)stations.size(), S = (int)st.slot_chan.size();
    seq = meta->get_fpga_seq_num();
    tdf = meta->get_time_downsampling_fpga();
    if (st.records == 0)
        write_empty();
    // The staging now becomes the record of what is applied to this frame.
    std::memcpy(&src_seq, staging + gnss_n2proj::BASIS_SRC_SEQ, sizeof(src_seq));
    std::memcpy(staging, &seq, sizeof(seq));
    k_applied.resize(S);
    int k_max = 0;
    for (int s = 0; s < S; ++s) {
        std::memcpy(&k_applied[s],
                    staging + gnss_n2proj::basis_head_bytes(n)
                        + s * gnss_n2proj::basis_slot_bytes(k_slot, n) + 4,
                    sizeof(int32_t));
        k_max = std::max(k_max, (int)k_applied[s]);
    }
    st.lag.set(src_seq >= 0 ? (double)(seq - src_seq) / samples_per_data_set : -1.0);

    auto* d_corr =
        (int32_t*)device.get_gpu_memory_array(corr_mem, gpu_frame_id, buffer_depth, corr_bytes);
    auto* d_rec =
        (uint8_t*)device.get_gpu_memory_array(basis_mem, gpu_frame_id, buffer_depth, rec_bytes);
    const size_t half = (size_t)S * gnss_cuda::N2P_WORDS;
    auto* d_live = (int32_t*)device.get_gpu_memory_array(live_mem, gpu_frame_id, buffer_depth,
                                                         2 * half * sizeof(int32_t));
    record_start_event();
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_rec, staging, rec_bytes, cudaMemcpyHostToDevice, stream));
    gnss_cuda::launch_n2proj_gather(d_corr, st.d_chan, S, d_live, stream);
    if (k_max > 0) {
        gnss_cuda::launch_n2proj_dense(d_corr, st.d_chan, d_rec, n, k_slot, S, k_slot, st.d_v,
                                       st.d_q, stream);
        cublasStatus_t bs = cublasSetStream(st.blas, stream);
        if (bs == CUBLAS_STATUS_SUCCESS)
            bs = gnss_cuda::n2proj_wz(st.blas, S, k_max, k_slot, st.d_v, st.d_q, st.d_w, st.d_m);
        if (bs != CUBLAS_STATUS_SUCCESS)
            FATAL_ERROR("cudaGnssN2Project: cuBLAS: {:s}", cublasGetStatusString(bs));
        gnss_cuda::launch_n2proj_update(d_corr, st.d_chan, d_rec, n, k_slot, S, k_max, k_slot,
                                        st.d_q, st.d_w, stream);
    }
    gnss_cuda::launch_n2proj_gather(d_corr, st.d_chan, S, d_live + half, stream);
    CHECK_CUDA_ERROR(cudaGetLastError());
    if (live_frame >= 0)
        CHECK_CUDA_ERROR(cudaMemcpyAsync(live->frames[live_frame] + gnss_n2proj::live_head_bytes(S),
                                         d_live, 2 * half * sizeof(int32_t), cudaMemcpyDeviceToHost,
                                         stream));
    if ((st.frames++ & 0x3FF) == 0)
        INFO("cudaGnssN2Project: frame seq {:d}, basis from seq {:d} (lag {:.0f}), up to {:d} "
             "directions; {:d} records taken, {:d} live frames dropped",
             seq, src_seq, src_seq >= 0 ? (double)(seq - src_seq) / samples_per_data_set : -1.0,
             k_max, st.records, st.drops);
    return record_end_event();
}

void cudaGnssN2Project::finalize_frame() {
    if (live_frame >= 0) {
        const cudaGnssN2ProjectState& st = state();
        const int S = (int)st.slot_chan.size();
        uint8_t* fr = live->frames[live_frame];
        std::memset(fr, 0, gnss_n2proj::live_head_bytes(S));
        const int32_t h[4] = {gnss_n2proj::LIVE_VERSION, S, gnss_cuda::N2P_WORDS,
                              gnss_n2proj::LIVE_MAGIC};
        std::memcpy(fr, &seq, sizeof(seq));
        std::memcpy(fr + 8, &src_seq, sizeof(src_seq));
        std::memcpy(fr + 16, h, sizeof(h));
        std::memcpy(fr + 64, k_applied.data(), 4 * (size_t)S);
        live->allocate_new_metadata_object(live_frame);
        const auto m = std::dynamic_pointer_cast<chordMetadata>(live->get_metadata(live_frame));
        if (m) {
            m->set_fpga_seq_num(seq);
            m->set_time_downsampling_fpga(tdf);
            m->set_coarse_freq(st.slot_fid);
        }
        live->mark_frame_full(unique_name, live_frame);
        live_frame = -1;
    }
    cudaCommand::finalize_frame();
}
