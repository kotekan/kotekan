#ifndef GNSS_CUDA_DESPREAD_HPP
#define GNSS_CUDA_DESPREAD_HPP

#include "gnssChannelizedDespread.hpp" // for DespreadResult
#include "gnssChannelizedReplica.hpp"  // for ChannelizedReplicaBank

#include <array>
#include <complex>
#include <memory>
#include <vector>

/**
 * Host-side driver for the fused GPU GNSS despread (lib/cuda/cudaGnssDespreadKernel.cu).
 *
 * Owns the device buffers and the per-PRN Doppler-bucketed Phi filter caches; the tracker calls
 * @c upload_window once per record, then @c despread3 per active PRN for the Early/Prompt/Late
 * triple. Replica semantics == ChannelizedReplicaBank::hoprate_stream (the validated closed form);
 * Phi tables cover ALL subband channels (a job's covering subset is a bitmask), rebuilt on the CPU
 * only when a sat's Doppler moves > refresh_hz. G1b of docs/gnss_gpu_migration.md: correctness
 * first (synchronous launches, one PRN per call); cross-PRN batching + cudaProcess stream overlap
 * is the G1c throughput step.
 *
 * Only built when USE_CUDA=ON (stages CMake guards the source + defines GNSS_CUDA for the tracker).
 */
class GnssCudaDespread {
public:
    /// @param bank the tracker's replica bank (code tables, filter builder, geometry)
    /// @param n_prn number of PRN slots  @param n_chan channels in this subband (<=64)
    /// @param refresh_hz rebuild a PRN's Phi bucket when its Doppler moves further than this
    GnssCudaDespread(gnss::ChannelizedReplicaBank& bank, int n_prn, int n_chan, int chan_offset,
                     int n_hops, double sample_rate, double f_offset, double refresh_hz = 100.0);
    ~GnssCudaDespread();

    /// Upload one record window ([hop][chan] interleaved complex float, as the tracker holds it).
    void upload_window(const std::complex<float>* window, long long window_start_sample);

    /// One (PRN x E/P/L) despread request against the uploaded window. @c covering holds this
    /// subband's LOCAL channel indices.
    struct Spec {
        int p;                     ///< PRN slot
        double cp_seed;            ///< commanded prompt code phase (chips, signal units)
        double spacing_chips;      ///< Early/Late offset
        double doppler_hz;         ///< replica carrier (the tracker's fixed f_ref)
        std::vector<int> covering; ///< local channel indices in this PRN's covering set
    };

    /// Batched despread: ALL requested PRNs' E/P/L triples in ONE kernel launch (G1c -- one
    /// launch per record instead of one per PRN). Results parallel to @c specs, each ordered
    /// {early, prompt, late}, channel-summed, matching gnss::channelized_despread's fields.
    std::vector<std::array<gnss::DespreadResult, 3>>
    despread_batch(const std::vector<Spec>& specs);

    /// Single-PRN convenience (= despread_batch of one Spec).
    std::array<gnss::DespreadResult, 3> despread3(int p, double cp_seed, double spacing_chips,
                                                  double doppler_hz,
                                                  const std::vector<int>& covering);

    /// Phase-F (cudaProcess chain) entry: despread one record window read IN PLACE from a
    /// DEVICE-resident channel-major array (row stride @c data_stride hops -- the internal
    /// ring), on a CALLER stream, results into CALLER device buffers, NO synchronization.
    /// @c d_jobs_slot must hold >= specs.size() jobs (a per-frame arena slice: each record's
    /// jobs get their own slice so multiple records stay in flight). Phi-bucket rebuilds
    /// (rare: Doppler moved > refresh_hz) still upload synchronously.
    /// JOBS AND OUTPUT ROWS DIFFER HERE: one job per spec emits FOUR rows (E, P, L, P_HEAD), and
    /// the return value is the ROW count (4*specs.size()) -- what indexes d_corr_out/d_energy_out
    /// and PrnCtl::job0. Advance @c d_jobs_slot by specs.size(), not by the return value.
    /// Opaque pointer types keep this header CUDA-free for the tracker include.
    int enqueue_batch_device(const void* d_window /*float2*/, int data_stride,
                             long long window_start_sample, const std::vector<Spec>& specs,
                             void* d_jobs_slot /*gnss_cuda::DespreadJob, [specs]*/,
                             void* d_corr_out /*double2, [4*specs][n_chan]*/,
                             void* d_energy_out /*double, [4*specs][n_chan]*/,
                             void* stream /*cudaStream_t*/);

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

#endif
