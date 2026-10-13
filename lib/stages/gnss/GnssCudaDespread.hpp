#ifndef GNSS_CUDA_DESPREAD_HPP
#define GNSS_CUDA_DESPREAD_HPP

#include "gnssChannelizedDespread.hpp" // for DespreadResult
#include "gnssChannelizedReplica.hpp"  // for ChannelizedReplicaBank

#include <array>
#include <complex>
#include <memory>
#include <vector>

/**
 * Host-side driver for the GPU GNSS despread kernels (lib/cuda/cudaGnssDespreadKernel.cu and
 * lib/cuda/cudaGnssChordDespread.cu).
 *
 * Owns the device code table and the per-PRN, Doppler-bucketed Phi filter tables, and builds the
 * per-PRN jobs. Replicas follow ChannelizedReplicaBank::hoprate_stream. Phi tables cover every
 * local channel (a job selects its covering subset with a 64-bit mask) and are rebuilt on the CPU
 * only when a PRN's Doppler moves more than refresh_hz. @ref despread_batch synchronizes; the
 * enqueue_* calls are ordered on the caller's stream and do not.
 *
 * Built only when USE_CUDA=ON.
 */
class GnssCudaDespread {
public:
    /// @param bank the replica bank (code tables, filter builder, geometry), held by reference
    /// @param n_prn number of PRN slots
    /// @param chan_ids the global bin index of every local channel, in local order (at most 64,
    ///        unique, inside [0, spectrum_length)). No contiguity, ordering, spacing or comb
    ///        structure is assumed: each local channel's replica is built at its own bin's
    ///        centre frequency, because the PFB filter is per bin. There is no offset/count
    ///        form, because an offset of 0 for a strided comb would silently build every
    ///        replica near DC; a contiguous subband passes its ids (std::iota).
    /// @param n_hops hops per record window
    /// @param sample_rate samples per second
    /// @param refresh_hz rebuild a PRN's Phi bucket when its Doppler moves further than this
    GnssCudaDespread(gnss::ChannelizedReplicaBank& bank, int n_prn,
                     const std::vector<int>& chan_ids, int n_hops, double sample_rate,
                     double refresh_hz = 100.0);
    ~GnssCudaDespread();

    // ---- Slot membership (docs/CHORD_LIVE_PRN_RECONFIG.md) ----

    /// Repoint slot @c p at a different satellite: updates the bank's code table for the slot
    /// (unless another engine sharing the bank already has), re-uploads that slot's code to the
    /// device, and drops the per-slot state held here (Phi cache, last_ang0, last_phi_ddop).
    ///
    /// The code upload is ordered on @c stream because the previous frame's kernels may still
    /// be reading the code table; an unordered copy could hand them a mix of two satellites'
    /// codes. Pass the stream the despread is enqueued on, or nullptr for this engine's own
    /// stream (the one @ref despread_batch uses).
    ///
    /// This is only the GPU half of the swap. Seeds, trims, power averages, element cal and
    /// anything else the caller keys by slot still describe the old satellite; see
    /// cudaGnssChordTrackState::apply_prn_swaps.
    ///
    /// Returns false, having changed nothing, if the slot is out of range, the PRN has no code
    /// for this signal, or the bank is FDMA (see ChannelizedReplicaBank::set_prn).
    bool set_prn(int p, int prn, void* stream /*cudaStream_t*/);

    /// Upload one record window ([hop][chan] complex float) for @ref despread_batch.
    void upload_window(const std::complex<float>* window, long long window_start_sample);

    /// One PRN's E/P/L despread request.
    struct Spec {
        int p;                     ///< PRN slot
        double cp_seed;            ///< prompt code-phase argument, component chips
        double spacing_chips;      ///< Early/Late offset from the prompt, component chips
        double doppler_hz;         ///< replica Doppler, Hz: sets the carrier and the code rate
        std::vector<int> covering; ///< local channel indices in this PRN's covering set
        /// The broker's carrier trim, Hz, added to the replica carrier only, never to the code
        /// rate: carrier and code are separate control paths. The exported commanded phase
        /// (PrnCtl::f_nco, GnssGpuRecordAssemble) integrates the same trim, so it must be
        /// applied here. Last and defaulted so that brace-initialized Specs listing only the
        /// members before it carry no trim.
        double ctrim_hz = 0.0;
    };

    /// Despread every spec's E/P/L in one kernel launch, against the uploaded window, and wait
    /// for the result. At most @ref max_batch_specs specs. Results are parallel to @c specs,
    /// each {early, prompt, late}, summed over channels: amplitude, correlation and
    /// replica_energy as gnss::channelized_despread computes them (per_channel left empty).
    std::vector<std::array<gnss::DespreadResult, 3>> despread_batch(const std::vector<Spec>& specs);

    /// Cap the per-hop chip gather at @c n chips (0 = the filter's full span, ~210 chips at
    /// CHORD: 65536 samples at 312.8 samples/chip). Synthesis cost is linear in the depth, but
    /// the span is the channel response, so a capped gather is a different replica. Applies on
    /// top of any cap set on the bank.
    void set_max_chips(int n);

    /// Centre the @ref set_max_chips window on the PFB span instead of starting it at chip 0.
    /// The prototype (a windowed sinc) peaks mid-span, so a one-sided cap shorter than about
    /// half the span cuts through the peak; a centered 105-chip window keeps 99.6% of the filter
    /// energy. Off by default. Every kernel this class launches honours the resulting d_first.
    void set_chips_centered(bool c);

    /// Store the per-PRN Phi tables as __half2, halving their device footprint (storage error
    /// ~3.3e-4 relative). Off by default. Returns whether it is in effect. While set,
    /// @ref despread_batch throws, because its kernel reads Phi as float2; @ref enqueue_waveform
    /// and @ref enqueue_batch_nm support it.
    bool set_phi_fp16(bool on);

    /// Largest @c specs.size() for @ref despread_batch, @ref build_jobs and the Spec-taking
    /// enqueue_* calls: the host job staging, and despread_batch's device job arena, hold n_prn
    /// jobs (one per PRN slot). Nothing checks the bound. It is not gnss_gpu::max_specs(n_prn),
    /// which sizes a whole frame's device arena across records. A caller batching many specs
    /// against one PRN (the search refine) must chunk against this.
    int max_batch_specs() const;

    /// The ang0 (DespreadJob::ang0, radians) this engine last gave the kernel for PRN slot
    /// @p p, read back from the job rather than recomputed. NaN until the slot builds a job,
    /// and again after @ref set_prn repoints the slot. Exported as gnss::REC_ANG0 (see there
    /// for why).
    double last_ang0(int p) const;

    /// Doppler of slot @p p's last job minus the Doppler its Phi table was built at, Hz; NaN as
    /// for @ref last_ang0. Phi is rebuilt only past refresh_hz, so this is per-instance state
    /// (gnss::REC_PHI_DDOP).
    double last_phi_ddop(int p) const;

    /// Build the per-PRN jobs for @c specs and upload them to @c d_jobs_slot on @c stream. Every
    /// enqueue_* call that takes Specs builds its jobs here, so their replicas cannot differ.
    void build_jobs(const std::vector<Spec>& specs, void* d_jobs_slot,
                    long long window_start_sample, void* stream);

    /// CHORD N x M: generate the replicas once, then correlate them against @c n_elem antennas.
    /// With fp32 tables the replicas equal the fused despread's (launch_despread); only the
    /// consumer differs (cuda_gnss_despread_test checks the two kernel paths agree exactly at
    /// N=1).
    ///
    /// @c d_frame is the 4+4b voltage in CHORD's native [hop][chan][elem] order, as
    /// GnssChordVoltageTap emits it, with strides @c frame_chan_stride and @c elem_stride;
    /// @c d_chan_scale is [n_chan] lsb-to-volts and @c d_chan_ids [n_chan] each local channel's
    /// index in the frame. @c d_wave is scratch for the replicas, [3*specs][n_chan][n_hops]
    /// float2. @c conjugate negates the imaginary part of every data sample
    /// (DespreadParams::conj_data).
    ///
    /// Correlations have an element axis, [4*specs][n_chan][n_elem]; energies do not,
    /// [4*specs][n_chan], because one replica serves every antenna. The record schema mirrors
    /// this: energies in the per-PRN header, correlations in the element blocks (gnssRecord.hpp).
    /// Returns 4 * specs.size().
    int enqueue_batch_nm(const void* d_frame, const void* d_chan_scale, const int* d_chan_ids,
                         void* d_wave, int n_elem, int elem_stride, int frame_chan_stride,
                         long long window_start_sample, const std::vector<Spec>& specs,
                         void* d_jobs_slot, void* d_corr_out /*double2 [4*specs][nchan][nelem]*/,
                         void* d_energy_out /*double [4*specs][nchan]*/, void* stream,
                         bool conjugate = false);

    /// Replicas only: the first half of @ref enqueue_batch_nm (same build_jobs, same
    /// launch_waveform) with no correlation. cudaGnssInject uses it: the correlation happens
    /// later in the N^2 kernel, against these waveforms quantized into synthetic 4+4b inputs by
    /// launch_pack44. @c d_wave is [3*specs][n_chan][n_hops] float2, @c d_energy_out
    /// [4*specs][n_chan] double, rows E/P/L/P_HEAD (row 3 is the head-gated prompt energy).
    /// Returns specs.size(), the job count.
    int enqueue_waveform(long long window_start_sample, const std::vector<Spec>& specs,
                         void* d_jobs_slot, void* d_wave,
                         void* d_energy_out /*double [4*specs][n_chan]*/, void* stream);

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

#endif
