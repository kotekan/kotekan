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
    /// @param n_prn number of PRN slots
    /// @param chan_ids the GLOBAL bin index of every local channel, IN LOCAL ORDER (<=64).
    ///        ARBITRARY: no contiguity, ordering, uniform spacing or comb structure is
    ///        assumed or checked -- whatever the front end presents is what gets synthesized.
    ///        Each local channel's replica is built at the sky frequency of ITS OWN bin,
    ///        because the PFB filter is per bin centre.
    /// @param refresh_hz rebuild a PRN's Phi bucket when its Doppler moves further than this
    ///
    /// There is deliberately NO offset/count form. There was one (`chan_offset + local`), it
    /// suited the airspy node's contiguous subband, and passing 0 for CHORD's stride-16 comb
    /// synthesized every replica near DC against data at 1176 MHz -- noise at every code
    /// phase, for weeks, invisible to every self-consistent test. A caller with a contiguous
    /// subband can say so in one line (std::iota); a caller with an arbitrary channel set now
    /// has no way to get it silently wrong.
    GnssCudaDespread(gnss::ChannelizedReplicaBank& bank, int n_prn,
                     const std::vector<int>& chan_ids, int n_hops, double sample_rate,
                     double f_offset, double refresh_hz = 100.0);
    ~GnssCudaDespread();

    /// Upload one record window ([hop][chan] interleaved complex float, as the tracker holds it).
    void upload_window(const std::complex<float>* window, long long window_start_sample);

    /// One (PRN x E/P/L) despread request against the uploaded window. @c covering holds this
    /// subband's LOCAL channel indices.
    struct Spec {
        int p;                ///< PRN slot
        double cp_seed;       ///< commanded prompt code phase (chips, signal units)
        double spacing_chips; ///< Early/Late offset
        double doppler_hz;    ///< replica carrier (the tracker's fixed f_ref)
        std::vector<int> covering; ///< local channel indices in this PRN's covering set
        /// Broker carrier trim (Hz), added to the replica CARRIER only -- never to the code
        /// rate, because carrier and code are separate control paths. Before 2026-08-04 this
        /// did not exist and `carrier_trim_hz` reached `c.f_nco` and stopped there, consumed
        /// only by GnssGpuRecordAssemble's exported commanded phase. So the despread ignored
        /// the trim while the ADR export already assumed it had been applied
        /// (`_phi += 2*pi*f_nco*dt`) -- the loop had no actuator, deep_rate_hz was invariant to
        /// the trim (measured gain +0.14 against an expected -1.00), and the broker's carrier
        /// loop integrated an error it could never null, to the +-40 Hz rail at any gain.
        /// Adding it here makes the despread and the export consistent for the first time.
        ///
        /// LAST, and defaulted, because several call sites brace-initialize the members before
        /// it positionally and legitimately have no trim: the host tracker applies its own via
        /// f_nco, and the offline harness commands none.
        double ctrim_hz = 0.0;
    };

    /// Batched despread: ALL requested PRNs' E/P/L triples in ONE kernel launch (G1c -- one
    /// launch per record instead of one per PRN). Results parallel to @c specs, each ordered
    /// {early, prompt, late}, channel-summed, matching gnss::channelized_despread's fields.
    std::vector<std::array<gnss::DespreadResult, 3>>
    despread_batch(const std::vector<Spec>& specs);

    /// BENCH ONLY: cap chip_gather's depth. The gather runs n_chips deep per hop, where
    /// n_chips is the chips the PFB filter spans -- 210 at CHORD against the 13 the kernel
    /// comment describes for airspy L5, because a 65536-sample span covers 210 chips at 312.8
    /// samples/chip. Capping it TRUNCATES THE FILTER: the despread output is WRONG and the run
    /// is for timing only. Its purpose is to answer one question before anyone rewrites the
    /// Phi layout -- is synthesis time proportional to the gather depth, or not?
    void set_max_chips(int n);

    /// BENCH: bracket the CHORD split path's two kernels with CUDA events, so the
    /// synthesis / correlation balance can be measured ON THE NODE at the real geometry.
    /// Measuring it in a synthetic bench does not work: chip_gather's depth (job.n_chips) comes
    /// from the PFB span -- ~8 chips per hop at the unit test's 40-sample hop against ~52 at
    /// CHORD's 16384 -- so a per-sample rate from one geometry does not transfer to the other.
    /// Off by default; costs two event records per frame when on.
    /// A/B ARM FOR TASK #52 -- ⚠️ TEMPORARY, remove with task #55. true (default) = carrier
    /// phase from DespreadJob::ang0; false = the pre-86349ac4d wc * n_abs expression. Exists so
    /// half the fleet can run each arm and be compared IN ONE POLL, because the sky churns
    /// faster than a before/after across two restarts can resolve.
    void set_carrier_phase_from_ref(bool on);

    void enable_split_timing(bool on);
    /// Last frame's split, milliseconds. False if timing is off or the events are not resolved.
    /// synthesis = launch_waveform (n_elem-INDEPENDENT: one replica per PRN/chan/hop, broadcast
    /// across elements), correlation = launch_correlate_nm (scales with n_elem). That asymmetry
    /// is why a 1-element measurement cannot be carried to CHORD's 32.
    bool split_timing_ms(double& synthesis_ms, double& correlation_ms) const;

    /// Largest `specs.size()` @ref despread_batch can accept: the DespreadJob arena is sized
    /// n_prn * MAX_REC on the tracker's one-spec-per-PRN-per-record assumption. A caller that
    /// batches many specs against ONE PRN (the A5 refine scan) must chunk against this, or the
    /// H2D fails with "jobs upload: invalid argument" a long way from the cause.
    int max_batch_specs() const;

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
    /// @c d_chan_scale selects the ring's sample type: nullptr = fp32 (d_window is float2);
    /// non-null = CHORD 4+4b (d_window is one byte per sample, and d_chan_scale is the device
    /// [n_chan] lsb->volts array the bytes were ENCODED with -- GnssChanMetadata::chan_scale,
    /// uploaded by the caller). Same kernel either way; only the voltage load differs.
    /// @c d_xcorr_out, when non-null, additionally receives the reference cross-correlations the
    /// analytic peel add-back needs: [4*specs][n_chan] double2, rows 4i+0/1 = <R_P,R_E>/<R_P,R_L>
    /// over the window and 4i+2/3 the same restricted to the head segment. Costs nothing when
    /// null (the accumulators are a template parameter and compile away).
    /// Build + upload the per-PRN jobs. Shared by the fused and N x M paths so the replicas
    /// cannot fork between them.
    void build_jobs(const std::vector<Spec>& specs, void* d_jobs_slot,
                    long long window_start_sample, void* stream);

    int enqueue_batch_device(const void* d_window /*float2 or uint8*/, int data_stride,
                             long long window_start_sample, const std::vector<Spec>& specs,
                             void* d_jobs_slot /*gnss_cuda::DespreadJob, [specs]*/,
                             void* d_corr_out /*double2, [4*specs][n_chan]*/,
                             void* d_energy_out /*double, [4*specs][n_chan]*/,
                             void* stream /*cudaStream_t*/,
                             const void* d_chan_scale = nullptr /*float, [n_chan]*/,
                             void* d_xcorr_out = nullptr /*double2, [4*specs][n_chan]*/,
                             int rows_spec = 4 /*corr/energy row stride per spec*/);

    /// CHORD N x M: generate the replicas ONCE, then correlate them against @c n_elem antennas.
    ///
    /// Same jobs, same replicas, same numerics as @ref enqueue_batch_device -- only the consumer
    /// differs, which is checked by an exact-equality test at N=1 (cuda_gnss_despread_test).
    ///
    /// @c d_frame is the voltage in CHORD's NATIVE [hop][chan][elem] order (element fastest), as
    /// GnssChordVoltageTap emits it -- no transpose anywhere. @c d_wave is scratch for the
    /// materialised replicas, [3*specs][n_chan][n_hops] float2.
    ///
    /// Correlations come back with an ELEMENT AXIS, [4*specs][n_chan][n_elem]; the energies do
    /// NOT ([4*specs][n_chan]), because one replica is correlated against every antenna. That
    /// asymmetry is deliberate and is what the record schema mirrors: energies in the per-PRN
    /// header, correlations in the element blocks (gnssRecord.hpp).
    int enqueue_batch_nm(const void* d_frame, const void* d_chan_scale, const int* d_chan_ids,
                         void* d_wave, int n_elem, int elem_stride, int frame_chan_stride,
                         long long window_start_sample, const std::vector<Spec>& specs,
                         void* d_jobs_slot, void* d_corr_out /*double2 [4*specs][nchan][nelem]*/,
                         void* d_energy_out /*double [4*specs][nchan]*/, void* stream,
                         bool conjugate = false);

    /// Replicas ONLY -- the first half of @ref enqueue_batch_nm with no correlation: same
    /// build_jobs, same launch_waveform, so the waveforms cannot fork from the tracker's.
    /// For GNSS path B (cudaGnssInject): the correlation happens later inside the N^2 kernel,
    /// against these waveforms quantized into synthetic 4+4b lanes (launch_pack44).
    /// @c d_wave [3*specs][n_chan][n_hops] float2, @c d_energy_out [4*specs][n_chan] double
    /// (rows E/P/L/P_HEAD -- row 3 is the head-gated prompt energy the pack's scale uses).
    /// Returns specs.size() (the JOB count; wave rows are 3x, energy rows 4x that).
    int enqueue_waveform(long long window_start_sample, const std::vector<Spec>& specs,
                         void* d_jobs_slot, void* d_wave,
                         void* d_energy_out /*double [4*specs][n_chan]*/, void* stream);

    /// One PRN's VOLTAGE PEEL request (docs/gnss_voltage_peel_live.md).
    ///
    /// The gains are FEED-FORWARD -- a slow EMA of previously tracked gain, carried forward, NOT
    /// fitted to this window -- and PER LOCAL CHANNEL, which is free (the despread already
    /// reports per channel) and absorbs the receiver bandpass that a single combined gain cannot.
    /// A ZERO gain means "do not peel this (PRN, channel)": the voltage passes through untouched,
    /// which is how an unconverged or dropped-lock satellite is gated out.
    ///
    /// Two gains because the window straddles a code-period boundary and the overlay/nav symbol
    /// flips sign exactly there. The boundary hop is NOT passed in -- it is derived from the same
    /// expression the despread's P_HEAD row uses, so the two cannot disagree.
    struct PeelSpec {
        int p;                                   ///< PRN slot
        double cp_seed;                          ///< commanded prompt code phase (chips)
        double doppler_hz;                       ///< replica carrier (the tracker's f_ref)
        std::vector<int> covering;               ///< local channel indices in the covering set
        std::vector<std::complex<float>> a_head; ///< [n_chan] gain before the boundary
        std::vector<std::complex<float>> a_tail; ///< [n_chan] gain at/after the boundary
    };

    /// Peel every spec's prompt waveform out of one record window, on a CALLER stream into a
    /// CALLER device buffer, NO synchronization -- the twin of @ref enqueue_batch_device, and
    /// meant to be enqueued immediately before it so the despread reads the residual.
    /// @c d_pjobs_slot >= specs.size() PeelJob; @c d_gain_slot >= 2*specs.size()*n_chan float2.
    /// @c d_resid_out is [n_chan][n_hops] float2, PACKED -- pass data_stride = n_hops to the
    /// following despread. Returns the job count.
    int enqueue_peel_device(const void* d_window /*float2 or uint8*/, int data_stride,
                            long long window_start_sample, const std::vector<PeelSpec>& specs,
                            void* d_pjobs_slot /*gnss_cuda::PeelJob, [specs]*/,
                            void* d_gain_slot /*float2, [2*specs][n_chan]*/,
                            void* d_resid_out /*float2, [n_chan][n_hops]*/,
                            void* stream /*cudaStream_t*/,
                            const void* d_chan_scale = nullptr /*float, [n_chan]*/);

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

#endif
