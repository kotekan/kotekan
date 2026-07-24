#ifndef CUDA_GNSS_TRACK_HPP
#define CUDA_GNSS_TRACK_HPP

#include "Config.hpp"
#include "GnssCudaDespread.hpp"
#include "bufferContainer.hpp"
#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"
#include "gnssChannelizedReplica.hpp"
#include "restServer.hpp"

#include <complex>
#include <string>
#include <memory>
#include <mutex>
#include <vector>

/**
 * Shared state for @ref cudaGnssTrack (one per stage, shared by the per-frame command
 * instances): the replica bank + GPU despread driver, the broker-facing seed set (REST
 * /set_seeds, same contract as GnssChannelizedTracker), the tracking control state (pinned
 * f_ref per PRN, fence anchors), and the device ring bookkeeping. All of pass-1's decisions
 * live HERE; the per-instance command objects are stateless dispatchers.
 */
class cudaGnssTrackState : public cudaCommandState {
public:
    cudaGnssTrackState(kotekan::Config& config, const std::string& unique_name,
                       kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device);
    ~cudaGnssTrackState();

    void set_seeds_callback(kotekan::connectionInstance& conn, nlohmann::json& request);

    // Geometry / config (immutable after construction).
    std::vector<int> prns;
    int n_prn = 0, n_chan = 0, chan_offset = 0, fft_len = 0, hops_per_record = 0;
    double sample_rate = 5e6, capture_utc0 = 0.0, doppler_margin_hz = 5000.0;
    double dll_spacing = 0.5, fll_reacq_hz = 200.0, max_anchor_age_s = 30.0;
    double f_offset_hz = 0.0; ///< carrier IF (bin-distance reference for max_cover_bins)
    int max_cover_bins = 0;   ///< DIAGNOSTIC (2026-07-21, L5 ADR-wander channel-width A/B):
                              ///< despread on only the N covering channels nearest the
                              ///< carrier. 0 (default) = the full covering set.
    long long ring_hops = 0; ///< multiple of hops_per_record (windows never straddle the wrap)
    bool quantized = false;  ///< input frames + device ring are CHORD 4+4b bytes (GnssQuantize44
                             ///< upstream); scales arrive via GnssChanMetadata::chan_scale

    // 4+4b dequantization scales (lsb -> volts): the last set seen on frame metadata, staged
    // here for the async upload to the device array (pageable staging = host-synchronous on
    // return, same pattern as the jobs). Empty until the quantizer freezes its bandpass.
    std::vector<float> chan_scale_host;
    bool chan_scale_valid = false; ///< device array holds chan_scale_host

    std::unique_ptr<gnss::ChannelizedReplicaBank> replica;
    std::unique_ptr<GnssCudaDespread> despread;

    // Broker seeds (REST-updated; snapshotted per GPU frame under the mutex).
    std::mutex seed_mtx;
    std::vector<double> dop, cp, cp_rate, dop_rate, ctrim;
    std::vector<long long> ref_hop;
    std::vector<uint8_t> active;

    // Tracking control state (pass-1 owns it; command instances execute in frame order).
    std::vector<double> f_ref;         ///< pinned replica carrier per PRN (NaN = unset)
    std::vector<long long> reacq_hop;  ///< ff_hz ramp anchor per PRN

    // ---- VOLTAGE PEEL (docs/gnss_voltage_peel_live.md) ----
    bool peel = false;         ///< subtract each seeded PRN's waveform before the despread
    double peel_alpha = 0.01;  ///< LMS/EMA rate on the DEROTATED gain (~1/alpha records)
    double peel_min_amp = 0.0; ///< |a| gate: below this the PRN is not peeled at all
    int peel_warmup = 100;     ///< gain updates before a PRN's gain is trusted enough to subtract

    /// Feed-forward gain per (PRN, channel), in the NCO-DEROTATED frame -- which is where it is
    /// SLOWLY VARYING, and therefore where averaging is legitimate. Per channel because it costs
    /// nothing (the despread already reports per channel) and it absorbs the receiver bandpass,
    /// which is not in the replica and which a single combined gain cannot represent.
    std::vector<std::complex<float>> gain; ///< [n_prn][n_chan]
    std::vector<int> gain_n;               ///< updates so far, per PRN (vs peel_warmup)
    /// P7a PREDICTED NAV BITS (broker nav_bits, from the LNAV decode-and-predict): the +-1
    /// sign per 20 ms bit on the capture clock, utc0-anchored. 0 = unknown -> the
    /// decision-directed fallback below. Replaces the one-frame-stale gain_sign wherever
    /// known -- the removal of the ~11 dB sign-lag depth ceiling on data signals.
    struct NavBits {
        double utc0 = 0.0, bit_s = 0.02;
        std::vector<int8_t> bits;
    };
    std::vector<NavBits> nav_bits; ///< [n_prn], seed-updated under seed_mtx

    /// Last MEASURED overlay/nav sign, per PRN, and the prediction used for the next record.
    /// The EMA above is DE-BITTED (the sign is divided out before averaging), so the sign has to
    /// be carried separately and re-applied at subtraction time. Predicting "same as last" is
    /// exact for a pilot and right ~19 records in 20 on GPS nav data.
    std::vector<double> gain_sign;         ///< [n_prn], +-1

    /// NCO phase MIRROR. The gain is averaged derotated but must be subtracted in the replica
    /// frame, so the peel needs this record's NCO phase -- the same integration
    /// GnssGpuRecordAssemble runs (re-pin fold included). Deliberately duplicated rather than
    /// plumbed across stages: every input (f_nco, fcar, reanchored, wstart) is computed HERE in
    /// pass 1, and a REST/shared-state hop would add a frame of latency to a per-record quantity.
    /// ⚠️ If the assembler's NCO integration changes, change this with it.
    std::vector<double> phi_nco;     ///< [n_prn] radians, wrapped
    std::vector<double> fcar_prev;   ///< [n_prn] previous record's replica f_ref (re-pin fold)
    std::vector<uint8_t> nco_ok;     ///< [n_prn] phase history valid
    std::vector<long long> wstart_prev_p; ///< [n_prn] previous record's window start (sample)

    /// PEEL FLL: the NCO mirror above carries only the COMMANDED carrier (ctrim + the ramp FF).
    /// The residual (true - commanded) is ~0.01 Hz once the broker's shared carrier loop is
    /// closed, but Hz-level before it converges -- and at Hz-level the gain EMA averages a
    /// ROTATING phasor: the magnitude survives at reduced scale while the phase lags ~90 deg, so
    /// the subtraction removes the right amount at the wrong angle, i.e. ~nothing (measured
    /// 2026-07-24 on the broker-less bench: |g| converged and stable, depth pinned at 0 dB).
    /// Track the residual with a per-PRN FLL on the record-to-record phase of the gain
    /// measurements (bit-robust squared discriminator, the codebase's standard), integrate it
    /// into the derotation, and the EMA sees a stationary phasor again. On the live node with
    /// ctrim closed this loop idles near zero; on a bench (or during broker convergence) it is
    /// what makes the peel work at all. Same design as the CPU peel v2's f_track/phi_track.
    double peel_fll_gain = 0.05;
    std::string peel_tag;      ///< chain identity for the health line (multi-chain node)
    /// GROUND TRUTH per PRN: EMA of |residual|/|full| measured on the SAME record, straight
    /// from the mirrored despread rows. 0 = perfect cancellation, 1 = nothing subtracted.
    /// Independent of the combiner's deep estimators, so it separates "the kernel is not
    /// subtracting" from "the depth measurement is wrong" -- the exact ambiguity that cost
    /// two wrong theories on 2026-07-24.
    std::vector<double> peel_ratio;
    std::vector<int> peel_ratio_n;
    std::vector<double> peel_f_track;   ///< [n_prn] residual carrier (Hz, derotated frame)
    std::vector<double> peel_phi_track; ///< [n_prn] integrated FLL phase (rad, wrapped)
    std::vector<std::complex<double>> peel_a_prev; ///< [n_prn] previous FLL-frame gain measurement
    std::vector<uint8_t> peel_prev_ok;             ///< [n_prn] discriminator history valid
    std::vector<long long> peel_prev_wstart;       ///< [n_prn] sample of the last gain update

    /// What the PREVIOUS frame's peel actually subtracted, so the next gain update can undo it
    /// exactly. Storing it beats recomputing: by update time the EMA has moved on.
    struct PeelUsed {
        uint8_t run = 0;
        int8_t s_head = 0;     ///< predicted bit sign used for the head segment (0 = none:
        int8_t s_tail = 0;     ///< the decision-directed fallback measured this record)
        int job0 = -1;
        double phi = 0.0;      ///< the NCO-mirror phase this record was peeled/measured at
        long long wstart = 0;  ///< window start (sample) -- the FLL's time axis
    };
    std::vector<PeelUsed> used;                 ///< [MAX_REC][n_prn]
    std::vector<int> used_prn;                  ///< [MAX_REC*n_prn] -> PRN slot, parallel to used
    int used_n_rec = 0;
    std::vector<double2> mir_corr;   ///< host mirror of the previous frame's corr rows
    std::vector<double> mir_energy;  ///< ... and energies
    int mir_rows = 0;
    int mir_rows_spec = 0;
    cudaEvent_t mir_evt = nullptr;   ///< signals the mirror D2H is complete
    bool mir_pending = false;
    long long _peel_diag_ctr = 0; ///< throttles the per-record full-vs-residual diag

    // Device ring bookkeeping (the ring memory itself is a named GPU allocation).
    bool ring_init = false;
    long long hop0 = 0;      ///< absolute hop of ring position 0 (window tiling anchor)
    long long next_hop = 0;  ///< absolute hop the next ingest is expected to start at
    long long next_rec = 0;  ///< next unprocessed record index k (window = hop0 + k*hpr)
};

/**
 * @class cudaGnssTrack
 * @brief cudaProcess command: GNSS channelized tracking despread (phase F0).
 *
 * Per GPU frame: transpose-ingest the staged channelized frame into an internal channel-major
 * device ring (zero-filling valve-drop gaps so ring position == absolute hop - hop0), run
 * pass-1 control for every complete record window (seed extrapolation with the quadratic
 * code-Doppler feed-forward, code-currency translation into the pinned f_ref, fence), then
 * ONE batched E/P/L despread launch per record reading the window in place from the ring.
 * Emits the gnssGpuChain.hpp frame (control block + raw per-channel correlations) into the
 * @c gpu_mem_output array for cudaOutputData; GnssGpuRecordAssemble builds tracker records
 * downstream. Semantics match GnssChannelizedTracker with carrier_shared (fll_gain must be 0).
 *
 * @conf All GnssChannelizedTracker geometry/config keys (signal, sample_rate, f_offset,
 *       spectrum_length, n_channels, channel_offset, num_taps, pfb_window, prns,
 *       hops_per_record, code_doppler_sign, dll_spacing_chips, fll_reacq_hz, capture_utc0,
 *       doppler_margin_hz), plus:
 * @conf gpu_mem_input   staged channelized frame: [n_hops_f][n_chan] complex float, or one
 *                       4+4b byte per sample when quantized_input is set
 * @conf gpu_mem_output  gnssGpuChain frame (frame_bytes(n_prn, n_chan))
 * @conf ring_records    device ring length in records (default 50)
 * @conf seed_endpoint   REST path for broker seeds (default "/track/set_seeds")
 * @conf quantized_input input frames + ring are CHORD 4+4b (GnssQuantize44 upstream; the
 *                       dequantization scales arrive on GnssChanMetadata::chan_scale).
 *                       in_frame_len is then in BYTES = hops * n_channels. Default false.
 */
class cudaGnssTrack : public cudaCommand {
public:
    cudaGnssTrack(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                  int instance_num, std::shared_ptr<cudaCommandState> state);
    ~cudaGnssTrack();

    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;

private:
    cudaGnssTrackState* st();

    std::string _gpu_mem_input, _gpu_mem_output;
    std::string _mem_ring, _mem_jobs, _mem_scale; // stage-namespaced device allocation names
    // Peel scratch, per frame (never leaves the GPU): the float2 residual window, the peel jobs,
    // the gains actually subtracted, and the reference cross-terms the add-back consumes.
    std::string _mem_resid, _mem_pjobs, _mem_gain, _mem_xcorr;
    int _rows_spec = 4; // gnss_gpu::ROWS_PLAIN, or ROWS_PEEL when this chain peels
    size_t _in_frame_len = 0, _out_frame_len = 0;
    int _n_hops_frame = 0;

    // Host staging for the control block (pageable: the async H2D is host-synchronous on
    // return, so per-instance reuse is safe).
    std::vector<char> _ctl_stage;
    long long _peel_dbg_ctr = 0; ///< throttles the peel INFO log
};

#endif
