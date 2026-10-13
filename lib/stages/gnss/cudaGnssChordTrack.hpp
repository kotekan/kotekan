/**
 * @file
 * @brief Broker seeds and per-PRN control for the GNSS GPU commands.
 *  - cudaGnssChordTrackState : public cudaCommandState
 */

#ifndef CUDA_GNSS_CHORD_TRACK_HPP
#define CUDA_GNSS_CHORD_TRACK_HPP

#include "Config.hpp"
#include "GnssCudaDespread.hpp"
#include "bufferContainer.hpp"
#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"
#include "gnssChannelizedReplica.hpp"
#include "restServer.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

/**
 * @class cudaGnssChordTrackState
 * @brief Broker seeds, fleet code trims and slot membership, shared by a GNSS command's
 *        instances, plus the replica bank and GPU despread engine they build from.
 *
 * @conf prns            List of Int. PRN in each slot; the slot count n_prn is fixed.
 * @conf signal          String. Signal descriptor name (gnssSignal.hpp).
 * @conf n_channels      Int. Covering channels.
 * @conf channel_ids     List of Int. Global PFB bin of each covering channel, in local order.
 * @conf n_elements      Int. Required, but only logged.
 * @conf samples_per_data_set Int. Hops per frame.
 * @conf hops_per_record Int. Hops per record; must divide samples_per_data_set. Default 2048.
 * @conf fft_length      Int. Real samples per hop. Default 16384.
 * @conf sample_rate     Double. Real sample rate, Hz. Default 3.2e9.
 * @conf f_offset_hz     Double. Carrier offset from band centre, Hz (on CHORD, the carrier
 *                       itself). Default 0.
 * @conf conjugate       Bool. The F-engine output is conjugated. Default false.
 * @conf dll_spacing     Double. Early/Late offset, chips. Default 0.5.
 * @conf frame0_utc      Double. UTC of absolute sample 0; overrides the telescope's. Default 0.
 * @conf seed_ttl_s      Double. Seed lifetime without a refresh, s; 0 = never. Default 60.
 * @conf trim_ttl_s      Double. Trim lifetime without a refresh, s; 0 = never. Default 0.
 * @conf trim_clamp      Double. Bound on |trim|, chips. Default 3.
 * @conf despread_max_chips      Int. Cap on the per-hop chip gather, a window centred on the
 *                               PFB span; 0 = full span. Default 0.
 * @conf despread_chips_centered Bool. Must be true when despread_max_chips is set; the capped
 *                               window is always centred. Default true.
 * @conf phi_fp16        Bool. Store the Phi tables as fp16. Default false.
 * @conf seed_endpoint   String. POST /set_seeds. Default "/chord_track/set_seeds".
 * @conf trim_endpoint   String. GET /get_trim. Default "/chord_track/get_trim".
 * @conf set_trim_endpoint String. POST /set_trim. Default "/chord_track/set_trim".
 * @conf get_prns_endpoint String. GET /get_prns. Default "/chord_track/get_prns".
 * @conf set_prns_endpoint String. POST /set_prns. Default "/chord_track/set_prns".
 */
class cudaGnssChordTrackState : public cudaCommandState {
public:
    cudaGnssChordTrackState(kotekan::Config& config, const std::string& unique_name,
                            kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device);

    /// POST endpoint: a JSON array of {prn, doppler_hz, code_phase_chips, code_phase_rate,
    /// doppler_rate_hz_s, carrier_trim_hz, ref_hop, code_phase_at_ref_chips}. The first three
    /// are required; each seed replaces the slot's whole seed. PRNs not in the map are ignored.
    void set_seeds_callback(kotekan::connectionInstance& conn, nlohmann::json& request);

    /// POST endpoint: the fleet controller's code trim, a JSON array of {prn, trim_chips}. See
    /// @ref trim_ttl_s for why this is not a /set_seeds field.
    void set_trim_callback(kotekan::connectionInstance& conn, nlohmann::json& request);

    /// GET endpoint: per-PRN code-trim state {prn, trim_chips, age_s, updates}, plus the post
    /// and expiry counters and the TTL and clamp.
    void get_trim_callback(kotekan::connectionInstance& conn);

    // ---- Slot membership (docs/CHORD_LIVE_PRN_RECONFIG.md) ----
    // The broker reassigns slots to satellites as the sky changes. The slot count n_prn never
    // changes: it sizes every buffer, GPU allocation and wire frame, so a resize is a restart.
    // Changing which PRN a slot holds leaves every size the same.
    //
    // Lock order: prn_mtx is taken before seed_mtx or trim_mtx, never after. The REST callbacks
    // that resolve a PRN to a slot hold it across their seed_mtx/trim_mtx section. Other
    // threads read @c prns only under prn_mtx. The GPU thread is its only writer
    // (apply_prn_swaps, under prn_mtx), so it may read @c prns without the lock
    // (snapshot_seeds, snapshot_trims, cudaGnssInject::execute).

    /// POST endpoint: the whole slot->PRN map in slot order, as {"prns": [...]}.
    /// The node diffs it against its own map, so re-sending the current map is a no-op and a
    /// lost post costs only latency. The length must equal n_prn exactly; a different length is
    /// refused, not padded. An optional "at_hop" schedules the swap (see @ref prn_at_hop).
    /// Staged, not applied: @ref apply_prn_swaps applies it at a frame boundary, so no record
    /// is built from half a map.
    void set_prns_callback(kotekan::connectionInstance& conn, nlohmann::json& request);

    /// GET endpoint: {prns, pending, last_hop}, the live map, whether a new one is staged and
    /// the newest frame hop seen.
    void get_prns_callback(kotekan::connectionInstance& conn);

    /// Apply the staged map if it is due. GPU thread only: it re-uploads device code tables on
    /// @c stream. Resets each swapped slot's seed, trim and trim diagnostics and, through
    /// GnssCudaDespread::set_prn, its device code table, Phi cache and last_ang0/last_phi_ddop
    /// records. Anything left over would attribute the old satellite's state to the new one.
    /// The fold history is not reset here: the slot's @ref slot_gen is bumped, and
    /// cudaGnssInject resets it from that.
    void apply_prn_swaps(void* stream /*cudaStream_t*/);

    std::mutex prn_mtx; ///< taken before seed_mtx/trim_mtx; guards @c prns (see above), the
                        ///< staging below and last_hop
    std::vector<int> pending_prns; ///< staged map; empty = nothing staged
    /// Per-slot swap counter, incremented on every swap of the slot. A holder of per-slot
    /// history compares it with the generation it last saw (FoldHist::slot_gen_seen) to know
    /// when to reset.
    std::vector<uint64_t> slot_gen;

    // ---- Scheduled swaps ----
    // A swap applied at each node's next frame boundary lands on a different frame on every
    // node, and the combiner would then fold a window whose instances disagree about which
    // satellite a slot holds. `at_hop` pins the swap to an absolute F-engine hop: the broker
    // picks one a few seconds ahead and every node applies it at its first frame boundary at
    // or after it, before any job of that frame is built.
    //
    // The deadline is on the F-engine's hop counter, which is the same on every node; wall time
    // is not. It is in hops, not samples (GnssChanMetadata::sample_seq is hop * fft_len); the
    // broker derives it from the combiner's hop axis.
    int64_t prn_at_hop = -1;    ///< apply at the first frame with hop0 >= this; <0 = next frame
    int64_t prn_stage_hop = -1; ///< last_hop when the map was staged (re-base guard)
    int64_t last_hop = -1;      ///< newest frame hop0 seen; the deadline is tested on it

    /// Record this frame's absolute first hop. Whatever owns the frame loop must call this at
    /// every frame boundary, before @ref apply_prn_swaps: without it last_hop stays -1 and
    /// every scheduled swap applies immediately, with no error.
    void note_frame_hop(long long hop);

    // Geometry. n_prn is immutable; @c prns is not (see above).
    std::vector<int> prns;
    int n_prn = 0, n_chan = 0, n_elem = 0;
    int hops_per_record = 0, n_hops_frame = 0, fft_len = 16384;
    double sample_rate = 3.2e9, f_offset_hz = 0.0, dll_spacing = 0.5;
    bool _conjugate = false; ///< F-engine conjugation (see DespreadParams::conj_data)
    double frame0_utc = 0.0; ///< UTC of absolute sample 0; 0 = unknown, and the record
                             ///< assembler then stamps records from the host clock

    /// One PRN's live seed. The broker owns these and refreshes them every cycle; one not
    /// refreshed within @ref seed_ttl_s expires.
    struct Seed {
        bool have = false;
        double doppler_hz = 0.0;
        double cp_chips = 0.0; ///< prompt code-phase argument at ref_hop (gnss::SeedState)
        double cp_rate = 0.0;  ///< residual code rate, chips per hop
        double dop_rate = 0.0; ///< Hz/s
        double ctrim_hz = 0.0; ///< broker carrier trim
        long long ref_hop = 0;
        double phase_ref_chips = -1.0; ///< physical code phase at ref_hop (-1 = derive from cp)
        double t_recv = 0.0;           ///< steady_clock seconds when this seed last arrived
    };
    /// Drop a seed not refreshed within this many seconds (config `seed_ttl_s`, 0 = never), so a
    /// satellite that has set stops being despread. The broker re-seeds every cycle, so the
    /// default is many refreshes of margin.
    double seed_ttl_s = 60.0;

    /// Snapshot the live trims, expiring any whose controller stopped posting.
    std::vector<double> snapshot_trims(std::vector<int>& expired);

    /// The expiry sweep itself. The caller must hold trim_mtx.
    ///
    /// snapshot_trims runs it before every copy, so the GPU thread never applies an expired
    /// trim. set_trim_callback runs it too, so each post also expires stale trims in the shared
    /// state, and in /get_trim, while the GPU thread is stalled.
    void expire_trims_locked(std::vector<int>& expired);

    /// Snapshot the live seeds, expiring stale ones in the shared state so every reader sees the
    /// live set. Expired PRN numbers land in @c expired for the caller to log outside the lock.
    std::vector<Seed> snapshot_seeds(std::vector<int>& expired);
    std::mutex seed_mtx;
    std::vector<Seed> seeds;

    // ---- Fleet code trim ----
    // `trim` is written by GnssFleetTrim through /set_trim and added to the propagated code
    // phase (gnss::propagate_seed).
    //
    // It is not a /set_seeds field because a seed post replaces the whole seed, zeroing any
    // field it omits, so a trim sender would have to resend the broker's exact seed. /set_trim
    // carries one number per PRN and touches nothing else, not even the seed's t_recv. It is
    // absolute, not a delta, so a lost post costs latency only.
    //
    // A trim expires: one not refreshed within trim_ttl_s goes to zero (a step of up to
    // trim_clamp) rather than stand as a fixed code offset once its controller has stopped,
    // which the broker's own DLL would then fight. 0 disables expiry.
    double trim_ttl_s = 0.0;
    std::vector<double> trim_t_recv; ///< steady-clock stamp of the last /set_trim, per PRN
    uint64_t trim_posts = 0;         ///< /set_trim requests accepted
    uint64_t trim_expired = 0;

    double trim_clamp = 3.0;  ///< |trim| bound, chips
    std::mutex trim_mtx;      ///< guards the trim vectors and counters
    std::vector<double> trim; ///< per-PRN cp trim, chips (applied cp = model + trim)
    /// Re-pin fold history per slot, shared by the command's instances; GPU thread only. The
    /// producer's dcyc = (applied - dop_prev) * t_abs is the carrier-phase step from the
    /// previous record. A cudaCommand has one instance per in-flight GPU frame and frames
    /// round-robin over them, so a history kept per instance would hold the record a whole
    /// buffer depth of frames back. gpuProcess runs every instance's execute() on one thread in
    /// frame order, so a history kept here is always the true previous record.
    struct FoldHist {
        std::vector<double> dop_prev; ///< previous record's applied carrier (dop + ctrim), Hz
        std::vector<uint8_t> ok;      ///< 0 = no previous record (arc start)
        std::vector<uint64_t> slot_gen_seen; ///< slot_gen already acknowledged (history reset once)
        void init(int n) {
            dop_prev.assign((size_t)n, 0.0);
            ok.assign((size_t)n, 0);
            slot_gen_seen.assign((size_t)n, 0);
        }
    };
    FoldHist fold;
    std::vector<long long> trim_n; ///< /set_trim updates per slot (diagnostics)


    std::unique_ptr<gnss::ChannelizedReplicaBank> replica;
    std::unique_ptr<GnssCudaDespread> despread;
    std::vector<int> covering; ///< local channel indices this signal occupies (0..n_chan-1)
};

#endif // CUDA_GNSS_CHORD_TRACK_HPP
