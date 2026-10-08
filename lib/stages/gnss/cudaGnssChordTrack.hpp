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
 * @brief Broker seeds + per-PRN control, shared across a GNSS command's instances.
 */
class cudaGnssChordTrackState : public cudaCommandState {
public:
    cudaGnssChordTrackState(kotekan::Config& config, const std::string& unique_name,
                            kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device);

    /// Broker seed contract (/set_seeds): a JSON array of
    /// {prn, doppler_hz, code_phase_chips, code_phase_rate, doppler_rate_hz_s,
    ///  carrier_trim_hz, ref_hop}.
    void set_seeds_callback(kotekan::connectionInstance& conn, nlohmann::json& request);

    /// POST endpoint: the FLEET controller's code trim (task #51 F2). See the long note on
    /// `trim_ttl_s` in the state class for why this is not `set_seeds` with one more field.
    void set_trim_callback(kotekan::connectionInstance& conn, nlohmann::json& request);

    /// GET endpoint: per-PRN code-trim state {prn, trim_chips, age_s, updates}, for watching
    /// the fleet trim without touching the record stream.
    void get_trim_callback(kotekan::connectionInstance& conn);

    // ================= LIVE PRN MEMBERSHIP (docs/CHORD_LIVE_PRN_RECONFIG.md) =============
    //
    // WHY. The node's PRN list was a hand-written string in config/gnss_fleet_chord.yaml and
    // the broker's view of the constellation comes from live BRDC every cycle. Nothing
    // reconciled them, so they drifted apart silently: measured 2026-08-26, Galileo carried 5
    // slots whose satellites no longer exist while E36 -- which transits at 83 deg elevation,
    // essentially through the main beam -- had no slot at all, and had been picked as a noise
    // probe the node could not represent, quietly dropping both gal chains from the q+p
    // presence gate to brightness-only. A startup-only list cannot track a constellation that
    // changes under it, and campaigns now run longer than the list stays true.
    //
    // ⚠️ THE SLOT COUNT NEVER CHANGES; ONLY MEMBERSHIP DOES. n_prn propagates into every
    // buffer, every GPU allocation and the frame sizes negotiated on the wire, so changing it
    // live is a fleet-wide re-plumb including bufferRecv. Changing WHICH satellite occupies a
    // slot at constant count leaves every size byte-identical. That is sufficient: Galileo
    // needs 28 of 32 slots, so the budget was never the binding constraint -- we simply had
    // the wrong 32.
    //
    // ⚠️ LOCK ORDER: prn_mtx IS THE OUTERMOST LOCK. Every REST callback that resolves a PRN to
    // a slot takes it FIRST and holds it across its seed_mtx/trim_mtx section; the GPU thread
    // takes it while applying a swap. Follow that and @ref prns can be read without further
    // ceremony inside; break it and this deadlocks against expire_trims_locked, which reads
    // @c prns with trim_mtx already held.

    /// POST endpoint: the WHOLE slot->PRN map, in slot order, as {"prns": [...]} or a bare
    /// array. DECLARATIVE AND IDEMPOTENT -- the broker sends what it wants the node to hold and
    /// the node diffs; a re-send of the current map is a no-op that costs nothing, so a lost
    /// message costs latency and not correctness (the same reasoning as /set_trim being
    /// absolute rather than a delta).
    ///
    /// ⚠️ THE LENGTH MUST EQUAL n_prn EXACTLY. A shorter or longer list is refused, not
    /// padded: a resize is the fleet-wide re-plumb above, and quietly accepting one here would
    /// present it as an ordinary swap.
    ///
    /// Staged, not applied: the swap lands at the next FRAME BOUNDARY in @ref apply_prn_swaps
    /// so no record is ever assembled from half a map.
    void set_prns_callback(kotekan::connectionInstance& conn, nlohmann::json& request);

    /// GET endpoint: the live map plus what has happened to it -- {prns, n_prn, swaps,
    /// pending, last_error, slot_gen}. The broker reads this to see what it is diffing
    /// against, and an operator reads it to answer "which satellite is slot 7?" without
    /// consulting a config file that may no longer be true.
    void get_prns_callback(kotekan::connectionInstance& conn);

    /// Apply any staged map AT A FRAME BOUNDARY. ⚠️ GPU THREAD ONLY -- it re-uploads device
    /// code tables on @c stream. Returns the slots that changed, so the CALLING COMMAND can
    /// clear its own per-slot history (each command instance keeps its own, and the state
    /// cannot reach it).
    ///
    /// Everything this state keys by slot is reset in here, in ONE place: seed, trim, trim
    /// diagnostics, power EMAs, and -- through GnssCudaDespread::set_prn -- the device code
    /// table, the Phi cache and the carrier-NCO accumulator. A survivor of any of these is the
    /// old satellite's state attributed to the new one, which is the accumulator-identity trap
    /// and is silent in every case.
    std::vector<int> apply_prn_swaps(void* stream /*cudaStream_t*/);

    /// A consistent copy of the slot->PRN map for a thread that does not hold @c prn_mtx.
    std::vector<int> prn_map();

    std::mutex prn_mtx;            ///< OUTERMOST. Guards @c prns and the staging below.
    std::vector<int> pending_prns; ///< staged map; empty = nothing staged
    bool prn_pending = false;
    /// Per-slot swap counter, monotone. A command instance compares its own remembered copy
    /// against this to learn which slots moved WHILE IT WAS NOT THE INSTANCE THAT APPLIED --
    /// cudaCommand runs several instances against one shared state, each with its own
    /// per-slot Doppler history, and only one of them gets the return value of
    /// @ref apply_prn_swaps.
    std::vector<uint64_t> slot_gen;
    uint64_t prn_swaps = 0;   ///< slots swapped since start (diagnostics)
    std::string prn_last_err; ///< why the last POST was refused, "" if none

    // ── SCHEDULED SWAPS: the same map, on the same FRAME, on every node ──────────────────
    // A swap posted "now" lands on whatever frame each node happens to be building, so twelve
    // nodes cross the discontinuity at twelve different instants. The combiner then folds one
    // window whose instances disagree about which satellite slot p IS -- an accumulator
    // identity error that no downstream check can see, because every row is individually
    // well-formed. `at_hop` fixes the swap to an ABSOLUTE F-engine HOP: the broker picks one a
    // couple of seconds ahead, every node stages the same number, and each applies it at the
    // first frame boundary at or after it. Same frame, fleet-wide, and never mid-record
    // because the test runs where the swap already ran -- before a single job is built.
    //
    // ⚠️ THE CLOCK IS THE F-ENGINE'S OWN COUNTER, NOT WALL TIME. It is the axis the records
    // are indexed by and it is identical on every node by construction; wall time is not
    // (measured lag spread across instances runs to seconds), and a wall deadline would put
    // the nodes back on twelve different frames.
    //
    // ⚠️⚠️ HOPS, NOT SAMPLES, AND THE UNIT IS IN THE NAME FOR A REASON. The first version of
    // this called the field `at_seq` and tested it against GnssChanMetadata::sample_seq, which
    // is hop*fft_len -- while the broker, whose only view of the axis is the combiner's
    // `pow_hop`, filled it in HOPS. A deadline 16384x too small is always already past, so
    // every swap took the apply-immediately degrade and the whole mechanism read as working.
    // The hop is the fleet's alignment key everywhere else in this pipeline
    // (GnssCoherentCombiner: "equal hop IS the same sky"); it is the currency here too.
    int64_t prn_at_hop = -1;    ///< apply at the first frame with hop0 >= this; <0 = ASAP
    int64_t prn_stage_hop = -1; ///< hop0 last seen when the map was staged (re-base guard)
    int64_t last_hop = -1;      ///< newest frame hop0 seen; the deadline is tested on it

    /// Record this frame's absolute first HOP. Called by whichever stage owns the frame loop,
    /// at the frame boundary; @ref apply_prn_swaps tests the scheduled deadline against it.
    ///
    /// ⚠️ THE PRODUCER MUST CALL THIS every frame. Without it last_hop stays -1, the deadline
    /// can never be tested, and no error is raised.
    void note_frame_hop(long long hop);

    // Geometry. n_prn is immutable; @c prns is NOT -- see the block above.
    std::vector<int> prns;
    int n_prn = 0, n_chan = 0, n_elem = 0;
    int hops_per_record = 0, n_hops_frame = 0, fft_len = 16384;
    double sample_rate = 3.2e9, f_offset_hz = 0.0, dll_spacing = 0.5;
    bool _conjugate = false; ///< F-engine conjugation (see DespreadParams::conj_data)
    double frame0_utc = 0.0; ///< GPS-disciplined UTC of absolute sample 0; 0 = unset (the
                             ///< assembler then stamps records with HOST time -- see the cpp)

    /// One PRN's live seed. Model-primary: the broker owns these and refreshes them every
    /// cycle, so there is no frozen-seed state to age or unfreeze here.
    struct Seed {
        bool have = false;
        double doppler_hz = 0.0;
        double cp_chips = 0.0; ///< prompt code phase at ref_hop
        double cp_rate = 0.0;  ///< chips per hop (the broker's measured l-a residual)
        double dop_rate = 0.0; ///< Hz/s
        double ctrim_hz = 0.0; ///< broker carrier trim
        long long ref_hop = 0;
        double phase_ref_chips = -1.0; ///< physical code phase at ref_hop (-1 = derive from cp)
        /// steady_clock seconds when this seed last arrived. A seed used to be LATCHED FOREVER:
        /// `have` was set true and never cleared, so a satellite that had set was still
        /// despread, and the spec count grew monotonically with everything that had ever risen.
        /// Measured 2026-08-05 on a 1 h soak: 14 active PRN slots against the 6-7 the broker
        /// seeds, kernel 14.20 -> 22.63 ms, GPU memory 561 -> 601 MiB, saturating toward the
        /// 32-PRN list -- and after ~4 h that is the 100% GPU / 24k-dropped-frame state.
        double t_recv = 0.0;
    };
    /// Drop a seed not refreshed within this many seconds (config `seed_ttl_s`, 0 = never, the
    /// old latching behaviour). The broker re-seeds every --interval (2 s live), so the default
    /// is many refreshes of margin while still retiring a set satellite promptly.
    double seed_ttl_s = 60.0;

    /// Snapshot the live trims, expiring any whose controller stopped posting.
    std::vector<double> snapshot_trims(std::vector<int>& expired);

    /// The expiry sweep itself. ⚠️ CALLER MUST HOLD trim_mtx.
    ///
    /// It is called from the REST callback as well as from execute(), and that is not
    /// belt-and-braces -- it is the only thing that covers the case the TTL exists for.
    /// Expiry that runs ONLY on the consumer's thread cannot protect against the consumer
    /// being dead. Measured on sky 2026-08-15: cx19's and cx43's GPU-0 chains were wedged
    /// (task #60) -- still answering REST, so /set_trim landed and `posts` climbed, but
    /// execute() never ran, so trims sat at 236 s and 949 s old against a 4 s TTL while the
    /// eight healthy instances expired theirs normally. A chain that later resumes would then
    /// apply a quarter-hour-old correction as its first act.
    void expire_trims_locked(std::vector<int>& expired);

    /// Snapshot the live seeds, expiring stale ones IN THE SHARED STATE (so /get_trim and
    /// every consumer sees the live set, not the high-water mark). Expired PRN numbers land
    /// in @c expired for the caller to log OUTSIDE the lock.
    std::vector<Seed> snapshot_seeds(std::vector<int>& expired);
    std::mutex seed_mtx;
    std::vector<Seed> seeds;

    // ---- THE FLEET CONTROLLER'S TRIM (task #51 F2, 2026-08-15) ---------------------------
    //
    // `trim` below is written by GnssFleetTrim through /set_trim and added to the model phase.
    //
    // ⚠️ WHY /set_trim AND NOT ONE MORE FIELD ON /set_seeds. A seed POST that omits a field
    // ZEROES it, so the Python
    // fast-trim thread had to copy the policy cycle's exact dict and substitute one value; an
    // actuator that can silently undo another loop's field is the worst failure this could
    // have. /set_trim carries ONE number and touches nothing else: not seeds, not t_recv. It
    // is ABSOLUTE, not a delta, so a dropped message costs latency and not
    // authority.
    //
    // ⚠️ AND IT EXPIRES. A frozen trim from a controller that died is a permanent, silent code
    // offset that the broker's own slow DLL would then fight -- and "latched forever" is
    // exactly the failure seed_ttl_s exists to have fixed (#13: the despread grew to 14 slots
    // against 6-7 seeded, kernel 14.2 -> 22.6 ms). On expiry the trim goes to ZERO and says
    // so: that is a step of up to `trim_clamp`, but the alternative is a wrong correction held
    // for as long as the process lives, and zero is the state the instrument ran in before
    // this existed. 0 disables.
    double trim_ttl_s = 0.0;
    std::vector<double> trim_t_recv; ///< steady-clock stamp of the last /set_trim, per PRN
    uint64_t trim_posts = 0;         ///< /set_trim requests accepted (the ACHIEVED post rate)
    uint64_t trim_expired = 0;

    double trim_clamp = 3.0;  ///< |trim| bound, chips
    std::mutex trim_mtx;      ///< guards the vectors below (REST getter thread)
    std::vector<double> trim; ///< per-PRN cp trim, chips (applied cp = model + trim)
    /// THE RE-PIN FOLD HISTORY, SHARED ACROSS THE PRODUCER'S INSTANCES. `dcyc` is
    /// (applied - dop_prev) * t_abs: the carrier-phase step between THIS record and the one
    /// immediately before it. cudaCommands are instantiated once per in-flight GPU frame
    /// (gpu_buffer_depth) and the frames round-robin over the instances, so a history kept on
    /// the command is the Doppler of the record THAT INSTANCE saw last -- a whole buffer depth
    /// of frames ago at every frame boundary. The assembler folded that as the one-record
    /// step, and the exported prompt jumped by a uniform random angle at the first record of
    /// every frame. Records are handed over in frame order on the process's one host thread, so
    /// a history that lives here is the true previous record for whichever instance takes the
    /// frame.
    struct FoldHist {
        std::vector<double> dop_prev; ///< previous record's applied carrier (dop + ctrim), Hz
        std::vector<double> t_prev;   ///< and the absolute time it was pinned at, s
        std::vector<uint8_t> ok;      ///< 0 = no previous record (arc start)
        std::vector<uint64_t> slot_gen_seen; ///< slot_gen already acknowledged (history reset once)
        void init(int n) {
            dop_prev.assign((size_t)n, 0.0);
            t_prev.assign((size_t)n, 0.0);
            ok.assign((size_t)n, 0);
            slot_gen_seen.assign((size_t)n, 0);
        }
    };
    FoldHist fold;
    std::vector<long long> trim_n; ///< updates applied, diagnostics


    std::unique_ptr<gnss::ChannelizedReplicaBank> replica;
    std::unique_ptr<GnssCudaDespread> despread;
    std::vector<int> covering;    ///< local channel indices this signal occupies (0..n_chan-1)
    std::vector<int> channel_ids; ///< GLOBAL bin of each local channel (sparse comb; @conf)
};

#endif // CUDA_GNSS_CHORD_TRACK_HPP
