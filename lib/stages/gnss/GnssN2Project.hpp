/**
 * @file
 * @brief Project the brightest spatially coherent components (the transiting GNSS
 *        satellites) out of the science N^2 per frame, before N2Accumulate.
 *  - GnssN2Project : public kotekan::Stage
 */
#ifndef GNSS_N2_PROJECT_HPP
#define GNSS_N2_PROJECT_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "gnssElemSteer.hpp"
#include "gnssFleetLane.hpp"
#include "gnssN2ProjectMath.hpp"
#include "gnssProjSubspace.hpp"
#include "restServer.hpp"

#include "json.hpp" // nlohmann::json for the REST callbacks

#include <atomic>
#include <complex>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

/**
 * @class GnssN2Project
 * @brief Per frame, per channel: EMA the live block's off-diagonal covariance, solve its top-k
 *        subspace, and write (I - QQ^H) V (I - QQ^H) for the frame's live block V into the
 *        output copy.
 *
 * The N^2 is linear in the visibilities, so projecting the correlator's own per-frame matrix
 * is exact and needs no second quantisation. The direction comes from the data: the EMA of the
 * live block's off-diagonal part, solved by ProjSubspace (which completes the diagonal from its
 * rank-k model), so per-input noise levels never enter the eigenproblem and no satellite
 * identity, ephemeris or health is needed. The applied direction lags by the EMA and up to
 * solve_every frames; a direction ~1 s old nulls within 0.1 dB of the current one, while 10 s
 * old loses 8-9 dB.
 *
 * Rank. k is gated per channel. Components with participation ratio below pr_min are skipped.
 * The first remaining one must carry frac_first_min of the off-diagonal energy left after the
 * skipped ones (0.8x that while already on) and lambda_min_rel mean autos: the quiet-sky GNSS
 * ensemble is full rank and its brightest member carries ~13% of the energy, so nothing is
 * projected until one emitter dominates. Further components must pass ProjSubspace's rel_min
 * and frac_next_min gates; k_max caps k. Each projected direction removes ~1/n_live of the sky
 * with it, so the cap stays small on a small array.
 *
 * Output. The projected live block is written into the output copy that N2Accumulate reads,
 * with the products between the stations and the other inputs projected on the station side
 * (X <- (I - QQ^H) X): the frame is that of the projected station voltages, and the other
 * inputs' own products are unchanged. The input frame stays raw, as the subspace estimator must
 * see it: a projected input would null its own source and oscillate.
 *
 * Reporting. Per-channel gauges labelled by freq_id: k; lambda0_rel and frac0, the first
 * spread component's eigenvalue in mean autos and its energy fraction (the trigger); pr0,
 * component 0's participation ratio; null_db, the strongest spread component left after
 * projection relative to the first projected one (a second ProjSubspace tracks the projected
 * matrix).
 *
 * @buffer in_buf   int32 n2k_correlation [Tc][F][blocks][16][16][2] (host_correlation_buffer)
 * @buffer out_buf  same shape: the projected copy for N2Accumulate
 *
 * @conf num_elements, num_local_freq, samples_per_data_set, sub_integration_ntime  as the N^2
 * @conf stations         Int list. Live station indices (correlator order).
 * @conf mode             String, default live, and live is the only value accepted.
 * @conf tau_s            Double. Covariance EMA time constant, s (0.5).
 * @conf k_max            Int. Rank cap (3).
 * @conf frac_first_min   Double. Trigger: the first spread component's energy fraction (0.4).
 * @conf lambda_min_rel   Double. Trigger: its eigenvalue / mean auto (3.0).
 * @conf live_freq_ids    Int list. The projected block is written only for these absolute
 *                        freq_ids, resolved against the first frame's coarse frequencies (fatal
 *                        if its metadata has none); must not be empty. Metrics cover every
 *                        channel unless eigen_channels is live. The contract with the cosmology
 *                        pipeline: live writes touch exactly the GNSS lobes exempted from
 *                        first-stage RFI excision (the config generator's "exempt").
 * @conf eigen_channels   String, all (default), live or list. With live, a channel outside
 *                        live_freq_ids is skipped entirely: no covariance, no solve and no
 *                        gauges. Its output is the input copy either way, and each channel's
 *                        subspace is its own, so the live channels' output is unchanged. The
 *                        cost is per channel and ~n^2 in the stations, so this is what pays for
 *                        a larger station set; the price is the other channels' diagnostics.
 *                        With list, the eigen path runs on eigen_freq_ids; a live channel
 *                        outside them gets the lane projection only (a lane target beside the
 *                        lobes), and null_db and the eigen gauges stay 0 there.
 * @conf eigen_freq_ids   Int list. With eigen_channels list: the eigen path's absolute
 *                        freq_ids, resolved like live_freq_ids.
 * @conf pr_min           Double. A component must be spread over the array: participation
 *                        ratio 1 / sum |q_i|^4 >= pr_min (6.0). A lone correlated input pair
 *                        (cross-talk, or one saturated dish's two pols) is a rank-1 term with
 *                        PR = 2 that the least-squares fit scores as a full component; a
 *                        satellite spans 20-40 of 48 inputs.
 * @conf solve_every      Int. Frames between solves per channel (4).
 * @conf metric_period_s  Double. Gauge update period (1.0).
 * @conf archive_period_s Double. Accepted; it has no effect.
 *
 * Known-satellite directions (#190). With source lanes or lanes+residual the stage first
 * projects, per live channel, the directions the chains publish on the process-wide ProjBoard
 * (src 2, gnssLaneDirection.hpp: each tracked PRN's averaged prompt row, in the tracker's
 * element frame), mapped onto the stations axis through tracker_stations, conjugated
 * (lane_conj: the N^2 is E_i conj(E_j) and a row is conj(E_e), so the N^2-frame direction is
 * the row's conjugate; the two Rayleigh quotients lane_rq_{conj,raw}_db settle it on sky),
 * ordered by row power, gated by coherence and Gram-Schmidt'd (a sibling chain's copy of the
 * same emitter is dropped). lanes+residual then runs the data-driven path above on what is
 * left, as the detector of unlisted emitters; lanes alone runs no eigen path. Each lane costs
 * 1/n of the sky and ~4 n^2 operations per channel-frame, so k_lanes_max is a budget.
 *
 * @conf source            String. eigen (default: the data-driven subspace only), lanes, or
 *                         lanes+residual.
 * @conf board_group       String. The ProjBoard group of this GPU's chains, e.g. "/gnss0"
 *                         (an assembler's unique_name up to its first '_'). Required with lanes.
 * @conf tracker_stations  Int list. The chains' element axis: station of tracker element e
 *                         (live_element_tiles[e >> 4] * 16 + (e & 15)). Required with lanes.
 * @conf k_lanes_max       Int. Lane directions per channel at most (8).
 * @conf lane_max_age_s    Double. A board entry older than this is ignored (1.0).
 * @conf lane_coh_min      Double. Minimum LaneDirection coherence of a usable row (0.08: five
 *                         times the noise floor ~1/n at 96 elements; a satellite at per-record
 *                         SNR 0.1 per element reads ~0.11).
 * @conf lane_min_new_frac Double. Gram-Schmidt: a direction with less than this fraction of
 *                         its energy outside the basis so far is a duplicate (0.1).
 * @conf lane_conj         Bool. Conjugate the row into the N^2 frame (true).
 *
 * Fleet lanes (docs/CHORD_FLEET_LANES.md). With lanes, GET <unique_name>/get_lanes serves the
 * board's measured lane directions (per owner the channel of best coherence, with the track
 * each was steered with) for the relay that carries them to the other nodes. With fleet_lanes,
 * POST <unique_name>/set_lanes takes the relay's set: each lane is evaluated on the live
 * channels its target ranges cover (gnssFleetLane.hpp: the lane_transfer_freq_ids transfer
 * with the line of sight moved along its track, every fleet_lane_rebuild_s) and appended to
 * build_lanes' candidates after the board's, so a local lane always wins the duplicate test
 * and the budget. A set older than fleet_lane_max_age_s is not used: a dead relay leaves the
 * local lanes.
 *
 * @conf fleet_lanes           Bool. Accept posted lanes (false).
 * @conf elem_positions_enu    Double list, [n][3]. The tracker elements' positions, metres
 *                             ENU (the assemblers' value). Required with fleet_lanes.
 * @conf elem_steer_sign       Double. The assemblers' steering sign (1.0).
 * @conf fleet_lane_max_age_s  Double. A posted set older than this is ignored (10.0).
 * @conf fleet_lane_rebuild_s  Double. How often the posted lanes are re-evaluated along their
 *                             tracks (0.25).
 *
 * auto_raw_db / auto_proj_db per freq_id: the mean auto over the live block before and
 * after projection, dB of correlator units, so a spectrum of what the projection removed
 * can be drawn from the gauges (null_db is a ratio of coherent components, not of power).
 */
class GnssN2Project : public kotekan::Stage {
public:
    GnssN2Project(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container);
    ~GnssN2Project() override = default;
    void main_thread() override;

private:
    using cd = std::complex<double>;

    Buffer* in_buf;
    Buffer* out_buf;

    const int num_elements;
    const int num_local_freq;
    const int samples_per_data_set;
    const int sub_integration_ntime;
    const int n_integrations;
    std::vector<int> stations;
    const std::vector<int> live_freq_ids;
    std::vector<char> live_ch; ///< per local channel: live writes allowed (set on frame 1)
    bool live_ch_resolved = false;
    std::vector<int> freq_ids; ///< per local channel: the absolute freq_id once labelled
    const double tau_s;
    const int k_max;
    const double frac_first_min;
    const double lambda_min_rel;
    const double pr_min;
    const int solve_every;
    const double metric_period_s;

    gnss_n2proj::Layout layout;
    gnss::ProjSubspace sub; ///< the raw live block's subspace
    gnss::ProjSubspace nul; ///< the projected block's (for null_db)
    std::vector<int> k_cur, on;
    std::vector<double> lam0_rel, frac0, null_db, mean_auto, pr0;
    std::vector<int> q_use; ///< per channel: the solved components the gates accepted, in order
    // #190 lane directions
    const std::string source;
    const bool use_lanes;            ///< source != eigen
    const bool use_residual;         ///< the eigen path runs (eigen, or on the residual)
    std::string eigen_channels;      ///< all, live or list
    std::vector<int> eigen_freq_ids; ///< eigen_channels list: the eigen path's absolute freq_ids
    std::vector<char> eigen_ch;      ///< per local channel: the eigen path runs (set on frame 1)
    const std::string board_group;
    std::vector<int> tracker_stations;
    const int k_lanes_max;
    const double lane_max_age_s;
    const double lane_coh_min;
    const double lane_min_new_frac;
    const bool lane_conj;
    std::vector<int> e2a;    ///< tracker element e -> index on the stations axis, -1 if absent
    gnss::ProjBasis lanes;   ///< the channel's lane basis, rebuilt every frame
    std::vector<int> lane_k; ///< per channel: lanes projected this frame
    std::vector<double> lane_rq_conj_db,
        lane_rq_raw_db; ///< per channel: top lane's q^H V q / mean auto
    std::vector<double> auto_raw,
        auto_proj; ///< per channel: mean live auto before / after projection
    uint64_t lane_dup = 0, lane_used = 0;
    struct LaneCand {
        double power, coh;
        std::vector<cd> v; ///< on the stations axis, conjugated per lane_conj
    };
    std::vector<LaneCand> cand;
    /// Read the board for local channel f, build `lanes` from the fresh, coherent entries in
    /// power order, then the fleet lanes that cover the channel; with @p diag also the two
    /// Rayleigh quotients of the strongest one against the raw block V. Returns the basis size.
    int build_lanes(int f, int freq_id, const cd* V, double now_s, bool diag);

    // Fleet lanes (docs/CHORD_FLEET_LANES.md)
    const bool fleet_lanes;
    const double fleet_lane_max_age_s;
    const double fleet_lane_rebuild_s;
    gnss::ElemSteer geo;                   ///< element positions and steering sign (shift() only)
    std::atomic<bool> labelled_pub{false}; ///< freq_ids and live_ch are resolved
    std::mutex fleet_mtx;                  ///< guards the pending set between REST and main_thread
    std::vector<gnss::FleetLane> fleet_pending;
    bool fleet_pending_set = false;
    std::vector<double> inst_pending; ///< per-element instrumental delay, s; empty = none
    bool inst_pending_set = false;
    std::vector<gnss::FleetLane> fleet; ///< main_thread's set
    std::vector<double> inst_delay_s;
    double fleet_t_rx = 0.0;    ///< steady seconds the set was taken
    double fleet_t_built = 0.0; ///< utc the directions were last evaluated at (0 = never)
    bool fleet_on = false;      ///< the set is fresh enough to use this frame
    std::vector<std::vector<std::vector<cd>>> fleet_v; ///< [lane][channel][stations]; empty
                                                       ///< where the lane does not cover it
    std::vector<gnss::ElemSteer::cf> fleet_sh;         ///< scratch
    std::vector<cd> fleet_tmp;                         ///< scratch [n_elem]
    std::vector<int> fleet_k;                          ///< per channel: fleet lanes in the basis
    void take_fleet(double now_s);
    void build_fleet(double t_utc);
    void get_lanes_callback(kotekan::connectionInstance& conn);
    void set_lanes_callback(kotekan::connectionInstance& conn, nlohmann::json& request);
};

#endif
