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
#include "gnssN2ProjectMath.hpp"
#include "gnssProjSubspace.hpp"

#include <complex>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

/**
 * @class GnssN2Project
 * @brief Per frame, per channel: EMA the live block's off-diagonal covariance, solve its top-k
 *        subspace, and (live mode) replace the frame's live block by (I - QQ^H) V (I - QQ^H).
 *
 * Phase 3a of the bright-satellite projection (fixtures/projtest/PROJECTION_PLAN.md): the N^2
 * is linear in the visibilities, so the projection the tracker applies to its rows is applied
 * here to the correlator's own per-frame matrix, EXACTLY and without a second quantisation --
 * one frame (42 ms) of latency, which the offline prototype showed costs nothing (a 1-s-old
 * direction equals the non-causal truth to 0.1 dB; 10 s costs 8-9 dB). The direction comes
 * from the data: the covariance's off-diagonal part (ProjSubspace, the solver the assembler's
 * projection already runs live, diagonal completed from the rank-k model) so the per-input
 * noise levels never enter the eigenproblem, and no satellite identity, ephemeris or health
 * is needed -- unhealthy and non-BRDC emitters are projected like any other.
 *
 * RANK. k is gated per channel: component 0 must carry `frac_first_min` of the off-diagonal
 * energy (a dominant emitter; the quiet-sky GNSS ensemble is FULL RANK at 26-48 inputs and
 * its brightest member carries ~13 %, so nothing is projected until something dominates) and
 * `lambda_min_rel` mean autos; further components need `rel_min` of the first eigenvalue and
 * `frac_next_min` of the remaining energy (ProjSubspace's gates); `k_max` caps it. Every
 * projected direction removes 1/n_live of the sky with it, so the cap is deliberate at the
 * pathfinder and lifts with the dish count.
 *
 * MODES. `shadow`: solve and report, write nothing (and pass frames through unchanged if an
 * out_buf is given). `live`: write the projected live block into the output copy that
 * N2Accumulate reads; the input frame stays raw, which is what the subspace estimator must
 * see (a projected input would null its own source and oscillate).
 *
 * REPORTING. Per channel gauges (labelled by freq_id): k, lambda0 in units of the mean auto,
 * frac0 (the trigger), null_db = the strongest coherent component left after projection
 * against the one before (a second ProjSubspace tracks the projected matrix). Optional JSONL
 * archive every `archive_period_s`.
 *
 * @buffer in_buf   int32 n2k_correlation [Tc][F][blocks][16][16][2] (host_correlation_buffer)
 * @buffer out_buf  optional, same shape: the (projected) copy for N2Accumulate
 *
 * @conf num_elements, num_local_freq, samples_per_data_set, sub_integration_ntime  as the N^2
 * @conf stations         Int list. Live station indices (correlator order); default all.
 * @conf mode             String. shadow | live (default shadow).
 * @conf tau_s            Double. Covariance EMA time constant, s (0.5).
 * @conf k_max            Int. Rank cap (3).
 * @conf frac_first_min   Double. Trigger: component 0's off-diagonal energy fraction (0.4).
 * @conf lambda_min_rel   Double. Trigger: lambda0 / mean auto (3.0).
 * @conf pr_min           Double. A component must be SPREAD over the array: participation
 *                        ratio 1 / sum |q_i|^4 >= pr_min (6.0). A lone correlated input pair
 *                        (cross-talk, a saturated dish's two pols: B04X/B04Y on 10-03) is a
 *                        rank-1 term with PR = 2 that the least-squares fit scores as a full
 *                        component; a satellite spans 20-40 of 48 inputs.
 * @conf rel_min, frac_next_min  Double. ProjSubspace's rank gates (0.2, 0.3).
 * @conf solve_every      Int. Frames between solves per channel (4).
 * @conf metric_period_s  Double. Gauge update period (1.0).
 * @conf archive_path     String. JSONL path, empty = none.
 * @conf archive_period_s Double. (10.0)
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
    Buffer* out_buf = nullptr;

    const int num_elements;
    const int num_local_freq;
    const int samples_per_data_set;
    const int sub_integration_ntime;
    const int n_integrations;
    std::vector<int> stations;
    const std::string mode;
    const bool live;
    const double tau_s;
    const int k_max;
    const double frac_first_min;
    const double lambda_min_rel;
    const double pr_min;
    const double rel_min;
    const double frac_next_min;
    const int solve_every;
    const double metric_period_s;
    const std::string archive_path;
    const double archive_period_s;

    gnss_n2proj::Layout layout;
    gnss::ProjSubspace sub; ///< the raw live block's subspace
    gnss::ProjSubspace nul; ///< the projected block's (for null_db)
    std::vector<int> k_cur, on;
    std::vector<double> lam0_rel, frac0, null_db, mean_auto, pr0;
    std::vector<int> q_use; ///< per channel: the solved components the gates accepted, in order
    std::ofstream archive;
};

#endif
