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
 * Modes. shadow: solve and report, write nothing (frames pass through unchanged if an out_buf
 * is given). live: write the projected live block into the output copy that N2Accumulate
 * reads. The input frame stays raw, as the subspace estimator must see it: a projected input
 * would null its own source and oscillate.
 *
 * Reporting. Per-channel gauges labelled by freq_id: k; lambda0_rel and frac0, the first
 * spread component's eigenvalue in mean autos and its energy fraction (the trigger); pr0,
 * component 0's participation ratio; null_db, the strongest spread component left after
 * projection relative to the first projected one (a second ProjSubspace tracks the projected
 * matrix). The same values go to an optional JSONL archive every archive_period_s.
 *
 * @buffer in_buf   int32 n2k_correlation [Tc][F][blocks][16][16][2] (host_correlation_buffer)
 * @buffer out_buf  optional, same shape: the (projected) copy for N2Accumulate
 *
 * @conf num_elements, num_local_freq, samples_per_data_set, sub_integration_ntime  as the N^2
 * @conf stations         Int list. Live station indices (correlator order); default all.
 * @conf mode             String. shadow | live (default shadow).
 * @conf tau_s            Double. Covariance EMA time constant, s (0.5).
 * @conf k_max            Int. Rank cap (3).
 * @conf frac_first_min   Double. Trigger: the first spread component's energy fraction (0.4).
 * @conf lambda_min_rel   Double. Trigger: its eigenvalue / mean auto (3.0).
 * @conf live_freq_ids    Int list. Live mode writes the projected block only for these absolute
 *                        freq_ids, resolved against the first frame's coarse frequencies (fatal
 *                        if its metadata has none); empty = every channel. Shadow metrics cover
 *                        every channel. The contract with the cosmology pipeline: live writes
 *                        touch exactly the GNSS lobes exempted from first-stage RFI excision
 *                        (the config generator's "exempt").
 * @conf pr_min           Double. A component must be spread over the array: participation
 *                        ratio 1 / sum |q_i|^4 >= pr_min (6.0). A lone correlated input pair
 *                        (cross-talk, or one saturated dish's two pols) is a rank-1 term with
 *                        PR = 2 that the least-squares fit scores as a full component; a
 *                        satellite spans 20-40 of 48 inputs.
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
    const std::vector<int> live_freq_ids;
    std::vector<char> live_ch; ///< per local channel: live writes allowed (set on frame 1)
    bool live_ch_resolved = false;
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
