#ifndef GNSS_FLEET_DLL_HPP
#define GNSS_FLEET_DLL_HPP
/**
 * @file gnssFleetDll.hpp
 * @brief The fleet code discriminator from the per-channel comb: the arithmetic, with no kotekan
 *        dependencies.
 *
 * C++ twin of python/scripts/gnss/gnss_broker/combdll.py: folds every channel of the lobe's
 * per-channel E/P/L into one discriminator per PRN per window. It is a header rather than part
 * of the GnssFleetTrim stage so that scripts/gnss/fleetdll.cpp can run the shipped arithmetic
 * on a fixture, and scripts/gnss/fleetdll_gate.py can compare it with the Python arm on the
 * same bytes.
 *
 *     per record, per PRN, over every channel c of the lobe (all senders, each sender's
 *     columns rotated by exp(+i*phi0), REC_PHI0, and weighted by 1 - REC_PROJ_COST):
 *                                        e = |SUM_c G^E_c|^2 / (SUM_c E^E_c)^2   (and p, l)
 *     per PRN:                           E = mean over the complete records of the newest
 *                                            n_win windows (and P, L)
 *                                        disc = (E-L)/(E+L) ,  q = 2P/(E+L)
 *
 * The coherence unit is the lobe. A sender is a `freq_id mod 16` grouping of channels (one
 * GPU's comb at 3.125 MHz stride), a transport artefact. A reduction that sums coherently
 * within a sender and adds powers across senders puts that grouping into the discriminator as
 * a 320 ns grating comb (3.27 chips at 10.23 Mcps) and an n_sender-fold noise floor. The sender
 * count enters only as completeness: a record fewer than `min_instances` senders reached is
 * not averaged in.
 *
 * Records are the unit of averaging. A record with no live comb is absent, not zero, and a
 * dropped frame leaves its record with fewer channels (same normalised signal, more noise).
 *
 * Presence (the noise floor, the probe and deep gates) is policy and stays in the Python
 * broker. This produces numbers, not verdicts.
 *
 * @author Keith Vanderlinde
 */

#include "gnssRecord.hpp"
#include "gnssTelem.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace gnss {

/// One PRN's fleet discriminator: combdll.fleet_dll_comb's row, without presence.
struct FleetDllRow {
    double disc = 0.0; ///< (E-L)/(E+L); 0 on peak, positive when E > L
    double q = 0.0;    ///< 2P/(E+L); 1 when E = P = L (no peak), ~4 on a clean peak at 0.5 spacing
    double e_pow = 0.0, p_pow = 0.0, l_pow = 0.0;
    double n_chan = 0.0;    ///< channels in the lobe sum, meaned over records
    int n_src = 0;          ///< senders behind the most complete record (completeness only)
    int n_rec = 0;          ///< records averaged
    int64_t hop = -1;       ///< newest F-engine hop in the average
    uint64_t win = 0;       ///< newest window in the average
    uint64_t n_updates = 0; ///< discriminators formed since start; its rate is the update rate
};

/// Result of FleetDll::fold(). BAD_HEADER: the frame's magic, version or shape does not match
/// this build, or it overruns `bytes`, so it was not parsed. LATE: its window is older than the
/// chain's newest and no longer open, so it was dropped.
enum class FoldStatus { OK, BAD_HEADER, LATE };

/// The integrator's constants. The Python arm uses the same expressions (combdll.dll_tau and
/// dll_integrate), and scripts/gnss/fleetdll_gate.py checks that the two trim series agree.
struct TrimPolicy {
    double gain = 0.25;   ///< step = gain * tau
    double leak = 0.05;   ///< per update, not per second; see dll_integrate
    double clamp = 3.0;   ///< |trim| bound, chips
    double spacing = 0.5; ///< tracker Early/Late spacing, chips
    /// Absolute prompt-power floor for the per-window information gate, in the rows' p_pow
    /// units. The broker sends its probe-anchored floor via /set_policy. 0 = no floor: every
    /// armed PRN gets leak-only steps (see FleetDll::integrate).
    double p_floor_abs = 0.0;
};

/// The code discriminator -> delay estimate, in chips. |tau| <= spacing/2 (0.25 chips at the
/// default 0.5 spacing) whatever `disc` is, because the clamp is on the discriminator. The
/// discriminator therefore moves the trim at most gain*spacing/2 chips per update however large
/// the error, and the maximum slew rate is that times the update rate.
inline double dll_tau(double disc, double spacing) {
    return -std::max(-1.0, std::min(1.0, disc)) / 4.0 * (spacing / 0.5);
}

/// One leaky-integrator update. `leak` is per update, so the loop bandwidth scales with the
/// update rate: in continuous form dT/dt = -leak*f*T + gain*f*tau, whose steady state
/// gain*tau/leak does not depend on f but whose closed-loop and noise bandwidths do. Constants
/// stated per second must be converted with the achieved rate before they reach here.
///
/// Under a railed discriminator the trim converges to gain*(spacing/2)/leak, 1.25 chips at the
/// defaults, and no update rate raises that; integration alone never reaches the default
/// 3-chip clamp.
inline double dll_integrate(double trim, double disc, const TrimPolicy& p) {
    const double t = (1.0 - p.leak) * trim + p.gain * dll_tau(disc, p.spacing);
    return std::max(-p.clamp, std::min(p.clamp, t));
}

/**
 * The fleet discriminator, accumulated per chain over a ring of absolute windows.
 *
 * Frames are collated by exact integer match on `win`, with no tolerance and no arrival-order
 * inference. A window closes when any sender on that chain reports a newer one, the counterpart
 * of the Python arm's TelemClient.windows(..., lag=1); nothing waits on a sender. A frame for
 * an already-closed window is counted as LATE and dropped: reopening a window would make the
 * aggregate depend on arrival order.
 */
class FleetDll {
public:
    /// `taps_win` is the window depth served by taps() and rec_series(), independent of the
    /// loop's `n_win`. The loop integrates only the freshest few windows (one update's authority
    /// is fixed, so depth buys it nothing), while the broker's policy cycle averages its
    /// `telem-windows` for the SNR its gates need. 0 = follow n_win.
    FleetDll(int n_win = 4, int min_instances = 2, int taps_win = 0) :
        _n_win(std::max(1, n_win)), _min_instances(min_instances),
        _taps_win(taps_win > 0 ? taps_win : std::max(1, n_win)) {}

    /// One channel's three powers for one PRN over the records of one window, keyed by the
    /// F-engine freq_id from the frame header, never by the sender's column index. Formed from
    /// that channel alone; a power needs no phase reference, so no rotation is applied.
    struct ChanTap {
        double e = 0.0, p = 0.0, l = 0.0;
        int n_rec = 0;
    };

    /// One record's complex partial sums for one PRN over every channel of the lobe that has
    /// reached this window so far; settle() turns it into a RecTap at close. Each sender's
    /// assembler applied exp(-i*phi0) to its comb with a phi0 whose zero is arbitrary per sender
    /// (REC_PHI0), so its columns are multiplied by exp(+i*phi0) before they are added; without
    /// that the cross-sender sum adds unrelated phasors.
    struct RecAcc {
        double gE_re = 0, gE_im = 0, gP_re = 0, gP_im = 0, gL_re = 0, gL_im = 0;
        double wE = 0, wP = 0, wL = 0;
        /// Check on the rotation, a diagnostic only: SUM_i a_i and SUM_i |a_i|^2 over senders
        /// of the derotated, energy-normalised prompt a_i = (G_P/W_P)*exp(+i*phi0). The
        /// cross-sender coherence (|SUM a|^2 - SUM |a|^2) / (SUM |a|^2 * (n-1)) is ~1 when the
        /// senders share one reference and ~0 when they do not.
        double aP_re = 0, aP_im = 0, aP_sq = 0;
        int n_chan = 0; ///< channels behind the sums
        int n_inst = 0; ///< senders behind the sums (completeness only)
        int64_t hop = -1;
    };

    /// One record's three lobe-coherent powers for one PRN, |SUM_c G_c|^2 / (SUM_c E_c)^2 with
    /// each tap on its own replica energy: the unit every downstream reduction averages. A
    /// record with no live comb has no RecTap rather than a zero one. rec_series() serves these
    /// un-averaged for the C/N0 estimator; `n_chan`/`n_inst` say how complete each record was.
    struct RecTap {
        double e = 0.0, p = 0.0, l = 0.0;
        /// cross-sender coherence of the derotated prompts (RecAcc); NaN below two senders
        double xcoh = std::numeric_limits<double>::quiet_NaN();
        int n_chan = 0, n_inst = 0;
        int64_t hop = -1;
    };

    struct WindowAcc {
        uint64_t win = 0;
        std::map<int, std::map<int, RecAcc>> acc;   ///< [record slot][prn], while open
        std::map<int, std::map<int, RecTap>> rec;   ///< [record slot][prn], once closed
        std::map<int, std::map<int, ChanTap>> chan; ///< [prn][freq_id]
        /// freq_id -> the sender that carried it this window. Each channel is routed to exactly
        /// one sender, so a second claimant is an upstream routing fault: the channel goes in
        /// `dup`, is counted (Chain::n_dup_chan) and is left out of the per-channel taps that
        /// taps() serves. It stays in the lobe sum.
        std::map<int, std::string> owner;
        std::set<int> dup;
    };

    /// One PRN's integrator state. `trim` corrects the broker's model, not a particular seed,
    /// so it survives the model being republished every policy cycle.
    struct TrimState {
        double trim = 0.0;
        uint64_t n_steps = 0;   ///< integrator updates; their rate is the achieved loop rate
        uint64_t n_railed = 0;  ///< updates that hit the clamp
        uint64_t n_skipped = 0; ///< leak-only windows: below the prompt-power floor, or no floor
        double last_disc = 0.0;
        uint64_t last_win = 0;
    };

    struct Chain {
        uint64_t newest = 0;
        /// Record length in hops, from the wire header; /get_rec_taps serves it so the C/N0
        /// estimator gets each record's integration time rather than assuming it.
        uint32_t hops_per_record = 0;
        bool have_newest = false;
        std::map<uint64_t, WindowAcc> open; ///< ordered, so "close everything older" is a walk
        std::deque<WindowAcc> closed;       ///< the newest max(n_win, taps_win) closed windows
        std::map<int, FleetDllRow> row;
        /// PRNs the broker armed via /set_policy; integrate() applies the discriminator only to
        /// these. Presence, floors and the arming verdict are policy and stay in the broker.
        std::set<int> armed;
        TrimPolicy policy;
        std::map<int, TrimState> trim;
        uint64_t n_closed = 0; ///< windows closed == integrator steps available
        uint64_t n_late = 0;   ///< frames for a window already closed: dropped, never folded
        uint64_t n_forced = 0; ///< windows force-closed by flush()
        uint64_t n_frames = 0;
        /// freq_ids two senders both carried in one window (see WindowAcc::owner)
        uint64_t n_dup_chan = 0;
        /// Consecutive frames more than _epoch_margin windows behind `newest`; _epoch_strikes
        /// of them re-anchor the chain (see fold()).
        uint64_t n_backwards = 0;
        uint64_t n_epoch_reset = 0; ///< re-anchors performed (the F-engine frame0 moved)
    };

    /// Fold one wire frame. `chain_out` is filled whenever the header parsed.
    FoldStatus fold(const void* frame, size_t bytes, std::string* chain_out = nullptr) {
        const auto* h = (const TelemHeader*)frame;
        // The same check GnssTelemGather applies: a sender on a different record layout would
        // parse at the wrong stride. `bytes` is the capacity the caller can read (the buffer's
        // frame size), not this frame's size: senders ship their own shapes into a buffer
        // sized for the widest.
        const bool ok = h->magic == TELEM_MAGIC && h->version == TELEM_VERSION && h->fft_len > 0
                        && telem_shape_ok(*h, telem_frame_bytes(*h))
                        && telem_frame_bytes(*h) <= bytes;
        if (!ok)
            return FoldStatus::BAD_HEADER;

        const std::string chain(h->chain, strnlen(h->chain, TELEM_NAME));
        const std::string inst(h->inst, strnlen(h->inst, TELEM_NAME));
        if (chain_out)
            *chain_out = chain;

        Chain& c = _chain[chain];
        c.n_frames++;
        c.hops_per_record = h->hops_per_record;
        // Epoch reset. `win` counts from the F-engine's frame0, so an F-engine restart makes
        // every sender's window index restart small, and the LATE rule alone would then drop
        // every later frame. A laggard is a few windows behind; a frame0 move puts the whole
        // stream far back and keeps it there. So a jump of more than `_epoch_margin` windows is
        // counted, and `_epoch_strikes` consecutive such frames re-anchor; any in-order frame
        // clears the run. Only `open` and `newest` are reset: the trims, arming and policy
        // belong to the broker's cycle, and wiping them would force a fleet-wide pull-in.
        if (c.have_newest && h->win < c.newest) {
            if (c.newest - h->win <= _epoch_margin || ++c.n_backwards < _epoch_strikes) {
                c.n_late++;
                return FoldStatus::LATE;
            }
            // Re-anchor on this frame and fold it.
            c.open.clear();
            c.newest = h->win;
            c.have_newest = true;
            c.n_backwards = 0;
            c.n_epoch_reset++;
        } else {
            c.n_backwards = 0;
        }

        const float* rows = telem_rows(frame);
        const int n_prn = h->n_prn;
        const int n_chan = h->n_chan;
        // The sender's row stride, from the wire: senders carry different comb widths, and a
        // compile-time width would read a narrow sender's rows at the wrong offsets.
        const int row_floats = h->n_row_total;

        // The PRN map is read from the data, from record slot 0's rows, as the Python client
        // does. GnssTelemPack compacts the rows to the PRNs despread in the window, so the row
        // order is not the configured PRN order, and fixes the map for the whole window, so
        // slot 0's map holds for every slot. Unused rows read PRN 0. A frame whose slot 0 was
        // not filled reads all zeros here and folds nothing.
        constexpr int MAX_ROWS = 256; // PRN-map ceiling; telem_max_prn is per chain and smaller
        int prn_of_row[MAX_ROWS];
        const int n_row_map = std::min(n_prn, MAX_ROWS);
        for (int p = 0; p < n_row_map; ++p) {
            const float v = rows[telem_row_offset(0, p, n_prn, row_floats) + REC_PRN];
            prn_of_row[p] = (v > 0.0f) ? (int)(v + 0.5f) : 0;
        }

        WindowAcc& w = c.open[h->win];
        w.win = h->win;

        for (int r = 0; r < (int)h->n_rec; ++r) {
            if (!(h->present & (1u << r)))
                continue;
            const int64_t hop =
                (h->wstart0 + (int64_t)r * h->hops_per_record * h->fft_len) / (int64_t)h->fft_len;
            for (int p = 0; p < n_row_map; ++p) {
                const int prn = prn_of_row[p];
                if (prn <= 0)
                    continue;
                const float* row = rows + telem_row_offset(r, p, n_prn, row_floats);

                // This sender's channels, summed raw. Each tap is normalised by its own replica
                // energy, and the three were element-combined and NCO-rotated identically
                // upstream. The Python arm accumulates (raw/E)*E per channel, which is the raw
                // complex.
                double gE_re = 0, gE_im = 0, gP_re = 0, gP_im = 0, gL_re = 0, gL_im = 0;
                double wE = 0, wP = 0, wL = 0;
                int used = 0;
                for (int ch = 0; ch < n_chan; ++ch) {
                    const float* cc = row + telem_chan_offset(ch);
                    const double eP = cc[CHAN_ENERGY];
                    if (eP <= 0.0)
                        continue; // no live comb for this channel this record
                    const double eE = cc[CHAN_E_ENERGY];
                    const double eL = cc[CHAN_L_ENERGY];
                    gE_re += cc[CHAN_E_RE];
                    gE_im += cc[CHAN_E_IM];
                    gP_re += cc[CHAN_RE];
                    gP_im += cc[CHAN_IM];
                    gL_re += cc[CHAN_L_RE];
                    gL_im += cc[CHAN_L_IM];
                    wE += eE;
                    wP += eP;
                    wL += eL;
                    used++;
                }
                if (used == 0 || wP <= 0.0)
                    continue;

                // Projection-cost weight (REC_PROJ_COST): a sender whose live weights lie in the
                // projected subspace by cos^2 = b kept (1 - b) of this satellite's amplitude, so
                // its partial enters the lobe sum weighted (1 - b), numerator and energies
                // alike: maximal-ratio across senders, and the unweighted sum when every sender
                // has the same b. -1 (not applicable) weighs 1; b = 1 is an absent measurement.
                const float pc = row[REC_PROJ_COST];
                const double pw = (pc >= 0.0f) ? 1.0 - std::min(1.0, (double)pc) : 1.0;
                if (pw <= 0.0)
                    continue;

                // Onto the common reference, then into the lobe sum for this record.
                const double phi0 = row[REC_PHI0];
                const double cr = std::cos(phi0), ci = std::sin(phi0);
                RecAcc& a = w.acc[r][prn];
                a.gE_re += (gE_re * cr - gE_im * ci) * pw;
                a.gE_im += (gE_re * ci + gE_im * cr) * pw;
                a.gP_re += (gP_re * cr - gP_im * ci) * pw;
                a.gP_im += (gP_re * ci + gP_im * cr) * pw;
                a.gL_re += (gL_re * cr - gL_im * ci) * pw;
                a.gL_im += (gL_re * ci + gL_im * cr) * pw;
                a.wE += wE * pw;
                a.wP += wP * pw;
                a.wL += wL * pw;
                // Normalise, then rotate, as the Python arm's `gP / eP * rot` does: equal in
                // exact arithmetic, not in floating point.
                {
                    const double nr = gP_re / wP, ni = gP_im / wP;
                    const double ar = nr * cr - ni * ci, ai = nr * ci + ni * cr;
                    a.aP_re += ar;
                    a.aP_im += ai;
                    a.aP_sq += ar * ar + ai * ai;
                }
                a.n_chan += used;
                a.n_inst++;
                a.hop = std::max(a.hop, hop);

                // Per channel, by freq_id, once the record has been admitted above. Normalise,
                // then take the magnitude, as Python's `abs(complex(re/e, im/e))**2` does;
                // (re*re + im*im)/(e*e) is a different float, and the arms are compared at 1e-9.
                auto& per_fid = w.chan[prn];
                for (int ch = 0; ch < n_chan; ++ch) {
                    const float* cc = row + telem_chan_offset(ch);
                    const double eP = cc[CHAN_ENERGY];
                    if (eP <= 0.0)
                        continue;
                    const int fid = (int)h->chan_id[ch];
                    auto own = w.owner.emplace(fid, inst);
                    if (!own.second && own.first->second != inst)
                        w.dup.insert(fid);
                    const double eE = cc[CHAN_E_ENERGY];
                    const double eL = cc[CHAN_L_ENERGY];
                    ChanTap& ct = per_fid[fid];
                    const double ae = std::hypot(cc[CHAN_E_RE] / eE, cc[CHAN_E_IM] / eE);
                    const double ap = std::hypot(cc[CHAN_RE] / eP, cc[CHAN_IM] / eP);
                    const double al = std::hypot(cc[CHAN_L_RE] / eL, cc[CHAN_L_IM] / eL);
                    ct.e += ae * ae;
                    ct.p += ap * ap;
                    ct.l += al * al;
                    ct.n_rec++;
                }
            }
        }


        if (!c.have_newest || h->win > c.newest) {
            c.newest = h->win;
            c.have_newest = true;
        }
        while (!c.open.empty() && c.open.begin()->first < c.newest)
            close_oldest(c, false);
        return FoldStatus::OK;
    }

    /// Force every open window closed. For the offline harness at end of file, so a fixture's
    /// last window is not missing from the answer. Not for the live path, where the next frame
    /// closes the window and a flush would race it.
    void flush() {
        for (auto& cv : _chain)
            while (!cv.second.open.empty())
                close_oldest(cv.second, true);
    }

    /// Apply the policy cycle's decisions: which PRNs may be trimmed, and with what constants.
    ///
    /// A PRN that leaves the armed set keeps its trim: zeroing it would step the commanded code
    /// phase by up to `clamp` chips just when policy stopped being sure. integrate() decays it
    /// through the leak instead, and the tracker's trim TTL removes it if this controller stops.
    void set_armed(const std::string& chain, const std::set<int>& prns, const TrimPolicy& p) {
        Chain& c = _chain[chain];
        c.armed = prns;
        c.policy = p;
    }

    /// {chain: {prn: trim_chips}} for every |trim| >= 1e-3: the only state here that cannot be
    /// rebuilt from the stream; without it a restart leaves every tracker off-peak by its
    /// standing trim. Counters are not included: they are rates since start, and carrying them
    /// across a restart would hide a stalled loop.
    std::map<std::string, std::map<int, double>> trim_snapshot() const {
        std::map<std::string, std::map<int, double>> out;
        for (const auto& cv : _chain)
            for (const auto& tv : cv.second.trim)
                if (std::abs(tv.second.trim) >= 1e-3)
                    out[cv.first][tv.first] = tv.second.trim;
        return out;
    }

    /// Adopt a restored trim for a PRN the broker has just armed; returns true if adopted.
    ///
    /// Adoption waits for arming because an unarmed trim leaks away (0.84 s time constant at
    /// leak 0.05 and 23.84 closes/s), so a trim restored at load could be erased before the
    /// broker's first /set_policy names its PRN. It also keeps the rule that nothing is trimmed
    /// unless the broker armed it.
    ///
    /// Refuses if the PRN already has a trim of its own (the live loop outranks a saved one), or
    /// if the value is not finite or exceeds the policy clamp.
    bool adopt_trim(const std::string& chain, int prn, double trim_chips) {
        auto ci = _chain.find(chain);
        if (ci == _chain.end() || !ci->second.armed.count(prn))
            return false;
        if (!std::isfinite(trim_chips) || std::abs(trim_chips) > ci->second.policy.clamp)
            return false;
        auto ti = ci->second.trim.find(prn);
        if (ti != ci->second.trim.end() && (ti->second.n_steps != 0 || ti->second.trim != 0.0))
            return false;
        ci->second.trim[prn].trim = trim_chips;
        return true;
    }

    /// One (window, record slot, PRN)'s lobe-coherent powers over the served window depth:
    /// combdll.prompt_cn0's `recs`, oldest window first.
    ///
    /// The code loop wants records averaged; the served C/N0 wants the series, because it gates
    /// individual records. Its gates stay in Python. This returns every record with a live
    /// comb, however partial and without the min_instances test; `n_chan`/`n_inst` let the
    /// consumer decline one.
    struct RecRow {
        uint64_t win;
        int slot, prn, n_inst, n_chan;
        double e, p, l;
    };
    std::map<std::string, std::vector<RecRow>> rec_series() const {
        std::map<std::string, std::vector<RecRow>> out;
        for (const auto& cv : _chain) {
            auto& rows = out[cv.first];
            const auto& ring = cv.second.closed;
            for (size_t k = ring.size() > (size_t)_taps_win ? ring.size() - _taps_win : 0;
                 k < ring.size(); ++k)
                for (const auto& sv : ring[k].rec)   // slot, ordered
                    for (const auto& pv : sv.second) // prn, ordered
                        rows.push_back({ring[k].win, sv.first, pv.first, pv.second.n_inst,
                                        pv.second.n_chan, pv.second.e, pv.second.p, pv.second.l});
        }
        return out;
    }

    const std::map<std::string, Chain>& chains() const {
        return _chain;
    }

    /// One PRN's lobe taps over the served window depth, meaned over records: what
    /// `combdll.lobe_taps` returns for one PRN.
    struct LobeTap {
        double e = 0.0, p = 0.0, l = 0.0, n_chan = 0.0;
        /// mean cross-sender coherence over the `n_xcoh` records two or more senders reached;
        /// NaN when none did
        double xcoh = std::numeric_limits<double>::quiet_NaN();
        int n_xcoh = 0;
        int n_rec = 0;  ///< records behind the mean
        int n_inst = 0; ///< senders behind the most complete record (completeness only)
        int64_t hop = -1;
        std::map<int, std::array<double, 4>> chan; ///< freq_id -> {e, p, l, n_rec}
    };

    /// [chain][prn]: the lobe-coherent and per-channel taps over the served window depth
    /// (`taps_win`), reduced here so the broker need not walk every (window, sender, record,
    /// PRN, channel) in Python. As in the Python arm, the aggregate divides by the PRN's record
    /// count and each channel by its own, so a channel live for half the records is a mean over
    /// that half. No policy is applied: records are filtered only on completeness
    /// (`min_instances`), and channels two senders claimed (WindowAcc::dup) are left out.
    std::map<std::string, std::map<int, LobeTap>> taps() const {
        std::map<std::string, std::map<int, LobeTap>> out;
        for (const auto& cv : _chain) {
            auto& per_prn = out[cv.first];
            const auto& ring = cv.second.closed;
            for (size_t k = ring.size() > (size_t)_taps_win ? ring.size() - _taps_win : 0;
                 k < ring.size(); ++k) {
                const WindowAcc& w = ring[k];
                std::set<int> complete; // PRNs with a record this window that passed the gate
                for (const auto& sv : w.rec)
                    for (const auto& pv : sv.second) {
                        const RecTap& r = pv.second;
                        if (r.n_inst < _min_instances)
                            continue;
                        complete.insert(pv.first);
                        LobeTap& t = per_prn[pv.first];
                        t.e += r.e;
                        t.p += r.p;
                        t.l += r.l;
                        t.n_chan += r.n_chan;
                        t.n_rec++;
                        t.n_inst = std::max(t.n_inst, r.n_inst);
                        t.hop = std::max(t.hop, r.hop);
                        if (std::isfinite(r.xcoh)) {
                            t.xcoh = (t.n_xcoh ? t.xcoh : 0.0) + r.xcoh;
                            t.n_xcoh++;
                        }
                    }
                for (const auto& pv : w.chan) {
                    // A PRN with no complete record this window gets no lobe tap from it, so
                    // its channels are skipped too: both tables cover the same windows.
                    if (!complete.count(pv.first))
                        continue;
                    LobeTap& t = per_prn[pv.first];
                    for (const auto& cv2 : pv.second) {
                        if (cv2.second.n_rec <= 0 || w.dup.count(cv2.first))
                            continue;
                        std::array<double, 4>& a = t.chan[cv2.first];
                        a[0] += cv2.second.e;
                        a[1] += cv2.second.p;
                        a[2] += cv2.second.l;
                        a[3] += cv2.second.n_rec;
                    }
                }
            }
            for (auto& pv : per_prn) {
                LobeTap& t = pv.second;
                const double n = t.n_rec ? (double)t.n_rec : 1.0;
                t.e /= n;
                t.p /= n;
                t.l /= n;
                t.n_chan /= n;
                if (t.n_xcoh)
                    t.xcoh /= (double)t.n_xcoh;
                for (auto& cv2 : t.chan) {
                    const double m = cv2.second[3] ? cv2.second[3] : 1.0;
                    cv2.second[0] /= m;
                    cv2.second[1] /= m;
                    cv2.second[2] /= m;
                }
            }
        }
        return out;
    }
    int n_win() const {
        return _n_win;
    }
    int taps_win() const {
        return _taps_win;
    }
    int min_instances() const {
        return _min_instances;
    }

private:
    /// The window's open accumulators become its record powers, once, at close, so a late
    /// sender cannot move a record that has already been averaged.
    static void settle(WindowAcc& w) {
        for (auto& sv : w.acc)
            for (auto& pv : sv.second) {
                const RecAcc& a = pv.second;
                if (a.n_inst <= 0 || a.wP <= 0.0)
                    continue;
                RecTap& r = w.rec[sv.first][pv.first];
                const double aE = std::hypot(a.gE_re, a.gE_im);
                const double aP = std::hypot(a.gP_re, a.gP_im);
                const double aL = std::hypot(a.gL_re, a.gL_im);
                r.e = a.wE > 0.0 ? (aE / a.wE) * (aE / a.wE) : 0.0;
                r.p = (aP / a.wP) * (aP / a.wP);
                r.l = a.wL > 0.0 ? (aL / a.wL) * (aL / a.wL) : 0.0;
                if (a.n_inst >= 2 && a.aP_sq > 0.0) {
                    const double num = (a.aP_re * a.aP_re + a.aP_im * a.aP_im) - a.aP_sq;
                    r.xcoh = num / (a.aP_sq * (double)(a.n_inst - 1));
                }
                r.n_chan = a.n_chan;
                r.n_inst = a.n_inst;
                r.hop = a.hop;
            }
        w.acc.clear();
        w.owner.clear();
    }

    void close_oldest(Chain& c, bool forced) {
        settle(c.open.begin()->second);
        c.n_dup_chan += c.open.begin()->second.dup.size();
        c.closed.push_back(std::move(c.open.begin()->second));
        c.open.erase(c.open.begin());
        // The ring keeps the deeper of the two depths; aggregate(), taps() and rec_series()
        // each read the newest slice they need.
        while ((int)c.closed.size() > std::max(_n_win, _taps_win))
            c.closed.pop_front();
        c.n_closed++;
        if (forced)
            c.n_forced++;
        aggregate(c);
    }

    /// The loop's discriminator: the lobe-coherent record powers of the newest _n_win windows,
    /// meaned over complete records, per PRN.
    void aggregate(Chain& c) {
        struct Acc {
            double e = 0, p = 0, l = 0, n_chan = 0;
            int n_rec = 0, n_inst = 0;
            int64_t hop = -1;
        };
        std::map<int, Acc> by_prn;
        uint64_t win_hi = 0;
        // The loop's own depth: the newest _n_win windows of the ring.
        for (size_t k = c.closed.size() > (size_t)_n_win ? c.closed.size() - _n_win : 0;
             k < c.closed.size(); ++k) {
            const WindowAcc& w = c.closed[k];
            win_hi = std::max(win_hi, w.win);
            for (const auto& sv : w.rec)
                for (const auto& pv : sv.second) {
                    const RecTap& r = pv.second;
                    // Completeness, not content: skip a record too few senders reached, as the
                    // Python arm does.
                    if (r.n_inst < _min_instances)
                        continue;
                    Acc& a = by_prn[pv.first];
                    a.e += r.e;
                    a.p += r.p;
                    a.l += r.l;
                    a.n_chan += r.n_chan;
                    a.n_rec++;
                    a.n_inst = std::max(a.n_inst, r.n_inst);
                    a.hop = std::max(a.hop, r.hop);
                }
        }

        for (const auto& kv : by_prn) {
            const Acc& a = kv.second;
            if (a.n_rec <= 0)
                continue;
            const double n = (double)a.n_rec;
            const double E = a.e / n, P = a.p / n, L = a.l / n;
            if (E + L <= 0.0)
                continue;
            FleetDllRow& s = c.row[kv.first];
            s.disc = (E - L) / (E + L);
            s.q = 2.0 * P / (E + L);
            s.e_pow = E;
            s.p_pow = P;
            s.l_pow = L;
            s.n_chan = a.n_chan / n;
            s.n_src = a.n_inst;
            s.n_rec = a.n_rec;
            s.hop = a.hop;
            s.win = win_hi;
            s.n_updates++;
        }
        integrate(c, win_hi);
    }

    /// One integrator step per window close for every armed PRN that got a discriminator this
    /// window, and a leak-only step for every disarmed PRN whose trim still stands, so leaving
    /// the armed set releases the trim gradually rather than as a step.
    ///
    /// Stepping per window (23.84 Hz) rather than per record (95.4 Hz) keeps the arithmetic the
    /// Python arm computes. At the defaults the discriminator moves the trim at most 0.0625
    /// chips per step, ~1.5 chips/s at 23.84 Hz, about 12x a ~0.12 chips/s code drift.
    ///
    /// An armed PRN with no row this window is not stepped, not even by the leak: an absent
    /// discriminator is not a measurement of zero, and leaking on it would walk the correction
    /// out during the dropouts it exists to ride through.
    void integrate(Chain& c, uint64_t win) {
        // The discriminator carries information only when there is a peak under the taps.
        // Off-peak, disc is noise that flips sign every window, and integrating it at full
        // authority random-walks the trim to the clamp.
        //
        // The gate is prompt power, not q: q is high only once the tap is already on the
        // peak, so a q gate would never let the loop pull in from the shoulder, where P is
        // still well above noise. This is a per-window information test, not the presence
        // verdict. The floor is absolute (the broker's probe-anchored floor), not relative to
        // other PRNs' powers, which would compare satellites with each other rather than with
        // noise. With no floor every window is leak-only: a trim not applied is recoverable, a
        // trim driven by noise is not.
        const double p_floor = c.policy.p_floor_abs;
        for (int prn : c.armed) {
            auto it = c.row.find(prn);
            if (it == c.row.end() || it->second.win != win)
                continue; // no discriminator formed for this PRN this window
            TrimState& t = c.trim[prn];
            if (p_floor <= 0.0 || it->second.p_pow < p_floor) {
                // Leak only: the trim mean-reverts instead of chasing noise, and the PRN stays
                // armed so a returning peak is caught on the next window.
                t.trim = dll_integrate(t.trim, 0.0, c.policy);
                t.last_win = win;
                t.n_skipped++;
                continue;
            }
            t.trim = dll_integrate(t.trim, it->second.disc, c.policy);
            if (std::abs(t.trim) >= c.policy.clamp * 0.999)
                t.n_railed++;
            t.last_disc = it->second.disc;
            t.last_win = win;
            t.n_steps++;
        }
        // Graceful release: a disarmed PRN's trim decays through the leak (disc treated as 0)
        // and is still posted until it falls below 1e-3 chips, then is erased. Erasing it at
        // once, or letting the tracker's TTL zero it, would turn every presence flicker into a
        // code step.
        for (auto it = c.trim.begin(); it != c.trim.end();) {
            if (!c.armed.count(it->first)) {
                TrimState& t = it->second;
                t.trim = dll_integrate(t.trim, 0.0, c.policy);
                t.last_win = win; // still posted, stamped with this window
                if (std::abs(t.trim) < 1e-3) {
                    it = c.trim.erase(it);
                    continue;
                }
            }
            ++it;
        }
    }

    int _n_win, _min_instances;
    // Members initialise in declaration order; keep it matching the constructor's list.
    int _taps_win = 1; ///< window depth served by taps() and rec_series()
    /// Epoch-reset thresholds (see fold()). 64 windows is ~2.7 s at 23.84 windows/s: more than
    /// a laggard falls behind, far less than a frame0 move. 8 strikes is under half a second of
    /// stream even from one sender, and no single far-behind frame can re-anchor.
    static constexpr uint64_t _epoch_margin = 64, _epoch_strikes = 8;
    std::map<std::string, Chain> _chain;
};

} // namespace gnss

#endif // GNSS_FLEET_DLL_HPP
