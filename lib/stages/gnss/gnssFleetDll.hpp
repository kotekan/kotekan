#ifndef GNSS_FLEET_DLL_HPP
#define GNSS_FLEET_DLL_HPP
/**
 * @file gnssFleetDll.hpp
 * @brief THE FLEET CODE DISCRIMINATOR, from the comb -- the arithmetic, with no kotekan in it.
 *
 * Task #51, milestone F1. This is the C++ twin of
 * python/scripts/gnss/gnss_broker/combdll.py: fold every (chain, instance)'s per-channel
 * E/P/L into one fleet discriminator per PRN per window.
 *
 * WHY IT IS A HEADER AND NOT A STAGE. Same reason gnssSeedTransport.hpp exists. The arithmetic
 * that matters here sits behind a kotekan buffer graph and a REST hop, so the only way to
 * exercise it in a stage would be to fly it -- and this codebase has repeatedly paid for
 * conventions that could only be tested in deployment order. Pulled out, `scripts/gnss/fleetdll`
 * drives the SHIPPED code against a byte-identical fixture in under a second, and the gate
 * compares it with the Python arm's answer on the same bytes. A harness that re-derives the
 * arithmetic instead of calling it tests the harness author's understanding, which is exactly
 * the thing already known to be unreliable.
 *
 * WHAT IT REPRODUCES, and where the Python arm is the reference:
 *
 *     per record, per (instance, PRN):   e = |SUM_c G^E_c|^2 / (SUM_c E^E_c)^2   (and p, l)
 *     per (instance, PRN):               mean over the records in the window ring
 *     per PRN:                           E = SUM_inst e ,  P, L likewise
 *                                        disc = (E-L)/(E+L) ,  q = 2P/(E+L)
 *
 * ⚠️ THE ORDER OF THOSE TWO REDUCTIONS IS NOT A DETAIL. Mean over records FIRST, sum over
 * instances SECOND. Pooling every record across instances instead would weight an instance by
 * how many records it happened to deliver, so an instance that dropped half its frames would
 * quietly stop counting -- which is a data-loss event silently rewriting the discriminator.
 *
 * ⚠️ PRESENCE IS NOT HERE, ON PURPOSE. apply_presence -- the k-sigma floor, the quality
 * fallback, the probe and deep gates -- is POLICY and stays in the Python broker
 * (docs/CHORD_FAST_TRIM.md 3). This produces numbers; it passes no verdicts.
 *
 * @author Keith Vanderlinde
 */

#include "gnssRecord.hpp"
#include "gnssTelem.hpp"

#include <algorithm>
#include <cmath>
#include <array>
#include <cstdint>
#include <cstring>
#include <deque>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace gnss {

/// What one PRN's fleet discriminator looks like. combdll.fleet_dll_comb's row, minus presence.
struct FleetDllRow {
    double disc = 0.0;   ///< (E-L)/(E+L). 0 = on peak; sign is "the tap is early/late".
    double q = 0.0;      ///< 2P/(E+L). EXACTLY 1.0 = no peak; ~4 = clean lock at 0.5 spacing.
    double e_pow = 0.0, p_pow = 0.0, l_pow = 0.0;
    double n_chan = 0.0;    ///< channels behind it, fleet-wide (the axis the tracker sum destroyed)
    int n_src = 0;          ///< instances that contributed
    int n_rec = 0;          ///< records the busiest instance contributed
    int64_t hop = -1;       ///< newest F-engine hop in the average
    uint64_t win = 0;       ///< newest window in the average
    uint64_t n_updates = 0; ///< discriminators formed for this PRN since start -- THE RATE
};

/// Why a frame was not folded. Anything but OK means the wire and this build disagree, and the
/// numbers that would have fallen out are plausible and wrong.
enum class FoldStatus { OK, BAD_HEADER, LATE };

/// The integrator's constants. ONE convention, and the Python arm calls the same expression
/// (gnss_broker.combdll.dll_tau / dll_integrate) so the two cannot drift.
struct TrimPolicy {
    double gain = 0.25;    ///< step = gain * tau
    double leak = 0.05;    ///< PER UPDATE -- see the warning on dll_integrate
    double clamp = 3.0;    ///< |trim| bound, chips
    double spacing = 0.5;  ///< tracker Early/Late spacing, chips
};

/// The code discriminator -> delay estimate, in chips.
///
/// ⚠️ |tau| <= 0.25 chips BY CONSTRUCTION, whatever `disc` is: the clamp is on the
/// discriminator, and it is divided by four. THIS IS THE WHOLE OF #51 -- the loop's authority
/// is a step per update, so no gain and no clamp can make one update slew further, and the
/// UPDATE RATE is the only lever. (Cutting the gain to "compensate" for a faster rate hands the
/// entire win straight back: same gain, faster rate.)
inline double dll_tau(double disc, double spacing) {
    return -std::max(-1.0, std::min(1.0, disc)) / 4.0 * (spacing / 0.5);
}

/// One leaky-integrator update.
///
/// ⚠️ `leak` IS PER UPDATE, SO LOOP BANDWIDTH SCALES WITH RATE. In continuous form this is
/// dT/dt = -leak*f*T + gain*f*tau: the steady state (gain*tau/leak) does NOT move with f, but
/// the closed-loop AND noise bandwidths both scale with it. Going 3.1 -> 23.8 Hz is therefore
/// ~8x the bandwidth at unchanged constants. Whoever sets these must convert from a
/// per-SECOND leak using the ACHIEVED rate; this function is deliberately the dumb primitive
/// so that conversion happens in exactly one, clearly-labelled place.
///
/// ⚠️ AND THE STEADY STATE IS A CEILING THE RATE CANNOT LIFT: under a railed discriminator the
/// trim converges to gain*0.25/leak, which at the shipped defaults (0.25, 0.05) is 1.25 chips
/// -- BELOW the residuals seen on sky, and far below the +-3.0 clamp, which is therefore
/// unreachable by construction. Measured on sky 2026-08-15: max |trim| 1.140 chips over 5174
/// updates in 8 hours, never once past 1.25. If the loop is pushing at a railed discriminator
/// without arriving, THIS is the reason and a faster loop will not fix it.
inline double dll_integrate(double trim, double disc, const TrimPolicy& p) {
    const double t = (1.0 - p.leak) * trim + p.gain * dll_tau(disc, p.spacing);
    return std::max(-p.clamp, std::min(p.clamp, t));
}

/**
 * The fleet discriminator, accumulated per chain over a ring of absolute windows.
 *
 * COLLATION is #59's exact-integer match on `win` -- no tolerance, no arrival-order inference.
 * COMPLETION mirrors the Python arm rather than inventing a rule: a window CLOSES when any
 * sender on that chain reports a newer one (TelemClient.windows(..., lag=1)). No timer, no
 * barrier, nothing waits on a sender. A frame for an already-closed window is counted as LATE
 * and DROPPED -- re-opening a window would make the aggregate depend on arrival order, which is
 * the whole bug class #59 removed.
 */
class FleetDll {
public:
    FleetDll(int n_win = 4, int min_instances = 2, int max_open_win = 8, double sig_k = 3.0) :
        _n_win(std::max(1, n_win)), _min_instances(min_instances),
        _max_open_win(std::max(2, max_open_win)), _sig_k(sig_k) {}

    /// One comb COLUMN's three powers for one (instance, PRN), over the records of one window.
    /// `fid` is the F-engine freq_id off the frame header -- never assumed, never configured
    /// (gnssTelem.hpp: a fit over unlabelled columns returns a confident wrong tau).
    struct ChanTap {
        int fid = 0;
        double e = 0.0, p = 0.0, l = 0.0;
        int n_rec = 0;
    };

    /// One (instance, PRN)'s three powers, accumulated over the records of ONE window.
    ///
    /// ⚠️ Records with no live comb contribute NOTHING rather than zeros. A zeroed record is not
    /// a measurement of no signal, it is the absence of one, and averaging it in dilutes the
    /// power exactly the way the deep fold's zero-padding did.
    struct Tap {
        double e = 0.0, p = 0.0, l = 0.0;
        double n_chan = 0.0;
        int n_rec = 0;
        int64_t hop = -1;
        /// ONE CHANNEL's own three powers, kept UNSUMMED -- indexed by the sender's comb
        /// column, labelled with the freq_id the frame header carries. This is
        /// combdll.instance_taps' `chan` dict, which the broker builds by walking every
        /// (window, instance, record, PRN, channel) in Python: ~140k channel-tuples per chain
        /// per cycle, ~700k across the fleet, each allocating Python complex objects. The
        /// arithmetic under that is ~1.4 MFLOP -- microseconds of real work wrapped in a
        /// second of interpreter, on a process already pinned at one core by the GIL.
        /// Formed here instead, where the frame already is.
        ///
        /// ⚠️ A channel reaches exactly ONE instance (freq_id mod 8 routing), so merging these
        /// across instances downstream is a merge and never a sum over duplicates -- see the
        /// duplicate handling in combdll.fleet_dll_comb, which names and DROPS a freq_id that
        /// two instances both claim rather than adding two measurements of different things.
        std::array<ChanTap, TELEM_MAX_CHAN> chan{};
    };

    struct WindowAcc {
        uint64_t win = 0;
        std::map<std::string, std::map<int, Tap>> tap; ///< [instance][prn]
    };

    /// One PRN's integrator state. `trim` is a correction to the BROKER'S MODEL, not to a
    /// particular seed, so it survives the model being republished every policy cycle.
    struct TrimState {
        double trim = 0.0;
        uint64_t n_steps = 0;  ///< integrator updates -- THE RATE, measured not assumed
        uint64_t n_railed = 0;  ///< updates that hit the clamp
        uint64_t n_skipped = 0; ///< windows with no signal under the taps: leak only
        double last_disc = 0.0, last_q = 0.0;
        uint64_t last_win = 0;
    };

    struct Chain {
        uint64_t newest = 0;
        bool have_newest = false;
        std::map<uint64_t, WindowAcc> open; ///< ordered, so "close everything older" is a walk
        std::deque<WindowAcc> closed;       ///< the last n_win closed windows
        std::map<int, FleetDllRow> row;
        /// WHO MAY BE TRIMMED, decided by the Python broker and never here. This class forms
        /// numbers and steps an integrator; presence, floors, the deep gate and the arming
        /// verdict are POLICY and stay on the 12 s cycle (docs/CHORD_FAST_TRIM.md 3).
        std::set<int> armed;
        TrimPolicy policy;
        std::map<int, TrimState> trim;
        uint64_t n_closed = 0; ///< windows closed == integrator steps available
        uint64_t n_late = 0;   ///< frames for a window already closed: dropped, never folded
        uint64_t n_forced = 0; ///< force-closed by max_open_win, i.e. a sender went away
        uint64_t n_frames = 0;
    };

    /// Fold one wire frame. `chain_out`/`inst_out` are filled whenever the header parsed.
    FoldStatus fold(const void* frame, size_t bytes, std::string* chain_out = nullptr,
                    std::string* inst_out = nullptr) {
        const auto* h = (const TelemHeader*)frame;
        // The same validation the gather applies, for the same reason: a sender on a different
        // record layout parses at the wrong stride, and this stage would close a loop on it.
        const bool ok = h->magic == TELEM_MAGIC && h->version == TELEM_VERSION
                        && h->n_row == RECORD_FLOATS && h->n_rec > 0 && h->n_rec <= TELEM_MAX_REC
                        && h->n_prn > 0 && h->n_chan <= TELEM_MAX_CHAN && h->fft_len > 0
                        && telem_frame_bytes(h->n_rec, h->n_prn) == bytes;
        if (!ok)
            return FoldStatus::BAD_HEADER;

        const std::string chain(h->chain, strnlen(h->chain, TELEM_NAME));
        const std::string inst(h->inst, strnlen(h->inst, TELEM_NAME));
        if (chain_out)
            *chain_out = chain;
        if (inst_out)
            *inst_out = inst;

        Chain& c = _chain[chain];
        c.n_frames++;
        if (c.have_newest && h->win < c.newest && !c.open.count(h->win)) {
            c.n_late++;
            return FoldStatus::LATE;
        }

        const float* rows = telem_rows(frame);
        const int n_prn = h->n_prn;
        const int n_chan = h->n_chan;

        // THE PRN MAP IS READ FROM THE DATA, from record slot 0's rows, exactly as the Python
        // client does. The assembler writes REC_PRN even for a PRN that did not run this window,
        // so it is there whether or not slot 0 was filled. A configured copy is one more thing
        // that can drift out of step with the node it describes -- and after #64's row
        // compaction the row order is not the configured PRN order at all.
        constexpr int MAX_ROWS = 256; // telem_max_prn is 16 today, was 40 before #64
        int prn_of_row[MAX_ROWS];
        const int n_row_map = std::min(n_prn, MAX_ROWS);
        for (int p = 0; p < n_row_map; ++p) {
            const float v = rows[telem_row_offset(0, p, n_prn) + REC_PRN];
            prn_of_row[p] = (v > 0.0f) ? (int)(v + 0.5f) : 0;
        }

        WindowAcc& w = c.open[h->win];
        w.win = h->win;
        auto& per_prn = w.tap[inst];

        for (int r = 0; r < (int)h->n_rec; ++r) {
            if (!(h->present & (1u << r)))
                continue;
            const int64_t hop = (h->wstart0 + (int64_t)r * h->hops_per_record * h->fft_len)
                                / (int64_t)h->fft_len;
            for (int p = 0; p < n_row_map; ++p) {
                const int prn = prn_of_row[p];
                if (prn <= 0)
                    continue;
                const float* row = rows + telem_row_offset(r, p, n_prn);

                // THE COMB, SUMMED ACROSS CHANNELS HERE AND NOWHERE ELSE (#63). Each tap is
                // normalised by ITS OWN replica energy, and the three were element-combined and
                // NCO-rotated identically upstream -- a discriminator built from taps combined
                // even slightly differently measures the difference between the combines rather
                // than the code offset.
                //
                // combdll.instance_taps accumulates (raw/E)*E per channel, which is the raw
                // complex, and sums the energies separately. Written that way directly here.
                double gE_re = 0, gE_im = 0, gP_re = 0, gP_im = 0, gL_re = 0, gL_im = 0;
                double wE = 0, wP = 0, wL = 0;
                int used = 0;
                for (int ch = 0; ch < n_chan; ++ch) {
                    const float* cc = row + telem_chan_offset(ch);
                    const double eP = cc[CHAN_ENERGY];
                    if (eP <= 0.0)
                        continue; // no live comb for this channel this record
                    // ⚠️ EXACTLY 0.0 falls back to the prompt energy, matching the Python arm's
                    // `a[CHAN_E_ENERGY] or eP`. It moves only the denominator -- the numerator
                    // is the raw complex either way -- and only for senders predating the E/L
                    // energies. Mirrored rather than tidied: the two arms have to agree.
                    const double eE = cc[CHAN_E_ENERGY] != 0.0f ? cc[CHAN_E_ENERGY] : eP;
                    const double eL = cc[CHAN_L_ENERGY] != 0.0f ? cc[CHAN_L_ENERGY] : eP;
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

                Tap& t = per_prn[prn];
                const double aE = std::hypot(gE_re, gE_im);
                const double aP = std::hypot(gP_re, gP_im);
                const double aL = std::hypot(gL_re, gL_im);
                t.e += wE > 0.0 ? (aE / wE) * (aE / wE) : 0.0;
                t.p += (aP / wP) * (aP / wP);
                t.l += wL > 0.0 ? (aL / wL) * (aL / wL) : 0.0;
                t.n_chan += used;
                t.n_rec++;
                t.hop = std::max(t.hop, hop);

                // PER-CHANNEL, formed by the IDENTICAL expression one column at a time.
                // Second pass over the same columns rather than folded into the loop above:
                // the aggregate is only accumulated once `used > 0` is known, and the Python
                // arm likewise creates its per-instance entry only for a record with a live
                // comb (`if not cmb: continue` precedes it). Splitting the passes keeps that
                // equivalence obvious instead of hidden in a guard.
                //
                // ⚠️ NORMALISE FIRST, THEN TAKE THE MAGNITUDE. Python computes
                // `abs(complex(re/e, im/e))**2`; (re*re + im*im)/(e*e) is the same number in
                // exact arithmetic and NOT the same float. The two arms are compared at 1e-9.
                for (int ch = 0; ch < n_chan; ++ch) {
                    const float* cc = row + telem_chan_offset(ch);
                    const double eP = cc[CHAN_ENERGY];
                    if (eP <= 0.0)
                        continue;
                    const double eE = cc[CHAN_E_ENERGY] != 0.0f ? cc[CHAN_E_ENERGY] : eP;
                    const double eL = cc[CHAN_L_ENERGY] != 0.0f ? cc[CHAN_L_ENERGY] : eP;
                    ChanTap& ct = t.chan[ch];
                    ct.fid = (int)h->chan_id[ch];
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
        // A sender that dies mid-window leaves that window open forever if nothing else ever
        // passes it. Bounded, and COUNTED: an unexplained n_forced is a sender that stopped.
        while ((int)c.open.size() > _max_open_win)
            close_oldest(c, true);
        return FoldStatus::OK;
    }

    /// Force every open window closed. For the offline harness at end-of-file, so the last
    /// window of a fixture is not silently missing from the answer. NEVER on the live path:
    /// there, the next frame is the completion signal and a flush would race it.
    void flush() {
        for (auto& cv : _chain)
            while (!cv.second.open.empty())
                close_oldest(cv.second, true);
    }

    /// Publish the policy cycle's decisions: who may be trimmed, and with what constants.
    ///
    /// ⚠️ A PRN THAT LEAVES THE ARMED SET KEEPS ITS TRIM. Zeroing it would step the commanded
    /// code phase by up to `clamp` chips at the moment policy stopped being sure -- a
    /// disturbance injected exactly when confidence is lowest. The trim decays through the
    /// leak instead, and the tracker's own TTL is what removes it if this controller dies.
    void set_armed(const std::string& chain, const std::set<int>& prns, const TrimPolicy& p) {
        Chain& c = _chain[chain];
        c.armed = prns;
        c.policy = p;
    }

    const std::map<std::string, Chain>& chains() const {
        return _chain;
    }

    /// ONE (instance, PRN)'s taps over the closed window set, MEANED -- i.e. exactly what
    /// `combdll.instance_taps` returns for one entry of `{prn: {inst: ...}}`.
    struct InstTap {
        double e = 0.0, p = 0.0, l = 0.0, n_chan = 0.0;
        int n_rec = 0;
        int64_t hop = -1;
        std::map<int, std::array<double, 4>> chan; ///< freq_id -> {e, p, l, n_rec}
    };

    /// [chain][prn][instance] -- the per-instance, per-channel taps over the closed windows.
    ///
    /// THIS IS WHY IT EXISTS: `combdll.instance_taps` builds the identical object in Python by
    /// walking every (window, instance, record, PRN, channel) of the gathered stream -- ~140k
    /// channel-tuples per chain per cycle, ~700k across the fleet. Profiled on the live broker
    /// it is ~18% of chain CPU, on a process pinned at 100% of ONE core by the GIL, where cycle
    /// time IS the sum of the five chains' Python CPU. The frames are already here, in C++, so
    /// the reduction belongs here and the broker should be handed the ~6k numbers that survive
    /// it rather than the 46 MB/s that do not.
    ///
    /// ⚠️ THE MEANS ARE TAKEN OVER DIFFERENT DENOMINATORS ON PURPOSE, mirroring the Python:
    /// the aggregate divides by the (instance, PRN)'s record count, and EACH CHANNEL divides by
    /// ITS OWN -- a channel that was live for half the records is a mean over that half, not a
    /// half-sized mean. Getting this wrong is invisible in the full-band numbers and shows up
    /// only per channel, which is precisely where nobody looks.
    ///
    /// ⚠️ POLICY IS NOT HERE AND MUST NOT COME HERE. Presence, the noise floor, the deep gate,
    /// who is armed -- all of that stays on the broker's cycle (GnssFleetTrim.hpp). This
    /// returns measurements.
    std::map<std::string, std::map<int, std::map<std::string, InstTap>>> taps() const {
        std::map<std::string, std::map<int, std::map<std::string, InstTap>>> out;
        for (const auto& cv : _chain) {
            auto& per_prn = out[cv.first];
            for (const WindowAcc& w : cv.second.closed)
                for (const auto& iv : w.tap)
                    for (const auto& pv : iv.second) {
                        InstTap& t = per_prn[pv.first][iv.first];
                        t.e += pv.second.e;
                        t.p += pv.second.p;
                        t.l += pv.second.l;
                        t.n_chan += pv.second.n_chan;
                        t.n_rec += pv.second.n_rec;
                        t.hop = std::max(t.hop, pv.second.hop);
                        for (const ChanTap& ct : pv.second.chan) {
                            if (ct.n_rec <= 0)
                                continue;
                            std::array<double, 4>& a = t.chan[ct.fid];
                            a[0] += ct.e;
                            a[1] += ct.p;
                            a[2] += ct.l;
                            a[3] += ct.n_rec;
                        }
                    }
            for (auto& pv : per_prn)
                for (auto& iv : pv.second) {
                    InstTap& t = iv.second;
                    const double n = t.n_rec ? (double)t.n_rec : 1.0;
                    t.e /= n;
                    t.p /= n;
                    t.l /= n;
                    t.n_chan /= n;
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
    int min_instances() const {
        return _min_instances;
    }

private:
    void close_oldest(Chain& c, bool forced) {
        c.closed.push_back(std::move(c.open.begin()->second));
        c.open.erase(c.open.begin());
        while ((int)c.closed.size() > _n_win)
            c.closed.pop_front();
        c.n_closed++;
        if (forced)
            c.n_forced++;
        aggregate(c);
    }

    void aggregate(Chain& c) {
        struct Acc {
            double e = 0, p = 0, l = 0, n_chan = 0;
            int n_rec = 0;
            int64_t hop = -1;
        };
        std::map<int, std::map<std::string, Acc>> by_prn;
        uint64_t win_hi = 0;
        for (const WindowAcc& w : c.closed) {
            win_hi = std::max(win_hi, w.win);
            for (const auto& iv : w.tap)
                for (const auto& pv : iv.second) {
                    Acc& a = by_prn[pv.first][iv.first];
                    a.e += pv.second.e;
                    a.p += pv.second.p;
                    a.l += pv.second.l;
                    a.n_chan += pv.second.n_chan;
                    a.n_rec += pv.second.n_rec;
                    a.hop = std::max(a.hop, pv.second.hop);
                }
        }

        for (const auto& kv : by_prn) {
            double E = 0, P = 0, L = 0, n_chan = 0;
            int n_src = 0, n_rec = 0;
            int64_t hop = -1;
            for (const auto& iv : kv.second) {
                const Acc& a = iv.second;
                if (a.n_rec <= 0)
                    continue;
                const double n = (double)a.n_rec; // MEAN over records, then SUM over instances
                E += a.e / n;
                P += a.p / n;
                L += a.l / n;
                n_chan += a.n_chan / n;
                n_rec = std::max(n_rec, a.n_rec);
                hop = std::max(hop, a.hop);
                n_src++;
            }
            if (n_src < _min_instances || E + L <= 0.0)
                continue;
            FleetDllRow& s = c.row[kv.first];
            s.disc = (E - L) / (E + L);
            s.q = 2.0 * P / (E + L);
            s.e_pow = E;
            s.p_pow = P;
            s.l_pow = L;
            s.n_chan = n_chan;
            s.n_src = n_src;
            s.n_rec = n_rec;
            s.hop = hop;
            s.win = win_hi;
            s.n_updates++;
        }
        integrate(c, win_hi);
    }

    /// ONE INTEGRATOR STEP PER WINDOW CLOSE, for every armed PRN that got a discriminator --
    /// and a LEAK-ONLY step for a disarmed PRN whose trim is still standing, so leaving the
    /// armed set is a graceful release rather than a commanded step. Without this the
    /// controller stopped posting a disarmed PRN, the tracker's TTL zeroed it 4 s later, and
    /// re-arming re-applied it: measured on sky 2026-08-15 as trims snapping N -> 0 -> N
    /// whenever presence flickered, ON TOP of the gain oscillation it was entangled with.
    ///
    /// THE RATE THIS ACHIEVES, and why it is not the 95.4/s the wire could support. A window is
    /// one frame, so this steps at 23.84 Hz = 12x the 1.94 Hz break-even against CHORD's
    /// measured 0.121 chips/s drift. Stepping per RECORD instead would give 95.4 Hz and 49x --
    /// but it would also stop being the same arithmetic the Python arm computes, and the
    /// byte-for-byte equivalence gate (scripts/gnss/fleetdll_gate.py) would go with it. 12x is
    /// margin enough; the gate is not worth spending for the other 4x until something measured
    /// says 12x is short.
    ///
    /// A PRN with no row this window is NOT stepped -- not even by the leak. An absent
    /// discriminator is the absence of a measurement, not a measurement of zero, and leaking a
    /// trim toward zero on windows where the satellite simply was not seen would walk the
    /// correction out during exactly the dropouts it exists to ride through.
    void integrate(Chain& c, uint64_t win) {
        // ⚠️ THE DISCRIMINATOR IS ONLY INFORMATION WHEN THERE IS A PEAK UNDER THE TAPS.
        // Off-peak, disc is noise that flips sign every window, and integrating it at full
        // authority RANDOM-WALKS THE TRIM TO THE CLAMP. Measured on sky 2026-08-15, PRN 23:
        // trim railed at -3.0000 chips, then ran 3.4 chips back to +0.38 in 32 s, sign-flipping
        // disc (+0.87, -0.96, +0.35, ...) all the way. And an ARMING HOLD makes it worse, not
        // better: holding a PRN armed across the off-peak half of the clock breathing is
        // exactly a licence to integrate tens of seconds of noise.
        //
        // The gate is PROMPT POWER against the same window's population, not q. That is the
        // distinction the broker learned on 2026-08-03 and wrote down: q is peak SHARPNESS,
        // high only once the tap is already on the peak, so gating on it says "only correct
        // the code once it is already correct" and the loop can never pull in from the
        // shoulder. Prompt power answers "is there signal here at all", which is independent
        // of WHERE on the correlation function we sit -- and on the shoulder P is still well
        // above noise. Self-calibrating: most PRNs are signal-free at any moment, so the
        // MEDIAN of this window's p_pow IS the no-signal level.
        //
        // This is a LOCAL INFORMATION TEST, not the presence verdict. Presence is policy and
        // stays in Python (it decides whether a satellite is worth tracking, over cycles);
        // this decides whether THIS WINDOW's number means anything.
        double p_floor = 0.0;
        {
            std::vector<double> pp;
            pp.reserve(c.row.size());
            for (const auto& rv : c.row)
                if (rv.second.win == win)
                    pp.push_back(rv.second.p_pow);
            if (pp.size() >= 4) {
                std::nth_element(pp.begin(), pp.begin() + pp.size() / 2, pp.end());
                p_floor = _sig_k * pp[pp.size() / 2];
            }
        }
        for (int prn : c.armed) {
            auto it = c.row.find(prn);
            if (it == c.row.end() || it->second.win != win)
                continue; // no discriminator formed for this PRN THIS window
            TrimState& t = c.trim[prn];
            if (p_floor > 0.0 && it->second.p_pow < p_floor) {
                // No signal under the taps this window: LEAK ONLY. The trim mean-reverts
                // instead of chasing noise, and the PRN stays armed so a returning peak is
                // caught on the very next window.
                t.trim = dll_integrate(t.trim, 0.0, c.policy);
                t.last_win = win;
                t.n_skipped++;
                continue;
            }
            t.trim = dll_integrate(t.trim, it->second.disc, c.policy);
            if (std::abs(t.trim) >= c.policy.clamp * 0.999)
                t.n_railed++;
            t.last_disc = it->second.disc;
            t.last_q = it->second.q;
            t.last_win = win;
            t.n_steps++;
        }
        // GRACEFUL RELEASE: a disarmed PRN's trim decays through the leak (disc treated as 0)
        // and keeps being posted until it is negligible, then drops out. Erasing it -- or
        // letting the tracker TTL zero it -- turns every presence flicker into a code step.
        for (auto it = c.trim.begin(); it != c.trim.end();) {
            if (!c.armed.count(it->first)) {
                TrimState& t = it->second;
                t.trim = dll_integrate(t.trim, 0.0, c.policy);
                t.last_win = win; // still commanded: the poster keys on this moving
                if (std::abs(t.trim) < 1e-3) {
                    it = c.trim.erase(it);
                    continue;
                }
            }
            ++it;
        }
    }

    int _n_win, _min_instances, _max_open_win;
    double _sig_k = 3.0; ///< prompt power must exceed this x the window median
    std::map<std::string, Chain> _chain;
};

} // namespace gnss

#endif // GNSS_FLEET_DLL_HPP
