#ifndef GNSS_FLEET_DLL_HPP
#define GNSS_FLEET_DLL_HPP
/**
 * @file gnssFleetDll.hpp
 * @brief THE FLEET CODE DISCRIMINATOR, from the comb -- the arithmetic, with no kotekan in it.
 *
 * Task #51, milestone F1. This is the C++ twin of
 * python/scripts/gnss/gnss_broker/combdll.py: fold every channel of the lobe's per-channel
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
 *     per record, per PRN, over EVERY channel c of the lobe (all senders, each sender's
 *     columns first multiplied by exp(+i*phi0), REC_PHI0):
 *                                        e = |SUM_c G^E_c|^2 / (SUM_c E^E_c)^2   (and p, l)
 *     per PRN:                           E = mean over the records in the window ring, P, L
 *                                        disc = (E-L)/(E+L) ,  q = 2P/(E+L)
 *
 * ⚠️ THE COHERENCE UNIT IS THE LOBE. Nothing in this arithmetic knows what a sender is: a
 * sender is a `freq_id mod 8` grouping of channels, a transport artefact, and any reduction
 * that treats it as a unit -- coherent within, powers added across -- puts that grouping into
 * the discriminator as a 3.27-chip grating comb and an n_sender-fold noise floor. The only
 * place a sender count survives is as COMPLETENESS: a record fewer than `min_instances`
 * senders reached is not averaged in. It is never an operand.
 *
 * ⚠️ RECORDS ARE THE UNIT OF AVERAGING, and a record with no live comb is absent, not zero. A
 * sender that drops a frame leaves that record less complete (fewer channels in the sum, the
 * same normalised signal power, more noise); it does not zero anything and it does not weight
 * anything.
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
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace gnss {

/// What one PRN's fleet discriminator looks like. combdll.fleet_dll_comb's row, minus presence.
struct FleetDllRow {
    double disc = 0.0;   ///< (E-L)/(E+L). 0 = on peak; sign is "the tap is early/late".
    double q = 0.0;      ///< 2P/(E+L). EXACTLY 1.0 = no peak; ~4 = clean lock at 0.5 spacing.
    double e_pow = 0.0, p_pow = 0.0, l_pow = 0.0;
    double n_chan = 0.0;    ///< channels in the lobe sum, meaned over records
    int n_src = 0;          ///< senders behind the most complete record -- completeness, not an operand
    int n_rec = 0;          ///< records averaged
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
    /// ABSOLUTE prompt-power floor for the per-window information gate (same currency as the
    /// rows' p_pow). 0 = the original behaviour: 3x the window's own population median.
    ///
    /// ⚠️ WHY THIS EXISTS (2026-08-26, the E3 chase). The peer-median gate's premise -- "most
    /// PRNs are signal-free at any moment, so the median IS the no-signal level" -- was true
    /// on the airspy prototype, where --noise-probes seeded genuine noise rows into the
    /// population. On CHORD the armed rows are mostly REAL satellites, so the median is a
    /// SIGNAL level and 3x it is a PEER COMPETITION: the bottom of the pack goes leak-only
    /// no matter how far above the actual noise floor it sits. Measured: gal_e5b PRN 33 spent
    /// 75.1% of its windows skipped; E3 was at ~20x the PROBE floor when the gate stood the
    /// loop down, its trim erased through the leak, and a 60 s fade became a 12 min outage.
    /// This is the same trap the broker's presence gate found and fixed with probe anchoring
    /// on 2026-08-14 -- the fix just never reached this loop.
    ///
    /// The value arrives from the BROKER's probe-anchored floor via /set_policy, margin
    /// already applied. Policy stays in Python; this is a number, not a decision.
    double p_floor_abs = 0.0;
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
    /// `taps_win` is the window depth SERVED BY taps(), independent of the loop's `n_win`.
    /// They are different questions and were being answered with one number: the code loop
    /// integrates the freshest 2 windows (authority is a step per update, so depth buys it
    /// nothing), while the broker's policy cycle averages 32 for the SNR its gates need
    /// (`telem-windows: 32`, matched to the REST feed's `record_export: 128`). 0 = follow
    /// n_win, which is what the offline harness wants so both arms window identically.
    /// ⚠️ `sig_k` IS GONE, NOT DEFAULTED OFF (2026-08-27). It multiplied a PEER MEDIAN; see
    /// integrate(). Removing the parameter is deliberate -- a knob that only ever selects
    /// between two wrong answers is one a future reader will try to tune.
    FleetDll(int n_win = 4, int min_instances = 2, int max_open_win = 8,
             int taps_win = 0, int epoch_margin = 64, int epoch_strikes = 8) :
        _n_win(std::max(1, n_win)), _min_instances(min_instances),
        _max_open_win(std::max(2, max_open_win)),
        _taps_win(taps_win > 0 ? taps_win : std::max(1, n_win)),
        // 64 windows is ~2.7 s at the 23.84 Hz close rate -- far beyond any laggard
        // (max_open_win is 8) and far below a frame0 move, which lands thousands back.
        // 8 strikes is under half a second of stream: fast enough that the fold recovers
        // inside one policy cycle, long enough that no single frame can re-anchor it.
        _epoch_margin(std::max(2, epoch_margin)), _epoch_strikes(std::max(1, epoch_strikes)) {}

    /// ONE CHANNEL's three powers for one PRN, over the records of one window -- keyed by the
    /// F-engine freq_id off the frame header, never by the sender's column index or by which
    /// sender carried it (gnssTelem.hpp: a fit over unlabelled columns returns a confident
    /// wrong tau). Formed from that channel alone, no rotation needed: a power is reference-free.
    struct ChanTap {
        double e = 0.0, p = 0.0, l = 0.0;
        int n_rec = 0;
    };

    /// ONE RECORD's complex partial sums for one PRN over EVERY channel of the lobe that has
    /// reached this window so far, each sender's columns rotated onto the common reference
    /// before they are added. Live only while the window is open; closes into a RecTap.
    ///
    /// ⚠️ THE COHERENCE UNIT IS THE LOBE, NOT THE SENDER. A sender is a `freq_id mod 8`
    /// grouping of channels -- a transport artefact -- and a discriminator that sums its
    /// channels coherently and then adds POWERS across senders inherits that grouping: seven
    /// channels at 3.125 MHz stride give a 3.27-chip grating comb in the correlation response
    /// and a noise floor ~n_sender times what the band can give. One complex sum over all
    /// channels, then one power, has neither. No sender appears in this arithmetic.
    ///
    /// ⚠️ THE ROTATION IS LOAD-BEARING. Each sender's assembler applied exp(-i*phi0) to its
    /// comb with a phi0 whose zero is arbitrary per sender (gnssRecord.hpp REC_PHI0); the
    /// per-record row carries that phi0, and multiplying the sender's columns by exp(+i*phi0)
    /// is what puts twelve senders on one reference. Without it the cross-sender sum is a sum
    /// of unrelated phasors and the lobe-coherent prompt is BELOW the per-sender one.
    struct RecAcc {
        double gE_re = 0, gE_im = 0, gP_re = 0, gP_im = 0, gL_re = 0, gL_im = 0;
        double wE = 0, wP = 0, wL = 0;
        /// THE ROTATION'S OWN CHECK. Each sender's energy-normalised, derotated prompt
        /// a_i = (G_P/W_P)*exp(+i*phi0): SUM_i a_i (complex) and SUM_i |a_i|^2. From these the
        /// record's cross-sender coherence (|SUM a|^2 - SUM |a|^2) / (SUM |a|^2 * (n-1)) says
        /// whether phi0 put the senders on ONE reference (~1) or the lobe sum is adding
        /// unrelated phasors (~0, and the prompt above is then BELOW the per-sender one). It
        /// is a diagnostic beside the sum, never an operand in it.
        double aP_re = 0, aP_im = 0, aP_sq = 0;
        int n_chan = 0;  ///< channels behind the sums
        int n_inst = 0;  ///< senders behind the sums -- completeness, never an operand
        int64_t hop = -1;
    };

    /// One RECORD's three lobe-coherent powers for one PRN: |SUM_c G_c|^2 / (SUM_c E_c)^2,
    /// each tap on its own replica energy. The unit every downstream reduction averages.
    ///
    /// ⚠️ Records with no live comb contribute NOTHING rather than zeros. A zeroed record is not
    /// a measurement of no signal, it is the absence of one, and averaging it in dilutes the
    /// power exactly the way the deep fold's zero-padding did.
    ///
    /// A different consumer wants this un-averaged: the SERVED C/N0 (#57) is a radiometric
    /// estimator whose whole point is that it fits nothing, so it needs the samples, not their
    /// mean. `n_chan`/`n_inst` say how complete the record was; a partial record carries the
    /// same normalised signal power and more noise, and the consumer decides whether to use it.
    struct RecTap {
        double e = 0.0, p = 0.0, l = 0.0;
        /// cross-sender coherence of the derotated prompts (RecAcc); NaN below two senders
        double xcoh = std::numeric_limits<double>::quiet_NaN();
        int n_chan = 0, n_inst = 0;
        int64_t hop = -1;
    };

    struct WindowAcc {
        uint64_t win = 0;
        std::map<int, std::map<int, RecAcc>> acc;  ///< [record slot][prn], while open
        std::map<int, std::map<int, RecTap>> rec;  ///< [record slot][prn], once closed
        std::map<int, std::map<int, ChanTap>> chan; ///< [prn][freq_id]
        /// freq_id -> the sender that carried it this window. A channel reaches exactly ONE
        /// sender (freq_id mod 8 routing); a second claimant is a misconfiguration upstream,
        /// and that channel is named, counted and DROPPED from the per-channel table rather
        /// than summed twice into a number that is neither sender's measurement.
        std::map<int, std::string> owner;
        std::set<int> dup;
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
        /// Record length in hops, off the wire header. Carried because the SERVED C/N0 needs
        /// the integration time per record and must not assume it (gnssTelem.hpp: mixing a
        /// sample index with a hop index is a 16384x error that looks like a clock).
        uint32_t hops_per_record = 0;
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
        /// freq_ids two senders both carried in one window -- a routing fault upstream,
        /// named and dropped from the per-channel table (see WindowAcc::dup).
        uint64_t n_dup_chan = 0;
        /// A1/THE EPOCH RESET. Consecutive frames landing FAR behind `newest` -- the
        /// signature of an F-engine frame0 move, never of a late sender.
        uint64_t n_backwards = 0;
        uint64_t n_epoch_reset = 0; ///< re-anchors performed; >0 means the axis moved
    };

    /// Fold one wire frame. `chain_out`/`inst_out` are filled whenever the header parsed.
    FoldStatus fold(const void* frame, size_t bytes, std::string* chain_out = nullptr,
                    std::string* inst_out = nullptr) {
        const auto* h = (const TelemHeader*)frame;
        // The same validation the gather applies, for the same reason: a sender on a different
        // record layout parses at the wrong stride, and this stage would close a loop on it.
        // `bytes` is the CAPACITY the caller can read (the buffer's frame), not this frame's
        // size: senders ship their own shapes into a buffer sized to the widest one.
        const bool ok = h->magic == TELEM_MAGIC && h->version == TELEM_VERSION && h->fft_len > 0
                        && telem_shape_ok(*h, telem_frame_bytes(*h))
                        && telem_frame_bytes(*h) <= bytes;
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
        c.hops_per_record = h->hops_per_record;
        // ── A1: THE EPOCH RESET (2026-08-26) ────────────────────────────────────────
        // `win` is ABSOLUTE from the F-engine's frame0, so an F-engine restart moves
        // frame0 and every sender's window index restarts SMALL. Before this, the LATE
        // rule below dropped every subsequent frame FOREVER: the fold froze, `row` kept
        // serving its last aggregate with an IDENTICAL `hop` on every row at 200 OK, the
        // 60 TCP connections stayed ESTABLISHED, and nothing looked down. It voided at
        // least three experiments.
        //
        // A LAGGARD IS NOT AN EPOCH CHANGE, and the discriminator is SIZE plus
        // PERSISTENCE. A late sender is a few windows behind and its peers are not; a
        // frame0 move lands the WHOLE stream hundreds of windows back and keeps doing it.
        // So a large backwards jump is COUNTED, not acted on, and only a run of
        // `_epoch_strikes` consecutive such frames re-anchors -- one corrupt header
        // cannot, and any in-order frame clears the run.
        //
        // THE TRIMS SURVIVE. Only `open`/`newest` are epoch-scoped. The integrator,
        // arming and policy belong to the broker's 12 s cycle, not to the stream, and
        // wiping them here would turn an F-engine restart into a fleet-wide pull-in --
        // the very cost d01c0c1be's trim store exists to avoid.
        if (c.have_newest && h->win < c.newest && !c.open.count(h->win)) {
            if (c.newest - h->win <= (uint64_t)_epoch_margin
                || ++c.n_backwards < (uint64_t)_epoch_strikes) {
                c.n_late++;
                return FoldStatus::LATE;
            }
            // Re-anchor ON this frame and fold it: the new epoch starts here.
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
        // THE SENDER'S STRIDE, off the wire. Senders carry different comb widths, so a
        // compile-time row width reads a narrow sender's rows at the wide sender's offsets --
        // every field lands in the previous row's comb and looks like data.
        const int row_floats = h->n_row_total;

        // THE PRN MAP IS READ FROM THE DATA, from record slot 0's rows, exactly as the Python
        // client does. The assembler writes REC_PRN even for a PRN that did not run this window,
        // so it is there whether or not slot 0 was filled. A configured copy is one more thing
        // that can drift out of step with the node it describes -- and after #64's row
        // compaction the row order is not the configured PRN order at all.
        constexpr int MAX_ROWS = 256; // telem_max_prn is per chain now (12-24), was 40 pre-#64
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
            const int64_t hop = (h->wstart0 + (int64_t)r * h->hops_per_record * h->fft_len)
                                / (int64_t)h->fft_len;
            for (int p = 0; p < n_row_map; ++p) {
                const int prn = prn_of_row[p];
                if (prn <= 0)
                    continue;
                const float* row = rows + telem_row_offset(r, p, n_prn, row_floats);

                // THIS SENDER'S CHANNELS, SUMMED RAW (#63). Each tap is normalised by ITS OWN
                // replica energy, and the three were element-combined and NCO-rotated
                // identically upstream -- a discriminator built from taps combined even
                // slightly differently measures the difference between the combines rather
                // than the code offset. The Python arm (combdll.lobe_taps) accumulates
                // (raw/E)*E per channel, which is the raw complex; written that way directly.
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

                // ONTO THE COMMON REFERENCE, then into the lobe's sum for this record.
                const double phi0 = row[REC_PHI0];
                const double cr = std::cos(phi0), ci = std::sin(phi0);
                RecAcc& a = w.acc[r][prn];
                a.gE_re += gE_re * cr - gE_im * ci;
                a.gE_im += gE_re * ci + gE_im * cr;
                a.gP_re += gP_re * cr - gP_im * ci;
                a.gP_im += gP_re * ci + gP_im * cr;
                a.gL_re += gL_re * cr - gL_im * ci;
                a.gL_im += gL_re * ci + gL_im * cr;
                a.wE += wE;
                a.wP += wP;
                a.wL += wL;
                // ⚠️ NORMALISE FIRST, THEN ROTATE, mirroring the Python `gP / eP * rot`: the
                // same number in exact arithmetic and not the same float otherwise.
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

                // PER-CHANNEL, by freq_id, formed by the IDENTICAL expression one column at a
                // time. A second pass over the same columns rather than folded into the loop
                // above: the record is only admitted once `used > 0` is known, and the Python
                // arm likewise skips a record with no live comb before touching its channels.
                //
                // ⚠️ NORMALISE FIRST, THEN TAKE THE MAGNITUDE. Python computes
                // `abs(complex(re/e, im/e))**2`; (re*re + im*im)/(e*e) is the same number in
                // exact arithmetic and NOT the same float. The two arms are compared at 1e-9.
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
                    const double eE = cc[CHAN_E_ENERGY] != 0.0f ? cc[CHAN_E_ENERGY] : eP;
                    const double eL = cc[CHAN_L_ENERGY] != 0.0f ? cc[CHAN_L_ENERGY] : eP;
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

    /// {chain: {prn: trim_chips}} -- THE ONLY STATE IN THIS PROCESS THAT CANNOT BE REBUILT
    /// FROM THE STREAM, and therefore the only thing a restart genuinely loses.
    ///
    /// Measured across one gather restart, 2026-08-23: q on armed PRNs fell 2.0-3.7 -> ~1.0
    /// fleet-wide with 0-1 "present" per chain, because every tracker was left off-peak by
    /// however much trim had been standing (one PRN was at the 3-chip clamp). It re-acquires
    /// on its own -- median q 0.90 -> 1.93 over a few minutes -- but everything derived from
    /// the prompt tap reads as a dead fleet meanwhile, which looks exactly like whatever was
    /// deployed having broken tracking.
    ///
    /// Counters are deliberately NOT here. n_steps/n_railed/n_skipped are rates since start;
    /// carrying them across a restart would make "updates per second" a lie in the direction
    /// that hides a stalled loop.
    std::map<std::string, std::map<int, double>> trim_snapshot() const {
        std::map<std::string, std::map<int, double>> out;
        for (const auto& cv : _chain)
            for (const auto& tv : cv.second.trim)
                if (std::abs(tv.second.trim) >= 1e-3)
                    out[cv.first][tv.first] = tv.second.trim;
        return out;
    }

    /// Adopt a restored trim for a PRN THE BROKER HAS JUST ARMED. Returns true if adopted.
    ///
    /// ⚠️ WHY ADOPTION IS GATED ON ARMING RATHER THAN DONE AT LOAD. A trim on an unarmed PRN
    /// decays through the graceful-release leak on every window close and is erased below
    /// 1e-3 chips: at leak 0.05 and 23.84 closes/s that is a 0.84 s time constant and ~5.6 s
    /// to erasure, while the broker's policy cycle is ~11 s. Restoring at load would put the
    /// trims back and let them evaporate before the first /set_policy ever named them --
    /// persistence that measures as working and buys nothing.
    ///
    /// It also keeps the rule intact: this stage still acts on no PRN the broker has not
    /// armed. A restored trim is a PROPOSAL, and arming is the acceptance.
    ///
    /// Refuses if the PRN already has a trim of its own -- the live loop outranks a saved one.
    bool adopt_trim(const std::string& chain, int prn, double trim_chips) {
        auto ci = _chain.find(chain);
        if (ci == _chain.end() || !ci->second.armed.count(prn))
            return false;
        if (!std::isfinite(trim_chips)
            || std::abs(trim_chips) > ci->second.policy.clamp)
            return false;
        auto ti = ci->second.trim.find(prn);
        if (ti != ci->second.trim.end() && (ti->second.n_steps != 0 || ti->second.trim != 0.0))
            return false;
        ci->second.trim[prn].trim = trim_chips;
        return true;
    }

    /// #92 THE HANDOVER: the broker re-based a seed by some step and the standing trim
    /// carrying the SAME chips must move with it in the same cycle, or the tap
    /// (seed + trim) transiently lands a chip off the sky, q craters, and the trim is
    /// wiped and rebuilt from scratch (E3's ~25-min sawtooth, docs/CHORD_BUGLIST.md #92).
    /// `delta_chips` is ADDED to the standing trim; the broker owns the sign (it posts
    /// -birth_step). Result is clamped to the chain's policy clamp.
    ///
    /// Refuses for a PRN that is not armed -- an unarmed trim is leak-decaying to
    /// erasure and adjusting it manufactures a correction no loop will maintain -- and
    /// for a non-finite or over-clamp delta: a step the trim could never have been
    /// carrying is not a handover, it is a wrong number (the multi-hundred-chip shared
    /// clock births land here and must be refused, not folded in).
    bool adjust_trim(const std::string& chain, int prn, double delta_chips) {
        auto ci = _chain.find(chain);
        if (ci == _chain.end() || !ci->second.armed.count(prn))
            return false;
        const double c = ci->second.policy.clamp;
        if (!std::isfinite(delta_chips) || std::abs(delta_chips) > c)
            return false;
        auto& ts = ci->second.trim[prn];
        ts.trim = std::max(-c, std::min(c, ts.trim + delta_chips));
        return true;
    }

    /// One (window, record slot, PRN)'s lobe-coherent powers, over the served window depth --
    /// combdll.prompt_cn0's `recs`, in time order.
    ///
    /// WHY A SECOND SHAPE OF THE SAME NUMBERS. The code loop and the comb DLL want records
    /// AVERAGED; the served C/N0 wants the SERIES. It is a radiometric estimator that fits
    /// nothing -- the rate is the tracker's, the tap is the loop's, the only arithmetic is a
    /// debiased power ratio -- so its inputs are the individual records, q-gated one at a time
    /// against a probe anchor. Meaning it away upstream would be exactly the #47 category error
    /// this estimator exists to have removed.
    ///
    /// ⚠️ The gates that USE this stay in Python: the q gate, the probe anchor, the Gamma-mean
    /// debias and the clip. This returns samples -- EVERY record with a live comb, however
    /// partial; `n_chan`/`n_inst` are there so the consumer can decline one.
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
                for (const auto& sv : ring[k].rec)       // slot, ordered
                    for (const auto& pv : sv.second)     // prn, ordered
                        rows.push_back({ring[k].win, sv.first, pv.first, pv.second.n_inst,
                                        pv.second.n_chan, pv.second.e, pv.second.p,
                                        pv.second.l});
        }
        return out;
    }

    const std::map<std::string, Chain>& chains() const {
        return _chain;
    }

    /// ONE PRN's lobe taps over the closed window set, MEANED over records -- exactly what
    /// `combdll.lobe_taps` returns for one PRN.
    struct LobeTap {
        double e = 0.0, p = 0.0, l = 0.0, n_chan = 0.0;
        /// mean cross-sender coherence over the `n_xcoh` records two or more senders reached;
        /// NaN when none did. The per-PRN answer to "are the senders on one reference".
        double xcoh = std::numeric_limits<double>::quiet_NaN();
        int n_xcoh = 0;
        int n_rec = 0;   ///< records behind the mean
        int n_inst = 0;  ///< senders behind the most complete record -- completeness, not an operand
        int64_t hop = -1;
        std::map<int, std::array<double, 4>> chan; ///< freq_id -> {e, p, l, n_rec}
    };

    /// [chain][prn] -- the lobe-coherent, per-channel taps over the served window depth.
    ///
    /// THIS IS WHY IT EXISTS: the Python arm builds the identical object by walking every
    /// (window, sender, record, PRN, channel) of the gathered stream -- ~140k channel-tuples
    /// per chain per cycle, ~700k across the fleet, on a process pinned at 100% of ONE core by
    /// the GIL, where cycle time IS the sum of the chains' Python CPU. The frames are already
    /// here, in C++, so the reduction belongs here and the broker is handed the ~1k numbers
    /// that survive it rather than the 46 MB/s that do not.
    ///
    /// ⚠️ THE MEANS ARE TAKEN OVER DIFFERENT DENOMINATORS ON PURPOSE, mirroring the Python:
    /// the aggregate divides by the PRN's record count, and EACH CHANNEL divides by ITS OWN --
    /// a channel that was live for half the records is a mean over that half, not a half-sized
    /// mean. Getting this wrong is invisible in the full-band numbers and shows up only per
    /// channel, which is precisely where nobody looks.
    ///
    /// ⚠️ POLICY IS NOT HERE AND MUST NOT COME HERE. Presence, the noise floor, the deep gate,
    /// who is armed -- all of that stays on the broker's cycle (GnssFleetTrim.hpp). This
    /// returns measurements. The one filter applied is COMPLETENESS: a record carried by fewer
    /// than `min_instances` senders is not averaged in, in both arms identically.
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
                    // A PRN whose every record THIS window was incomplete contributes no
                    // lobe tap from it, and its channels must not appear without one: the
                    // two tables describe the same records, window by window.
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
    /// The window's open accumulators become its record powers. Done ONCE, at close, so a
    /// late-arriving sender cannot move a record that has already been averaged.
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
        // ⚠️ THE RING IS THE DEEPER OF THE TWO CONSUMERS, and each takes the newest slice it
        // asked for. Trimming to _n_win here (as this did while taps() did not exist) would
        // silently cap the served depth at the loop's, and the broker would get a 2-window
        // average where it configured 32 -- a 4x SNR loss that looks like the sky got worse.
        while ((int)c.closed.size() > std::max(_n_win, _taps_win))
            c.closed.pop_front();
        c.n_closed++;
        if (forced)
            c.n_forced++;
        aggregate(c);
    }

    /// The loop's discriminator: the lobe-coherent record powers of the newest _n_win windows,
    /// MEANED over records, per PRN. Nothing here knows what a sender is.
    void aggregate(Chain& c) {
        struct Acc {
            double e = 0, p = 0, l = 0, n_chan = 0;
            int n_rec = 0, n_inst = 0;
            int64_t hop = -1;
        };
        std::map<int, Acc> by_prn;
        uint64_t win_hi = 0;
        // THE LOOP'S OWN DEPTH: the newest _n_win of the ring, never all of it.
        for (size_t k = c.closed.size() > (size_t)_n_win ? c.closed.size() - _n_win : 0;
             k < c.closed.size(); ++k) {
            const WindowAcc& w = c.closed[k];
            win_hi = std::max(win_hi, w.win);
            for (const auto& sv : w.rec)
                for (const auto& pv : sv.second) {
                    const RecTap& r = pv.second;
                    // COMPLETENESS, not content: a record too few senders reached is not
                    // averaged in. The same test, in the same place, in the Python arm.
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
        // The gate is PROMPT POWER, not q. That is the distinction the broker learned on
        // 2026-08-03 and wrote down: q is peak SHARPNESS, high only once the tap is already
        // on the peak, so gating on it says "only correct the code once it is already
        // correct" and the loop can never pull in from the shoulder. Prompt power answers
        // "is there signal here at all", which is independent of WHERE on the correlation
        // function we sit -- on the shoulder P is still well above noise.
        //
        // This is a LOCAL INFORMATION TEST, not the presence verdict. Presence is policy and
        // stays in Python (it decides whether a satellite is worth tracking, over cycles);
        // this decides whether THIS WINDOW's number means anything.
        //
        // ⚠️⚠️ THE REFERENCE IS THE PROBES, AND THERE IS NO LONGER ANY OTHER OPTION.
        // This used to fall back to `3 x the MEDIAN of this window's own p_pow` whenever the
        // policy carried no absolute floor, on the premise that "most PRNs are signal-free at
        // any moment, so the median IS the no-signal level". That premise was true on the
        // airspy prototype, where --noise-probes seeded genuine signal-free rows, and is
        // FALSE on CHORD, where the armed rows are mostly real satellites. So the bar was a
        // signal level and the gate was a PEER COMPETITION -- a race a satellite loses the
        // moment it starts drifting, which is exactly when it needs the loop.
        // Measured 2026-08-27 on chains still running it, satellites unambiguously ON the
        // peak: gps_l5 PRN 18 (q 2.85) losing 45.7% of its windows, PRN 20 (q 3.25) 34.5%,
        // PRN 27 (q 3.28) 24.4%. And E3's 12-minute outage, where the trim was erased while
        // the satellite sat ~20x above the real probe floor.
        //
        // KV, 2026-08-27, and it is the reason this branch is DELETED rather than defaulted
        // off: "peer comparisons can never tell us about a given signal, they only ever tell
        // us fleet-relative properties, which aren't relevant." A fleet-relative ratio cannot
        // answer "does this window's discriminator mean anything". See
        // docs/CHORD_PEER_COMPARISON_PURGE.md.
        //
        // NO FLOOR => REFUSE. Not "no gate": ungated, an off-peak discriminator random-walks
        // the trim to the clamp (measured, PRN 23, 3.4 chips in 32 s with disc sign-flipping
        // all the way). Leak-only is the conservative direction -- a trim not applied is
        // recoverable, a trim applied to noise is not -- and it mirrors what presence does
        // without a probe anchor, which is to say UNANCHORED and admit nobody.
        const double p_floor = c.policy.p_floor_abs;
        for (int prn : c.armed) {
            auto it = c.row.find(prn);
            if (it == c.row.end() || it->second.win != win)
                continue; // no discriminator formed for this PRN THIS window
            TrimState& t = c.trim[prn];
            if (p_floor <= 0.0 || it->second.p_pow < p_floor) {
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
    /// ⚠️ DECLARATION ORDER IS THE INITIALISER ORDER -- members initialise in declaration
    /// order, so this disagreeing with the constructor's list is a -Wreorder warning today
    /// and a real read-before-init the moment one initialiser references another member.
    int _taps_win = 1; ///< window depth served by taps() -- the POLICY cycle's, not the loop's
    /// A1's epoch-reset knobs -- LAST, for the declaration-order reason directly above.
    int _epoch_margin = 64, _epoch_strikes = 8;
    std::map<std::string, Chain> _chain;
};

} // namespace gnss

#endif // GNSS_FLEET_DLL_HPP
