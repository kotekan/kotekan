#include "GnssCoherentCombiner.hpp"

#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE
#include "visUtil.hpp"      // for frameID

#include "gnssChannelizedDespread.hpp" // for overlay_wipe
#include "beidouB1CCode.hpp"           // for generate_b1cp_secondary (per-PRN B1C overlay)
#include "galileoE1Code.hpp"           // for E1C_CS25 (shared Galileo pilot overlay)
#include "gpsL1CCode.hpp"              // for generate_l1co_code (per-PRN L1C-P overlay)
#include "gpsL5Code.hpp"               // for L5_NH10/NH20 (secondary overlay sequences)

#include "json.hpp"   // for json
#include <algorithm> // for max
#include <cmath>      // for hypot, sqrt
#include <functional> // for bind
#include <string>     // for string

using kotekan::Config;
using kotekan::bufferContainer;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssCoherentCombiner);

GnssCoherentCombiner::GnssCoherentCombiner(Config& config, const std::string& unique_name,
                                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssCoherentCombiner::main_thread, this)) {
    in_bufs = get_buffer_array("in_bufs");
    for (Buffer* b : in_bufs)
        b->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    // Records per frame: configured, else inferred from the first input frame size.
    _n_prn = config.get_default<int>(unique_name, "n_prn", 0);
    if (_n_prn <= 0 && !in_bufs.empty())
        _n_prn = in_bufs[0]->frame_size / (int)sizeof(float) / RECORD_FLOATS;

    _integration_length = std::max(1, config.get_default<int>(unique_name, "integration_length", 1));
    const std::string mode = config.get_default<std::string>(unique_name, "integration_mode", "block");
    _rolling = (mode == "rolling");
    // Rolling output cadence: every output_every records (default ~10 outputs per time
    // constant). Block mode emits once per integration_length records (ignores this).
    _emit_every = std::max(1, config.get_default<int>(unique_name, "output_every",
                                                      std::max(1, _integration_length / 10)));
    _navwipe_bit_records = config.get_default<int>(unique_name, "navwipe_bit_records", 0);
    // Known secondary overlay (the L5 Neuman-Hofman pilot overlay) -> deep coherent integration
    // by searching its alignment, instead of the squaring nav-bit estimate. Mutually exclusive
    // with navwipe_bit_records (a pilot has no nav bit). Both buffer the per-record A.
    const std::string sec = config.get_default<std::string>(unique_name, "secondary_overlay", "");
    if (sec == "L5_NH20")
        _secondary.assign(gps::L5_NH20.begin(), gps::L5_NH20.end());
    else if (sec == "L5_NH10")
        _secondary.assign(gps::L5_NH10.begin(), gps::L5_NH10.end());
    else if (sec == "COHERENT")
        // Dataless pilot with NO overlay at all (L2C CL: the 1.5 s code is the only modulation
        // and records are consecutive segments of it, phase-continuous by construction): a
        // length-1 all-ones "overlay" turns overlay_wipe into a plain gap-robust coherent sum
        // with the same SNR estimate and auto-coherence ladder -- deep coherent integration
        // with no bit estimate and no alignment search.
        _secondary.assign(1, (int8_t)1);
    else if (sec == "E1_CS25")
        // Galileo E1-C pilot: the 25-chip CS25 secondary, SAME sequence for every satellite
        // (like the L5 NH overlays, just longer). One chip per 4 ms primary period; the
        // 25-phase alignment search is well-determined within a 250-record (1 s) window.
        _secondary.assign(galileo::E1C_CS25.begin(), galileo::E1C_CS25.end());
    else if (sec == "B1C") {
        // BeiDou-3 B1C pilot: PER-PRN 1800-chip Weil overlay -- structurally identical to
        // L1CO (same length, same per-PRN pick-at-wipe-time), reusing its cache/selection
        // path. Alignment floor ~sqrt(2 ln 1800) ~ 3.9 sigma, same caveat as L1CO.
        _l1co.resize(63);
        for (int prn = 1; prn <= 63; ++prn) {
            const auto o = beidou::generate_b1cp_secondary(prn);
            _l1co[(size_t)(prn - 1)].assign(o.begin(), o.end());
        }
    } else if (sec == "L1CO") {
        // L1C-P overlay is PER-PRN (not one shared NH sequence), so cache all 32 here and pick
        // by the slot's current PRN at wipe time. NOTE: it is 1800 symbols long (vs NH20's 20),
        // so the overlay_wipe phase search is only well-determined once the integration window
        // holds a comparable number of records -- run this with a LONG rolling integration and
        // read the deep_snr against a higher floor (~sqrt(2 ln 1800) ~ 3.9 sigma, vs NH20's 2.4).
        _l1co.resize(32);
        for (int prn = 1; prn <= 32; ++prn) {
            const auto o = gps::generate_l1co_code(prn);
            _l1co[(size_t)(prn - 1)].assign(o.begin(), o.end());
        }
    }
    _wipe_buffer = (_navwipe_bit_records > 0 || !_secondary.empty() || !_l1co.empty());
    // Auto-coherence (default ON, the clock_profile philosophy: automatic unless overridden):
    // integration_length is a MAX -- at emit the deep wipe is also evaluated over trailing
    // sub-windows (full, 1/2, 1/4, ... octaves) and the best deep SNR wins, so the deep |A|
    // reads the longest span the clock (+ sky) actually COHERES over instead of a decohered
    // full-window average. The chosen span is exported as coherence_s via get_status -- a live
    // measurement of the receiver clock's usable coherence time.
    _auto_coherence = config.get_default<bool>(unique_name, "auto_coherence", true);
    // Shared-carrier-loop observable: measure the full-band cross-record phase walk. Data
    // signals need the bit-robust SQUARED product (+-1 symbol flips cancel; unambiguous to
    // +-1/(4*dt)); a dataless pilot (L2C CL) sets carrier_pilot: true for the unsquared
    // product -- twice the unaliased range, no squaring loss at low SNR.
    _carrier_pilot = config.get_default<bool>(unique_name, "carrier_pilot", false);
    if (_wipe_buffer) {
        _navbuf.assign(_n_prn, {});
        _navutc.assign(_n_prn, {});
    }

    _st_prn.assign(_n_prn, 0);
    _st_amp.assign(_n_prn, 0.0f);
    _st_coh.assign(_n_prn, 0.0f);
    _st_deep.assign(_n_prn, 0.0f);
    _st_deep_snr.assign(_n_prn, 0.0f);
    _st_amp_snr.assign(_n_prn, 0.0f);
    _st_amp_dbi.assign(_n_prn, 0.0f);
    _st_nh_phase.assign(_n_prn, -1);
    _st_dop.assign(_n_prn, 0.0f);
    _st_cp.assign(_n_prn, 0.0f);
    _st_dll_disc.assign(_n_prn, 0.0f);
    _st_car_resid.assign(_n_prn, 0.0f);
    _st_coh_s.assign(_n_prn, 0.0f);
    _st_deep_floor.assign(_n_prn, 0.0f);
    _st_deep_pow.assign(_n_prn, 0.0f);
    _dr_phase.assign(_n_prn, -1);
    _dr_utc.assign(_n_prn, 0.0);
    _dr_prn.assign(_n_prn, -1);
    _st_deep_rec.assign(_n_prn, 0);
}

void GnssCoherentCombiner::main_thread() {
    using namespace std::placeholders;
    kotekan::restServer::instance().register_get_callback(
        unique_name + "/get_status",
        std::bind(&GnssCoherentCombiner::get_status_callback, this, _1));

    std::vector<frameID> in_ids;
    for (Buffer* b : in_bufs)
        in_ids.emplace_back(b);
    frameID out_id(out_buf);

    // Per-PRN integration accumulators. block: running SUMS over _integration_length records.
    // rolling: exponential moving AVERAGES (alpha = 1/_integration_length), bias-corrected at
    // emit by dividing by (1 - (1-alpha)^n_roll) so they read true running means from record 1.
    std::vector<double> acc_pow(_n_prn), acc_ar(_n_prn), acc_ai(_n_prn), acc_nchan(_n_prn);
    std::vector<double> acc_pow2(_n_prn); // <|A|^4>, for the noise-debiased incoherent significance
    std::vector<double> acc_epow(_n_prn), acc_lpow(_n_prn); // <|E|^2>, <|L|^2> for the DLL disc
    // Shared carrier loop: vector-averaged cross-record phase product per PRN over the emit
    // window. S += (A_k conj A_{k-1})^{2|1}; at emit resid_hz = arg(S)/{2|1}/(2 pi dt_med).
    std::vector<std::complex<double>> car_S(_n_prn);
    std::vector<double> car_dt(_n_prn);
    std::vector<int> car_n(_n_prn);
    std::vector<std::complex<double>> car_prev(_n_prn);
    std::vector<double> car_prev_utc(_n_prn, -1.0);
    std::vector<float> ref_prn(_n_prn), ref_dop(_n_prn), ref_cp(_n_prn);
    std::vector<double> ref_utc(_n_prn);
    int n_acc = 0;              // block: records in the current block
    long long n_roll = 0;      // rolling: total records seen (for EMA bias correction)
    int since_emit = 0;        // rolling: records since the last emit
    const double alpha = 1.0 / (double)_integration_length; // rolling EMA weight

    while (!stop_thread) {
        // Gather the i-th frame of every subband (same window, same PRN order).
        std::vector<float*> ins;
        bool stopping = false;
        for (size_t i = 0; i < in_bufs.size(); ++i) {
            float* in = (float*)in_bufs[i]->wait_for_full_frame(unique_name, in_ids[i]);
            if (in == nullptr) {
                stopping = true;
                break;
            }
            ins.push_back(in);
        }
        if (stopping)
            break;

        for (int p = 0; p < _n_prn; ++p) {
            // Sum the un-normalized correlation and replica energy across subbands
            // (the cross-channel coherent combine), then the full-band amplitude.
            double gr = 0.0, gi = 0.0, energy = 0.0, nchan = 0.0;
            double er = 0.0, ei = 0.0, e_energy = 0.0, lr = 0.0, li = 0.0, l_energy = 0.0;
            const float* ref = ins[0] + (size_t)p * RECORD_FLOATS; // PRN/dop/cp/UTC reference
            for (float* in : ins) {
                const float* rec = in + (size_t)p * RECORD_FLOATS;
                gr += rec[3];
                gi += rec[4];
                energy += rec[5];
                nchan += rec[6];
                // Early/Late correlators (gnssRecord.hpp): additive across subbands exactly
                // like the prompt -- sum raw G and energy, normalize full-band below. Only the
                // POWERS feed the DLL discriminator, so per-subband NCO phases are irrelevant.
                er += rec[gnss::REC_E_RE];
                ei += rec[gnss::REC_E_IM];
                e_energy += rec[gnss::REC_E_ENERGY];
                lr += rec[gnss::REC_L_RE];
                li += rec[gnss::REC_L_IM];
                l_energy += rec[gnss::REC_L_ENERGY];
            }
            const double ar = energy > 0.0 ? gr / energy : 0.0;
            const double ai = energy > 0.0 ? gi / energy : 0.0;
            const double e2 = e_energy > 0.0 ? (er * er + ei * ei) / (e_energy * e_energy) : 0.0;
            const double l2 = l_energy > 0.0 ? (lr * lr + li * li) / (l_energy * l_energy) : 0.0;

            // Accumulate over time: incoherent power (|A|^2) + its square (|A|^4, for the noise
            // debias) + coherent complex (A).
            const double utc_p = *reinterpret_cast<const double*>(ref + RECORD_UTC_SLOT);
            const double p2 = ar * ar + ai * ai;
            {
                // Full-band carrier phase walk vs the previous record (gap-gated: a valve drop
                // would alias the product). Squared for data signals, plain for pilots.
                const std::complex<double> A(ar, ai);
                const double dt_c = utc_p - car_prev_utc[p];
                if (car_prev_utc[p] > 0.0 && dt_c > 0.0 && dt_c < 0.050) {
                    std::complex<double> prod = A * std::conj(car_prev[p]);
                    if (!_carrier_pilot)
                        prod *= prod;
                    car_S[p] += prod;
                    car_dt[p] += dt_c;
                    car_n[p] += 1;
                }
                car_prev[p] = A;
                car_prev_utc[p] = utc_p;
            }
            if (_rolling) { // exponential moving average (no reset; integrates indefinitely)
                acc_pow[p] += alpha * (p2 - acc_pow[p]);
                acc_pow2[p] += alpha * (p2 * p2 - acc_pow2[p]);
                acc_epow[p] += alpha * (e2 - acc_epow[p]);
                acc_lpow[p] += alpha * (l2 - acc_lpow[p]);
                acc_ar[p] += alpha * (ar - acc_ar[p]);
                acc_ai[p] += alpha * (ai - acc_ai[p]);
                acc_nchan[p] += alpha * (nchan - acc_nchan[p]);
                if (_wipe_buffer && energy > 0.0) {
                    // Sliding window of the last _integration_length recs (deep wipe). Records
                    // with NO despread (PRN inactive this window: broker drop/coast churn) are
                    // NOT buffered -- the wipe is UTC-indexed and gap-robust, so they become
                    // gaps exactly like valve drops. Buffering their exact zeros instead
                    // diluted the amplitude while contributing NOTHING to the orthogonal-noise
                    // estimate: a mostly-zero window reported a tiny deep |A| at an enormous
                    // fake significance (observed 500 sigma at |A|=0.04 on flapping CL sats).
                    _navbuf[p].emplace_back(ar, ai);
                    _navutc[p].push_back(utc_p);
                    if ((int)_navbuf[p].size() > _integration_length) {
                        _navbuf[p].erase(_navbuf[p].begin());
                        _navutc[p].erase(_navutc[p].begin());
                    }
                }
                ref_prn[p] = ref[0]; // reference = the latest record (rolling has no block start)
                ref_dop[p] = ref[1];
                ref_cp[p] = ref[2];
                ref_utc[p] = utc_p;
            } else { // block: running sums, reset at emit
                acc_pow[p] += p2;
                acc_pow2[p] += p2 * p2;
                acc_epow[p] += e2;
                acc_lpow[p] += l2;
                acc_ar[p] += ar;
                acc_ai[p] += ai;
                acc_nchan[p] += nchan;
                if (_wipe_buffer && energy > 0.0) { // per-record (A, UTC) for the deep wipe;
                    // zero-energy (inactive) records are gaps, not zeros -- see the rolling
                    // branch comment (fake-significance pathology otherwise).
                    _navbuf[p].emplace_back(ar, ai);
                    _navutc[p].push_back(utc_p);
                }
                if (n_acc == 0) { // window reference from the first record of the block
                    ref_prn[p] = ref[0];
                    ref_dop[p] = ref[1];
                    ref_cp[p] = ref[2];
                    ref_utc[p] = utc_p;
                }
            }
        }

        for (size_t i = 0; i < in_bufs.size(); ++i)
            in_bufs[i]->mark_frame_empty(unique_name, in_ids[i]++);

        if (_rolling) {
            ++n_roll;
            if (++since_emit < _emit_every)
                continue; // EMA keeps integrating; emit only every _emit_every records
            since_emit = 0;
        } else if (++n_acc < _integration_length) {
            continue; // keep accumulating
        }

        float* out = (float*)out_buf->wait_for_empty_frame(unique_name, out_id);
        if (out == nullptr)
            break;
        // block: divide sums by the block count. rolling: the EMA is already a mean, so just
        // undo the warm-up bias (1-(1-alpha)^n_roll) -> a true running mean from record 1.
        const double inv = _rolling ? 1.0 / (1.0 - std::pow(1.0 - alpha, (double)n_roll))
                                    : 1.0 / (double)n_acc;
        std::vector<double> deep_snr(_n_prn, 0.0); // significance of the deep detection (for REST)
        std::vector<double> amp_snr(_n_prn, 0.0);  // noise-debiased incoherent significance (REST)
        std::vector<int> nh_phase(_n_prn, -1);     // secondary-overlay alignment found (L5; -1 = n/a)
        std::vector<double> amp_dbi(_n_prn, 0.0);  // noise-debiased (unbiased) signal amplitude
        std::vector<double> coh_s(_n_prn, 0.0);    // measured coherence: span of the winning window
        std::vector<int> deep_rec(_n_prn, 0);      // records in the winning deep window
        std::vector<double> deep_floor(_n_prn, 0.0); // rectification floor at the reported rung
        // FIXED-WINDOW, noise-debiased coherent power (Hz) for the beam map: the coherent
        // twin of the incoherent x = s^2/N. Computed from the FULL window's wipe every emit
        // -- NO ladder, NO detection gate, so a pixel average over emits is UNBIASED at
        // arbitrarily weak signal (the map's radiometry rule: average the linear estimator
        // INCLUDING its noise; never threshold-then-average). E[snr^2] under noise is the
        // wipe's selection mean (overlay: max over n_align trials of ~Exp(2) -> 2(ln n +
        // gamma); navwipe: floor^2 + 1); subtracting it centers noise on ZERO, and the
        // map's below-horizon pedestal absorbs the approximation residue exactly as it
        // does x's clip bias. Coherent integration buys sqrt(T_coh/T_rec) sensitivity --
        // ~15 dB (GPS) / 12 (E1C) / 10 (B1C) deeper sidelobes per unit dwell.
        std::vector<double> deep_pow(_n_prn, 0.0);
        // A wiped rung whose SNR is indistinguishable from its NOISE floor must not be
        // reported as coherence: the wipe's free parameters rectify pure noise to a
        // window-length-dependent floor, so an unlocked sat otherwise reports coh = the
        // FULL window at deep ~= floor -- a convincing false lock (observed 2026-07-12
        // when the carrier loop briefly ran open). Rungs must clear FLOOR_MARGIN x floor
        // to be eligible; when none does, coherence_s = 0 (the honest "no coherent
        // detection") and the full-window numbers are reported for diagnostics only.
        constexpr double FLOOR_MARGIN = 2.0;
        // Auto-coherence ladder: deep-wipe window lengths to try -- the full buffer (always,
        // legacy behaviour) plus up to 4 octave-shorter trailing suffixes (>= min_len). Walked
        // longest-first; a shorter window must beat the incumbent deep SNR by >5%, so ties
        // integrate DEEP and the winner sits at the knee of the coherence curve.
        auto ladder = [&](size_t n, size_t min_len) {
            std::vector<size_t> lens;
            if (n > 0)
                lens.push_back(n);
            if (_auto_coherence)
                for (size_t l = n / 2; l >= std::max<size_t>(min_len, 2) && lens.size() < 5; l /= 2)
                    lens.push_back(l);
            return lens;
        };
        const double Keff = _rolling ? (double)_integration_length : (double)n_acc;
        for (int p = 0; p < _n_prn; ++p) {
            float* rec = out + (size_t)p * RECORD_FLOATS;
            for (int f = 0; f < RECORD_FLOATS; ++f)
                rec[f] = 0.0f;
            rec[0] = ref_prn[p];
            rec[1] = ref_dop[p];
            rec[2] = ref_cp[p];
            rec[3] = (float)std::sqrt(acc_pow[p] * inv);             // |A|_incoh = sqrt<|A|^2>
            rec[4] = (float)(acc_ar[p] * inv);                       // <A>.re (coherent mean)
            rec[5] = (float)(acc_ai[p] * inv);                       // <A>.im
            rec[6] = (float)std::hypot(acc_ar[p], acc_ai[p]) * inv;  // |<A>|_coh
            rec[7] = (float)(acc_nchan[p] * inv);                    // covering channels used
            // DLL observables (gnssRecord.hpp): window-averaged full-band Early/Late powers and
            // the normalized early-minus-late discriminator. ~Linear in the code-phase error
            // over ~+-dll_spacing (tau ~ disc/4 chips at 0.5-chip spacing); the BROKER closes
            // the loop from these at ~Hz. Noise averages out over the window, so no per-record
            // decisions anywhere (R1).
            {
                const double e_pow = acc_epow[p] * inv;
                const double l_pow = acc_lpow[p] * inv;
                rec[gnss::CMB_E_POW] = (float)e_pow;
                rec[gnss::CMB_L_POW] = (float)l_pow;
                rec[gnss::CMB_DLL_DISC] =
                    (e_pow + l_pow) > 0.0 ? (float)((e_pow - l_pow) / (e_pow + l_pow)) : 0.0f;
                // Shared carrier observable: full-band residual carrier (Hz). The broker
                // integrates this into carrier_trim_hz for every subband tracker -- ONE loop at
                // full-band SNR instead of N noise-driven per-channel FLLs. PREFER a bit-robust
                // phase-SLOPE fit over the whole deep-wipe window (_navbuf): its long baseline
                // gives a low-variance frequency, the clean measurement the loop needs to hold
                // <0.5 Hz over a 1 s coherent window (OFFLINE-VALIDATED: with this residual removed,
                // the L1 deep builds 77 sigma@1 s / 94@2 s vs a ~20 ms wall without). The old
                // consecutive-record product (car_S) is short-baseline and doubly-noisy (squaring)
                // -> it made the shared loop noise-inject; kept only as the fallback for non-wipe
                // configs where _navbuf isn't populated.
                double resid_hz = 0.0;
                if (_wipe_buffer && _navbuf[p].size() >= 16) {
                    resid_hz = carrier_resid_hz(_navbuf[p], _navutc[p]);
                } else if (car_n[p] > 0 && std::abs(car_S[p]) > 0.0) {
                    const double dphi = std::arg(car_S[p]) / (_carrier_pilot ? 1.0 : 2.0);
                    resid_hz = dphi / (2.0 * M_PI * (car_dt[p] / car_n[p]));
                }
                rec[gnss::CMB_CARRIER_RESID] = (float)resid_hz;
                car_S[p] = {0.0, 0.0};
                car_dt[p] = 0.0;
                car_n[p] = 0;
            }
            // Noise-debiased incoherent significance: <|A|^2> = s^2 + N is biased by the noise
            // floor N (so |A| ~= floor for weak sats), but the moments separate them --
            // s^2 = sqrt(<|A|^2>^2 - Var(|A|^2)), N = <|A|^2> - s^2 -- giving a signal-vs-noise SNR
            // that is ~0 for noise (not the floor) and grows ~sqrt(K) for a real sat. The honest
            // "is it really there", independent of nav-bit wipe.
            {
                const double m2 = acc_pow[p] * inv, m4 = acc_pow2[p] * inv;
                const double var = std::max(0.0, m4 - m2 * m2);
                const double s2 = std::sqrt(std::max(0.0, m2 * m2 - var));
                const double noise = m2 - s2;
                amp_dbi[p] = std::sqrt(s2); // unbiased signal amplitude (~0 for noise, =signal for a sat)
                // Normalise by the H0 std of the s^2 estimator (~N/K^{1/4}) so the significance is
                // ~1 for noise REGARDLESS of K (a stable lock threshold) and grows for real signal.
                amp_snr[p] = noise > 1e-12 ? s2 * std::pow(Keff, 0.25) / noise : 0.0;
            }
            if (!_secondary.empty() || !_l1co.empty()) {
                // Dataless pilot: wipe the KNOWN secondary overlay at its best-fitting alignment
                // -> deep coherent |A| past the primary period (no nav-bit estimate). L5 Q5 uses
                // one PRN-independent NH overlay; L1C-P uses the per-PRN L1CO, picked by the
                // slot's current PRN (a slot's PRN can change as the search re-assigns channels).
                const std::vector<int8_t>* ov = &_secondary;
                if (_l1co.empty()) {
                    // L5: _secondary already points at the shared overlay.
                } else {
                    const int prn = (int)std::lround(ref_prn[p]);
                    ov = (prn >= 1 && prn <= (int)_l1co.size()) ? &_l1co[(size_t)(prn - 1)] : nullptr;
                }
                if (ov && !ov->empty()) {
                    const auto& ab = _navbuf[p];
                    const auto& ub = _navutc[p];
                    // Overlay-wipe floor: best of ~n_align alignment trials of a Rayleigh
                    // sum -> noise rectifies to ~sqrt(2 ln n) sigma, window-length-free.
                    const double flr =
                        std::sqrt(2.0 * std::log((double)std::max<size_t>(ov->size(), 2))) + 1.0;
                    deep_floor[p] = flr;
                    bool cleared = false;
                    gnss::OverlayWipeResult full{};
                    size_t full_len = 0;
                    for (size_t len : ladder(ab.size(), std::max<size_t>(2 * ov->size(), 64))) {
                        gnss::OverlayWipeResult ow;
                        if (len == ab.size()) {
                            ow = gnss::overlay_wipe(ab, ub, *ov);
                        } else {
                            const std::vector<std::complex<double>> as(ab.end() - (long)len,
                                                                       ab.end());
                            const std::vector<double> us(ub.end() - (long)len, ub.end());
                            ow = gnss::overlay_wipe(as, us, *ov);
                        }
                        if (full_len == 0) { // ladder walks longest-first
                            full = ow;
                            full_len = len;
                        }
                        if (ow.snr < FLOOR_MARGIN * flr)
                            continue; // indistinguishable from noise rectification
                        if (!cleared || ow.snr > deep_snr[p] * 1.05) {
                            rec[8] = (float)ow.amplitude;
                            deep_snr[p] = ow.snr;
                            nh_phase[p] = ow.phase;
                            deep_rec[p] = (int)len;
                            coh_s[p] = ub.empty() ? 0.0 : ub.back() - ub[ub.size() - len];
                            cleared = true;
                        }
                    }
                    if (!cleared && full_len > 0) {
                        rec[8] = (float)full.amplitude;
                        deep_snr[p] = full.snr;
                        nh_phase[p] = full.phase;
                        deep_rec[p] = (int)full_len;
                        coh_s[p] = 0.0; // no rung beat its floor: no coherent detection
                    }
                    // DEAD-RECKON ANCHOR (selection-free coherent power): a floor-cleared
                    // win pins the overlay alignment at a capture-UTC; it advances one chip
                    // per primary period deterministically, so later emits can wipe at the
                    // PREDICTED phase -- one straight coherent sum, no max-over-alignments,
                    // E[snr^2] = 2 exactly. That removes the extreme-value variance that
                    // made the coherent single-emit floors WORSE than the incoherent ones
                    // (2026-07-12: C 26 vs 20.3 dB-Hz) and is what unlocks the coherent
                    // observable's sqrt(T_coh/T_rec) sidelobe advantage.
                    if (cleared) {
                        _dr_phase[p] = nh_phase[p];
                        _dr_utc[p] = ub[ub.size() - (size_t)deep_rec[p]];
                        _dr_prn[p] = (int)std::lround(ref_prn[p]);
                    } else if (_dr_prn[p] != (int)std::lround(ref_prn[p])) {
                        _dr_phase[p] = -1; // slot reassigned to another sat: anchor invalid
                    }
                    if (full_len > 1 && ub.size() >= full_len) {
                        const double t0f = ub[ub.size() - full_len];
                        const double span = ub.back() - t0f;
                        const double T = span * (double)full_len / (double)(full_len - 1);
                        const double rec_dt = span / (double)(full_len - 1);
                        bool dr_ok = _dr_phase[p] >= 0 && rec_dt > 0.0
                                     && _dr_prn[p] == (int)std::lround(ref_prn[p]);
                        if (dr_ok) {
                            const long long steps =
                                (long long)std::llround((t0f - _dr_utc[p]) / rec_dt);
                            const int L_ov = (int)ov->size();
                            const int ph = (int)(((_dr_phase[p] + steps) % L_ov + L_ov) % L_ov);
                            const auto dw = gnss::overlay_wipe_at(ab, ub, *ov, ph);
                            if (T > 0.0 && dw.snr > 0.0)
                                deep_pow[p] = (dw.snr * dw.snr - 2.0) / T;
                            else
                                dr_ok = false;
                        }
                        if (!dr_ok) { // no anchor yet: searched full window, selection-debiased
                            const double bias = 2.0
                                * (std::log((double)std::max<size_t>(ov->size(), 2)) + 0.5772);
                            if (T > 0.0)
                                deep_pow[p] = (full.snr * full.snr - bias) / T;
                        }
                    }
                }
                if (!_rolling) { _navbuf[p].clear(); _navutc[p].clear(); }
            } else if (_navwipe_bit_records > 0) {
                const auto& ab = _navbuf[p];
                const auto& ub = _navutc[p];
                // Navwipe floor GROWS with window length: nb free +-1 bit signs rectify
                // noise to ~sqrt(2 nb / pi) sigma, plus the bit-phase selection over
                // _navwipe_bit_records alignments (~sqrt(ln br)). The ~7 sigma "GPS deep
                // floor" at 50 bits is exactly this -- and it is why an unlocked sat's
                // ladder drifts to the LONGEST rung (bigger floor) if not gated.
                const double sel = std::sqrt(std::log((double)std::max(_navwipe_bit_records, 2)));
                auto nav_floor = [&](size_t len) {
                    const double nb = std::max(1.0, (double)len / (double)_navwipe_bit_records);
                    return std::sqrt(2.0 * nb / M_PI) + sel;
                };
                bool cleared = false;
                double full_amp = 0.0, full_snr = 0.0;
                size_t full_len = 0;
                for (size_t len :
                     ladder(ab.size(), std::max<size_t>(4 * (size_t)_navwipe_bit_records, 64))) {
                    double snr = 0.0, amp;
                    if (len == ab.size()) {
                        amp = navwipe_amplitude(ab, ub, &snr); // deep |A|
                    } else {
                        const std::vector<std::complex<double>> as(ab.end() - (long)len, ab.end());
                        const std::vector<double> us(ub.end() - (long)len, ub.end());
                        amp = navwipe_amplitude(as, us, &snr);
                    }
                    if (full_len == 0) { // ladder walks longest-first
                        full_amp = amp;
                        full_snr = snr;
                        full_len = len;
                    }
                    if (snr < FLOOR_MARGIN * nav_floor(len))
                        continue; // indistinguishable from noise rectification
                    if (!cleared || snr > deep_snr[p] * 1.05) {
                        rec[8] = (float)amp;
                        deep_snr[p] = snr;
                        deep_rec[p] = (int)len;
                        deep_floor[p] = nav_floor(len);
                        coh_s[p] = ub.empty() ? 0.0 : ub.back() - ub[ub.size() - len];
                        cleared = true;
                    }
                }
                if (!cleared && full_len > 0) {
                    rec[8] = (float)full_amp;
                    deep_snr[p] = full_snr;
                    deep_rec[p] = (int)full_len;
                    deep_floor[p] = nav_floor(full_len);
                    coh_s[p] = 0.0; // no rung beat its floor: no coherent detection
                }
                if (full_len > 1 && ub.size() >= full_len) {
                    const double span = ub.back() - ub[ub.size() - full_len];
                    const double T = span * (double)full_len / (double)(full_len - 1);
                    const double flr0 = nav_floor(full_len);
                    if (T > 0.0)
                        deep_pow[p] = (full_snr * full_snr - (flr0 * flr0 + 1.0)) / T;
                }
                if (!_rolling) { // block clears the window; rolling slides it (capped above)
                    _navbuf[p].clear();
                    _navutc[p].clear();
                }
            }
            *reinterpret_cast<double*>(rec + RECORD_UTC_SLOT) = ref_utc[p];
            if (!_rolling)
                acc_pow[p] = acc_pow2[p] = acc_ar[p] = acc_ai[p] = acc_nchan[p] = acc_epow[p] = acc_lpow[p] = 0.0; // block: reset
        }
        if (!_rolling)
            n_acc = 0;

        // Publish the latest full-band amplitudes for the broker's drop decisions.
        {
            std::lock_guard<std::mutex> lk(_st_mtx);
            for (int p = 0; p < _n_prn; ++p) {
                const float* rec = out + (size_t)p * RECORD_FLOATS;
                _st_prn[p] = (int)std::lround(rec[0]);
                _st_amp[p] = rec[3];
                _st_coh[p] = rec[6];
                _st_deep[p] = rec[8];
                _st_deep_snr[p] = (float)deep_snr[p];
                _st_nh_phase[p] = nh_phase[p];
                _st_amp_snr[p] = (float)amp_snr[p];
                _st_amp_dbi[p] = (float)amp_dbi[p];
                _st_dop[p] = rec[1];
                _st_cp[p] = rec[2];
                _st_coh_s[p] = (float)coh_s[p];
                _st_deep_rec[p] = deep_rec[p];
                _st_deep_floor[p] = (float)deep_floor[p];
                _st_deep_pow[p] = (float)deep_pow[p];
                _st_dll_disc[p] = rec[gnss::CMB_DLL_DISC];
                _st_car_resid[p] = rec[gnss::CMB_CARRIER_RESID];
            }
        }
        out_buf->mark_frame_full(unique_name, out_id++);
    }
}

void GnssCoherentCombiner::get_status_callback(kotekan::connectionInstance& conn) {
    nlohmann::json reply = nlohmann::json::array();
    std::lock_guard<std::mutex> lk(_st_mtx);
    for (int p = 0; p < _n_prn; ++p)
        reply.push_back({{"prn", _st_prn[p]},
                         {"amplitude", _st_amp[p]},
                         {"coh_amplitude", _st_coh[p]},
                         {"deep_amplitude", _st_deep[p]},
                         {"deep_snr", _st_deep_snr[p]},
                         {"amp_snr", _st_amp_snr[p]},
                         {"nh_phase", _st_nh_phase[p]},
                         {"unbiased_amplitude", _st_amp_dbi[p]},
                         {"doppler_hz", _st_dop[p]},
                         {"code_phase_chips", _st_cp[p]},
                         {"coherence_s", _st_coh_s[p]},
                         {"deep_records", _st_deep_rec[p]},
                         {"deep_floor", _st_deep_floor[p]},
                         {"deep_pow_hz", _st_deep_pow[p]},
                         {"dll_disc", _st_dll_disc[p]},
                         {"carrier_hz_resid", _st_car_resid[p]}});
    conn.send_json_reply(reply);
}

double
GnssCoherentCombiner::navwipe_amplitude(const std::vector<std::complex<double>>& a,
                                        const std::vector<double>& utc, double* snr_out) const {
    const int br = _navwipe_bit_records;
    const int nrec = (int)a.size();
    if (br <= 0 || nrec < 2 * br)
        return 0.0;
    using cd = std::complex<double>;

    // Absolute code-period index per record from capture-UTC -- a valve drop just skips an
    // index, so bit epochs stay aligned (vs binning by buffer position, which a gap shifts).
    std::vector<double> dt;
    dt.reserve(nrec - 1);
    for (int r = 1; r < nrec; ++r)
        dt.push_back(utc[r] - utc[r - 1]);
    std::nth_element(dt.begin(), dt.begin() + dt.size() / 2, dt.end());
    const double rec_dt = dt[dt.size() / 2]; // median step = the no-drop record period
    if (!(rec_dt > 0.0))
        return 0.0;
    std::vector<long long> cpi(nrec);
    for (int r = 0; r < nrec; ++r)
        cpi[r] = (long long)std::llround((utc[r] - utc[0]) / rec_dt);

    // Per-bit coherent sums for a given epoch phase: records are sorted by cpi, so a bit is
    // a run of equal floor((cpi+phase)/br). Returns the sums (and their summed |.| via out).
    auto bit_sums = [&](int phase, std::vector<cd>* out, double* powsum) {
        cd s(0.0, 0.0);
        long long cur = 0;
        bool have = false;
        double g = 0.0;
        int nb = 0;
        for (int r = 0; r < nrec; ++r) {
            const long long bi = (cpi[r] + phase) / br;
            if (have && bi != cur) {
                if (out)
                    out->push_back(s);
                g += std::abs(s);
                ++nb;
                s = cd(0.0, 0.0);
            }
            s += a[r];
            cur = bi;
            have = true;
        }
        if (have) {
            if (out)
                out->push_back(s);
            g += std::abs(s);
            ++nb;
        }
        if (powsum)
            *powsum = nb >= 2 ? g / nb : -1.0;
    };

    // Bit sync: the phase (0..br-1) maximising the mean per-bit coherent power.
    int best_phase = 0;
    double best_g = -1.0;
    for (int phase = 0; phase < br; ++phase) {
        double g;
        bit_sums(phase, nullptr, &g);
        if (g > best_g) {
            best_g = g;
            best_phase = phase;
        }
    }
    std::vector<cd> s;
    bit_sums(best_phase, &s, nullptr);
    if (s.size() < 2)
        return 0.0;

    // Estimate the +-1 per bit by squaring (theta0 = 1/2 arg sum s^2; the global sign cancels
    // in |.|), wipe, and coherently sum -> the deep coherent gain (mean of the wiped A). The
    // component ORTHOGONAL to the aligned signal (the imaginary part after rot) carries no signal,
    // so it measures the per-bit noise -> the deep's uncertainty and a significance SNR.
    cd sumsq(0.0, 0.0);
    for (const cd& v : s)
        sumsq += v * v;
    const cd rot = std::polar(1.0, -0.5 * std::arg(sumsq));
    cd deep(0.0, 0.0);
    double noise2 = 0.0; // sum of squared orthogonal (noise) components, one per bit
    for (const cd& v : s) {
        const cd vr = v * rot; // align the signal onto the real axis
        deep += (std::real(vr) >= 0.0 ? 1.0 : -1.0) * v;
        noise2 += std::imag(vr) * std::imag(vr);
    }
    const int nb = (int)s.size();
    const double sigma_bit = std::sqrt(noise2 / (double)nb);    // per-bit noise (1 axis)
    const double noise_sum = sigma_bit * std::sqrt((double)nb); // noise std of the coherent sum
    // SNR = significance of the deep detection (coherent sum / its noise); the deep's standard
    // error is noise_sum/nrec, so SNR == deep / its uncertainty. Real lock SNR >> 1; noise ~1.
    if (snr_out)
        *snr_out = noise_sum > 0.0 ? std::abs(deep) / noise_sum : 0.0;
    return std::abs(deep) / (double)nrec; // coherent mean of the wiped per-record A
}

double
GnssCoherentCombiner::carrier_resid_hz(const std::vector<std::complex<double>>& a,
                                       const std::vector<double>& utc) const {
    const int n = (int)a.size();
    if (n < 16)
        return 0.0;
    using cd = std::complex<double>;
    // Bit-robust phase-SLOPE fit: square A to cancel the +-1 nav-bit pi flips (a data signal), or
    // fit the raw phase for a dataless pilot. TWO STAGES, because a sequential record-to-record
    // unwrap alone is fragile: at 1 ms records (GPS) the window has ~1000 unwrap steps at 2x the
    // per-record phase noise, and ONE deep fade mid-window slips the accumulated phase by 2*pi,
    // biasing the slope by several Hz. The broker loop then injects that error into the NCO
    // (observed 2026-07-12: per-emit resid swings +-1..7 Hz on exactly the low-record-energy
    // sats, capping GPS coherence at 0.125-0.25 s while E/C's 4/10 ms records converged).
    //   Stage 1 -- slip-free coarse frequency: lag-L delay-and-multiply, ONE arg() of a
    //   weighted complex sum (no accumulation, a fade only down-weights its pairs).
    //   Stage 2 -- demodulate by the coarse frequency, then the weighted (|A|^2) LSQ of the
    //   sequentially-unwrapped phase vs CAPTURE-UTC (uneven gaps contribute their true span).
    //   Post-demod the true slope is ~0, so steps sit deep inside +-pi and cannot slip.
    const double mult = _carrier_pilot ? 1.0 : 2.0; // squared phase advances at mult * the carrier
    // Stage 1: coarse f over pairs L records apart. L=4 trades variance against unambiguous
    // range: +-1/(2*L*dt*mult) (+-62 Hz at 1 ms data records) covers any post-seed residual.
    const int L = 4;
    cd pair_sum(0.0, 0.0);
    double dt_sum = 0.0;
    int n_pair = 0;
    for (int r = L; r < n; ++r) {
        const double w = std::norm(a[r]) * std::norm(a[r - L]);
        if (!(w > 0.0))
            continue;
        const cd vr = _carrier_pilot ? a[r] : a[r] * a[r];
        const cd vp = _carrier_pilot ? a[r - L] : a[r - L] * a[r - L];
        pair_sum += w * vr * std::conj(vp);
        dt_sum += utc[r] - utc[r - L];
        ++n_pair;
    }
    double f_coarse = 0.0;
    if (n_pair > 0 && std::abs(pair_sum) > 0.0 && dt_sum > 0.0)
        f_coarse = std::arg(pair_sum) / (2.0 * M_PI * (dt_sum / n_pair) * mult);
    // Stage 2: LSQ slope of the unwrapped DEMODULATED phase.
    double phi = 0.0, raw_prev = 0.0;
    bool have = false;
    double Sw = 0, Swt = 0, Swtt = 0, Swp = 0, Swtp = 0;
    const double t0 = utc[0];
    for (int r = 0; r < n; ++r) {
        const double w = std::norm(a[r]); // |A|^2
        if (!(w > 0.0))
            continue;
        const cd v = (_carrier_pilot ? a[r] : a[r] * a[r])
                     * std::polar(1.0, -2.0 * M_PI * f_coarse * mult * (utc[r] - t0));
        const double cur = std::arg(v); // (-pi, pi]
        if (have) {
            double d = cur - raw_prev;
            while (d > M_PI)
                d -= 2.0 * M_PI;
            while (d < -M_PI)
                d += 2.0 * M_PI;
            phi += d;
        } else {
            phi = cur;
            have = true;
        }
        raw_prev = cur;
        const double t = utc[r] - t0;
        Sw += w;
        Swt += w * t;
        Swtt += w * t * t;
        Swp += w * phi;
        Swtp += w * t * phi;
    }
    const double det = Sw * Swtt - Swt * Swt;
    if (!(std::fabs(det) > 0.0))
        return 0.0;
    const double slope = (Sw * Swtp - Swt * Swp) / det; // d(phi)/dt (rad/s)
    // Total residual = the slip-free coarse stage PLUS the fine LSQ slope of what it left.
    return f_coarse + slope / mult / (2.0 * M_PI);      // -> residual carrier (Hz)
}
