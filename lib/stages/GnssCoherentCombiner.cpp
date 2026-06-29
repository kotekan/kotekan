#include "GnssCoherentCombiner.hpp"

#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE
#include "visUtil.hpp"      // for frameID

#include "json.hpp"   // for json
#include <algorithm> // for max
#include <cmath>      // for hypot, sqrt
#include <functional> // for bind

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
    if (_navwipe_bit_records > 0) {
        _navbuf.assign(_n_prn, {});
        _navutc.assign(_n_prn, {});
    }

    _st_prn.assign(_n_prn, 0);
    _st_amp.assign(_n_prn, 0.0f);
    _st_coh.assign(_n_prn, 0.0f);
    _st_deep.assign(_n_prn, 0.0f);
    _st_deep_snr.assign(_n_prn, 0.0f);
    _st_amp_snr.assign(_n_prn, 0.0f);
    _st_dop.assign(_n_prn, 0.0f);
    _st_cp.assign(_n_prn, 0.0f);
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
            const float* ref = ins[0] + (size_t)p * RECORD_FLOATS; // PRN/dop/cp/UTC reference
            for (float* in : ins) {
                const float* rec = in + (size_t)p * RECORD_FLOATS;
                gr += rec[3];
                gi += rec[4];
                energy += rec[5];
                nchan += rec[6];
            }
            const double ar = energy > 0.0 ? gr / energy : 0.0;
            const double ai = energy > 0.0 ? gi / energy : 0.0;

            // Accumulate over time: incoherent power (|A|^2) + its square (|A|^4, for the noise
            // debias) + coherent complex (A).
            const double utc_p = *reinterpret_cast<const double*>(ref + RECORD_UTC_SLOT);
            const double p2 = ar * ar + ai * ai;
            if (_rolling) { // exponential moving average (no reset; integrates indefinitely)
                acc_pow[p] += alpha * (p2 - acc_pow[p]);
                acc_pow2[p] += alpha * (p2 * p2 - acc_pow2[p]);
                acc_ar[p] += alpha * (ar - acc_ar[p]);
                acc_ai[p] += alpha * (ai - acc_ai[p]);
                acc_nchan[p] += alpha * (nchan - acc_nchan[p]);
                if (_navwipe_bit_records > 0) { // sliding window of the last _integration_length recs
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
                acc_ar[p] += ar;
                acc_ai[p] += ai;
                acc_nchan[p] += nchan;
                if (_navwipe_bit_records > 0) { // per-record (A, capture-UTC) for the nav-bit wipe
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
                // Normalise by the H0 std of the s^2 estimator (~N/K^{1/4}) so the significance is
                // ~1 for noise REGARDLESS of K (a stable lock threshold) and grows for real signal.
                amp_snr[p] = noise > 1e-12 ? s2 * std::pow(Keff, 0.25) / noise : 0.0;
            }
            if (_navwipe_bit_records > 0) {
                rec[8] = (float)navwipe_amplitude(_navbuf[p], _navutc[p], &deep_snr[p]); // deep |A|
                if (!_rolling) { // block clears the window; rolling slides it (capped above)
                    _navbuf[p].clear();
                    _navutc[p].clear();
                }
            }
            *reinterpret_cast<double*>(rec + RECORD_UTC_SLOT) = ref_utc[p];
            if (!_rolling)
                acc_pow[p] = acc_pow2[p] = acc_ar[p] = acc_ai[p] = acc_nchan[p] = 0.0; // block: reset
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
                _st_amp_snr[p] = (float)amp_snr[p];
                _st_dop[p] = rec[1];
                _st_cp[p] = rec[2];
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
                         {"doppler_hz", _st_dop[p]},
                         {"code_phase_chips", _st_cp[p]}});
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
