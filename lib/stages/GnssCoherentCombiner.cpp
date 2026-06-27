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
    _navwipe_bit_records = config.get_default<int>(unique_name, "navwipe_bit_records", 0);
    if (_navwipe_bit_records > 0)
        _navbuf.assign(_n_prn, {});

    _st_prn.assign(_n_prn, 0);
    _st_amp.assign(_n_prn, 0.0f);
    _st_deep.assign(_n_prn, 0.0f);
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

    // Per-PRN integration accumulators over _integration_length records.
    std::vector<double> acc_pow(_n_prn), acc_ar(_n_prn), acc_ai(_n_prn), acc_nchan(_n_prn);
    std::vector<float> ref_prn(_n_prn), ref_dop(_n_prn), ref_cp(_n_prn);
    std::vector<double> ref_utc(_n_prn);
    int n_acc = 0;

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

            // Accumulate over time: incoherent power (|A|^2) + coherent complex (A).
            acc_pow[p] += ar * ar + ai * ai;
            acc_ar[p] += ar;
            acc_ai[p] += ai;
            acc_nchan[p] += nchan;
            if (_navwipe_bit_records > 0)
                _navbuf[p].emplace_back(ar, ai); // per-record A for the nav-bit wipe
            if (n_acc == 0) { // window reference from the first record of the block
                ref_prn[p] = ref[0];
                ref_dop[p] = ref[1];
                ref_cp[p] = ref[2];
                ref_utc[p] = *reinterpret_cast<const double*>(ref + RECORD_UTC_SLOT);
            }
        }

        for (size_t i = 0; i < in_bufs.size(); ++i)
            in_bufs[i]->mark_frame_empty(unique_name, in_ids[i]++);

        if (++n_acc < _integration_length)
            continue; // keep accumulating

        float* out = (float*)out_buf->wait_for_empty_frame(unique_name, out_id);
        if (out == nullptr)
            break;
        const double inv = 1.0 / (double)n_acc;
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
            if (_navwipe_bit_records > 0) {
                rec[8] = (float)navwipe_amplitude(_navbuf[p]);       // deep |A| past the nav bit
                _navbuf[p].clear();
            }
            *reinterpret_cast<double*>(rec + RECORD_UTC_SLOT) = ref_utc[p];
            acc_pow[p] = acc_ar[p] = acc_ai[p] = acc_nchan[p] = 0.0; // reset
        }
        n_acc = 0;

        // Publish the latest full-band amplitudes for the broker's drop decisions.
        {
            std::lock_guard<std::mutex> lk(_st_mtx);
            for (int p = 0; p < _n_prn; ++p) {
                const float* rec = out + (size_t)p * RECORD_FLOATS;
                _st_prn[p] = (int)std::lround(rec[0]);
                _st_amp[p] = rec[3];
                _st_deep[p] = rec[8];
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
                         {"deep_amplitude", _st_deep[p]},
                         {"doppler_hz", _st_dop[p]},
                         {"code_phase_chips", _st_cp[p]}});
    conn.send_json_reply(reply);
}

double
GnssCoherentCombiner::navwipe_amplitude(const std::vector<std::complex<double>>& a) const {
    const int br = _navwipe_bit_records;
    const int nrec = (int)a.size();
    if (br <= 0 || nrec < 2 * br)
        return 0.0;
    using cd = std::complex<double>;
    // Bit sync: the +-1 data bit edge falls on a code-period (= record) boundary; pick the
    // record offset (0..br-1) that maximises the mean per-bit coherent power.
    int best_off = 0;
    double best_g = -1.0;
    for (int off = 0; off < br; ++off) {
        const int nb = (nrec - off) / br;
        if (nb < 2)
            continue;
        double g = 0.0;
        for (int b = 0; b < nb; ++b) {
            cd s(0.0, 0.0);
            for (int r = 0; r < br; ++r)
                s += a[off + b * br + r];
            g += std::abs(s);
        }
        if (g / nb > best_g) {
            best_g = g / nb;
            best_off = off;
        }
    }
    const int off = best_off;
    const int nb = (nrec - off) / br;
    // Per-bit coherent sums, then estimate the +-1 by squaring (theta0 = 1/2 arg sum s^2;
    // the global sign cancels in |.|), wipe, and coherently sum -> the deep coherent gain.
    std::vector<cd> s(nb);
    for (int b = 0; b < nb; ++b) {
        cd acc(0.0, 0.0);
        for (int r = 0; r < br; ++r)
            acc += a[off + b * br + r];
        s[b] = acc;
    }
    cd sumsq(0.0, 0.0);
    for (const cd& v : s)
        sumsq += v * v;
    const cd rot = std::polar(1.0, -0.5 * std::arg(sumsq));
    cd deep(0.0, 0.0);
    for (const cd& v : s)
        deep += (std::real(v * rot) >= 0.0 ? 1.0 : -1.0) * v;
    return std::abs(deep) / (double)(nb * br); // coherent mean of the wiped per-record A
}
