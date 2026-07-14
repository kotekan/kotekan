#include "GnssBeamCube.hpp"

#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE
#include "visUtil.hpp"      // for frameID

#include <algorithm>  // for max
#include <cmath>      // for hypot, sqrt, pow
#include <functional> // for bind

using kotekan::Config;
using kotekan::bufferContainer;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssBeamCube);

GnssBeamCube::GnssBeamCube(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&GnssBeamCube::main_thread, this)) {
    in_bufs = get_buffer_array("in_bufs");
    for (Buffer* b : in_bufs)
        b->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    _n_chan = (int)in_bufs.size();
    _n_prn = config.get_default<int>(unique_name, "n_prn", 0);
    if (_n_prn <= 0 && !in_bufs.empty())
        _n_prn = in_bufs[0]->frame_size / (int)sizeof(float) / IN_RECORD_FLOATS;
    _channel0 = config.get_default<int>(unique_name, "channel0", 0);

    _out_record_floats = OUT_HEADER + _n_chan + 2; // + UTC float64
    _out_utc_slot = OUT_HEADER + _n_chan;

    _integration_length = std::max(1, config.get_default<int>(unique_name, "integration_length", 1));
    const std::string mode = config.get_default<std::string>(unique_name, "integration_mode", "block");
    _rolling = (mode == "rolling");
    _emit_every = std::max(1, config.get_default<int>(unique_name, "output_every",
                                                      std::max(1, _integration_length / 10)));
}

void GnssBeamCube::main_thread() {
    std::vector<frameID> in_ids;
    for (Buffer* b : in_bufs)
        in_ids.emplace_back(b);
    frameID out_id(out_buf);

    // Per-PRN per-channel accumulators. block: running SUM of |A_c|^2 over the block. rolling:
    // EMA of |A_c|^2 (alpha = 1/_integration_length), bias-corrected at emit by /(1-(1-a)^n_roll).
    std::vector<std::vector<double>> acc(_n_prn, std::vector<double>(_n_chan, 0.0));
    std::vector<float> ref_prn(_n_prn), ref_dop(_n_prn), ref_cp(_n_prn);
    std::vector<double> ref_utc(_n_prn);
    int n_acc = 0;
    long long n_roll = 0;
    int since_emit = 0;
    const double alpha = 1.0 / (double)_integration_length;

    while (!stop_thread) {
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
            const float* ref = ins[0] + (size_t)p * IN_RECORD_FLOATS; // PRN/dop/cp/UTC reference
            const double utc_p = *reinterpret_cast<const double*>(ref + IN_UTC_SLOT);
            for (int c = 0; c < _n_chan; ++c) {
                const float* rec = ins[c] + (size_t)p * IN_RECORD_FLOATS;
                const double gr = rec[3], gi = rec[4], e = rec[5];
                const double a = e > 0.0 ? std::hypot(gr, gi) / e : 0.0; // |A_c| = |G_c|/E_c
                const double a2 = a * a;
                if (_rolling)
                    acc[p][c] += alpha * (a2 - acc[p][c]); // EMA of |A_c|^2
                else
                    acc[p][c] += a2; // running sum
            }
            if (_rolling || n_acc == 0) { // rolling: latest; block: first record of the block
                ref_prn[p] = ref[0];
                ref_dop[p] = ref[1];
                ref_cp[p] = ref[2];
                ref_utc[p] = utc_p;
            }
        }

        for (size_t i = 0; i < in_bufs.size(); ++i)
            in_bufs[i]->mark_frame_empty(unique_name, in_ids[i]++);

        if (_rolling) {
            ++n_roll;
            if (++since_emit < _emit_every)
                continue;
            since_emit = 0;
        } else if (++n_acc < _integration_length) {
            continue;
        }

        float* out = (float*)out_buf->wait_for_empty_frame(unique_name, out_id);
        if (out == nullptr)
            break;
        // block: mean = sum/n_acc. rolling: EMA is already a mean; undo the warm-up bias.
        const double inv = _rolling ? 1.0 / (1.0 - std::pow(1.0 - alpha, (double)n_roll))
                                    : 1.0 / (double)n_acc;
        for (int p = 0; p < _n_prn; ++p) {
            float* rec = out + (size_t)p * _out_record_floats;
            rec[0] = ref_prn[p];
            rec[1] = ref_dop[p];
            rec[2] = ref_cp[p];
            rec[3] = (float)_channel0;
            for (int c = 0; c < _n_chan; ++c)
                rec[OUT_HEADER + c] = (float)std::sqrt(acc[p][c] * inv); // sqrt<|A_c|^2> per bin
            *reinterpret_cast<double*>(rec + _out_utc_slot) = ref_utc[p];
            if (!_rolling)
                for (int c = 0; c < _n_chan; ++c)
                    acc[p][c] = 0.0; // block: reset
        }
        if (!_rolling)
            n_acc = 0;
        out_buf->mark_frame_full(unique_name, out_id++);
    }
}
