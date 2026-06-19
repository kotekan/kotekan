#include "GnssChannelizedCorrelator.hpp"

#include "StageFactory.hpp"           // for REGISTER_KOTEKAN_STAGE
#include "gnssBandPlan.hpp"           // for ChannelBand, covering_channels
#include "gnssChannelizedAcquire.hpp" // for channelized_acquire
#include "gnssChannelizedDespread.hpp" // for channelized_despread
#include "gpsCACode.hpp"              // for generate_ca_code
#include "fftwPlannerLock.hpp"        // for fftw_planner_mutex
#include "kotekanLogging.hpp"         // for FATAL_ERROR

#include <chrono>   // for system_clock
#include <cmath>    // for cos, sin, floor, M_PI
#include <cstring>  // for memcpy
#include <set>      // for set

using kotekan::Config;
using kotekan::bufferContainer;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssChannelizedCorrelator);

GnssChannelizedCorrelator::GnssChannelizedCorrelator(Config& config, const std::string& unique_name,
                                                     bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssChannelizedCorrelator::main_thread, this)),
    _fold(nullptr), _spec(nullptr), _p_fwd(nullptr) {

    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    const std::string signal = config.get_default<std::string>(unique_name, "signal", "GPS_L1CA");
    const gnss::SignalDescriptor* sig = gnss::signal_by_name(signal);
    if (sig == nullptr) {
        FATAL_ERROR("GnssChannelizedCorrelator: unknown signal '{:s}'", signal);
        return;
    }
    _sig = *sig;

    _sample_rate = config.get_default<double>(unique_name, "sample_rate", 5e6);
    _f_offset = config.get_default<double>(unique_name, "f_offset", 0.0);
    _capture_utc0 = config.get_default<double>(unique_name, "capture_utc0", 0.0);
    _doppler_margin_hz = config.get_default<double>(unique_name, "doppler_margin_hz", 5000.0);

    _N = config.get<int>(unique_name, "spectrum_length");
    _num_taps = config.get_default<int>(unique_name, "num_taps", 4);
    const std::string win = config.get_default<std::string>(unique_name, "pfb_window", "hamming");
    try {
        _window_type = dsp::window_from_string(win);
        _proto = dsp::pfb_prototype(_N, _num_taps, _window_type);
    } catch (const std::exception& e) {
        FATAL_ERROR("GnssChannelizedCorrelator: {:s}", e.what());
        return;
    }

    _prns = config.get<std::vector<int>>(unique_name, "prns");
    if (_prns.empty()) {
        FATAL_ERROR("GnssChannelizedCorrelator: 'prns' is empty");
        return;
    }
    const int n_prn = (int)_prns.size();

    // Seeded Doppler / code phase per PRN (a scalar broadcasts to all PRNs).
    auto seed = [&](const std::string& key, std::vector<double>& out) {
        const auto v = config.get_default<std::vector<double>>(unique_name, key, {});
        out.assign(n_prn, 0.0);
        for (int i = 0; i < n_prn; ++i)
            out[i] = v.empty() ? 0.0 : (v.size() == 1 ? v[0] : v.at(i));
    };
    seed("doppler_hz", _doppler);
    seed("code_phase_chips", _code_phase);

    // One code period of measurement window unless overridden.
    const int period_hops = (int)std::llround(_sig.code_period_s * _sample_rate) / _N;
    _hops_per_record = config.get_default<int>(unique_name, "hops_per_record", period_hops);

    // Self-seeding acquisition.
    _acquire = config.get_default<bool>(unique_name, "acquire", false);
    _acquire_snr = config.get_default<double>(unique_name, "acquire_snr", 12.0);
    const double dmin = config.get_default<double>(unique_name, "doppler_min", -6000.0);
    const double dmax = config.get_default<double>(unique_name, "doppler_max", 6000.0);
    const double dstep = config.get_default<double>(unique_name, "doppler_step", 500.0);
    for (double f = dmin; f <= dmax + 1e-6; f += dstep)
        _doppler_grid.push_back(f);
    _locked.assign(n_prn, _acquire ? 0 : 1); // without acquire, configured seeds are "locked"

    // Cache the spreading code per PRN.
    _full_code.resize(n_prn);
    for (int p = 0; p < n_prn; ++p) {
        auto a = gps::generate_ca_code(_prns[p]); // L1 C/A; other signals: future work
        _full_code[p].assign(a.begin(), a.end());
    }

    const auto active_prns = config.get_default<std::vector<int>>(unique_name, "active_prns", {});
    _active.assign(n_prn, 1);
    if (!active_prns.empty()) {
        const std::set<int> on(active_prns.begin(), active_prns.end());
        for (int i = 0; i < n_prn; ++i)
            _active[i] = on.count(_prns[i]) ? 1 : 0;
    }

    std::lock_guard<std::mutex> planner_lock(fftw_planner_mutex());
    _fold = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * _N);
    _spec = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * _N);
    _p_fwd = fftwf_plan_dft_1d(_N, _fold, _spec, -1, FFTW_ESTIMATE);
}

GnssChannelizedCorrelator::~GnssChannelizedCorrelator() {
    if (_p_fwd) {
        std::lock_guard<std::mutex> planner_lock(fftw_planner_mutex());
        fftwf_destroy_plan(_p_fwd);
    }
    if (_fold)
        fftwf_free(_fold);
    if (_spec)
        fftwf_free(_spec);
}

int8_t GnssChannelizedCorrelator::code_chip(int p, double chip_phase) const {
    const long len = _sig.code_length;
    long idx = (long)std::floor(chip_phase);
    idx %= len;
    if (idx < 0)
        idx += len;
    return _full_code[p][idx];
}

std::vector<int> GnssChannelizedCorrelator::covering_bins(int p, double doppler_hz) const {
    (void)p; // covering set depends on the carrier + Doppler, not the PRN
    // Local fftshifted bin grid: bin j is centred at (j - N/2) * Fs/N relative
    // to band centre; the carrier sits at f_offset there.
    std::vector<gnss::ChannelBand> chans(_N);
    const double bin_w = _sample_rate / _N;
    for (int j = 0; j < _N; ++j)
        chans[j] = {(gnss::freq_id_t)j, (j - _N / 2) * bin_w, bin_w};

    gnss::SignalDescriptor local = _sig;
    local.carrier_hz = _f_offset;
    const double max_dop = std::abs(doppler_hz) + _doppler_margin_hz;

    const auto ids = gnss::covering_channels(chans, local, max_dop);
    return std::vector<int>(ids.begin(), ids.end());
}

std::vector<std::vector<std::complex<float>>>
GnssChannelizedCorrelator::replica_channels(int p, long long window_start_sample,
                                            double code_phase_chips, double doppler_hz) {
    const int warm = _num_taps - 1; // hops to prime the (continuous) F-engine delay line
    const int total = _hops_per_record + warm;
    const long long start = window_start_sample - (long long)warm * _N;

    _replica_hist.assign((size_t)_N * _num_taps, std::complex<float>(0.0f, 0.0f));
    std::vector<std::complex<float>> block(_N);
    std::vector<std::vector<std::complex<float>>> chan(_N,
        std::vector<std::complex<float>>(_hops_per_record));

    const double chip_per_sample = _sig.chip_rate_hz / _sample_rate;
    const double wcarrier = 2.0 * M_PI * (_f_offset + doppler_hz) / _sample_rate;

    for (int h = 0; h < total; ++h) {
        for (int i = 0; i < _N; ++i) {
            const long long n = start + (long long)h * _N + i;
            if (n < 0) {
                block[i] = std::complex<float>(0.0f, 0.0f); // pre-stream: matches F-engine zero init
                continue;
            }
            const int8_t c = code_chip(p, code_phase_chips + (double)n * chip_per_sample);
            const double ph = wcarrier * (double)n;
            block[i] = std::complex<float>(c * (float)std::cos(ph), c * (float)std::sin(ph));
        }
        dsp::pfb_push(_replica_hist.data(), block.data(), _N, _num_taps);
        dsp::pfb_fold(_replica_hist.data(), _proto.data(), _N, _num_taps,
                      reinterpret_cast<std::complex<float>*>(_fold));
        fftwf_execute(_p_fwd);
        if (h < warm)
            continue;
        const int m = h - warm;
        // fftshift to match fftwEngine: out[j] = spec[(j + N/2) mod N].
        auto* spec = reinterpret_cast<std::complex<float>*>(_spec);
        for (int j = 0; j < _N; ++j)
            chan[j][m] = spec[(j + _N / 2) % _N];
    }
    return chan;
}

gnss::DespreadResult
GnssChannelizedCorrelator::despread_at(int p, long long window_start_sample, double code_phase_chips,
                                       double doppler_hz, const std::vector<int>& bins,
                                       const std::vector<std::complex<float>>& window) {
    const auto repl = replica_channels(p, window_start_sample, code_phase_chips, doppler_hz);
    std::vector<std::vector<std::complex<float>>> data_ch, repl_ch;
    data_ch.reserve(bins.size());
    repl_ch.reserve(bins.size());
    for (int b : bins) {
        std::vector<std::complex<float>> col(_hops_per_record);
        for (int m = 0; m < _hops_per_record; ++m)
            col[m] = window[(size_t)m * _N + b];
        data_ch.push_back(std::move(col));
        repl_ch.push_back(repl[b]);
    }
    return gnss::channelized_despread(data_ch, repl_ch);
}

void GnssChannelizedCorrelator::main_thread() {
    frame_in = 0;
    frame_out = 0;

    using namespace std::placeholders;
    kotekan::restServer& rest = kotekan::restServer::instance();
    rest.register_get_callback(unique_name + "/get_mask",
                               std::bind(&GnssChannelizedCorrelator::get_mask_callback, this, _1));
    rest.register_post_callback(
        unique_name + "/set_mask",
        std::bind(&GnssChannelizedCorrelator::set_mask_callback, this, _1, _2));

    const int n_prn = (int)_prns.size();
    // Rolling measurement window of [hop][channel] voltages.
    std::vector<std::complex<float>> window((size_t)_hops_per_record * _N);
    int hops_filled = 0;
    long long window_index = 0;
    int in_pos_hops = 0; // hop cursor within the current input frame

    while (!stop_thread) {
        auto* in_local =
            (std::complex<float>*)in_buf->wait_for_full_frame(unique_name, frame_in);
        if (in_local == nullptr)
            break;
        const int frame_hops = in_buf->frame_size / (int)sizeof(fftwf_complex) / _N;

        for (in_pos_hops = 0; in_pos_hops < frame_hops; ++in_pos_hops) {
            std::memcpy(&window[(size_t)hops_filled * _N], &in_local[(size_t)in_pos_hops * _N],
                        sizeof(std::complex<float>) * _N);
            if (++hops_filled < _hops_per_record)
                continue;
            hops_filled = 0;

            // --- full window: measure each PRN ---
            const long long window_start = window_index * (long long)_hops_per_record * _N;
            std::vector<uint8_t> active;
            {
                std::lock_guard<std::mutex> lk(_mask_mutex);
                active = _active;
            }

            float* out = (float*)out_buf->wait_for_empty_frame(unique_name, frame_out);
            if (out == nullptr)
                return;
            const double utc =
                (_capture_utc0 > 0.0)
                    ? _capture_utc0 + (double)window_start / _sample_rate
                    : std::chrono::duration<double>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();

            for (int p = 0; p < n_prn; ++p) {
                float* rec = out + (size_t)p * RECORD_FLOATS;
                rec[0] = (float)_prns[p];
                for (int f = 1; f < RECORD_UTC_SLOT; ++f)
                    rec[f] = 0.0f;
                *reinterpret_cast<double*>(rec + RECORD_UTC_SLOT) = utc;
                if (!active[p])
                    continue;

                // Self-seed: acquire code phase + Doppler from the channels.
                if (_acquire && !_locked[p]) {
                    std::vector<std::vector<std::complex<float>>> dch(
                        _N, std::vector<std::complex<float>>(_hops_per_record));
                    for (int c = 0; c < _N; ++c)
                        for (int m = 0; m < _hops_per_record; ++m)
                            dch[c][m] = window[(size_t)m * _N + c];

                    const double dmax = _doppler_grid.empty() ? 0.0 : _doppler_grid.back();
                    const auto acov = covering_bins(p, std::abs(dmax));
                    const auto repl0 = replica_channels(p, window_start, 0.0, 0.0);
                    std::vector<int> cfreq;
                    cfreq.reserve(acov.size());
                    for (int b : acov)
                        cfreq.push_back(b - _N / 2); // fftshifted bank: channel <-> freq (b - N/2)

                    const auto a = gnss::channelized_acquire(
                        dch, repl0, acov, _doppler_grid, _sample_rate, _sig.chip_rate_hz, _N,
                        _sig.code_length, cfreq);
                    if (a.snr >= _acquire_snr) {
                        // Acquisition localizes to ~1 chip; refine the code phase
                        // to sample resolution with exact despreads (the
                        // acquisition->tracking handoff).
                        const auto rbins = covering_bins(p, a.doppler_hz);
                        const double cps = _sig.chip_rate_hz / _sample_rate;
                        const int span = 2 * (int)std::lround(_sample_rate / _sig.chip_rate_hz);
                        double best_cp = a.code_phase_chips, best_pw = -1.0;
                        for (int off = -span; off <= span; ++off) {
                            const double cp = a.code_phase_chips + off * cps;
                            const double pw = std::norm(
                                despread_at(p, window_start, cp, a.doppler_hz, rbins, window)
                                    .amplitude);
                            if (pw > best_pw) {
                                best_pw = pw;
                                best_cp = cp;
                            }
                        }
                        _code_phase[p] =
                            std::fmod(best_cp, (double)_sig.code_length) + (best_cp < 0 ? _sig.code_length : 0);
                        _doppler[p] = a.doppler_hz;
                        _locked[p] = 1;
                    }
                }

                const auto bins = covering_bins(p, _doppler[p]);
                const auto res = despread_at(p, window_start, _code_phase[p], _doppler[p], bins,
                                             window);
                rec[1] = (float)_doppler[p];
                rec[2] = (float)_code_phase[p];
                rec[3] = (float)std::abs(res.amplitude);
                rec[4] = (float)res.amplitude.real();
                rec[5] = (float)res.amplitude.imag();
                rec[6] = (float)std::abs(res.amplitude); // SNR proxy (amplitude magnitude)
            }

            out_buf->mark_frame_full(unique_name, frame_out);
            frame_out = (frame_out + 1) % out_buf->num_frames;
            window_index++;
        }

        in_buf->mark_frame_empty(unique_name, frame_in);
        frame_in = (frame_in + 1) % in_buf->num_frames;
    }
}

void GnssChannelizedCorrelator::get_mask_callback(kotekan::connectionInstance& conn) {
    nlohmann::json reply;
    std::lock_guard<std::mutex> lk(_mask_mutex);
    reply["prns"] = _prns;
    std::vector<bool> act(_active.begin(), _active.end());
    reply["active"] = act;
    conn.send_json_reply(reply);
}

void GnssChannelizedCorrelator::set_mask_callback(kotekan::connectionInstance& conn,
                                                  nlohmann::json& request) {
    try {
        std::lock_guard<std::mutex> lk(_mask_mutex);
        if (request.count("active_prns")) {
            const auto list = request["active_prns"].get<std::vector<int>>();
            const std::set<int> on(list.begin(), list.end());
            for (size_t i = 0; i < _prns.size(); ++i)
                _active[i] = on.count(_prns[i]) ? 1 : 0;
        } else if (request.count("prn")) {
            const int prn = request["prn"];
            const bool on = request.value("active", true);
            for (size_t i = 0; i < _prns.size(); ++i)
                if (_prns[i] == prn)
                    _active[i] = on ? 1 : 0;
        } else {
            conn.send_error("expected 'active_prns' list or 'prn'+'active'",
                            kotekan::HTTP_RESPONSE::BAD_REQUEST);
            return;
        }
    } catch (const std::exception& e) {
        conn.send_error(e.what(), kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    get_mask_callback(conn);
}
