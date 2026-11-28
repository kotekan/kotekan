#include "GnssCskDespread.hpp"

#include "GnssChanMetadata.hpp" // for get_gnss_chan_metadata, metadata_is_gnss_chan
#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE
#include "dspPfbPrototype.hpp"  // for window_from_string
#include "gnssRecord.hpp"       // for record_stride, REC_*, RECORD_UTC_SLOT
#include "gnssSignal.hpp"       // for signal_by_name
#include "guowang1202Code.hpp"  // for generate_gw1202_code, GW1202_*
#include "kotekanLogging.hpp"   // for FATAL_ERROR, INFO
#include "visUtil.hpp"          // for frameID

#include <algorithm>  // for sort, nth_element, min, max
#include <chrono>     // for steady_clock
#include <cmath>      // for floor, ceil, llround, log10, norm
#include <cstring>    // for memcpy
#include <functional> // for bind

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using cf = std::complex<float>;

REGISTER_KOTEKAN_STAGE(GnssCskDespread);

namespace {
constexpr int CODE = guowang::GW1202_CODE_LENGTH;   // 2046 chips, the data half
constexpr int FRAME = guowang::GW1202_FRAME_LENGTH; // 4092 chips, 2 ms
constexpr int N_SHIFT = 64;
constexpr double FRAME_S = 2e-3;
constexpr double PW_ALPHA = 0.02; // the prompt-power average's weight per record (~50 records)

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
} // namespace

GnssCskDespread::GnssCskDespread(Config& config, const std::string& unique_name,
                                 bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&GnssCskDespread::main_thread, this)) {
    _rec_buf = get_buffer("rec_buf");
    _rec_buf->register_consumer(unique_name);
    _chan_buf = get_buffer("chan_buf");
    _chan_buf->register_consumer(unique_name);
    if (!metadata_is_gnss_chan(_rec_buf) || !metadata_is_gnss_chan(_chan_buf))
        FATAL_ERROR("GnssCskDespread[{:s}]: rec_buf and chan_buf need a GnssChanMetadata pool",
                    unique_name);

    const std::string sig_name = config.get<std::string>(unique_name, "signal");
    const gnss::SignalDescriptor* sig = gnss::signal_by_name(sig_name);
    if (!sig || sig_name != "GW_1202")
        FATAL_ERROR("GnssCskDespread[{:s}]: signal must be GW_1202, not '{:s}'", unique_name,
                    sig_name);
    _carrier = sig->carrier_hz;
    const std::vector<int> prns = config.get<std::vector<int>>(unique_name, "prns");
    _n_prn = (int)prns.size();
    _n_elem = config.get_default<int>(unique_name, "n_elements", 0);
    _fft_len = config.get<int>(unique_name, "fft_length");
    _fs = config.get<double>(unique_name, "sample_rate");
    _rec_samples = (long long)config.get<int>(unique_name, "hops_per_record") * _fft_len;
    _shift_sign = config.get_default<int>(unique_name, "shift_sign", 1);
    _min_snr = config.get_default<double>(unique_name, "min_snr_db", 6.0);
    _n_workers = std::max(1, config.get_default<int>(unique_name, "n_workers", 2));
    _keep = std::max(1, config.get_default<int>(unique_name, "symbol_export", 600));
    _max_queue = 256; // records a worker may fall behind (256 ms) before it drops the oldest
    const int ring_frames = std::max(4, config.get_default<int>(unique_name, "ring_frames", 16));
    if (_shift_sign != 1 && _shift_sign != -1)
        FATAL_ERROR("GnssCskDespread[{:s}]: shift_sign must be +1 or -1", unique_name);

    const std::vector<int> col_ids = config.get<std::vector<int>>(unique_name, "channel_ids");
    _n_cols = (int)col_ids.size();
    _want = config.get<std::vector<int>>(unique_name, "csk_channels");
    for (int ch : _want) {
        const auto it = std::find(col_ids.begin(), col_ids.end(), ch);
        if (it == col_ids.end())
            FATAL_ERROR("GnssCskDespread[{:s}]: csk channel {:d} is not in channel_ids",
                        unique_name, ch);
        _cols.push_back((int)(it - col_ids.begin()));
    }

    const std::string win = config.get_default<std::string>(unique_name, "pfb_window", "hamming");
    const bool iq = config.get_default<std::string>(unique_name, "input_type", "real") != "real";
    try {
        const gnss::Channelization chz{
            _fs,
            config.get<int>(unique_name, "spectrum_length"),
            config.get_default<int>(unique_name, "num_taps", 4),
            dsp::window_from_string(win),
            iq,
            config.get_default<bool>(unique_name, "pfb_time_order", false)};
        _bank = std::make_unique<gnss::ChannelizedReplicaBank>(
            *sig, chz, config.get<double>(unique_name, "f_offset"), prns,
            config.get_default<std::vector<int>>(unique_name, "bds3s_prns", {}));
    } catch (const std::exception& e) {
        FATAL_ERROR("GnssCskDespread[{:s}]: {:s}", unique_name, e.what());
        return;
    }
    if (_bank->comb_mult() != 1 || _bank->fft_len() != _fft_len)
        FATAL_ERROR("GnssCskDespread[{:s}]: the bank's hop ({:d}) or comb ({:d}) does not match",
                    unique_name, _bank->fft_len(), _bank->comb_mult());

    if (config.exists(unique_name, "csk_g2")) {
        for (const auto& pg : config.get<std::vector<std::vector<int>>>(unique_name, "csk_g2")) {
            if (pg.size() != 2 || pg[1] < 0 || pg[1] >= guowang::GW1202_G2P_PERIOD)
                FATAL_ERROR("GnssCskDespread[{:s}]: csk_g2 entries are [prn, g2_phase 0..2046]",
                            unique_name);
            // Two periods, so that each shift reads one contiguous run of chips.
            const auto code = guowang::generate_gw1202_code(pg[1]);
            std::vector<float>& c = _csk[pg[0]];
            for (int rep = 0; rep < 2; ++rep)
                for (int8_t v : code)
                    c.push_back((float)v);
            _csk_g2[pg[0]] = pg[1];
        }
    }

    const size_t need = (size_t)_n_prn * gnss::record_stride(_n_elem) * sizeof(float);
    if ((size_t)_rec_buf->frame_size < need)
        FATAL_ERROR("GnssCskDespread[{:s}]: record frames under {:d} B", unique_name, need);

    _ring.resize((size_t)ring_frames);
    _slots.resize((size_t)_n_prn);
    _pw_avg.assign((size_t)_n_prn, 0.0);
    _pw_prn.assign((size_t)_n_prn, 0);
    _queues.resize((size_t)_n_workers);
    INFO("GnssCskDespread[{:s}]: {:d} slots, channels {:d}..{:d}, {:d} CSK codes, {:d} workers",
         unique_name, _n_prn, _want.front(), _want.back(), (int)_csk.size(), _n_workers);

    kotekan::restServer::instance().register_get_callback(
        unique_name + "/get_status",
        std::bind(&GnssCskDespread::get_status_callback, this, std::placeholders::_1));
    kotekan::restServer::instance().register_post_callback(
        unique_name + "/capture", std::bind(&GnssCskDespread::capture_callback, this,
                                            std::placeholders::_1, std::placeholders::_2));
}

GnssCskDespread::~GnssCskDespread() {
    kotekan::restServer::instance().remove_get_callback(unique_name + "/get_status");
    kotekan::restServer::instance().remove_json_callback(unique_name + "/capture");
    for (Slot& s : _slots)
        if (s.cap)
            fclose(s.cap);
}

void GnssCskDespread::main_thread() {
    _chan_th = std::thread(&GnssCskDespread::chan_thread, this);
    for (int w = 0; w < _n_workers; ++w)
        _workers.emplace_back(&GnssCskDespread::worker, this, w);

    frameID rec_id(_rec_buf);
    while (!stop_thread) {
        const uint8_t* rec = _rec_buf->wait_for_full_frame(unique_name, rec_id);
        if (rec == nullptr)
            break;
        push_record((const float*)rec, get_gnss_chan_metadata(_rec_buf, rec_id)->sample_seq);
        _rec_buf->mark_frame_empty(unique_name, rec_id++);
    }

    {
        std::lock_guard<std::mutex> lk(_q_m);
        _q_cv.notify_all();
    }
    for (auto& t : _workers)
        t.join();
    _chan_th.join();
}

// Copy the despread channels of each chan_buf frame into the ring and release it at once.
void GnssCskDespread::chan_thread() {
    frameID id(_chan_buf);
    while (!stop_thread) {
        const cf* in = (const cf*)_chan_buf->wait_for_full_frame(unique_name, id);
        if (in == nullptr)
            break;
        const GnssChanMetadata* mi = get_gnss_chan_metadata(_chan_buf, id);
        const int n_hops = (int)(_chan_buf->frame_size / sizeof(cf) / (size_t)_n_cols);
        if (mi != nullptr && mi->sample_seq >= 0) {
            std::unique_lock<std::shared_mutex> lk(_ring_m);
            _ring_end = std::max<long long>(_ring_end, mi->sample_seq / _fft_len + n_hops);
            ChanFrame& f = _ring[_ring_next];
            _ring_next = (_ring_next + 1) % _ring.size();
            f.first_hop = mi->sample_seq / _fft_len;
            f.n_hops = n_hops;
            f.data.resize(_cols.size());
            for (size_t c = 0; c < _cols.size(); ++c) {
                f.data[c].resize((size_t)n_hops);
                const cf* src = in + _cols[c];
                for (int h = 0; h < n_hops; ++h)
                    f.data[c][(size_t)h] = src[(size_t)h * (size_t)_n_cols];
            }
        }
        _chan_buf->mark_frame_empty(unique_name, id++);
        std::lock_guard<std::mutex> lk(_new_m);
        _new_cv.notify_all();
    }
}

int GnssCskDespread::gather(long long h0, int n, std::vector<std::vector<cf>>& out) {
    out.resize(_cols.size());
    for (auto& row : out)
        row.resize((size_t)n);
    std::shared_lock<std::shared_mutex> lk(_ring_m);
    if (h0 + n > _ring_end)
        return 1;
    long long h = h0;
    while (h < h0 + n) {
        const ChanFrame* f = nullptr;
        for (const ChanFrame& r : _ring)
            if (r.first_hop >= 0 && h >= r.first_hop && h < r.first_hop + r.n_hops) {
                f = &r;
                break;
            }
        if (f == nullptr)
            return -1;
        const size_t src = (size_t)(h - f->first_hop), dst = (size_t)(h - h0);
        const int cnt = (int)(std::min<long long>(f->first_hop + f->n_hops, h0 + n) - h);
        for (size_t c = 0; c < _cols.size(); ++c)
            std::memcpy(&out[c][dst], &f->data[c][src], (size_t)cnt * sizeof(cf));
        h += cnt;
    }
    return 0;
}

// Keep each slot's prompt-power average, and hand the record's models to the workers.
void GnssCskDespread::push_record(const float* rec, long long wstart) {
    const int stride = gnss::record_stride(_n_elem);
    auto b = std::make_shared<RecordBatch>();
    b->wstart = wstart;
    b->prn.resize((size_t)_n_prn);
    b->dop.resize((size_t)_n_prn);
    b->arg.resize((size_t)_n_prn);
    b->utc.resize((size_t)_n_prn);
    b->active.assign((size_t)_n_prn, 0);
    std::vector<double> pw;
    for (int p = 0; p < _n_prn; ++p) {
        const float* r = rec + (size_t)p * stride;
        const int prn = (int)r[gnss::REC_PRN];
        b->prn[(size_t)p] = prn;
        b->dop[(size_t)p] = r[gnss::REC_DOPPLER];
        b->arg[(size_t)p] = r[gnss::REC_CP];
        double utc;
        std::memcpy(&utc, &r[gnss::RECORD_UTC_SLOT], sizeof(double));
        b->utc[(size_t)p] = utc;
        if (prn != _pw_prn[(size_t)p]) {
            _pw_prn[(size_t)p] = prn;
            _pw_avg[(size_t)p] = 0.0;
        }
        // A slot with a job has covering channels; its replica energy can be 0 when the record
        // falls in the CSK half, which then says nothing about the slot's power.
        const float e = r[gnss::REC_P_ENERGY];
        if (prn > 0 && r[gnss::REC_NCHAN] > 0.0f && e > 0.0f) {
            const double v = ((double)r[gnss::REC_P_RE] * r[gnss::REC_P_RE]
                              + (double)r[gnss::REC_P_IM] * r[gnss::REC_P_IM])
                             / (double)e;
            double& a = _pw_avg[(size_t)p];
            a = a > 0.0 ? a + PW_ALPHA * (v - a) : v;
        }
        if (_pw_avg[(size_t)p] > 0.0)
            pw.push_back(_pw_avg[(size_t)p]);
    }
    if (pw.size() < 3)
        return;
    std::nth_element(pw.begin(), pw.begin() + (long)pw.size() / 2, pw.end());
    const double bar = pw[pw.size() / 2] * std::pow(10.0, _min_snr / 10.0);
    bool any = false;
    for (int p = 0; p < _n_prn; ++p)
        if (b->prn[(size_t)p] > 0 && _pw_avg[(size_t)p] > bar) {
            b->active[(size_t)p] = 1;
            any = true;
        }
    if (!any)
        return;
    std::lock_guard<std::mutex> lk(_q_m);
    for (int w = 0; w < _n_workers; ++w) {
        auto& q = _queues[(size_t)w];
        if ((int)q.size() >= _max_queue) {
            q.pop_front();
            std::lock_guard<std::mutex> ok(_m);
            for (int p = w; p < _n_prn; p += _n_workers)
                if (b->active[(size_t)p])
                    _slots[(size_t)p].n_drop++;
        }
        q.push_back(b);
    }
    _q_cv.notify_all();
}

void GnssCskDespread::worker(int w) {
    std::vector<std::complex<double>> z((size_t)CODE);
    std::vector<std::vector<cf>> scratch;
    while (true) {
        std::shared_ptr<const RecordBatch> b;
        {
            std::unique_lock<std::mutex> lk(_q_m);
            _q_cv.wait(lk, [&] { return stop_thread || !_queues[(size_t)w].empty(); });
            if (stop_thread)
                return;
            b = _queues[(size_t)w].front();
            _queues[(size_t)w].pop_front();
        }
        for (int p = w; p < _n_prn; p += _n_workers) {
            Slot& s = _slots[(size_t)p];
            if (b->prn[(size_t)p] != s.prn) {
                std::lock_guard<std::mutex> lk(_m);
                s.prn = b->prn[(size_t)p];
                s.last_id = -1;
                s.filt_dop = 1e30;
                const auto it = _csk.find(s.prn);
                s.csk = it == _csk.end() ? nullptr : &it->second;
                s.sym.clear();
            }
            // A slot without a CSK code is despread only for a capture, which finds its code.
            if (b->active[(size_t)p] && (s.csk || s.cap))
                despread(s, *b, p, z, scratch);
        }
    }
}

// The data halves whose middle chip falls in this record, under this record's model.
void GnssCskDespread::despread(Slot& s, const RecordBatch& b, int p,
                               std::vector<std::complex<double>>& z,
                               std::vector<std::vector<cf>>& scratch) {
    const double dop = b.dop[(size_t)p];
    if (std::fabs(dop - s.filt_dop) > 25.0) {
        s.filt = _bank->hoprate_filter(_want, dop, -1);
        s.filt_dop = dop;
    }
    const long double cps = (long double)_bank->eff_chip_rate() / (long double)_fs
                            * (1.0L + (long double)_bank->code_doppler_sign * dop / _carrier);
    const long double arg = b.arg[(size_t)p];
    const long double c_lo = arg + (long double)b.wstart * cps;
    const long double c_hi = arg + (long double)(b.wstart + _rec_samples) * cps;
    const int lf = _bank->num_taps() * _fft_len;
    std::vector<float> zr((size_t)CODE), zi((size_t)CODE);
    const long double frame_samples = (long double)FRAME_S * (long double)_fs;

    for (long long F = (long long)std::floor((c_lo - (CODE + CODE / 2)) / FRAME);; ++F) {
        const long double mid = (long double)F * FRAME + CODE + CODE / 2;
        if (mid >= c_hi)
            break;
        if (mid < c_lo)
            continue;
        const long long chip_lo = F * FRAME + CODE;
        const long double n_s = ((long double)chip_lo - arg) / cps;
        const long double n_e = ((long double)(chip_lo + CODE) - arg) / cps;
        // Already despread under the previous record's model.
        if (s.last_id >= 0 && n_s < s.last_n + frame_samples / 2)
            continue;
        const long long h0 = (long long)std::floor(n_s / _fft_len) - 1;
        const int n_hops = (int)((long long)std::ceil((n_e + lf) / _fft_len) + 1 - h0);
        // A record can come before the chan_buf frame holding the end of its last data half.
        int got = gather(h0, n_hops, scratch);
        for (int tries = 0; got > 0 && tries < 10 && !stop_thread; ++tries) {
            std::unique_lock<std::mutex> lk(_new_m);
            _new_cv.wait_for(lk, std::chrono::milliseconds(10));
            lk.unlock();
            got = gather(h0, n_hops, scratch);
        }
        if (got != 0)
            continue;
        std::vector<const cf*> ptr;
        for (auto& row : scratch)
            ptr.push_back(row.data());
        std::fill(z.begin(), z.end(), std::complex<double>(0.0, 0.0));
        _bank->hoprate_adjoint(s.filt, h0 * _fft_len, (double)arg, dop, n_hops, ptr, chip_lo, z);

        long long id;
        if (s.last_id < 0 || n_s - s.last_n > (long double)_fs)
            id = std::llround(n_s / frame_samples);
        else
            id = s.last_id + std::llround((n_s - s.last_n) / frame_samples);
        s.last_id = id;
        s.last_n = n_s;
        const double utc = b.utc[(size_t)p] + (double)((n_s - (long double)b.wstart) / _fs);

        Symbol sym{id, {0, 0, 0, 0}, {0, 0, 0, 0}};
        if (s.csk) {
            std::array<double, N_SHIFT> pk;
            double total = 0.0;
            for (int j = 0; j < CODE; ++j) {
                zr[(size_t)j] = (float)z[(size_t)j].real();
                zi[(size_t)j] = (float)z[(size_t)j].imag();
            }
            for (int k = 0; k < N_SHIFT; ++k) {
                // shift k: chip j reads code chip j + shift_sign * k
                const float* c = s.csk->data() + ((_shift_sign * k) % CODE + CODE) % CODE;
                float ar = 0.0f, ai = 0.0f;
#pragma omp simd reduction(+ : ar, ai)
                for (int j = 0; j < CODE; ++j) {
                    ar += c[j] * zr[(size_t)j];
                    ai += c[j] * zi[(size_t)j];
                }
                pk[(size_t)k] = (double)ar * ar + (double)ai * ai;
                total += pk[(size_t)k];
            }
            std::array<int, N_SHIFT> order;
            for (int k = 0; k < N_SHIFT; ++k)
                order[(size_t)k] = k;
            std::partial_sort(order.begin(), order.begin() + 4, order.end(),
                              [&](int a, int bb) { return pk[(size_t)a] > pk[(size_t)bb]; });
            double top = 0.0;
            for (int i = 0; i < 4; ++i)
                top += pk[(size_t)order[(size_t)i]];
            const double noise = std::max((total - top) / (N_SHIFT - 4), 1e-30);
            for (int i = 0; i < 4; ++i) {
                sym.k[(size_t)i] = order[(size_t)i];
                sym.db[(size_t)i] =
                    (float)(10.0 * std::log10(pk[(size_t)order[(size_t)i]] / noise));
            }
        }

        std::lock_guard<std::mutex> lk(_m);
        if (s.csk) {
            s.sym.push_back(sym);
            while ((int)s.sym.size() > _keep)
                s.sym.pop_front();
            s.n_sym++;
            s.snr_db += 0.02 * (sym.db[0] - s.snr_db);
            s.utc_ref = utc;
            s.id_ref = id;
        }
        if (s.cap) {
            if (now_s() > s.cap_until) {
                fclose(s.cap);
                s.cap = nullptr;
                INFO("GnssCskDespread[{:s}]: capture of PRN {:d} done", unique_name, s.prn);
            } else {
                std::vector<std::complex<float>> zf(z.begin(), z.end());
                fwrite(&id, sizeof(id), 1, s.cap);
                fwrite(&utc, sizeof(utc), 1, s.cap);
                fwrite(&dop, sizeof(dop), 1, s.cap);
                fwrite(zf.data(), sizeof(std::complex<float>), zf.size(), s.cap);
            }
        }
    }
}

void GnssCskDespread::get_status_callback(kotekan::connectionInstance& conn) {
    // Copied under _m and serialised after it is released: the reply runs to megabytes, and
    // building it under the lock stalled the despread workers.
    struct Row {
        int prn;
        double snr_db, utc_ref;
        long long n_sym, n_drop, id_ref;
        std::vector<Symbol> sym;
    };
    std::vector<Row> snap;
    {
        std::lock_guard<std::mutex> lk(_m);
        for (const Slot& s : _slots)
            if (s.prn > 0 && !s.sym.empty())
                snap.push_back({s.prn, s.snr_db, s.utc_ref, s.n_sym, s.n_drop, s.id_ref,
                                std::vector<Symbol>(s.sym.begin(), s.sym.end())});
    }
    // A slot no longer despread keeps its last symbols: one whose newest is STATUS_STALE_S older
    // than the newest slot's is left out, so the reply carries only the slots with new data.
    constexpr double STATUS_STALE_S = 2.0;
    auto last_utc = [](const Row& r) {
        return r.utc_ref + (double)(r.sym.back().id - r.id_ref) * FRAME_S;
    };
    double newest = -1e300;
    for (const Row& r : snap)
        newest = std::max(newest, last_utc(r));
    nlohmann::json rows = nlohmann::json::array();
    for (const Row& r : snap) {
        if (last_utc(r) < newest - STATUS_STALE_S)
            continue;
        nlohmann::json sym = nlohmann::json::array();
        for (const Symbol& y : r.sym) {
            nlohmann::json row = {y.id};
            for (int i = 0; i < 4; ++i) {
                row.push_back(y.k[(size_t)i]);
                // rounded in double, so it serialises as 6.4 and not 6.400000095367432
                row.push_back(std::round((double)y.db[(size_t)i] * 10.0) / 10.0);
            }
            sym.push_back(row);
        }
        const auto g = _csk_g2.find(r.prn);
        rows.push_back({{"prn", r.prn},
                        {"csk_g2", g == _csk_g2.end() ? -1 : g->second},
                        {"snr_db", r.snr_db},
                        {"n_sym", r.n_sym},
                        {"n_drop", r.n_drop},
                        {"utc_ref", r.utc_ref},
                        {"id_ref", r.id_ref},
                        {"sym", sym}});
    }
    conn.send_json_reply(rows);
}

void GnssCskDespread::capture_callback(kotekan::connectionInstance& conn, nlohmann::json& body) {
    int prn;
    double seconds;
    std::string path;
    try {
        prn = body.at("prn").get<int>();
        seconds = body.at("seconds").get<double>();
        path = body.at("path").get<std::string>();
    } catch (const std::exception& e) {
        conn.send_error(std::string("capture wants {prn, seconds, path}: ") + e.what(),
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    std::lock_guard<std::mutex> lk(_m);
    for (Slot& s : _slots) {
        if (s.prn != prn)
            continue;
        if (s.cap)
            fclose(s.cap);
        s.cap = fopen(path.c_str(), "wb");
        if (!s.cap) {
            conn.send_error("cannot open " + path, kotekan::HTTP_RESPONSE::BAD_REQUEST);
            return;
        }
        s.cap_until = now_s() + seconds;
        INFO("GnssCskDespread[{:s}]: capturing PRN {:d} for {:g} s to {:s}", unique_name, prn,
             seconds, path);
        conn.send_empty_reply(kotekan::HTTP_RESPONSE::OK);
        return;
    }
    conn.send_error("PRN " + std::to_string(prn) + " is in no slot",
                    kotekan::HTTP_RESPONSE::BAD_REQUEST);
}
