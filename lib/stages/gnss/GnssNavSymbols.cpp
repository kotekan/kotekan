#include "GnssNavSymbols.hpp"

#include "GnssChanMetadata.hpp" // for get_gnss_chan_metadata, metadata_is_gnss_chan
#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE
#include "gnssNavSym.hpp"       // for NavSymRow, nav_sym_frame_bytes, NAV_SYM_MAX
#include "gnssRecord.hpp"       // for record_stride, REC_*
#include "kotekanLogging.hpp"   // for FATAL_ERROR, INFO, WARN
#include "visUtil.hpp"          // for frameID, current_time

#include <algorithm>  // for min
#include <cmath>      // for M_PI, llround
#include <complex>    // for complex, polar
#include <cstring>    // for memcpy, memset
#include <functional> // for bind

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssNavSymbols);

GnssNavSymbols::GnssNavSymbols(Config& config, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&GnssNavSymbols::main_thread, this)) {
    _data_buf = get_buffer("data_buf");
    _data_buf->register_consumer(unique_name);
    _pilot_buf = config.exists(unique_name, "pilot_buf") ? get_buffer("pilot_buf") : nullptr;
    if (_pilot_buf)
        _pilot_buf->register_consumer(unique_name);
    if (!metadata_is_gnss_chan(_data_buf) || (_pilot_buf && !metadata_is_gnss_chan(_pilot_buf)))
        FATAL_ERROR("GnssNavSymbols[{:s}]: pilot_buf and data_buf need a GnssChanMetadata pool",
                    unique_name);

    _n_prn = config.get<int>(unique_name, "n_prn");
    _n_elem = config.get<int>(unique_name, "n_elements");
    _symbol_records = config.get<int>(unique_name, "symbol_records");
    const int periods = config.get_default<int>(unique_name, "periods_per_symbol", 1);
    const int hops = config.get<int>(unique_name, "hops_per_record");
    const int fft_len = config.get<int>(unique_name, "fft_length");
    const double fs = config.get<double>(unique_name, "sample_rate");
    const double offset_deg = _pilot_buf ? config.get<double>(unique_name, "offset_deg") : 0.0;
    const int keep = config.get_default<int>(unique_name, "symbol_export", 1500);
    _rec_samples = (long long)hops * fft_len;
    _rec_dt = (double)_rec_samples / fs;
    if (_symbol_records < 1 || keep < 1)
        FATAL_ERROR("GnssNavSymbols[{:s}]: symbol_records and symbol_export must be >= 1",
                    unique_name);
    // A pilot references one period per symbol; without one a symbol is one period or two.
    if (periods < 1 || periods > (_pilot_buf ? 1 : 2) || _symbol_records % periods != 0)
        FATAL_ERROR("GnssNavSymbols[{:s}]: periods_per_symbol {:d} with symbol_records {:d}: "
                    "1 with a pilot, 1 or 2 without, dividing symbol_records",
                    unique_name, periods, _symbol_records);
    const size_t need = (size_t)_n_prn * gnss::record_stride(_n_elem) * sizeof(float);
    if ((size_t)_data_buf->frame_size < need
        || (_pilot_buf && (size_t)_pilot_buf->frame_size < need))
        FATAL_ERROR("GnssNavSymbols[{:s}]: record frames under {:d} B", unique_name, need);

    for (int p = 0; p < _n_prn; ++p)
        _tracks.emplace_back(_symbol_records, periods, _rec_samples, _rec_dt, _pilot_buf != nullptr,
                             offset_deg * M_PI / 180.0, (size_t)keep);
    _prn.assign((size_t)_n_prn, 0);
    _sent_n.assign((size_t)_n_prn, 0);
    _sent_t0.assign((size_t)_n_prn, -1.0);

    if (config.exists(unique_name, "out_buf")) {
        _out_buf = get_buffer("out_buf");
        _out_buf->register_producer(unique_name);
        if ((size_t)_out_buf->frame_size < gnss::nav_sym_frame_bytes(_n_prn))
            FATAL_ERROR("GnssNavSymbols[{:s}]: out_buf frames under {:d} B", unique_name,
                        gnss::nav_sym_frame_bytes(_n_prn));
    }
}

GnssNavSymbols::~GnssNavSymbols() {
    kotekan::restServer::instance().remove_get_callback(unique_name + "/get_status");
}

void GnssNavSymbols::push_records(const float* pilot, const float* data, long long seq) {
    const int stride = gnss::record_stride(_n_elem);
    std::lock_guard<std::mutex> lk(_m);
    for (int p = 0; p < _n_prn; ++p) {
        const float* d = data + (size_t)p * stride;
        // Without a pilot the data record stands in for it in the checks below.
        const float* q = pilot ? pilot + (size_t)p * stride : d;
        const int prn = (int)q[gnss::REC_PRN];
        gnss::NavSymbolTrack& t = _tracks[(size_t)p];
        if (prn != _prn[(size_t)p]) {
            t.reset();
            _prn[(size_t)p] = prn;
        }
        // A slot with a job has covering channels. Its prompt energy can still be 0: GW_1202's
        // replica is zero over the CSK half, which can fill a record.
        if (prn <= 0 || (int)d[gnss::REC_PRN] != prn || !(q[gnss::REC_NCHAN] > 0.0f)
            || !(d[gnss::REC_NCHAN] > 0.0f)) {
            t.reset();
            continue;
        }
        // The pilot's assembler rotation (gnss::REC_PHI0) follows the carrier from record to
        // record. The data's assembler has its own origin, so swap its rotation for the pilot's.
        const std::complex<double> to_pilot =
            std::polar(1.0, (double)d[gnss::REC_PHI0] - (double)q[gnss::REC_PHI0]);
        gnss::NavSymbolTrack::Rec r;
        r.seq = seq;
        std::memcpy(&r.utc, q + gnss::RECORD_UTC_SLOT, sizeof(double));
        if (pilot)
            r.pilot = std::complex<double>(q[gnss::REC_P_RE], q[gnss::REC_P_IM]);
        r.data = std::complex<double>(d[gnss::REC_P_RE], d[gnss::REC_P_IM]) * to_pilot;
        r.head = std::complex<double>(d[gnss::REC_PH_RE], d[gnss::REC_PH_IM]) * to_pilot;
        // A boundary starts the code period's energy, so a record with none holds none.
        r.head_frac = d[gnss::REC_P_ENERGY] > 0.0f
                          ? (double)d[gnss::REC_PH_ENERGY] / (double)d[gnss::REC_P_ENERGY]
                          : 1.0;
        t.push(r);
    }
}

void GnssNavSymbols::main_thread() {
    kotekan::restServer::instance().register_get_callback(
        unique_name + "/get_status",
        std::bind(&GnssNavSymbols::get_status_callback, this, std::placeholders::_1));

    frameID pilot_id(_pilot_buf ? _pilot_buf : _data_buf);
    frameID data_id(_data_buf);
    uint8_t* pilot = nullptr;
    uint8_t* data = nullptr;
    while (!stop_thread && !_pilot_buf) {
        if (!(data = _data_buf->wait_for_full_frame(unique_name, data_id)))
            break;
        push_records(nullptr, (const float*)data,
                     get_gnss_chan_metadata(_data_buf, data_id)->sample_seq);
        _data_buf->mark_frame_empty(unique_name, data_id++);
        emit_symbols();
    }
    while (!stop_thread && _pilot_buf) {
        if (!pilot && !(pilot = _pilot_buf->wait_for_full_frame(unique_name, pilot_id)))
            break;
        if (!data && !(data = _data_buf->wait_for_full_frame(unique_name, data_id)))
            break;
        const long long sp = get_gnss_chan_metadata(_pilot_buf, pilot_id)->sample_seq;
        const long long sd = get_gnss_chan_metadata(_data_buf, data_id)->sample_seq;
        // The buffers carry the same records; release the older one until they pair up.
        if (sp == sd) {
            push_records((const float*)pilot, (const float*)data, sp);
            emit_symbols();
        }
        if (sp <= sd) {
            _pilot_buf->mark_frame_empty(unique_name, pilot_id++);
            pilot = nullptr;
        }
        if (sd <= sp) {
            _data_buf->mark_frame_empty(unique_name, data_id++);
            data = nullptr;
        }
    }
}

void GnssNavSymbols::emit_symbols() {
    if (!_out_buf)
        return;
    std::vector<gnss::NavSymRow> rows;
    {
        std::lock_guard<std::mutex> lk(_m);
        const double now = current_time();
        for (int p = 0; p < _n_prn; ++p) {
            const gnss::NavSymbolTrack& t = _tracks[(size_t)p];
            if (t.t0 != _sent_t0[(size_t)p]) { // a new stream
                _sent_t0[(size_t)p] = t.t0;
                _sent_n[(size_t)p] = 0;
            }
            const long long n = std::min({t.n_sym - _sent_n[(size_t)p], (long long)t.sym.size(),
                                          (long long)gnss::NAV_SYM_MAX});
            _sent_n[(size_t)p] = t.n_sym;
            if (_prn[(size_t)p] <= 0 || n <= 0)
                continue;
            gnss::NavSymRow r{};
            r.prn = _prn[(size_t)p];
            r.n = (int32_t)n;
            r.first = t.n_sym - n;
            r.stream_id = std::llround(t.t0 * 1000.0);
            r.t0 = t.t0;
            r.sym_dt = (double)_symbol_records * _rec_dt;
            r.t_dec = now;
            for (long long i = 0; i < n; ++i)
                r.sign[i] = t.sym[t.sym.size() - (size_t)n + (size_t)i];
            rows.push_back(r);
        }
    }
    if (rows.empty())
        return;
    if (!_out_buf->is_frame_empty(_out_frame)) {
        if (_n_dropped++ % 1000 == 0)
            WARN("GnssNavSymbols[{:s}]: out_buf full, {:d} frame(s) of symbols dropped",
                 unique_name, _n_dropped);
        return;
    }
    uint8_t* frame = _out_buf->wait_for_empty_frame(unique_name, _out_frame);
    if (frame == nullptr)
        return;
    const int32_t n_rows = (int32_t)rows.size();
    std::memset(frame, 0, 8);
    std::memcpy(frame, &n_rows, sizeof(n_rows));
    std::memcpy(frame + 8, rows.data(), rows.size() * sizeof(gnss::NavSymRow));
    _out_buf->mark_frame_full(unique_name, _out_frame);
    _out_frame = (_out_frame + 1) % _out_buf->num_frames;
}

void GnssNavSymbols::get_status_callback(kotekan::connectionInstance& conn) {
    nlohmann::json rows = nlohmann::json::array();
    {
        std::lock_guard<std::mutex> lk(_m);
        for (int p = 0; p < _n_prn; ++p) {
            const gnss::NavSymbolTrack& t = _tracks[(size_t)p];
            if (_prn[(size_t)p] <= 0 || t.sym.empty())
                continue;
            // As the combiner's emits do, each starts at its oldest symbol, so successive emits
            // differ and the decoder merges them by their overlap.
            nlohmann::json bits = nlohmann::json::array();
            int k = 0;
            for (int8_t s : t.sym)
                bits.push_back({k++, (int)s});
            const long long first = t.n_sym - (long long)t.sym.size();
            rows.push_back({{"prn", _prn[(size_t)p]},
                            {"n_sym", t.n_sym},
                            {"n_restart", t.n_restart},
                            {"t0", t.t0},
                            {"stream_id", std::llround(t.t0 * 1000.0)},
                            {"snr_db", t.snr_db()},
                            {"residual_deg", t.residual_rad * 180.0 / M_PI},
                            {"carrier_hz", t.carrier_hz},
                            {"data_pilot_db", t.data_pilot_db()},
                            {"n_records", t.n_records},
                            {"n_boundary", t.n_boundary},
                            {"nav_obs",
                             {{"utc_ref", t.t0 + (double)(first * _symbol_records) * _rec_dt},
                              {"rec_dt", _rec_dt},
                              {"phase", 0},
                              {"br", _symbol_records},
                              {"bits", bits}}}});
        }
    }
    conn.send_json_reply(rows);
}
