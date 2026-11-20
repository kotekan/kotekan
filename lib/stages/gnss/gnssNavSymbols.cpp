#include "gnssNavSymbols.hpp"

#include <cmath> // for log10, sqrt, arg, polar

namespace gnss {

// Weight of each new symbol in the running means of the residual angle and the SNR, and of each
// record in those of the prompts' power: ~1 s.
static constexpr double EMA_W = 0.01;
static constexpr double EMA_W_REC = 0.001;

NavSymbolTrack::NavSymbolTrack(int symbol_records, long long rec_samples, double rec_dt,
                               double offset_rad, size_t keep) :
    _symbol_records(symbol_records), _rec_samples(rec_samples), _rec_dt(rec_dt),
    _unturn(std::polar(1.0, -offset_rad)), _keep(keep) {}

void NavSymbolTrack::reset() {
    _next_seq = -1;
    n_records = 0;
    n_boundary = 0;
    _pow_data = 0.0;
    _pow_pilot = 0.0;
    restart();
}

void NavSymbolTrack::restart() {
    if (n_sym > 0)
        ++n_restart;
    n_sym = 0;
    sym.clear();
    soft.clear();
    _in_symbol = false;
    _cur_records = 0;
    _ended.clear();
}

void NavSymbolTrack::push(const Rec& r) {
    if (_next_seq >= 0 && r.seq != _next_seq)
        restart();
    _next_seq = r.seq + _rec_samples;
    ++n_records;
    _pow_data += EMA_W_REC * (std::norm(r.data) - _pow_data);
    _pow_pilot += EMA_W_REC * (std::norm(r.pilot) - _pow_pilot);

    if (r.head_frac >= 1.0) {
        if (_in_symbol) {
            _cur.data += r.data;
            _cur.pilot += r.pilot;
            ++_cur_records;
        }
        return;
    }
    ++n_boundary;
    // A boundary: the head ends the current symbol and the tail starts the next. The pilot,
    // which has no head of its own, is split between them by the head's share of the energy,
    // so each symbol's reference covers the same time as its data.
    if (_in_symbol) {
        _cur.data += r.head;
        _cur.pilot += r.head_frac * r.pilot;
        end_symbol(_cur, _cur_records + 1);
    }
    _in_symbol = true;
    _cur = {r.data - r.head, (1.0 - r.head_frac) * r.pilot, r.utc + r.head_frac * _rec_dt};
    _cur_records = 0;
}

void NavSymbolTrack::end_symbol(const Sym& s, int n_rec) {
    // A symbol spans symbol_records records, plus one when its ends split records. Any other
    // length means a boundary was missed or doubled, and the count is lost.
    if (n_rec < _symbol_records || n_rec > _symbol_records + 1) {
        restart();
        return;
    }
    if (_ended.size() == 2) {
        // Decide the newest ended symbol against the pilot centred on it.
        const Sym& k = _ended.back();
        const std::complex<double> ref = _ended.front().pilot + k.pilot + s.pilot;
        const double ref_amp = std::abs(ref);
        if (ref_amp > 0.0) {
            const std::complex<double> y = k.data * std::conj(ref) * _unturn / ref_amp;
            const double y_amp = std::abs(y);
            if (y_amp > 0.0) {
                const std::complex<double> u = y / y_amp;
                _sq += EMA_W * (u * u - _sq);
            }
            residual_rad = 0.5 * std::arg(_sq);
            // In units of the pilot's amplitude over one symbol; the reference sums three.
            const double proj = std::real(y * std::polar(1.0, -residual_rad));
            const double v = 3.0 * proj / ref_amp;
            if (n_sym == 0)
                t0 = k.t_start;
            sym.push_back(proj >= 0.0 ? 1 : -1);
            soft.push_back((float)v);
            ++n_sym;
            while (sym.size() > _keep) {
                sym.pop_front();
                soft.pop_front();
            }
            _m1 += EMA_W * (std::abs(v) - _m1);
            _m2 += EMA_W * (v * v - _m2);
        }
    }
    _ended.push_back(s);
    if (_ended.size() > 2)
        _ended.pop_front();
}

double NavSymbolTrack::data_pilot_db() const {
    if (!(_pow_data > 0.0) || !(_pow_pilot > 0.0))
        return 0.0;
    return 10.0 * std::log10(_pow_data / _pow_pilot);
}

double NavSymbolTrack::snr_db() const {
    const double var = _m2 - _m1 * _m1;
    if (!(var > 0.0) || !(_m1 > 0.0))
        return 0.0;
    return 10.0 * std::log10(_m1 * _m1 / var);
}

} // namespace gnss
