#include "gnssNavSymbols.hpp"

#include <cmath> // for log10, sqrt, arg, polar, remainder

namespace gnss {

// Weight of each new symbol in the running means of the residual angle and the SNR, and of each
// record in those of the prompts' power: ~1 s.
static constexpr double EMA_W = 0.01;
static constexpr double EMA_W_REC = 0.001;
// Without a pilot: weight of each period in the frequency estimate (~100 periods), and the pairs
// of each parity seen before the symbols' pairing is chosen.
static constexpr double EMA_W_FLL = 0.01;
static constexpr long long PARITY_MIN = 25;

NavSymbolTrack::NavSymbolTrack(int symbol_records, int periods_per_symbol, long long rec_samples,
                               double rec_dt, bool pilot, double offset_rad, size_t keep) :
    _period_records(symbol_records / periods_per_symbol), _periods_per_symbol(periods_per_symbol),
    _rec_samples(rec_samples), _rec_dt(rec_dt), _pilot(pilot),
    _unturn(std::polar(1.0, -offset_rad)), _keep(keep) {}

void NavSymbolTrack::reset() {
    _next_seq = -1;
    n_records = 0;
    n_boundary = 0;
    _pow_data = 0.0;
    _pow_pilot = 0.0;
    _fll = {};
    _have_sq = false;
    _phase = 0.0;
    carrier_hz = 0.0;
    restart();
}

void NavSymbolTrack::restart() {
    if (n_sym > 0)
        ++n_restart;
    n_sym = 0;
    sym.clear();
    soft.clear();
    _in_period = false;
    _cur_records = 0;
    _ended.clear();
    _n_period = 0;
    _have_last = false;
    _pair_e[0] = _pair_e[1] = 0.0;
    _pair_n[0] = _pair_n[1] = 0;
    _parity = -1;
    _have_th = false;
}

void NavSymbolTrack::push(const Rec& r) {
    if (_next_seq >= 0 && r.seq != _next_seq)
        restart();
    _next_seq = r.seq + _rec_samples;
    ++n_records;
    _pow_data += EMA_W_REC * (std::norm(r.data) - _pow_data);
    _pow_pilot += EMA_W_REC * (std::norm(r.pilot) - _pow_pilot);

    if (r.head_frac >= 1.0) {
        if (_in_period) {
            _cur.data += r.data;
            _cur.pilot += r.pilot;
            ++_cur_records;
        }
        return;
    }
    ++n_boundary;
    // A boundary: the head ends the current period and the tail starts the next. The pilot,
    // which has no head of its own, is split between them by the head's share of the energy,
    // so each period's reference covers the same time as its data.
    if (_in_period) {
        _cur.data += r.head;
        _cur.pilot += r.head_frac * r.pilot;
        end_period(_cur, _cur_records + 1);
    }
    _in_period = true;
    _cur = {r.data - r.head, (1.0 - r.head_frac) * r.pilot, r.utc + r.head_frac * _rec_dt};
    _cur_records = 0;
}

void NavSymbolTrack::end_period(const Sym& s, int n_rec) {
    // A period spans its records, plus one when its ends split records. Any other length means
    // a boundary was missed or doubled, and the count is lost.
    if (n_rec < _period_records || n_rec > _period_records + 1) {
        restart();
        return;
    }
    if (_pilot) {
        end_symbol(s);
        return;
    }
    // Without a pilot the carrier left over turns the symbols: a frequency-locked loop on the
    // squared periods, which the data sign does not reach, measures it, and each period is
    // turned back before the periods pair into symbols.
    const std::complex<double> sq = s.data * s.data;
    if (_have_sq) {
        const std::complex<double> d = sq * std::conj(_last_sq);
        const double a = std::abs(d);
        if (a > 0.0)
            _fll += EMA_W_FLL * (d / a - _fll);
    }
    _last_sq = sq;
    _have_sq = true;
    const double step = 0.5 * std::arg(_fll); // radians per period
    _phase = std::remainder(_phase + step, 2.0 * M_PI);
    carrier_hz = step / (2.0 * M_PI * _period_records * _rec_dt);
    Sym p = s;
    p.data *= std::polar(1.0, -_phase);
    ++_n_period;
    if (_periods_per_symbol == 1) {
        end_symbol(p);
        return;
    }
    // Two periods to a symbol, paired the way that gives the larger symbols: the sign changes
    // only between symbols, so the other pairing sums across half of its changes.
    if (_have_last) {
        const std::complex<double> z = _last.data + p.data;
        const int q = (int)((_n_period - 1) & 1); // parity of the pair's first period
        _pair_e[q] += EMA_W * (std::norm(z) - _pair_e[q]);
        ++_pair_n[q];
        if (_pair_n[0] >= PARITY_MIN && _pair_n[1] >= PARITY_MIN) {
            const int best = _pair_e[1] > _pair_e[0] ? 1 : 0;
            if (_parity >= 0 && best != _parity) {
                // The decisions so far were paired wrongly: start the stream again on the right
                // pairing, keeping what was learnt about it.
                const double e[2] = {_pair_e[0], _pair_e[1]};
                const long long n[2] = {_pair_n[0], _pair_n[1]};
                const long long np = _n_period;
                restart();
                _pair_e[0] = e[0];
                _pair_e[1] = e[1];
                _pair_n[0] = n[0];
                _pair_n[1] = n[1];
                _n_period = np;
            }
            _parity = best;
            if (q == _parity)
                end_symbol({z, {}, _last.t_start});
        }
    }
    _last = p;
    _have_last = true;
}

void NavSymbolTrack::end_symbol(const Sym& s) {
    if (_ended.size() == 2) {
        // Decide the newest ended symbol against a reference centred on it.
        const Sym& k = _ended.back();
        if (_pilot) {
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
                decide(k.t_start, proj, 3.0 * proj / ref_amp);
            }
        } else {
            // The data's own phase from the three squared symbols, kept continuous across the
            // squaring's half-turn ambiguity, so only the stream's overall sign is open.
            const std::complex<double> f = _ended.front().data, c = s.data;
            const std::complex<double> S = f * f + k.data * k.data + c * c;
            const double amp = std::sqrt(std::abs(S) / 3.0);
            if (amp > 0.0) {
                const double th = 0.5 * std::arg(S);
                _th = _have_th ? _th + std::remainder(th - _th, M_PI) : th;
                _have_th = true;
                const double proj = std::real(k.data * std::polar(1.0, -_th));
                decide(k.t_start, proj, proj / amp);
            }
        }
    }
    _ended.push_back(s);
    if (_ended.size() > 2)
        _ended.pop_front();
}

void NavSymbolTrack::decide(double t_start, double proj, double v) {
    if (n_sym == 0)
        t0 = t_start;
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
