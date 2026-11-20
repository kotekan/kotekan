#define BOOST_TEST_MODULE "test_gnss_nav_symbols"

#include "gnssNavSymbols.hpp" // for NavSymbolTrack

#include <boost/test/included/unit_test.hpp>
#include <cmath>   // for sqrt, abs
#include <complex> // for complex, polar
#include <random>  // for mt19937, normal_distribution
#include <vector>  // for vector

using cd = std::complex<double>;
using gnss::NavSymbolTrack;

namespace {

constexpr int SYM_REC = 10;          // records per symbol (L5: 10 ms symbols, 1 ms records)
constexpr long long REC_SAMP = 1000; // samples per record
constexpr double REC_DT = 1e-3;
constexpr double OFFSET = M_PI / 2; // data carrier ahead of the pilot's

// Records of one satellite: unit pilot and data amplitude per record, a residual carrier common
// to both, symbol boundaries at boundary0 + k * SYM_REC records, and complex noise of rms sigma
// per record and component.
struct Sky {
    double boundary0 = 3.37;  // records
    double resid_hz = 7.0;    // carrier left after the tracker, common to both components
    double sigma = 0.0;       // noise per record
    std::vector<int> symbols; // sign of symbol k, which starts at boundary0 + (k - 1) * SYM_REC
    std::mt19937 rng{1};

    int sign_at(double rec) const {
        const int k = (int)std::floor((rec - boundary0) / SYM_REC) + 1;
        return symbols[(size_t)k];
    }
    cd noise(double var) {
        std::normal_distribution<double> n(0.0, std::sqrt(var / 2));
        return {n(rng), n(rng)};
    }
    NavSymbolTrack::Rec record(long long r) {
        NavSymbolTrack::Rec out;
        out.seq = r * REC_SAMP;
        out.utc = 1000.0 + r * REC_DT;
        const cd carrier = std::polar(1.0, 0.4 + 2 * M_PI * resid_hz * (r + 0.5) * REC_DT);
        out.pilot = carrier + noise(sigma * sigma);
        // The boundary in this record, if any, at fraction f of it.
        const double b = boundary0 + std::ceil((r - boundary0) / SYM_REC) * SYM_REC;
        const double f = (b >= r && b < r + 1) ? b - r : 1.0;
        const cd turn = carrier * std::polar(1.0, OFFSET);
        const cd head = turn * (double)sign_at(r) * f + noise(sigma * sigma * f);
        const cd tail =
            f < 1.0 ? turn * (double)sign_at(b) * (1.0 - f) + noise(sigma * sigma * (1.0 - f))
                    : cd(0.0, 0.0);
        out.head = head;
        out.data = head + tail;
        out.head_frac = f;
        return out;
    }
};

std::vector<int> random_symbols(int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<int> s((size_t)n);
    for (int& v : s)
        v = (rng() & 1) ? 1 : -1;
    return s;
}

// The decisions against the sky's symbols, symbol 0 of the stream being the sky's symbol
// first_k: the count of disagreements after the overall sign that fits best.
int errors(const NavSymbolTrack& t, const std::vector<int>& sky, int first_k) {
    int agree = 0, n = 0;
    long long k = t.n_sym - (long long)t.sym.size();
    for (int8_t s : t.sym) {
        agree += (s == sky[(size_t)(first_k + k)]) ? 1 : 0;
        ++k;
        ++n;
    }
    return std::min(agree, n - agree);
}

} // namespace

// Noiseless: every whole symbol is decided, in order, from the second whole one (the first leads
// its reference), and the stream starts at its boundary.
BOOST_AUTO_TEST_CASE(noiseless_symbols_in_order) {
    Sky sky;
    sky.symbols = random_symbols(200, 2);
    NavSymbolTrack t(SYM_REC, 1, REC_SAMP, REC_DT, true, OFFSET, 1000);
    const int n_rec = 1500;
    for (long long r = 0; r < n_rec; ++r)
        t.push(sky.record(r));
    // Symbol 1 (from boundary0) is the first whole one. Symbols 2 to the next-to-last whole one
    // are decided: each waits for the next to end.
    const int whole = (int)std::floor((n_rec - sky.boundary0) / SYM_REC);
    BOOST_CHECK_EQUAL(t.n_sym, whole - 2);
    BOOST_CHECK_EQUAL(errors(t, sky.symbols, 2), 0);
    BOOST_CHECK_CLOSE(t.t0, 1000.0 + (sky.boundary0 + SYM_REC) * REC_DT, 1e-9);
    // The reference is centred on each symbol, so the carrier left over turns it as it turns the
    // data, and only shortens it: 7 Hz over 30 ms keeps 93% of its length.
    BOOST_CHECK_SMALL(t.residual_rad, 1e-9);
    BOOST_CHECK_EQUAL(t.n_restart, 0);
    for (float v : t.soft)
        BOOST_CHECK_CLOSE(std::abs(v), 1.0 / 0.93, 1.0);
}

// The decisions keep their sign convention whatever carrier is left over and wherever the
// boundary falls, including at a record's start (head 0).
BOOST_AUTO_TEST_CASE(residual_carrier_and_boundary_positions) {
    for (double b0 : {0.0, 0.5, 0.999}) {
        for (double resid : {-20.0, 3.0, 20.0}) {
            Sky sky;
            sky.boundary0 = b0;
            sky.resid_hz = resid;
            sky.symbols = random_symbols(120, 3);
            NavSymbolTrack t(SYM_REC, 1, REC_SAMP, REC_DT, true, OFFSET, 1000);
            for (long long r = 0; r < 1000; ++r)
                t.push(sky.record(r));
            BOOST_TEST_CONTEXT("boundary0 " << b0 << " resid " << resid) {
                BOOST_CHECK_GT(t.n_sym, 90);
                BOOST_CHECK_EQUAL(errors(t, sky.symbols, 2), 0);
            }
        }
    }
}

// Per-symbol SNR of ~13 dB (10 records of unit amplitude in noise of 1 per record): few errors,
// and the SNR estimate is in the right range.
BOOST_AUTO_TEST_CASE(noisy_symbols) {
    Sky sky;
    sky.sigma = 1.0;
    sky.symbols = random_symbols(1100, 4);
    NavSymbolTrack t(SYM_REC, 1, REC_SAMP, REC_DT, true, OFFSET, 2000);
    for (long long r = 0; r < 10000; ++r)
        t.push(sky.record(r));
    BOOST_CHECK_GT(t.n_sym, 990);
    BOOST_CHECK_LT(errors(t, sky.symbols, 2), 10);
    BOOST_CHECK_GT(t.snr_db(), 5.0);
    BOOST_CHECK_LT(t.snr_db(), 20.0);
}

// A missing record (a dropped frame) starts a new stream, at the next whole symbol.
BOOST_AUTO_TEST_CASE(gap_restarts) {
    Sky sky;
    sky.symbols = random_symbols(200, 5);
    NavSymbolTrack t(SYM_REC, 1, REC_SAMP, REC_DT, true, OFFSET, 1000);
    for (long long r = 0; r < 500; ++r)
        t.push(sky.record(r));
    BOOST_CHECK_GT(t.n_sym, 40);
    for (long long r = 505; r < 1000; ++r)
        t.push(sky.record(r));
    BOOST_CHECK_EQUAL(t.n_restart, 1);
    // The first boundary after the gap, at boundary0 + 510, starts symbol 52; symbol 53 is the
    // new stream's symbol 0.
    BOOST_CHECK_CLOSE(t.t0, 1000.0 + (sky.boundary0 + 520) * REC_DT, 1e-9);
    BOOST_CHECK_EQUAL(errors(t, sky.symbols, 53), 0);
}

// A boundary the records do not show makes a 20-record symbol, which restarts the stream.
BOOST_AUTO_TEST_CASE(missed_boundary_restarts) {
    Sky sky;
    sky.symbols = random_symbols(200, 6);
    NavSymbolTrack t(SYM_REC, 1, REC_SAMP, REC_DT, true, OFFSET, 1000);
    for (long long r = 0; r < 1000; ++r) {
        NavSymbolTrack::Rec rec = sky.record(r);
        if (r == 403) // the boundary at 403.37
            rec.head_frac = 1.0;
        t.push(rec);
    }
    // Symbols 40 and 41 run together; the stream restarts with symbol 42, so 43 is its first.
    BOOST_CHECK_EQUAL(t.n_restart, 1);
    BOOST_CHECK_EQUAL(errors(t, sky.symbols, 43), 0);
}

namespace {

// GW_1202-like records without a pilot: 2-record code periods whose first half (the ranging
// code) carries the data sign and whose second (the CSK code, zero in the replica) carries
// nothing, so a record holding a boundary has no head energy. A symbol is two periods, starting
// at period pair0, and a residual carrier turns every record.
struct GwSky {
    double boundary0 = 0.37; // first period boundary, records; period j starts at boundary0 + 2j
    int pair0 = 1;           // symbols start at periods pair0, pair0 + 2, ...
    double resid_hz = 40.0;
    double sigma = 0.0;
    std::vector<int> symbols; // symbols[i + 1] covers periods pair0 + 2i and pair0 + 2i + 1
    std::mt19937 rng{7};

    cd noise(double var) {
        std::normal_distribution<double> n(0.0, std::sqrt(var / 2));
        return {n(rng), n(rng)};
    }
    int sign_of_period(int j) const {
        return symbols[(size_t)((j - pair0 + 2) / 2)];
    }
    NavSymbolTrack::Rec record(long long r) {
        NavSymbolTrack::Rec out;
        out.seq = r * REC_SAMP;
        out.utc = 1000.0 + r * REC_DT;
        const cd carrier = std::polar(1.0, 0.4 + 2 * M_PI * resid_hz * (r + 0.5) * REC_DT);
        // The ranging halves [b, b + 1) of the periods that can overlap record [r, r + 1).
        out.head_frac = 1.0;
        for (int j = (int)std::floor((r - boundary0) / 2) - 1; j <= (int)((r - boundary0) / 2) + 1;
             ++j) {
            const double b = boundary0 + 2.0 * j;
            if (j < 0)
                continue;
            const double len = std::min(b + 1.0, r + 1.0) - std::max(b, (double)r);
            if (len > 0.0)
                out.data += carrier * (double)sign_of_period(j) * len + noise(sigma * sigma * len);
            if (b >= r && b < r + 1)
                out.head_frac = 0.0; // the head is the previous period's CSK half
        }
        return out;
    }
};

// The sky symbol a stream's symbol 0 is, from t0: its first period starts in the record t0
// labels, and that period's parity must be the symbols' own.
int gw_first_symbol(const NavSymbolTrack& t, const GwSky& sky) {
    const double r = std::round((t.t0 - 1000.0) / REC_DT);
    const int j = (int)std::lround((r - sky.boundary0) / 2.0);
    BOOST_CHECK_EQUAL((j - sky.pair0) % 2, 0);
    return (j - sky.pair0) / 2 + 1;
}

} // namespace

// Without a pilot: the pairing is found, the loop measures the carrier, and every symbol is
// decided, up to the stream's overall sign.
BOOST_AUTO_TEST_CASE(no_pilot_pairs_and_tracks_the_carrier) {
    for (double b0 : {0.37, 0.0, 0.93}) {
        for (int pair0 : {0, 1}) {
            GwSky sky;
            sky.boundary0 = b0;
            sky.pair0 = pair0;
            sky.symbols = random_symbols(1100, 8);
            NavSymbolTrack t(4, 2, REC_SAMP, REC_DT, false, 0.0, 2000);
            for (long long r = 0; r < 4000; ++r)
                t.push(sky.record(r));
            BOOST_TEST_CONTEXT("boundary0 " << b0 << " pair0 " << pair0) {
                BOOST_CHECK_GT(t.n_sym, 900);
                BOOST_CHECK_EQUAL(errors(t, sky.symbols, gw_first_symbol(t, sky)), 0);
                BOOST_CHECK_CLOSE(t.carrier_hz, 40.0, 1.0);
            }
        }
    }
}

// Without a pilot, ~12 dB per symbol and a carrier of -55 Hz: few errors.
BOOST_AUTO_TEST_CASE(no_pilot_noisy) {
    GwSky sky;
    sky.sigma = 0.35;
    sky.resid_hz = -55.0;
    sky.symbols = random_symbols(3000, 9);
    NavSymbolTrack t(4, 2, REC_SAMP, REC_DT, false, 0.0, 3000);
    for (long long r = 0; r < 11000; ++r)
        t.push(sky.record(r));
    BOOST_CHECK_GT(t.n_sym, 2500);
    BOOST_CHECK_LT(errors(t, sky.symbols, gw_first_symbol(t, sky)), 0.01 * t.n_sym);
    BOOST_CHECK_CLOSE(t.carrier_hz, -55.0, 2.0);
}
