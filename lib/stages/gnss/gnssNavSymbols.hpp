/**
 * @file
 * @brief One satellite's data symbols, demodulated as records arrive.
 *  - gnss::NavSymbolTrack
 */

#ifndef GNSS_NAV_SYMBOLS_HPP
#define GNSS_NAV_SYMBOLS_HPP

#include <complex> // for complex
#include <cstdint> // for int8_t
#include <deque>   // for deque

namespace gnss {

/**
 * @class NavSymbolTrack
 * @brief One satellite's data symbols from its records, decided as they arrive.
 *
 * The records are summed per code period: a record holding a code-period boundary (head energy
 * below its prompt's) ends one period at its head, and its tail starts the next.
 *
 * With a pilot, the data component (GPS_L5_I_NH) is despread at the pilot's code phase and
 * carrier (cudaGnssTrack's data block), so each record's data prompt is the pilot prompt turned
 * by the components' fixed carrier offset (90 degrees between L5's I5 and Q5) times the symbol's
 * sign. The caller puts both on one carrier reference that follows the carrier from record to
 * record: the pilot's assembler rotation (gnss::REC_PHI0). A period is one symbol, and the
 * boundary record's pilot is split between the two by the head's share. Symbol k is decided once
 * k+1 has ended, against the pilot summed over symbols k-1 to k+1: the reference is centred on
 * the symbol, so a residual carrier, which turns both components alike, cancels. The decision is
 * the sign of the data along the pilot turned by the expected offset and by the residual angle
 * measured from the squared symbols.
 *
 * Without a pilot (GW_1202, whose ranging half carries the data sign), a frequency-locked loop
 * on the squared periods, which the sign does not reach, measures the carrier the tracker left,
 * and each period is turned back by it. A symbol is one period or two; two are paired the way
 * that gives the larger symbols, since the sign changes only between symbols. Symbol k is decided
 * once k+1 has ended, against the phase of the squared symbols k-1 to k+1, kept continuous
 * across the squaring's half-turn ambiguity.
 *
 * Either way the decisions' overall sign is open, and a decoder resolves it (a CRC).
 *
 * A record that does not follow the previous one, a period of the wrong length, a change of
 * pairing, or @ref reset starts a new stream: the decisions so far are dropped, and symbol 0 is
 * the second whole symbol after it, the first one being its reference's lead.
 */
class NavSymbolTrack {
public:
    /// One record, on the pilot's carrier reference if there is one.
    struct Rec {
        long long seq = 0;          ///< the record's first sample
        double utc = 0.0;           ///< the record's capture UTC
        std::complex<double> pilot; ///< pilot prompt; unused without a pilot
        std::complex<double> data;  ///< data prompt
        std::complex<double> head;  ///< data prompt before the code-period boundary
        double head_frac = 1.0;     ///< data head energy over prompt energy; 1 = no boundary
    };

    /// @p symbol_records records per symbol in @p periods_per_symbol code periods (1 with a
    /// pilot), @p rec_samples samples and @p rec_dt seconds per record, @p offset_rad the data
    /// carrier's offset from the pilot's, and @p keep the decisions held for export.
    NavSymbolTrack(int symbol_records, int periods_per_symbol, long long rec_samples, double rec_dt,
                   bool pilot, double offset_rad, size_t keep);

    /// Add the next record.
    void push(const Rec& r);
    /// Start a new stream (the satellite lost its pilot or data, or its slot changed).
    void reset();

    double t0 = 0.0;        ///< UTC of the start of symbol 0 of this stream
    long long n_sym = 0;    ///< symbols decided in this stream
    std::deque<int8_t> sym; ///< the newest decisions, +-1; back() is symbol n_sym - 1
    /// Their projections on the decision axis, in units of the reference's amplitude over a
    /// symbol (+-1 for data and pilot of equal power, as L5's).
    std::deque<float> soft;
    long long n_restart = 0;   ///< streams started after the first
    double residual_rad = 0.0; ///< with a pilot: the measured offset beyond offset_rad
    double carrier_hz = 0.0;   ///< without one: the carrier the frequency-locked loop measures

    /// The decisions' SNR, dB, from the mean and spread of the soft values' magnitude.
    double snr_db() const;
    /// Data prompt power over pilot prompt power per record, dB: ~0 for components of equal
    /// power, and the data's SNR below that when its despread finds nothing.
    double data_pilot_db() const;
    long long n_records = 0;  ///< records pushed
    long long n_boundary = 0; ///< of which held a code-period boundary

private:
    struct Sym {
        std::complex<double> data, pilot;
        double t_start;
    };
    void restart();
    void end_period(const Sym& s, int n_rec);
    void end_symbol(const Sym& s);
    void decide(double t_start, double proj, double v);

    const int _period_records;
    const int _periods_per_symbol;
    const long long _rec_samples;
    const double _rec_dt;
    const bool _pilot;
    const std::complex<double> _unturn; ///< e^{-i offset_rad}
    const size_t _keep;

    long long _next_seq = -1;
    bool _in_period = false; ///< a boundary has been seen: _cur started at one
    Sym _cur{};
    int _cur_records = 0;
    std::deque<Sym> _ended;   ///< the last two symbols ended: the newest awaits its decision
    std::complex<double> _sq; ///< running mean of the squared unit projections
    double _m1 = 0.0, _m2 = 0.0;
    double _pow_data = 0.0, _pow_pilot = 0.0; ///< running means of the prompts' power

    // Without a pilot.
    std::complex<double> _fll;     ///< running mean of the squared periods' unit phase steps
    std::complex<double> _last_sq; ///< the previous period squared
    bool _have_sq = false;
    double _phase = 0.0;     ///< the loop's carrier phase at the latest period
    long long _n_period = 0; ///< periods in this stream
    Sym _last{};             ///< the previous period, turned back
    bool _have_last = false;
    double _pair_e[2] = {0.0, 0.0}; ///< running mean of the paired symbols' power, per parity
    long long _pair_n[2] = {0, 0};
    int _parity = -1; ///< the pairing in use; -1 until chosen
    double _th = 0.0; ///< the reference phase, continuous
    bool _have_th = false;
};

} // namespace gnss

#endif // GNSS_NAV_SYMBOLS_HPP
