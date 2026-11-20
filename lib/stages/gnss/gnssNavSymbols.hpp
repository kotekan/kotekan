/**
 * @file
 * @brief One satellite's data symbols, demodulated against its pilot as records arrive.
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
 * @brief One satellite's data symbols from its data and pilot records, decided as they arrive.
 *
 * The data component (GPS_L5_I_NH) is despread at the pilot's code phase and carrier
 * (cudaGnssTrack's data block), so each record's data prompt is the pilot prompt turned by the
 * components' fixed carrier offset (90 degrees between L5's I5 and Q5) times the symbol's sign.
 * The caller puts both on one carrier reference that follows the carrier from record to record:
 * the pilot's assembler rotation (gnss::REC_PHI0). The data code is one symbol long: a record
 * holding a code-period boundary (head energy below its prompt's) ends one symbol at its head, and
 * its tail starts the next. That record's pilot is split between the two by the head's share.
 *
 * Symbol k is decided once k+1 has ended, against the pilot summed over symbols k-1 to k+1:
 * the reference is centred on the symbol, so a residual carrier, which turns both components
 * alike, cancels. The decision is the sign of the data along the pilot turned by the expected
 * offset and by the residual angle measured from the squared symbols. Its overall sign is the
 * sign convention of the component pair, which a decoder resolves (CNAV's CRC).
 *
 * A record that does not follow the previous one, a symbol of the wrong length, or @ref reset
 * starts a new stream: the decisions so far are dropped, and symbol 0 is the second whole
 * symbol after it, the first one being its reference's lead.
 */
class NavSymbolTrack {
public:
    /// One record of the pair, both on the pilot's carrier reference.
    struct Rec {
        long long seq = 0;          ///< the record's first sample
        double utc = 0.0;           ///< the record's capture UTC
        std::complex<double> pilot; ///< pilot prompt
        std::complex<double> data;  ///< data prompt
        std::complex<double> head;  ///< data prompt before the code-period boundary
        double head_frac = 1.0;     ///< data head energy over prompt energy; 1 = no boundary
    };

    /// @p symbol_records records per symbol, @p rec_samples samples and @p rec_dt seconds per
    /// record, @p offset_rad the data carrier's offset from the pilot's, and @p keep the
    /// decisions held for export.
    NavSymbolTrack(int symbol_records, long long rec_samples, double rec_dt, double offset_rad,
                   size_t keep);

    /// Add the next record.
    void push(const Rec& r);
    /// Start a new stream (the satellite lost its pilot or data, or its slot changed).
    void reset();

    double t0 = 0.0;        ///< UTC of the start of symbol 0 of this stream
    long long n_sym = 0;    ///< symbols decided in this stream
    std::deque<int8_t> sym; ///< the newest decisions, +-1; back() is symbol n_sym - 1
    /// Their projections on the decision axis, in units of the pilot's amplitude over a symbol
    /// (+-1 for data and pilot of equal power, as L5's).
    std::deque<float> soft;
    long long n_restart = 0;   ///< streams started after the first
    double residual_rad = 0.0; ///< the measured offset beyond offset_rad

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
    void end_symbol(const Sym& s, int n_rec);

    const int _symbol_records;
    const long long _rec_samples;
    const double _rec_dt;
    const std::complex<double> _unturn; ///< e^{-i offset_rad}
    const size_t _keep;

    long long _next_seq = -1;
    bool _in_symbol = false; ///< a boundary has been seen: _cur started at one
    Sym _cur{};
    int _cur_records = 0;
    std::deque<Sym> _ended;   ///< the last two symbols ended: the newest awaits its decision
    std::complex<double> _sq; ///< running mean of the squared unit projections
    double _m1 = 0.0, _m2 = 0.0;
    double _pow_data = 0.0, _pow_pilot = 0.0; ///< running means of the prompts' power
};

} // namespace gnss

#endif // GNSS_NAV_SYMBOLS_HPP
