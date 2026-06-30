/**
 * @file
 * @brief GNSS signal descriptors -- the seam for multi-constellation support.
 *
 * A @ref SignalDescriptor captures everything the correlator/peeler needs that
 * is *signal-specific* (carrier, chip rate, code period/length, modulation,
 * pilot-vs-data, secondary codes) so the DSP core stays generic. The per-PRN
 * spreading code itself comes from a matching @c ReplicaSource (e.g.
 * gpsCACode for GPS L1 C/A), keeping the chip-value tables -- which must be
 * transcribed exactly from the interface specs -- isolated and unit-testable.
 *
 * Design notes for the distributed-band / multi-constellation roadmap:
 *  - The matched filter is already frequency-domain (FFT(data) * conj(FFT(code))),
 *    so a descriptor-driven replica naturally PFB-folds and bin-splits later.
 *  - @c code_period_s drives the block size Ns = round(sample_rate * period);
 *    this replaces the GPS-L1 hard-coded 1 ms. Long-period signals (L2C CL =
 *    1.5 s) need time-assisted acquisition rather than a full-period search --
 *    a correlator concern, not a descriptor one.
 *  - For peeling/calibration only the replica + tracking are needed; the nav
 *    message decode (LNAV/CNAV/...) is a separate, optional product. Dataless
 *    *pilot* components (@c pilot=true) are fully known and the easiest to peel.
 */

#ifndef GNSS_SIGNAL_HPP
#define GNSS_SIGNAL_HPP

#include <string> // for string (name lookup)

namespace gnss {

/// Modulation type of the spreading waveform.
enum class Modulation {
    BPSK,  ///< plain bi-phase code (GPS L1 C/A, L2C, BeiDou B1I, ...)
    BOC,   ///< binary offset carrier: code x square-wave subcarrier (L1C, E1, ...)
};

/// Signal-specific parameters. Public, spec-derived constants only -- the
/// per-PRN code chips live in the matching ReplicaSource, not here.
struct SignalDescriptor {
    const char* name;        ///< e.g. "GPS_L1CA", "GPS_L2C_CL"
    double carrier_hz;       ///< nominal sky carrier
    double chip_rate_hz;     ///< combined (bandwidth-setting) chip rate
    long code_length;        ///< chips in one primary-code period
    double code_period_s;    ///< primary-code period -> sets Ns in the correlator

    Modulation mod;          ///< BPSK or BOC
    int boc_m;               ///< BOC subcarrier order (subcarrier = boc_m * 1.023 MHz); 0 if BPSK
    int boc_n;               ///< BOC code order (code = boc_n * 1.023 Mcps); 0 if BPSK

    bool pilot;              ///< dataless component (modulation fully known)?
    double nav_symbol_s;     ///< data symbol period (post-FEC); 0 for pilot
    int secondary_length;    ///< overlay/secondary code length in primary periods; 0 if none

    bool time_multiplexed;   ///< component chip-interleaved with a sibling (L2C CM/CL)
    int tdm_phase;           ///< which combined-chip parity carries this code (0=even, 1=odd);
                             ///< only meaningful when time_multiplexed
    bool time_assisted;      ///< period too long to search blind: correlate a short coherent
                             ///< window at a time-predicted code phase (L2C CL) rather than a
                             ///< full-period FFT search
    int prn_min, prn_max;    ///< valid PRN range
};

// ---- known descriptors -------------------------------------------------------
// Parameters are public spec values; only the ReplicaSource code tables are
// spec-table transcriptions (kept separate, per-signal, and unit-tested).

/// GPS L1 C/A (1575.42 MHz) -- the existing, validated signal. ReplicaSource:
/// gpsCACode (full IS-GPS-200 G2 tap table).
inline constexpr SignalDescriptor GPS_L1CA = {
    "GPS_L1CA", 1575.42e6, 1.023e6, 1023, 1e-3,
    Modulation::BPSK, 0, 0,
    /*pilot=*/false, /*nav_symbol_s=*/20e-3, /*secondary_length=*/0,
    /*time_multiplexed=*/false, /*tdm_phase=*/0, /*time_assisted=*/false, 1, 32,
};

/// GPS L1C-P (1575.42 MHz, modernized civil) -- the dataless *pilot*: 10230-chip Weil code at
/// 1.023 Mcps (10 ms), BOC(1,1). ReplicaSource: gpsL1CCode (Legendre + Weil + 7-chip insertion).
/// secondary_length 0 for now -- the 1800-symbol L1CO overlay generator is a follow-on.
inline constexpr SignalDescriptor GPS_L1C_P = {
    "GPS_L1C_P", 1575.42e6, 1.023e6, 10230, 10e-3,
    Modulation::BOC, 1, 1, // BOC(1,1)
    /*pilot=*/true, /*nav_symbol_s=*/0.0, /*secondary_length=*/0,
    /*time_multiplexed=*/false, /*tdm_phase=*/0, /*time_assisted=*/false, 1, 32,
};

/// GPS L2C CM (1227.6 MHz) -- the *data* component: 10230 chips at 511.5 kcps
/// (20 ms), chip-interleaved with CL to a 1.023 Mcps stream; carries CNAV
/// (25 bps + FEC -> 50 sps). ReplicaSource: gpsL2CCode (27-stage LFSR + CM
/// initial-state table) -- TO POPULATE from IS-GPS-200.
inline constexpr SignalDescriptor GPS_L2C_CM = {
    "GPS_L2C_CM", 1227.6e6, 511.5e3, 10230, 20e-3,
    Modulation::BPSK, 0, 0,
    /*pilot=*/false, /*nav_symbol_s=*/20e-3, /*secondary_length=*/0,
    /*time_multiplexed=*/true, /*tdm_phase=*/0, /*time_assisted=*/false, 1, 32,  // CM on even
};

/// GPS L2C CL (1227.6 MHz) -- the dataless *pilot*: 767250 chips at 511.5 kcps
/// (1.5 s). Best target for peeling (fully known modulation). ReplicaSource:
/// gpsL2CCode (27-stage LFSR + CL initial-state table) -- TO POPULATE from
/// IS-GPS-200. Acquisition is time-assisted (the 1.5 s period is too long to
/// search blind).
inline constexpr SignalDescriptor GPS_L2C_CL = {
    "GPS_L2C_CL", 1227.6e6, 511.5e3, 767250, 1.5,
    Modulation::BPSK, 0, 0,
    /*pilot=*/true, /*nav_symbol_s=*/0.0, /*secondary_length=*/0,
    /*time_multiplexed=*/true, /*tdm_phase=*/1, /*time_assisted=*/true, 1, 32,  // CL on odd
};

/// GPS L5 in-phase (1176.45 MHz) -- the *data* component (CNAV): 10230 chips at
/// 10.23 Mcps (1 ms), with a 10-chip Neuman-Hofman overlay (NH10, 10 ms).
/// ReplicaSource: gpsL5Code (XA x per-PRN-shifted XB).
inline constexpr SignalDescriptor GPS_L5_I = {
    "GPS_L5_I", 1176.45e6, 10.23e6, 10230, 1e-3,
    Modulation::BPSK, 0, 0,
    /*pilot=*/false, /*nav_symbol_s=*/20e-3, /*secondary_length=*/10,
    /*time_multiplexed=*/false, /*tdm_phase=*/0, /*time_assisted=*/false, 1, 32,
};

/// GPS L5 quadrature (1176.45 MHz) -- the dataless *pilot*: same 1 ms primary
/// code (different XB advance) with a 20-chip Neuman-Hofman overlay (NH20,
/// 20 ms). Best L5 target for peeling (fully known modulation).
inline constexpr SignalDescriptor GPS_L5_Q = {
    "GPS_L5_Q", 1176.45e6, 10.23e6, 10230, 1e-3,
    Modulation::BPSK, 0, 0,
    /*pilot=*/true, /*nav_symbol_s=*/0.0, /*secondary_length=*/20,
    /*time_multiplexed=*/false, /*tdm_phase=*/0, /*time_assisted=*/false, 1, 32,
};

/// Look up a descriptor by its @c name (config string). Returns nullptr if
/// unknown. The full transmitted L2C signal is CM and CL combined; the two
/// descriptors let the correlator target either the data (CM) or the dataless
/// pilot (CL) component. Likewise L5 splits into I5 (data) and Q5 (pilot).
inline const SignalDescriptor* signal_by_name(const std::string& name) {
    for (const SignalDescriptor* s :
         {&GPS_L1CA, &GPS_L1C_P, &GPS_L2C_CM, &GPS_L2C_CL, &GPS_L5_I, &GPS_L5_Q})
        if (name == s->name)
            return s;
    return nullptr;
}

} // namespace gnss

#endif // GNSS_SIGNAL_HPP
