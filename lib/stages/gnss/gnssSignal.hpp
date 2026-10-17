/**
 * @file
 * @brief GNSS signal descriptors: the per-signal constants the replica, search and tracking
 *        code read.
 *
 * A @ref SignalDescriptor holds what differs between signals (carrier, chip rate, code length
 * and period, modulation, secondary code) so the DSP core stays generic. The
 * per-PRN code chips come from each signal's code generator (gpsCACode, galileoE5aCode, ...),
 * selected by descriptor name in signal_code() (gnssChannelizedReplica.cpp).
 */

#ifndef GNSS_SIGNAL_HPP
#define GNSS_SIGNAL_HPP

#include <string> // for string (name lookup)

namespace gnss {

/// Modulation type of the spreading waveform.
enum class Modulation {
    BPSK, ///< plain bi-phase code (GPS L1 C/A, L2C, BeiDou B1I, ...)
    BOC,  ///< binary offset carrier: code x square-wave subcarrier (L1C, E1, ...)
};

/// Signal-specific parameters: public spec constants only. The per-PRN code chips come from the
/// signal's code generator, not from here; ChannelizedReplicaBank (the bank) expands them for TDM
/// and BOC signals.
struct SignalDescriptor {
    const char* name;     ///< e.g. "GPS_L1CA", "GPS_L2C_CL"
    double carrier_hz;    ///< nominal sky carrier
    double chip_rate_hz;  ///< chip rate of the code table, before the bank's TDM/BOC expansion
    long code_length;     ///< chips in one code period, before the bank's TDM/BOC expansion
    double code_period_s; ///< code period

    Modulation mod; ///< BPSK or BOC
    int boc_m;      ///< BOC subcarrier order (subcarrier = boc_m * 1.023 MHz); 0 if BPSK
    int boc_n;      ///< BOC code order (code = boc_n * 1.023 Mcps); 0 if BPSK

    int secondary_length; ///< secondary (overlay) length in primary periods; 0 if none or tiled
                          ///< into the code. The bank applies a shared overlay only for signals
                          ///< it lists by name; per-PRN overlays are tiled into the *_CS codes.
                          ///< overlay_registry_check() compares the combiner's gnssOverlay.hpp
                          ///< rows (shared overlays under a data component) against this field.

    bool time_multiplexed; ///< component chip-interleaved with a sibling (L2C CM/CL)
    int tdm_phase;         ///< which combined-chip parity carries this code (0=even, 1=odd);
                           ///< only meaningful when time_multiplexed
};

// ---- Known descriptors ----
// Descriptors suffixed _NH or _CS have the secondary code tiled into the code table: the code
// period becomes the overlay period, secondary_length is 0, and a code phase includes the
// overlay phase. Trackers use them because a CHORD record (2048 hops, 10.49 ms) spans several
// primary periods, which add coherently only with the overlay signs applied; a record then
// straddles at most one code-period boundary, as the P_HEAD row assumes. The search uses the
// plain descriptor.
//
// gnss_broker/signals.py reads these initializers by regex: keep the name and the next four
// fields as plain literals in this order, and keep comments inside an initializer short
// (signals.py looks for secondary_length within 400 characters of the code period).

/// GPS L1 C/A (1575.42 MHz). Code: gpsCACode (IS-GPS-200 G2 tap table).
inline constexpr SignalDescriptor GPS_L1CA = {
    "GPS_L1CA",
    1575.42e6,
    1.023e6,
    1023,
    1e-3,
    Modulation::BPSK,
    0,
    0,
    /*secondary_length=*/0,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// GPS L1C-P (1575.42 MHz), the dataless pilot: 10230-chip Weil code at 1.023 Mcps (10 ms),
/// BOC(1,1). The per-PRN 1800-chip L1CO overlay (18 s, generate_l1co_code) is not applied: no
/// tiled descriptor carries it. Code: gpsL1CCode (Legendre + Weil + 7-chip insertion).
inline constexpr SignalDescriptor GPS_L1C_P = {
    "GPS_L1C_P",
    1575.42e6,
    1.023e6,
    10230,
    10e-3,
    Modulation::BOC,
    1,
    1, // BOC(1,1)
    /*secondary_length=*/1800,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// GPS L1C-D (1575.42 MHz), the L1C data channel carrying CNAV-2: same 10230-chip BOC(1,1)
/// primary at 10 ms as L1C-P (its own Weil code), no secondary of its own. CNAV-2 is 50 bps
/// through rate-1/2 LDPC = 100 sps, one symbol per code period. Code-synchronous with L1C-P.
/// Decoder: gps_cnav2.py. Code: gpsL1CCode generate_l1cd_code().
inline constexpr SignalDescriptor GPS_L1C_D = {
    "GPS_L1C_D",
    1575.42e6,
    1.023e6,
    10230,
    10e-3,
    Modulation::BOC,
    1,
    1, // BOC(1,1)
    /*secondary_length=*/0,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// GPS L2C CM (1227.6 MHz), the data component: 10230 chips at 511.5 kcps (20 ms),
/// interleaved with CL on the even chips of a 1.023 Mcps stream; carries CNAV (25 bps through
/// rate-1/2 FEC = 50 sps). Code: gpsL2CCode (27-stage LFSR, CM initial states).
inline constexpr SignalDescriptor GPS_L2C_CM = {
    "GPS_L2C_CM",
    1227.6e6,
    511.5e3,
    10230,
    20e-3,
    Modulation::BPSK,
    0,
    0,
    /*secondary_length=*/0,
    /*time_multiplexed=*/true,
    /*tdm_phase=*/0,
};

/// GPS L2C CL (1227.6 MHz), the dataless pilot: 767250 chips at 511.5 kcps (1.5 s) on the odd
/// chips of the interleaved stream. Acquisition is time-assisted: the 1.5 s period is too long
/// to search blind. Code: gpsL2CCode (27-stage LFSR, CL initial states).
inline constexpr SignalDescriptor GPS_L2C_CL = {
    "GPS_L2C_CL",
    1227.6e6,
    511.5e3,
    767250,
    1.5,
    Modulation::BPSK,
    0,
    0,
    /*secondary_length=*/0,
    /*time_multiplexed=*/true,
    /*tdm_phase=*/1,
};

/// GPS L5 I5 (1176.45 MHz), the data component (CNAV): 10230 chips at 10.23 Mcps (1 ms), with
/// the shared NH10 overlay (10 ms). Code: gpsL5Code (XA x per-PRN-shifted XB).
inline constexpr SignalDescriptor GPS_L5_I = {
    "GPS_L5_I",
    1176.45e6,
    10.23e6,
    10230,
    1e-3,
    Modulation::BPSK,
    0,
    0,
    // L5 CNAV is 50 bps through rate-1/2 FEC = 100 sps: a 10 ms symbol, exactly one NH10
    // period. A nav-bit wipe over 20 ms would straddle two symbols.
    /*secondary_length=*/10,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// GPS L5 Q5 (1176.45 MHz), the dataless pilot: the same 1 ms primary construction as I5
/// (a different XB advance) with the shared NH20 overlay (20 ms).
inline constexpr SignalDescriptor GPS_L5_Q = {
    "GPS_L5_Q",
    1176.45e6,
    10.23e6,
    10230,
    1e-3,
    Modulation::BPSK,
    0,
    0,
    /*secondary_length=*/20,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// GPS L5 Q5 with NH20 tiled into the code: one 204600-chip, 20 ms code. Without the overlay in
/// the table, the ~10 NH20 chips a CHORD record spans largely cancel. The NH20 phase follows
/// from GPS time, since the overlay is locked to the satellite's code periods.
inline constexpr SignalDescriptor GPS_L5_Q_NH = {
    "GPS_L5_Q_NH",
    1176.45e6,
    10.23e6,
    204600,
    20e-3,
    Modulation::BPSK,
    0,
    0,
    /*secondary_length=*/0,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// Galileo E1-C (1575.42 MHz, the GPS L1 carrier), the OS dataless pilot: 4092-chip memory
/// code at 1.023 Mcps (4 ms). Transmitted CBOC(6,1,1/11) and modelled as BOC(1,1), which loses
/// ~0.4 dB. Shared CS25 secondary (100 ms). Code: galileoE1Code (ICD Annex C tables).
inline constexpr SignalDescriptor GAL_E1C = {
    "GAL_E1C",
    1575.42e6,
    1.023e6,
    4092,
    4e-3,
    Modulation::BOC,
    1,
    1, // BOC(1,1)
    /*secondary_length=*/25,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// Galileo E1-B (1575.42 MHz), the OS data component (I/NAV, 250 sps: one 4 ms symbol per code
/// period). Same geometry as E1-C, no secondary code.
inline constexpr SignalDescriptor GAL_E1B = {
    "GAL_E1B",
    1575.42e6,
    1.023e6,
    4092,
    4e-3,
    Modulation::BOC,
    1,
    1, // BOC(1,1)
    /*secondary_length=*/0,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// BeiDou-3 B1C pilot (1575.42 MHz, the GPS L1 carrier): 10230-chip Weil code at 1.023 Mcps
/// (10 ms). Transmitted QMBOC(6,1,4/33) and modelled as BOC(1,1), which loses the 4/33 of pilot
/// power in the quadrature BOC(6,1) term (~0.6 dB). The per-PRN 1800-chip overlay
/// (generate_b1cp_secondary) is not applied: no tiled descriptor carries it. BDS-3 only. Code:
/// beidouB1CCode (Legendre/Weil).
inline constexpr SignalDescriptor BDS_B1C_P = {
    "BDS_B1C_P",
    1575.42e6,
    1.023e6,
    10230,
    10e-3,
    Modulation::BOC,
    1,
    1, // BOC(1,1)
    /*secondary_length=*/1800,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// BeiDou-3 B1C data (1575.42 MHz), carrying B-CNAV1: same 10230-chip BOC(1,1) primary at
/// 10 ms (its own Weil tables), no secondary. B-CNAV1 is 50 bps through rate-1/2 LDPC = 100 sps,
/// one symbol per code period. Code-synchronous with B1C-P. Decoder: beidou_bcnav1.py.
inline constexpr SignalDescriptor BDS_B1C_D = {
    "BDS_B1C_D",
    1575.42e6,
    1.023e6,
    10230,
    10e-3,
    Modulation::BOC,
    1,
    1, // BOC(1,1)
    /*secondary_length=*/0,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// Galileo E5a-Q (1176.45 MHz, the GPS L5 carrier), the E5a dataless pilot: 10230-chip primary
/// at 10.23 Mcps (1 ms, two 14-stage LFSRs), per-PRN CS100 secondary (100 ms).
/// Code: galileoE5aCode (PocketSDR-sourced tables).
inline constexpr SignalDescriptor GAL_E5A_Q = {
    "GAL_E5A_Q",
    1176.45e6,
    10.23e6,
    10230,
    1e-3,
    Modulation::BPSK,
    0,
    0,
    /*secondary_length=*/100,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// Galileo E5a-Q with each PRN's CS100 tiled into the code: one 1023000-chip, 100 ms code that
/// differs per satellite. The CS100 phase follows from Galileo system time; the search cannot
/// find it, because per-PRN overlays are not in the bank's overlay slot, so nh_search on
/// GAL_E5A_Q has no effect. Trackers on this signal are seeded by the broker's dead reckoning
/// instead: broadcast ephemeris plus a receiver clock adopted from other chains.
inline constexpr SignalDescriptor GAL_E5A_Q_CS = {
    "GAL_E5A_Q_CS",
    1176.45e6,
    10.23e6,
    1023000,
    100e-3,
    Modulation::BPSK,
    0,
    0,
    /*secondary_length=*/0,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// Galileo E5a-I (1176.45 MHz), the E5a data channel carrying F/NAV: same 10230-chip primary at
/// 1 ms (its own X2 start table), shared CS20 secondary. F/NAV is 25 bps through rate-1/2 FEC =
/// 50 sps, so CS20 spans exactly one 20 ms symbol. Code-synchronous with E5a-Q.
/// Decoder: galileo_fnav.py.
inline constexpr SignalDescriptor GAL_E5A_I = {
    "GAL_E5A_I",
    1176.45e6,
    10.23e6,
    10230,
    1e-3,
    Modulation::BPSK,
    0,
    0,
    /*secondary_length=*/20,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// BeiDou-3 B2a pilot (1176.45 MHz): 10230-chip primary at 10.23 Mcps (1 ms, two 13-stage
/// LFSRs), per-PRN 100-chip Weil-1021 secondary (100 ms). BDS-3 only.
/// Code: beidouB2aCode (PocketSDR-sourced tables).
inline constexpr SignalDescriptor BDS_B2A_P = {
    "BDS_B2A_P",
    1176.45e6,
    10.23e6,
    10230,
    1e-3,
    Modulation::BPSK,
    0,
    0,
    /*secondary_length=*/100,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// BeiDou-3 B2a pilot with each PRN's secondary tiled into the code: one 1023000-chip, 100 ms
/// code, built and seeded as GAL_E5A_Q_CS; the secondary phase follows from BeiDou time.
inline constexpr SignalDescriptor BDS_B2A_P_CS = {
    "BDS_B2A_P_CS",
    1176.45e6,
    10.23e6,
    1023000,
    100e-3,
    Modulation::BPSK,
    0,
    0,
    /*secondary_length=*/0,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// BeiDou-3 B2a data (1176.45 MHz), carrying B-CNAV2: same 10230-chip primary at 1 ms (its own
/// generator polynomials), shared 5-chip secondary. B-CNAV2 is 100 bps through GF(64) LDPC =
/// 200 sps, so the secondary spans exactly one 5 ms symbol. Code-synchronous with B2a-P.
/// Decoder: beidou_bcnav2.py.
inline constexpr SignalDescriptor BDS_B2A_D = {
    "BDS_B2A_D",
    1176.45e6,
    10.23e6,
    10230,
    1e-3,
    Modulation::BPSK,
    0,
    0,
    /*secondary_length=*/5,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// BeiDou-3 B2b-I (1207.14 MHz), the open PPP-B2b data channel carrying B-CNAV3: 10230-chip
/// primary at 10.23 Mcps (1 ms) from two 13-stage LFSRs. There is no open pilot, so deep
/// integration needs a nav-bit wipe, as on GPS L1 C/A. B-CNAV3 is 1000 sps, one symbol per code
/// period, with no secondary. BDS-3 only. Code: beidouB2bCode (checked by b2b_code_check.py).
inline constexpr SignalDescriptor BDS_B2B_I = {
    "BDS_B2B_I",
    1207.14e6,
    10.23e6,
    10230,
    1e-3,
    Modulation::BPSK,
    0,
    0,
    /*secondary_length=*/0,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// Galileo E5b-Q (1207.14 MHz, the BeiDou B2b carrier), the E5b dataless pilot: 10230-chip
/// primary at 10.23 Mcps (1 ms, two 14-stage LFSRs with the E5b polynomials), per-PRN CS100
/// secondary (100 ms). Code: galileoE5bCode (checked by e5b_code_check.py).
inline constexpr SignalDescriptor GAL_E5B_Q = {
    "GAL_E5B_Q",
    1207.14e6,
    10.23e6,
    10230,
    1e-3,
    Modulation::BPSK,
    0,
    0,
    /*secondary_length=*/100,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// Galileo E5b-Q with each PRN's CS100 (galileo::e5bq_secondary) tiled into the code: one
/// 1023000-chip, 100 ms code, built and seeded as GAL_E5A_Q_CS.
inline constexpr SignalDescriptor GAL_E5B_Q_CS = {
    "GAL_E5B_Q_CS",
    1207.14e6,
    10.23e6,
    1023000,
    100e-3,
    Modulation::BPSK,
    0,
    0,
    /*secondary_length=*/0,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// Galileo E5b-I (1207.14 MHz), the E5b data channel carrying I/NAV (the E1-B message): same
/// 10230-chip primary at 1 ms (its own X2 start table), shared CS4 secondary. I/NAV is 125 bps
/// through rate-1/2 FEC = 250 sps, so CS4 spans exactly one 4 ms symbol. Code-synchronous with
/// E5b-Q. Decoder: galileo_inav.py.
inline constexpr SignalDescriptor GAL_E5B_I = {
    "GAL_E5B_I",
    1207.14e6,
    10.23e6,
    10230,
    1e-3,
    Modulation::BPSK,
    0,
    0,
    /*secondary_length=*/4,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// Galileo E6-C (1278.75 MHz), the E6 dataless pilot: 5115-chip memory code at 5.115 Mcps
/// (1 ms), BPSK(5), with a per-PRN 100-chip secondary (100 ms), unlike E1-C's shared CS25.
/// Code: galileoE6Code (E6-B/C Codes Technical Note tables; checked by e6_code_check.py).
inline constexpr SignalDescriptor GAL_E6_C = {
    "GAL_E6_C",
    1278.75e6,
    5.115e6,
    5115,
    1e-3,
    Modulation::BPSK,
    0,
    0,
    /*secondary_length=*/100,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// Galileo E6-C with each PRN's 100-chip secondary tiled into the code: one 511500-chip, 100 ms
/// code, built and seeded as GAL_E5A_Q_CS.
inline constexpr SignalDescriptor GAL_E6_C_CS = {
    "GAL_E6_C_CS",
    1278.75e6,
    5.115e6,
    511500,
    100e-3,
    Modulation::BPSK,
    0,
    0,
    /*secondary_length=*/0,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// Galileo E6-B (1278.75 MHz), the E6 data channel carrying the High Accuracy Service (HAS)
/// message at 1000 sps: one symbol per code period and no secondary (the 100-chip secondary is
/// E6-C's). Code-synchronous with E6-C.
inline constexpr SignalDescriptor GAL_E6_B = {
    "GAL_E6_B",
    1278.75e6,
    5.115e6,
    5115,
    1e-3,
    Modulation::BPSK,
    0,
    0,
    /*secondary_length=*/0,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// BeiDou B1I (1561.098 MHz), broadcast by BDS-2 and BDS-3: 2046-chip Gold code at 2.046 Mcps
/// (1 ms), BPSK(2). Data only, so it is acquired directly and nav-wiped, as GPS L1 C/A. The
/// 50 bps D1 message is spread by a shared 20-chip NH secondary, one chip per code period, so a
/// symbol is 20 ms. Code: beidouB1ICode (checked by b1i_b3i_code_check.py).
inline constexpr SignalDescriptor BDS_B1I = {
    "BDS_B1I",
    1561.098e6,
    2.046e6,
    2046,
    1e-3,
    Modulation::BPSK,
    0,
    0,
    /*secondary_length=*/20,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// BeiDou B3I (1268.52 MHz), carrying the same D1 message as B1I: 10230-chip code at
/// 10.23 Mcps (1 ms), BPSK(10), from two 13-stage LFSRs with G1 reset to all ones as the ICD
/// specifies. Data only, with B1I's shared NH20 and 20 ms symbol. Code: beidouB3ICode.
inline constexpr SignalDescriptor BDS_B3I = {
    "BDS_B3I",
    1268.52e6,
    10.23e6,
    10230,
    1e-3,
    Modulation::BPSK,
    0,
    0,
    /*secondary_length=*/20,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// BeiDou B3I with the shared NH20 tiled into the code: one 204600-chip, 20 ms code, built as
/// GPS_L5_Q_NH. What remains is the D1 nav sign, constant over each 20 ms code period.
inline constexpr SignalDescriptor BDS_B3I_NH = {
    "BDS_B3I_NH",
    1268.52e6,
    10.23e6,
    204600,
    20e-3,
    Modulation::BPSK,
    0,
    0,
    /*secondary_length=*/0,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// BeiDou B2I (1207.14 MHz, the B2b and Galileo E5b centre), BDS-2's legacy second signal. It
/// uses B1I's ranging code (the ICD defines one generator for C_B1I and C_B2I), so only the
/// carrier and the broadcasting satellites differ: same 2046 chips at 2.046 Mcps (1 ms), D1
/// message and 20-chip NH. BDS-2 only; BDS-3 broadcasts B2a and B2b instead.
inline constexpr SignalDescriptor BDS_B2I = {
    "BDS_B2I",
    1207.14e6,
    2.046e6,
    2046,
    1e-3,
    Modulation::BPSK,
    0,
    0,
    /*secondary_length=*/20,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// GLONASS L3OC-p (1202.025 MHz), the dataless pilot of GLONASS's CDMA signal: 10230-chip code
/// at 10.23 Mcps (1 ms), BPSK(10), the same geometry as BDS_B2B_I and BDS_B3I. Shared NH10
/// secondary (10 ms). GLONASS-K satellites only. The PRN is the code index (1..63), which is not
/// assumed to equal the orbital slot (see glonassL3OCCode.hpp). Code: glonassL3OCCode.
inline constexpr SignalDescriptor GLO_L3OC_P = {
    "GLO_L3OC_P",
    1202.025e6,
    10.23e6,
    10230,
    1e-3,
    Modulation::BPSK,
    0,
    0,
    /*secondary_length=*/10,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// GLONASS L3OC-p with the shared NH10 tiled into the code: one 102300-chip, 10 ms code, built
/// as GPS_L5_Q_NH and BDS_B3I_NH (gnssChannelizedReplica.cpp), so a 10.5 ms record despreads
/// coherently instead of summing NH chips that largely cancel (NH10 sums to +2 of 10). The pilot
/// is dataless, so nothing remains above the 10 ms period. The tracker code of the glo_l3oc
/// chain; the broker pairs it with GLO_L3OC_P (gnss_broker/signals.py).
inline constexpr SignalDescriptor GLO_L3OC_P_NH = {
    "GLO_L3OC_P_NH",
    1202.025e6,
    10.23e6,
    102300,
    10e-3,
    Modulation::BPSK,
    0,
    0,
    /*secondary_length=*/0,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// GLONASS L3OC-d (1202.025 MHz), the data component beside L3OC-p: the CDMA navigation message
/// at 100 bps through rate-1/2 coding = 200 sps, spread by the shared 5-chip Barker secondary,
/// so a symbol is 5 ms (the BDS_B2A_D shape). Code-synchronous with L3OC-p.
inline constexpr SignalDescriptor GLO_L3OC_D = {
    "GLO_L3OC_D",
    1202.025e6,
    10.23e6,
    10230,
    1e-3,
    Modulation::BPSK,
    0,
    0,
    /*secondary_length=*/5,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// GLONASS L2OF (~1246 MHz), the legacy FDMA open signal and the only signal here whose
/// satellites are separated by frequency rather than code: one 511-chip m-sequence at 511 kcps
/// (1 ms), BPSK, shared by every satellite. Frequency channel k sits at 1246.0 + k*0.4375 MHz,
/// k = -7..+6 (a 5.69 MHz comb), so @c carrier_hz is the k = 0 centre and each satellite's
/// offset is set with ChannelizedReplicaBank::set_prn_freq_offsets() (GnssChannelizedSearch
/// config prn_freq_offset_hz). k is reassigned as satellites are replaced, so it must come from
/// broadcast ephemeris (read by gnss_ephemeris.glonass_freq_channels()), never a hardcoded table.
/// Data only: 50 bps Manchester-coded to 100 sps, so each 20 ms bit is a +-/-+ pair of 10 ms
/// symbols. There is no secondary; a wipe at the 10 ms symbol also removes the Manchester
/// structure. The PRN is the orbital slot (1..32); since the code is shared, only the carrier
/// offset depends on it. Code: glonassCACode (checked by glonass_ca_code_check.py).
inline constexpr SignalDescriptor GLO_L2OF = {
    "GLO_L2OF",
    1246.0e6,
    511e3,
    511,
    1e-3,
    Modulation::BPSK,
    0,
    0,
    /*secondary_length=*/0,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// GLONASS L2OC-p (1248.06 MHz), the modernised CDMA pilot, 2.06 MHz above the L2OF k = 0
/// centre; GLONASS-K satellites only. Its base ranging code is L3OC-p's (glonassL2OCCode expands
/// generate_l3ocp_code()): each base chip becomes [0, 0, +chip, -chip], BOC(1,1) plus data/pilot
/// TDM, so the transmitted code is 40920 chips at 2.046 Mcps (20 ms) with half the chips zero.
/// The expansion is in the code table, so this is a plain BPSK descriptor at the expanded rate.
/// Shared 50-chip OC2 secondary, one chip per 20 ms period (1 s). Acquired by a blind
/// full-period search, like L2C CM. Code: glonassL2OCCode (checked by l2oc_code_check.py).
inline constexpr SignalDescriptor GLO_L2OC_P = {
    "GLO_L2OC_P",
    1248.06e6,
    2.046e6,
    40920,
    20e-3,
    Modulation::BPSK,
    0,
    0,
    /*secondary_length=*/50,
    /*time_multiplexed=*/false,
    /*tdm_phase=*/0,
};

/// Look up a descriptor by its @c name (the config string); nullptr if unknown.
inline const SignalDescriptor* signal_by_name(const std::string& name) {
    for (const SignalDescriptor* s :
         {&GPS_L1CA,     &GPS_L1C_P,  &GPS_L2C_CM,   &GPS_L2C_CL,   &GPS_L5_I,     &GPS_L5_Q,
          &GPS_L5_Q_NH,  &GAL_E1C,    &GAL_E1B,      &BDS_B1C_P,    &BDS_B1C_D,    &GAL_E5A_Q,
          &GAL_E5A_Q_CS, &GAL_E5A_I,  &GAL_E5B_Q_CS, &BDS_B2A_P,    &BDS_B2A_P_CS, &BDS_B2A_D,
          &BDS_B2B_I,    &GAL_E5B_Q,  &GAL_E5B_I,    &GPS_L1C_D,    &GAL_E6_C,     &GAL_E6_B,
          &BDS_B1I,      &BDS_B3I,    &BDS_B3I_NH,   &BDS_B2I,      &GLO_L3OC_P,   &GLO_L3OC_D,
          &GLO_L2OF,     &GLO_L2OC_P, &GAL_E6_C_CS,  &GLO_L3OC_P_NH})
        if (name == s->name)
            return s;
    return nullptr;
}

} // namespace gnss

#endif // GNSS_SIGNAL_HPP
