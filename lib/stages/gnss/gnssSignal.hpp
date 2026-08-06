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
    int secondary_length;    ///< overlay/secondary code length in primary periods; 0 if none.
                             ///< DOCUMENTATION: the runtime overlay chips come from the
                             ///< gnssOverlay.hpp registry (which cross-checks this field at
                             ///< combiner init); consumers must not branch on it for per-PRN
                             ///< overlays (the bank's single-sequence slot can't carry them)

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
/// The 1800-symbol PER-PRN L1CO overlay (18 s) is wiped by the combiner via the
/// gnssOverlay.hpp registry ("L1CO"); secondary_length documents its length only.
inline constexpr SignalDescriptor GPS_L1C_P = {
    "GPS_L1C_P", 1575.42e6, 1.023e6, 10230, 10e-3,
    Modulation::BOC, 1, 1, // BOC(1,1)
    /*pilot=*/true, /*nav_symbol_s=*/0.0, /*secondary_length=*/1800,
    /*time_multiplexed=*/false, /*tdm_phase=*/0, /*time_assisted=*/false, 1, 32,
};

/// GPS L1C-D (1575.42 MHz) -- the L1C DATA channel carrying CNAV-2: same 10230-chip BOC(1,1)
/// primary at 10 ms as L1C-P (its own Weil code), NO overlay of its own (the 1800-symbol overlay
/// is the pilot's). CNAV-2 is 50 bps through rate-1/2 LDPC = 100 sps, so a symbol is 10 ms = one
/// code period -> navwipe_bit_records 1 (the B1C-D / L5-I discipline). DERIVED from the L1C-P
/// pilot; decoded by gps_cnav2 (binary LDPC, 18 s frame). ReplicaSource: gpsL1CCode::generate_l1cd.
inline constexpr SignalDescriptor GPS_L1C_D = {
    "GPS_L1C_D", 1575.42e6, 1.023e6, 10230, 10e-3,
    Modulation::BOC, 1, 1, // BOC(1,1)
    /*pilot=*/false, /*nav_symbol_s=*/10e-3, /*secondary_length=*/0,
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
    // 10 ms, NOT 20 (corrected 2026-07-29). L5 CNAV is 50 bps through the rate-1/2 FEC =
    // 100 sps, so a post-FEC SYMBOL is 10 ms = ten 1 ms I5 code periods. L2C is 25 bps ->
    // 50 sps -> 20 ms, which is where the 20e-3 came from. The giveaway is on this very line:
    // secondary_length 10 means NH10 spans 10 ms, and NH10 covers exactly ONE symbol by
    // construction -- a 20 ms symbol beside a 10 ms overlay was internally inconsistent.
    // Documentary field (no logic reads it), but the combiner's navwipe_bit_records is set
    // from this understanding, and at 20 the wipe straddles TWO symbols whose signs differ
    // half the time -- which is what blocked the L5 CNAV decode (g2 never locked).
    /*pilot=*/false, /*nav_symbol_s=*/10e-3, /*secondary_length=*/10,
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

/// GPS L5 Q5 with the NH20 secondary BAKED INTO the code table: the 10230-chip primary tiled
/// 20x with the overlay signs, one 204600-chip / 20 ms "code". FOR TRACKERS whose records
/// span multiple primary periods (CHORD: 2048-hop = 10.5 ms records = ~10 NH20 chips, whose
/// +-1 partial sums mostly CANCEL -- NH20 is designed to cancel; measured 2026-07-31, a
/// snr-40 satellite invisible per record). With the overlay in the table the record is
/// coherent by construction, the code-period boundary is the 20 ms overlay boundary (so a
/// record straddles at most ONE -- the P_HEAD design assumption becomes true), and the seed's
/// cp lives mod 204600 with the NH phase implied (computable exactly from GPS time: the
/// overlay is locked to the SV's code periods). secondary_length=0: nothing left to overlay.
/// The SEARCH keeps GPS_L5_Q + its nh_search; only trackers switch.
inline constexpr SignalDescriptor GPS_L5_Q_NH = {
    "GPS_L5_Q_NH", 1176.45e6, 10.23e6, 204600, 20e-3,
    Modulation::BPSK, 0, 0,
    /*pilot=*/true, /*nav_symbol_s=*/0.0, /*secondary_length=*/0,
    /*time_multiplexed=*/false, /*tdm_phase=*/0, /*time_assisted=*/false, 1, 32,
};

/// Galileo E1-C (1575.42 MHz) -- the OS dataless *pilot*: 4092-chip memory code at 1.023 Mcps
/// (4 ms), BOC(1,1) (the CBOC(6,1,1/11) high-frequency component is ~1/11 of the power;
/// modeling BOC(1,1) costs ~0.4 dB -- standard practice). 25-chip CS25 secondary (100 ms).
/// SAME sky carrier as GPS L1 -- one airspy tune covers both constellations.
/// ReplicaSource: galileoE1Code (ICD Annex C memory-code tables).
inline constexpr SignalDescriptor GAL_E1C = {
    "GAL_E1C", 1575.42e6, 1.023e6, 4092, 4e-3,
    Modulation::BOC, 1, 1, // BOC(1,1)
    /*pilot=*/true, /*nav_symbol_s=*/0.0, /*secondary_length=*/25,
    /*time_multiplexed=*/false, /*tdm_phase=*/0, /*time_assisted=*/false, 1, 50,
};

/// Galileo E1-B (1575.42 MHz) -- the OS *data* component (I/NAV, 250 sps = 4 ms symbols,
/// one symbol per primary period). Same geometry as E1-C, no secondary code.
inline constexpr SignalDescriptor GAL_E1B = {
    "GAL_E1B", 1575.42e6, 1.023e6, 4092, 4e-3,
    Modulation::BOC, 1, 1, // BOC(1,1)
    /*pilot=*/false, /*nav_symbol_s=*/4e-3, /*secondary_length=*/0,
    /*time_multiplexed=*/false, /*tdm_phase=*/0, /*time_assisted=*/false, 1, 50,
};

/// BeiDou-3 B1C pilot (1575.42 MHz -- the SAME carrier as GPS L1 / Galileo E1): 10230-chip
/// Weil code at 1.023 Mcps (10 ms), transmitted QMBOC(6,1,4/33), modeled BOC(1,1) (the
/// 4/33-power BOC(6,1) part is in quadrature -- negligible loss). Dataless pilot; the
/// 1800-chip PER-PRN overlay is wiped by the combiner via the gnssOverlay.hpp registry
/// ("B1C"); secondary_length documents its length only (the bank's single-sequence overlay
/// slot can't carry per-PRN codes, and its name ladder deliberately skips this signal).
/// ReplicaSource: beidouB1CCode (Legendre/Weil, algorithmic). BDS-3 only, PRN 19..63 mostly.
inline constexpr SignalDescriptor BDS_B1C_P = {
    "BDS_B1C_P", 1575.42e6, 1.023e6, 10230, 10e-3,
    Modulation::BOC, 1, 1, // BOC(1,1)
    /*pilot=*/true, /*nav_symbol_s=*/0.0, /*secondary_length=*/1800,
    /*time_multiplexed=*/false, /*tdm_phase=*/0, /*time_assisted=*/false, 1, 63,
};

/// BeiDou-3 B1C DATA (1575.42 MHz) -- the B1C data channel carrying B-CNAV1: same 10230-chip
/// BOC(1,1) primary at 10 ms (its own Weil tables), NO secondary (like E1B). B-CNAV1 is 50 bps
/// through the rate-1/2 LDPC = 100 sps, so a symbol is 10 ms = one code period ->
/// navwipe_bit_records 1, no overlay. The 4th/LAST S5 D-component; DERIVED from the B1C-P
/// pilot, decoded by beidou_bcnav1 (its 18 s frame is SF1 + block-interleaved SF2/SF3, GF(64)).
inline constexpr SignalDescriptor BDS_B1C_D = {
    "BDS_B1C_D", 1575.42e6, 1.023e6, 10230, 10e-3,
    Modulation::BOC, 1, 1, // BOC(1,1)
    /*pilot=*/false, /*nav_symbol_s=*/10e-3, /*secondary_length=*/0,
    /*time_multiplexed=*/false, /*tdm_phase=*/0, /*time_assisted=*/false, 1, 63,
};

/// Galileo E5a-Q (1176.45 MHz) -- the E5a dataless *pilot*: 10230-chip primary at
/// 10.23 Mcps (1 ms, two 14-stage LFSRs), per-PRN 100-chip CS100 secondary (100 ms).
/// SAME sky carrier as GPS L5 -- one airspy tune covers GPS/Galileo/BeiDou here.
/// ReplicaSource: galileoE5aCode (PocketSDR-sourced tables, sky-validated 2026-07-15).
inline constexpr SignalDescriptor GAL_E5A_Q = {
    "GAL_E5A_Q", 1176.45e6, 10.23e6, 10230, 1e-3,
    Modulation::BPSK, 0, 0,
    /*pilot=*/true, /*nav_symbol_s=*/0.0, /*secondary_length=*/100,
    /*time_multiplexed=*/false, /*tdm_phase=*/0, /*time_assisted=*/false, 1, 50,
};

/// Galileo E5a-I (1176.45 MHz) -- the E5a DATA channel carrying F/NAV: same 10230-chip
/// primary at 1 ms (its own X2 start table), a 20-chip CS20 secondary shared by all sats.
/// F/NAV is 25 bps through the rate-1/2 FEC = 50 sps, so a symbol is 20 ms = twenty 1 ms
/// code periods, and CS20 covers exactly ONE symbol (same discipline as GPS_L5_I / NH10).
/// The 2nd S5 D-component; DERIVED from the E5a-Q pilot (no search), decoded by galileo_fnav.
inline constexpr SignalDescriptor GAL_E5A_I = {
    "GAL_E5A_I", 1176.45e6, 10.23e6, 10230, 1e-3,
    Modulation::BPSK, 0, 0,
    /*pilot=*/false, /*nav_symbol_s=*/20e-3, /*secondary_length=*/20,
    /*time_multiplexed=*/false, /*tdm_phase=*/0, /*time_assisted=*/false, 1, 50,
};

/// BeiDou-3 B2a-pilot (1176.45 MHz) -- dataless: 10230-chip primary at 10.23 Mcps
/// (1 ms, two 13-stage LFSRs), per-PRN 100-chip Weil-1021 secondary (100 ms).
/// BDS-3 only (B2a not on BDS-2 -- same capability gate as B1C).
/// ReplicaSource: beidouB2aCode (PocketSDR-sourced tables, sky-validated 2026-07-15).
inline constexpr SignalDescriptor BDS_B2A_P = {
    "BDS_B2A_P", 1176.45e6, 10.23e6, 10230, 1e-3,
    Modulation::BPSK, 0, 0,
    /*pilot=*/true, /*nav_symbol_s=*/0.0, /*secondary_length=*/100,
    /*time_multiplexed=*/false, /*tdm_phase=*/0, /*time_assisted=*/false, 1, 63,
};

/// BeiDou-3 B2a DATA (1176.45 MHz) -- the B2a data channel carrying B-CNAV2: same 10230-chip
/// primary at 1 ms (its own generator polys), a 5-chip secondary shared by all sats. B-CNAV2 is
/// 100 sps through the NON-BINARY LDPC(GF64) = 200 sps, so a symbol is 5 ms = five 1 ms code
/// periods, and the 5-chip secondary covers exactly ONE symbol (the GPS_L5_I / GAL_E5A_I
/// discipline). The 3rd S5 D-component; DERIVED from the B2a-P pilot, decoded by beidou_bcnav2.
inline constexpr SignalDescriptor BDS_B2A_D = {
    "BDS_B2A_D", 1176.45e6, 10.23e6, 10230, 1e-3,
    Modulation::BPSK, 0, 0,
    /*pilot=*/false, /*nav_symbol_s=*/5e-3, /*secondary_length=*/5,
    /*time_multiplexed=*/false, /*tdm_phase=*/0, /*time_assisted=*/false, 1, 63,
};

/// BeiDou-3 B2b-I (1207.14 MHz) -- the OPEN PPP-B2b DATA channel carrying B-CNAV3: 10230-chip
/// primary at 10.23 Mcps (1 ms) from two 13-stage LFSRs (its own G1 tap + per-PRN G2 start
/// table). DATA-only: unlike B2a/B1C there is no dataless pilot in the open service, so deep
/// integration needs nav-bit wipe (like GPS L1 C/A), not a pilot peel. B-CNAV3 runs at 1000 sps
/// = exactly one symbol per 1 ms code period, so there is NO secondary overlay (navwipe_bit_
/// records 1). SEPARATE carrier from GPS L2C (1227.6) -- one airspy tune covers B2b alone.
/// ReplicaSource: beidouB2bCode (PocketSDR-sourced tables, bit-exact-verified; b2b_code_check.py).
/// BDS-3 only (PPP-B2b is a BDS-3 service).
inline constexpr SignalDescriptor BDS_B2B_I = {
    "BDS_B2B_I", 1207.14e6, 10.23e6, 10230, 1e-3,
    Modulation::BPSK, 0, 0,
    /*pilot=*/false, /*nav_symbol_s=*/1e-3, /*secondary_length=*/0,
    /*time_multiplexed=*/false, /*tdm_phase=*/0, /*time_assisted=*/false, 1, 63,
};

/// Galileo E5b-Q (1207.14 MHz) -- the E5b dataless *pilot*: 10230-chip primary at 10.23 Mcps
/// (1 ms, two 14-stage LFSRs; E5b register polys, distinct from E5a), per-PRN 100-chip CS100
/// secondary (100 ms). SAME sky carrier as BeiDou B2b -- one mid-band airspy tune covers both.
/// ReplicaSource: galileoE5bCode (PocketSDR-sourced tables, bit-exact-verified; e5b_code_check.py).
inline constexpr SignalDescriptor GAL_E5B_Q = {
    "GAL_E5B_Q", 1207.14e6, 10.23e6, 10230, 1e-3,
    Modulation::BPSK, 0, 0,
    /*pilot=*/true, /*nav_symbol_s=*/0.0, /*secondary_length=*/100,
    /*time_multiplexed=*/false, /*tdm_phase=*/0, /*time_assisted=*/false, 1, 50,
};

/// Galileo E5b-I (1207.14 MHz) -- the E5b DATA channel carrying I/NAV (the SAME message as
/// E1-B): same 10230-chip primary at 1 ms (its own X2 start table), a 4-chip CS4 secondary
/// shared by all sats. I/NAV is 125 bps through the rate-1/2 FEC = 250 sps, so a symbol is
/// 4 ms = four 1 ms code periods, and CS4 covers exactly ONE symbol (the GAL_E5A_I / GPS_L5_I
/// discipline). A future E5b D-component; DERIVED from the E5b-Q pilot, decoded by galileo_inav.
inline constexpr SignalDescriptor GAL_E5B_I = {
    "GAL_E5B_I", 1207.14e6, 10.23e6, 10230, 1e-3,
    Modulation::BPSK, 0, 0,
    /*pilot=*/false, /*nav_symbol_s=*/4e-3, /*secondary_length=*/4,
    /*time_multiplexed=*/false, /*tdm_phase=*/0, /*time_assisted=*/false, 1, 50,
};

/// Galileo E6-C (1278.75 MHz) -- the E6 dataless *pilot*: 5115-chip MEMORY code at 5.115 Mcps
/// (1 ms), BPSK(5), with a PER-PRN 100-chip secondary (100 ms). ⚠️ per-PRN, unlike E1-C's
/// SHARED CS25 -- E6 follows the E5a-Q / E5b-Q pattern instead. Its own front-end tune
/// (1278.75); B3I at 1268.52 is 10.23 MHz away and does NOT fit the same 10 MHz window.
/// ReplicaSource: galileoE6Code (EU E6-B/C Codes Technical Note tables; e6_code_check.py).
inline constexpr SignalDescriptor GAL_E6_C = {
    "GAL_E6_C", 1278.75e6, 5.115e6, 5115, 1e-3,
    Modulation::BPSK, 0, 0,
    /*pilot=*/true, /*nav_symbol_s=*/0.0, /*secondary_length=*/100,
    /*time_multiplexed=*/false, /*tdm_phase=*/0, /*time_assisted=*/false, 1, 50,
};

/// Galileo E6-B (1278.75 MHz) -- the E6 DATA channel carrying the High Accuracy Service (HAS)
/// message at 1000 sps, so a symbol is 1 ms = EXACTLY ONE code period -> navwipe_bit_records 1
/// and NO secondary of its own (the 100-chip secondary is the E6-C PILOT's). HAS has been
/// operational since Jan 2023, so unlike GPS L1C-D there IS a live message here -- confirm it
/// with the decoder's `trans` (symbol transition rate) before building a page decoder.
/// DERIVED from the E6-C pilot (no search, seeded verbatim), the B1C-D / L1C-D discipline.
inline constexpr SignalDescriptor GAL_E6_B = {
    "GAL_E6_B", 1278.75e6, 5.115e6, 5115, 1e-3,
    Modulation::BPSK, 0, 0,
    /*pilot=*/false, /*nav_symbol_s=*/1e-3, /*secondary_length=*/0,
    /*time_multiplexed=*/false, /*tdm_phase=*/0, /*time_assisted=*/false, 1, 50,
};

/// BeiDou B1I (1561.098 MHz) -- LEGACY BDS, broadcast by BDS-2 and BDS-3 alike: 2046-chip
/// ALGORITHMIC Gold code at 2.046 Mcps (1 ms), BPSK(2). DATA-ONLY -- no pilot component, so it
/// is acquired directly and nav-wiped (the GPS L1 C/A discipline). The D1 NAV message is 50 bps
/// spread by a 20-chip NH secondary, one chip per 1 ms code period => 20 ms/symbol (the
/// GPS_L5_I / BDS_B2A_D shape: secondary_length 20 AND nav_symbol_s 20 ms, wiped as a composed
/// overlay+navwipe). ⚠️ 1561.098 is 14.3 MHz from L1 -- it does NOT fit an L1 tune's ~10 MHz
/// window, so it needs its own front end. ReplicaSource: beidouB1ICode (b1i_b3i_code_check.py).
inline constexpr SignalDescriptor BDS_B1I = {
    "BDS_B1I", 1561.098e6, 2.046e6, 2046, 1e-3,
    Modulation::BPSK, 0, 0,
    /*pilot=*/false, /*nav_symbol_s=*/20e-3, /*secondary_length=*/20,
    /*time_multiplexed=*/false, /*tdm_phase=*/0, /*time_assisted=*/false, 1, 63,
};

/// BeiDou B3I (1268.52 MHz) -- the B1I sibling one band down, same legacy D1 message: 10230-chip
/// code at 10.23 Mcps (1 ms), BPSK(10), from two 13-stage LFSRs with the ICD's G1 all-ones reset.
/// DATA-ONLY, same 20-chip NH secondary and 20 ms symbol as B1I. Its 10.23 Mcps / 1 ms record
/// geometry is IDENTICAL to BDS_B2B_I, so the front-end and record shapes are already proven --
/// only the code generator differs. ⚠️ 1268.52 is 10.23 MHz from Galileo E6 (1278.75), just
/// outside a ~10 MHz window, so the two cannot share a tune. ReplicaSource: beidouB3ICode.
inline constexpr SignalDescriptor BDS_B3I = {
    "BDS_B3I", 1268.52e6, 10.23e6, 10230, 1e-3,
    Modulation::BPSK, 0, 0,
    /*pilot=*/false, /*nav_symbol_s=*/20e-3, /*secondary_length=*/20,
    /*time_multiplexed=*/false, /*tdm_phase=*/0, /*time_assisted=*/false, 1, 63,
};

/// BeiDou B2I (1207.14 MHz) -- BDS-2's legacy second signal. ★ It reuses the B1I RANGING CODE
/// VERBATIM: the ICD draws one generator for C_B1I and C_B2I (same 11-stage LFSR pair, same
/// per-PRN phase table), so the only differences from BDS_B1I are the carrier and the satellites
/// that broadcast it. Same 2.046 Mcps / 2046 chips / 1 ms, same D1 message, same 20-chip NH.
/// ★ 1207.14 is EXACTLY the B2b / Galileo E5b centre, and a 4 MHz lobe fits well inside that
/// 10 MHz window -- so B2I STACKS on the existing l2c tune for free, no retune.
/// ⚠️ BDS-2 ONLY: BDS-3 replaced B2I with B2a/B2b, and from Canada only the BDS-2 MEOs rise
/// (its GEO/IGSO sit over ~58-160 E). Expect a couple of satellites, not a constellation.
inline constexpr SignalDescriptor BDS_B2I = {
    "BDS_B2I", 1207.14e6, 2.046e6, 2046, 1e-3,
    Modulation::BPSK, 0, 0,
    /*pilot=*/false, /*nav_symbol_s=*/20e-3, /*secondary_length=*/20,
    /*time_multiplexed=*/false, /*tdm_phase=*/0, /*time_assisted=*/false, 1, 63,
};

/// GLONASS L3OC-p (1202.025 MHz) -- the dataless *pilot* of GLONASS's CDMA signal, and our first
/// GLONASS signal of any kind. 10230-chip ALGORITHMIC code at 10.23 Mcps (1 ms), BPSK(10): the
/// SAME record geometry as BDS_B2B_I / BDS_B3I, so the front end, record shapes and channelizer
/// plan are already proven. 10-chip NH secondary (10 ms). ⚠️ GLONASS-K SATELLITES ONLY (7 of 28
/// operational as of 2026-08) -- Celestrak marks them with a trailing 'K' in the Uragan number,
/// which is where the capability filter reads it from, the same trick as the GPS block names.
/// ⚠️ 1202.025 does NOT fit the l2c tune (1207.14 covers 1202.14-1212.14, missing L3OC's centre
/// by 0.115 MHz) -- it needs its own front end. ReplicaSource: glonassL3OCCode.
/// ★ The PRN here is a CODE INDEX whose relationship to the orbital slot number is to be settled
/// by measurement, not assumed -- hence the full 1..63 range. See glonassL3OCCode.hpp.
inline constexpr SignalDescriptor GLO_L3OC_P = {
    "GLO_L3OC_P", 1202.025e6, 10.23e6, 10230, 1e-3,
    Modulation::BPSK, 0, 0,
    /*pilot=*/true, /*nav_symbol_s=*/0.0, /*secondary_length=*/10,
    /*time_multiplexed=*/false, /*tdm_phase=*/0, /*time_assisted=*/false, 1, 63,
};

/// GLONASS L3OC-d (1202.025 MHz) -- the DATA component beside the L3OC-p pilot, carrying the
/// CDMA navigation message at 100 bps through a rate-1/2 code = 200 sym/s. Its secondary is the
/// 5-chip BARKER code, so a symbol is 5 ms = five 1 ms code periods: exactly the BDS_B2A_D /
/// GAL_E5A_I shape (secondary_length 5 AND nav_symbol_s 5 ms, wiped as a composed
/// overlay+navwipe). DERIVED from the pilot -- no search of its own, seeded verbatim.
inline constexpr SignalDescriptor GLO_L3OC_D = {
    "GLO_L3OC_D", 1202.025e6, 10.23e6, 10230, 1e-3,
    Modulation::BPSK, 0, 0,
    /*pilot=*/false, /*nav_symbol_s=*/5e-3, /*secondary_length=*/5,
    /*time_multiplexed=*/false, /*tdm_phase=*/0, /*time_assisted=*/false, 1, 63,
};

/// GLONASS L2OF (~1246 MHz) -- the legacy FDMA open signal, and the ONLY signal we carry whose
/// satellites are separated by FREQUENCY rather than by code. 511-chip m-sequence at 511 kcps
/// (1 ms), BPSK, and EVERY satellite transmits the SAME code: satellite with frequency-channel
/// number k sits at 1246.0 MHz + k*0.4375 MHz, k = -7..+6. So @c carrier_hz here is the comb's
/// NOMINAL CENTRE (k=0), and each satellite's true carrier comes from
/// ChannelizedReplicaBank::set_prn_freq_offset() with k supplied by the broker
/// (gnss_ephemeris.glonass_freq_channels(), read from BRDC -- k is reassigned as satellites are
/// replaced, so it must never be hardcoded).
/// ★ The whole comb spans 1242.94-1248.63 = 5.69 MHz and FITS ONE 10 MHz tune (~1245.8), so FDMA
/// costs no retuning here -- only the per-PRN carrier offset. L2OC at 1248.06 rides the same tune.
/// DATA-ONLY (no pilot): 50 bps MANCHESTER-encoded to 100 sym/s, so a symbol is 10 ms = ten 1 ms
/// code periods and each 20 ms bit is a +-/-+ symbol pair. There is NO secondary overlay -- wipe
/// at the 10 ms symbol (navwipe_bit_records 10) and the Manchester structure comes out for free,
/// since each half-bit is a constant +-1. ReplicaSource: glonassCACode (glonass_ca_code_check.py).
/// PRN here is the ORBITAL SLOT (1..32 covers the ~28 in use), as it is for L3OC.
inline constexpr SignalDescriptor GLO_L2OF = {
    "GLO_L2OF", 1246.0e6, 511e3, 511, 1e-3,
    Modulation::BPSK, 0, 0,
    /*pilot=*/false, /*nav_symbol_s=*/10e-3, /*secondary_length=*/0,
    /*time_multiplexed=*/false, /*tdm_phase=*/0, /*time_assisted=*/false, 1, 32,
};

/// GLONASS L2OC-p (1248.06 MHz) -- the modernised CDMA PILOT, riding the SAME ~1246 tune as L2OF
/// (+2.06 MHz off the k=0 nominal). GLONASS-K only, the same 7 satellites as L3OC. Its base
/// ranging code is BIT-IDENTICAL to L3OC-p's (glonassL2OCCode reuses generate_l3ocp_code); the
/// difference is a per-chip [0,0,+1,-1] BOC(1,1)+TDM expansion, so the TRANSMITTED code is 40920
/// chips at 2.046 Mcps, 20 ms period, HALF of them zero (the data component's TDM slots). The
/// expansion is baked into the code table, so this is a plain BPSK descriptor at the expanded
/// rate -- no TDM/BOC mechanism, the zeros carry the structure. Shared 50-chip OC2 secondary
/// (one chip per 20 ms period, 1 s cycle). ReplicaSource: glonassL2OCCode (l2oc_code_check.py).
/// ⚠️ 20 ms code period: acquired the L2C-CM way (blind full-period search, not time-assisted).
inline constexpr SignalDescriptor GLO_L2OC_P = {
    "GLO_L2OC_P", 1248.06e6, 2.046e6, 40920, 20e-3,
    Modulation::BPSK, 0, 0,
    /*pilot=*/true, /*nav_symbol_s=*/0.0, /*secondary_length=*/50,
    /*time_multiplexed=*/false, /*tdm_phase=*/0, /*time_assisted=*/false, 1, 63,
};

/// Look up a descriptor by its @c name (config string). Returns nullptr if
/// unknown. The full transmitted L2C signal is CM and CL combined; the two
/// descriptors let the correlator target either the data (CM) or the dataless
/// pilot (CL) component. Likewise L5 splits into I5 (data) and Q5 (pilot).
inline const SignalDescriptor* signal_by_name(const std::string& name) {
    for (const SignalDescriptor* s :
         {&GPS_L1CA, &GPS_L1C_P, &GPS_L2C_CM, &GPS_L2C_CL, &GPS_L5_I, &GPS_L5_Q, &GPS_L5_Q_NH,
          &GAL_E1C, &GAL_E1B, &BDS_B1C_P, &BDS_B1C_D, &GAL_E5A_Q, &GAL_E5A_I, &BDS_B2A_P,
          &BDS_B2A_D, &BDS_B2B_I, &GAL_E5B_Q, &GAL_E5B_I, &GPS_L1C_D, &GAL_E6_C, &GAL_E6_B,
          &BDS_B1I, &BDS_B3I, &BDS_B2I, &GLO_L3OC_P, &GLO_L3OC_D, &GLO_L2OF, &GLO_L2OC_P})
        if (name == s->name)
            return s;
    return nullptr;
}

} // namespace gnss

#endif // GNSS_SIGNAL_HPP
