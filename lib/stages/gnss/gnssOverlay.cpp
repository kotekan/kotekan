#include "gnssOverlay.hpp"

#include "gnssSignal.hpp"    // for signal_by_name (the documentary cross-check)
#include "beidouB1CCode.hpp" // for generate_b1cp_secondary (per-PRN B1C overlay)
#include "beidouB1ICode.hpp" // for B1I_NH20 (shared legacy-BeiDou D1 overlay)
#include "beidouB2aCode.hpp" // for b2ap_secondary (per-PRN B2a overlay)
#include "beidouB3ICode.hpp" // for B3I_NH20 (same sequence, own registry row)
#include "galileoE1Code.hpp" // for E1C_CS25 (shared Galileo pilot overlay)
#include "galileoE5aCode.hpp" // for e5aq_secondary (per-PRN E5a-Q overlay)
#include "galileoE5bCode.hpp" // for e5bq_secondary (per-PRN E5b-Q overlay)
#include "galileoE6Code.hpp" // for generate_e6c_secondary (per-PRN E6-C overlay)
#include "gpsL1CCode.hpp"    // for generate_l1co_code (per-PRN L1C-P overlay)
#include "glonassL3OCCode.hpp" // for L3OC_NH10 / L3OC_BC5 (shared GLONASS CDMA overlays)
#include "gpsL5Code.hpp"     // for L5_NH10/NH20 (shared Neuman-Hofman overlays)

namespace gnss {

// ---- generators (uniform std::vector<int8_t>(int prn) signature) -----------------
// Thin adapters over the per-signal code sources, so the registry rows can hold one
// function-pointer type regardless of the source's array/vector return convention.

static std::vector<int8_t> gen_l5_nh20(int) {
    return {gps::L5_NH20.begin(), gps::L5_NH20.end()};
}
static std::vector<int8_t> gen_l5_nh10(int) {
    return {gps::L5_NH10.begin(), gps::L5_NH10.end()};
}
static std::vector<int8_t> gen_coherent(int) {
    return std::vector<int8_t>(1, (int8_t)1); // length-1 all-ones (see the COHERENT row)
}
static std::vector<int8_t> gen_e1c_cs25(int) {
    return {galileo::E1C_CS25.begin(), galileo::E1C_CS25.end()};
}
static std::vector<int8_t> gen_b1cp(int prn) {
    return beidou::generate_b1cp_secondary(prn);
}
static std::vector<int8_t> gen_e5aq(int prn) {
    const auto o = galileo::e5aq_secondary(prn);
    return {o.begin(), o.end()};
}
static std::vector<int8_t> gen_e5ai_cs20(int) {
    const auto o = galileo::e5ai_secondary(); // shared 20-chip CS20 (E5a-I data channel)
    return {o.begin(), o.end()};
}
static std::vector<int8_t> gen_b2ap(int prn) {
    const auto o = beidou::b2ap_secondary(prn);
    return {o.begin(), o.end()};
}
static std::vector<int8_t> gen_b2ad_cs5(int) {
    const auto o = beidou::b2ad_secondary(); // shared 5-chip CS5 (B2a data channel)
    return {o.begin(), o.end()};
}
static std::vector<int8_t> gen_e5bq(int prn) {
    const auto o = galileo::e5bq_secondary(prn);
    return {o.begin(), o.end()};
}
static std::vector<int8_t> gen_b1i_nh20(int) {
    return {beidou::B1I_NH20.begin(), beidou::B1I_NH20.end()}; // shared 20-chip D1 NH
}
static std::vector<int8_t> gen_b3i_nh20(int) {
    return {beidou::B3I_NH20.begin(), beidou::B3I_NH20.end()}; // identical sequence to B1I's
}
static std::vector<int8_t> gen_e6c(int prn) {
    const auto o = galileo::generate_e6c_secondary(prn);
    return {o.begin(), o.end()};
}
static std::vector<int8_t> gen_e5bi_cs4(int) {
    const auto o = galileo::e5bi_secondary(); // shared 4-chip CS4 (E5b-I data channel)
    return {o.begin(), o.end()};
}
static std::vector<int8_t> gen_l3oc_nh10(int) {
    return {glonass::L3OC_NH10.begin(), glonass::L3OC_NH10.end()};
}
static std::vector<int8_t> gen_l3oc_bc5(int) {
    return {glonass::L3OC_BC5.begin(), glonass::L3OC_BC5.end()};
}
static std::vector<int8_t> gen_l1co(int prn) {
    const auto o = gps::generate_l1co_code(prn);
    return {o.begin(), o.end()};
}

// ---- the registry ----------------------------------------------------------------
// Names are the EXACT `secondary_overlay` config strings (config compatibility).
// Alignment-search floors scale as ~sqrt(2 ln length): NH20 ~2.4 sigma, CS25 ~2.5,
// CS100 ~3.0, the 1800-chip B1C/L1CO overlays ~3.9 -- run the long ones with a LONG
// rolling integration and read deep_snr against the higher floor.
static const OverlayDescriptor OVERLAY_REGISTRY[] = {
    // GPS L5 Q5 pilot: 20-chip Neuman-Hofman, one shared sequence for every satellite.
    {"L5_NH20", /*per_prn=*/false, 1, 20, gen_l5_nh20, "GPS_L5_Q"},
    // GPS L5 I5 data: 10-chip Neuman-Hofman (the CNAV symbol rides on top).
    {"L5_NH10", /*per_prn=*/false, 1, 10, gen_l5_nh10, "GPS_L5_I"},
    // Dataless pilot with NO overlay at all (L2C CL: the 1.5 s code is the only modulation
    // and records are consecutive segments of it, phase-continuous by construction): a
    // length-1 all-ones "overlay" turns overlay_wipe into a plain gap-robust coherent sum
    // with the same SNR estimate and auto-coherence ladder -- deep coherent integration
    // with no bit estimate and no alignment search. Synthetic (signal=nullptr): the
    // GPS_L2C_CL descriptor correctly documents secondary_length 0.
    {"COHERENT", /*per_prn=*/false, 1, 1, gen_coherent, nullptr},
    // Galileo E1-C pilot: the 25-chip CS25 secondary, SAME sequence for every satellite
    // (like the L5 NH overlays, just longer). One chip per 4 ms primary period; the
    // 25-phase alignment search is well-determined within a 250-record (1 s) window.
    {"E1_CS25", /*per_prn=*/false, 1, 25, gen_e1c_cs25, "GAL_E1C"},
    // BeiDou-3 B1C pilot: PER-PRN 1800-chip Weil overlay -- structurally identical to
    // L1CO (same length, same per-PRN pick-at-wipe-time path).
    {"B1C", /*per_prn=*/true, 63, 1800, gen_b1cp, "BDS_B1C_P"},
    // Galileo E5a-Q pilot: PER-PRN 100-chip CS100 secondary.
    {"E5A_CS100", /*per_prn=*/true, 50, 100, gen_e5aq, "GAL_E5A_Q"},
    // Galileo E5a-I DATA: 20-chip CS20, one SHARED sequence (the F/NAV symbol rides on top) --
    // structurally the L5_NH10 case, just 20 ms/symbol. Floor ~sqrt(2 ln 20) ~2.4 sigma.
    {"E5A_CS20", /*per_prn=*/false, 1, 20, gen_e5ai_cs20, "GAL_E5A_I"},
    // BeiDou-3 B2a-pilot: PER-PRN 100-chip Weil secondary.
    {"B2A_CS100", /*per_prn=*/true, 63, 100, gen_b2ap, "BDS_B2A_P"},
    // BeiDou-3 B2a DATA: 5-chip secondary, one SHARED sequence (the B-CNAV2 symbol rides on
    // top) -- the GAL_E5A_I case, just 5 ms/symbol. Floor ~sqrt(2 ln 5) ~1.8 sigma.
    {"B2A_CS5", /*per_prn=*/false, 1, 5, gen_b2ad_cs5, "BDS_B2A_D"},
    // Galileo E5b-Q pilot: PER-PRN 100-chip CS100 secondary -- structurally the E5a-Q case.
    {"E5B_CS100", /*per_prn=*/true, 50, 100, gen_e5bq, "GAL_E5B_Q"},
    // Galileo E5b-I DATA: 4-chip CS4, one SHARED sequence (the I/NAV symbol rides on top) --
    // the GAL_E5A_I case, just 4 ms/symbol. Floor ~sqrt(2 ln 4) ~1.7 sigma.
    {"E5B_CS4", /*per_prn=*/false, 1, 4, gen_e5bi_cs4, "GAL_E5B_I"},
    // GPS L1C-P pilot: PER-PRN 1800-symbol L1CO overlay (18 s).
    {"L1CO", /*per_prn=*/true, 32, 1800, gen_l1co, "GPS_L1C_P"},
    // Galileo E6-C pilot: PER-PRN 100-chip secondary (100 ms) -- structurally the E5a-Q /
    // E5b-Q case. ⚠️ per-PRN, NOT the shared-CS25 shape E1-C uses. E6-B (the HAS data
    // channel) has NO overlay: its 1000 sps symbol IS one 1 ms code period.
    {"E6_CS100", /*per_prn=*/true, 50, 100, gen_e6c, "GAL_E6_C"},
    // LEGACY BeiDou B1I/B3I: the 20-chip D1 NH secondary, one SHARED sequence for every
    // satellite AND identical between the two signals (kept as two rows so each chain's
    // registry entry names its own signal). One chip per 1 ms period, 20 ms per NAV symbol --
    // structurally the GPS L5_NH10 / E5A_CS20 / B2A_CS5 case. Floor ~sqrt(2 ln 20) ~2.4 sigma.
    {"B1I_NH20", /*per_prn=*/false, 1, 20, gen_b1i_nh20, "BDS_B1I"},
    {"B3I_NH20", /*per_prn=*/false, 1, 20, gen_b3i_nh20, "BDS_B3I"},
    // B2I: same D1 NH20 again (and the same ranging code as B1I -- see the descriptor).
    {"B2I_NH20", /*per_prn=*/false, 1, 20, gen_b1i_nh20, "BDS_B2I"},
    // GLONASS L3OC pilot: 10-chip Neuman-Hofman, one SHARED sequence -- the GPS L5_NH20 shape
    // at half the length. Floor ~sqrt(2 ln 10) ~2.1 sigma.
    {"L3OC_NH10", /*per_prn=*/false, 1, 10, gen_l3oc_nh10, "GLO_L3OC_P"},
    // GLONASS L3OC DATA: the 5-chip BARKER secondary, shared (the 200 sps nav symbol rides on
    // top) -- structurally the BDS_B2A_D / GAL_E5A_I case at 5 ms/symbol, and the same length
    // as B2A_CS5 though a different sequence. Floor ~sqrt(2 ln 5) ~1.8 sigma.
    {"L3OC_BC5", /*per_prn=*/false, 1, 5, gen_l3oc_bc5, "GLO_L3OC_D"},
};

const OverlayDescriptor* overlay_by_name(const std::string& name) {
    for (const OverlayDescriptor& od : OVERLAY_REGISTRY)
        if (name == od.name)
            return &od;
    return nullptr;
}

std::vector<std::string> overlay_registry_check() {
    std::vector<std::string> out;
    for (const OverlayDescriptor& od : OVERLAY_REGISTRY) {
        if (!od.signal)
            continue; // synthetic overlay: no spec secondary to document
        const SignalDescriptor* sd = signal_by_name(od.signal);
        if (!sd) {
            out.push_back(std::string(od.name) + ": signal descriptor '" + od.signal
                          + "' not found");
            continue;
        }
        if (sd->secondary_length != od.length)
            out.push_back(std::string(od.name) + ": registry length "
                          + std::to_string(od.length) + " != " + od.signal
                          + ".secondary_length " + std::to_string(sd->secondary_length));
    }
    return out;
}

} // namespace gnss
