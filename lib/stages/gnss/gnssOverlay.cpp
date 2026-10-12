#include "gnssOverlay.hpp"

#include "beidouB1CCode.hpp"   // for generate_b1cp_secondary (per-PRN B1C overlay)
#include "beidouB1ICode.hpp"   // for B1I_NH20 (shared legacy-BeiDou D1 overlay)
#include "beidouB2aCode.hpp"   // for b2ap_secondary (per-PRN B2a overlay)
#include "beidouB3ICode.hpp"   // for B3I_NH20 (same sequence, own registry row)
#include "galileoE1Code.hpp"   // for E1C_CS25 (shared Galileo pilot overlay)
#include "galileoE5aCode.hpp"  // for e5aq_secondary (per-PRN E5a-Q overlay)
#include "galileoE5bCode.hpp"  // for e5bq_secondary (per-PRN E5b-Q overlay)
#include "galileoE6Code.hpp"   // for generate_e6c_secondary (per-PRN E6-C overlay)
#include "glonassL2OCCode.hpp" // for L2OC_OC2 (shared 50-chip L2OC-p overlay)
#include "glonassL3OCCode.hpp" // for L3OC_NH10 / L3OC_BC5 (shared GLONASS CDMA overlays)
#include "gnssSignal.hpp"      // for signal_by_name, SignalDescriptor
#include "gpsL1CCode.hpp"      // for generate_l1co_code (per-PRN L1C-P overlay)
#include "gpsL5Code.hpp"       // for L5_NH10/NH20 (shared Neuman-Hofman overlays)

namespace gnss {

// Adapters giving every code source the one generator signature the registry holds.

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
    return {beidou::B3I_NH20.begin(), beidou::B3I_NH20.end()}; // same sequence as B1I_NH20
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
static std::vector<int8_t> gen_l2oc_oc2(int) {
    return {glonass::L2OC_OC2.begin(), glonass::L2OC_OC2.end()}; // shared 50-chip L2OC-p overlay
}
static std::vector<int8_t> gen_l3oc_bc5(int) {
    return {glonass::L3OC_BC5.begin(), glonass::L3OC_BC5.end()};
}
static std::vector<int8_t> gen_l1co(int prn) {
    const auto o = gps::generate_l1co_code(prn);
    return {o.begin(), o.end()};
}

// Names are the `secondary_overlay` config strings. The alignment search's noise maximum
// grows as ~sqrt(2 ln length) sigma: ~2.4 for 20 chips, ~3.0 for 100, ~3.9 for 1800. Run the
// long overlays with a long rolling integration and read deep_snr against their higher floor.
static const OverlayDescriptor OVERLAY_REGISTRY[] = {
    // GPS L5 Q5 pilot: 20-chip Neuman-Hofman, one shared sequence for every satellite.
    {"L5_NH20", /*per_prn=*/false, 1, 20, gen_l5_nh20, "GPS_L5_Q"},
    // GPS L5 I5 data: 10-chip Neuman-Hofman under the CNAV symbol.
    {"L5_NH10", /*per_prn=*/false, 1, 10, gen_l5_nh10, "GPS_L5_I"},
    // A dataless pilot with no overlay (L2C CL, whose records are consecutive, phase-continuous
    // segments of the 1.5 s code): a length-1 all-ones overlay makes overlay_wipe a plain
    // coherent sum with the same SNR estimate and auto-coherence ladder, and no bit estimate or
    // alignment search. Synthetic, so no signal: GPS_L2C_CL has secondary_length 0.
    {"COHERENT", /*per_prn=*/false, 1, 1, gen_coherent, nullptr},
    // Galileo E1-C pilot: the shared 25-chip CS25 secondary, one chip per 4 ms primary period.
    {"E1_CS25", /*per_prn=*/false, 1, 25, gen_e1c_cs25, "GAL_E1C"},
    // BeiDou-3 B1C pilot: per-PRN 1800-chip Weil overlay, the same shape as L1CO.
    {"B1C", /*per_prn=*/true, 63, 1800, gen_b1cp, "BDS_B1C_P"},
    // Galileo E5a-Q pilot: per-PRN 100-chip CS100 secondary.
    {"E5A_CS100", /*per_prn=*/true, 50, 100, gen_e5aq, "GAL_E5A_Q"},
    // Galileo E5a-I data: shared 20-chip CS20 under the F/NAV symbol (20 ms per symbol).
    {"E5A_CS20", /*per_prn=*/false, 1, 20, gen_e5ai_cs20, "GAL_E5A_I"},
    // BeiDou-3 B2a pilot: per-PRN 100-chip Weil secondary.
    {"B2A_CS100", /*per_prn=*/true, 63, 100, gen_b2ap, "BDS_B2A_P"},
    // BeiDou-3 B2a data: shared 5-chip secondary under the B-CNAV2 symbol (5 ms per symbol).
    {"B2A_CS5", /*per_prn=*/false, 1, 5, gen_b2ad_cs5, "BDS_B2A_D"},
    // Galileo E5b-Q pilot: per-PRN 100-chip CS100 secondary.
    {"E5B_CS100", /*per_prn=*/true, 50, 100, gen_e5bq, "GAL_E5B_Q"},
    // Galileo E5b-I data: shared 4-chip CS4 under the I/NAV symbol (4 ms per symbol).
    {"E5B_CS4", /*per_prn=*/false, 1, 4, gen_e5bi_cs4, "GAL_E5B_I"},
    // GPS L1C-P pilot: per-PRN 1800-chip L1CO overlay (18 s).
    {"L1CO", /*per_prn=*/true, 32, 1800, gen_l1co, "GPS_L1C_P"},
    // Galileo E6-C pilot: per-PRN 100-chip secondary (100 ms), unlike E1-C's shared CS25.
    // E6-B has no overlay: its 1000 sps symbol is one 1 ms code period.
    {"E6_CS100", /*per_prn=*/true, 50, 100, gen_e6c, "GAL_E6_C"},
    // BeiDou B1I/B3I: the shared 20-chip D1 NH secondary, one chip per 1 ms period under a
    // 20 ms nav symbol. The sequence is the same on both signals; each has its own row so the
    // row names its own signal.
    {"B1I_NH20", /*per_prn=*/false, 1, 20, gen_b1i_nh20, "BDS_B1I"},
    {"B3I_NH20", /*per_prn=*/false, 1, 20, gen_b3i_nh20, "BDS_B3I"},
    // B2I: the same D1 NH20.
    {"B2I_NH20", /*per_prn=*/false, 1, 20, gen_b1i_nh20, "BDS_B2I"},
    // GLONASS L3OC pilot: shared 10-chip Neuman-Hofman.
    {"L3OC_NH10", /*per_prn=*/false, 1, 10, gen_l3oc_nh10, "GLO_L3OC_P"},
    // GLONASS L3OC data: shared 5-chip Barker secondary under the 200 sps nav symbol (a
    // different sequence from B2A_CS5).
    {"L3OC_BC5", /*per_prn=*/false, 1, 5, gen_l3oc_bc5, "GLO_L3OC_D"},
    // GLONASS L2OC pilot: shared 50-chip OC2 secondary, one chip per 20 ms primary period
    // (1 s cycle), so resolving its alignment takes seconds of rolling integration.
    {"L2OC_OC2", /*per_prn=*/false, 1, 50, gen_l2oc_oc2, "GLO_L2OC_P"},
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
            out.push_back(std::string(od.name) + ": registry length " + std::to_string(od.length)
                          + " != " + od.signal + ".secondary_length "
                          + std::to_string(sd->secondary_length));
    }
    return out;
}

} // namespace gnss
