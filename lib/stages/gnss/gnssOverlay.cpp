#include "gnssOverlay.hpp"

#include "gnssSignal.hpp"    // for signal_by_name (the documentary cross-check)
#include "beidouB1CCode.hpp" // for generate_b1cp_secondary (per-PRN B1C overlay)
#include "beidouB2aCode.hpp" // for b2ap_secondary (per-PRN B2a overlay)
#include "galileoE1Code.hpp" // for E1C_CS25 (shared Galileo pilot overlay)
#include "galileoE5aCode.hpp" // for e5aq_secondary (per-PRN E5a-Q overlay)
#include "gpsL1CCode.hpp"    // for generate_l1co_code (per-PRN L1C-P overlay)
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
static std::vector<int8_t> gen_b2ap(int prn) {
    const auto o = beidou::b2ap_secondary(prn);
    return {o.begin(), o.end()};
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
    // BeiDou-3 B2a-pilot: PER-PRN 100-chip Weil secondary.
    {"B2A_CS100", /*per_prn=*/true, 63, 100, gen_b2ap, "BDS_B2A_P"},
    // GPS L1C-P pilot: PER-PRN 1800-symbol L1CO overlay (18 s).
    {"L1CO", /*per_prn=*/true, 32, 1800, gen_l1co, "GPS_L1C_P"},
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
