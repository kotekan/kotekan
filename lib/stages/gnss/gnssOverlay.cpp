#include "gnssOverlay.hpp"

#include "beidouB1ICode.hpp"   // for B1I_NH20 (shared legacy-BeiDou D1 overlay)
#include "beidouB2aCode.hpp"   // for b2ad_secondary (shared B2a data overlay)
#include "beidouB3ICode.hpp"   // for B3I_NH20 (same sequence, own registry row)
#include "galileoE5aCode.hpp"  // for e5ai_secondary (shared E5a-I overlay)
#include "galileoE5bCode.hpp"  // for e5bi_secondary (shared E5b-I overlay)
#include "glonassL3OCCode.hpp" // for L3OC_BC5 (shared GLONASS L3OC data overlay)
#include "gnssSignal.hpp"      // for signal_by_name, SignalDescriptor
#include "gpsL5Code.hpp"       // for L5_NH10 (shared Neuman-Hofman overlay)

namespace gnss {

// Adapters giving every code source the one generator signature the registry holds.

static std::vector<int8_t> gen_l5_nh10() {
    return {gps::L5_NH10.begin(), gps::L5_NH10.end()};
}
static std::vector<int8_t> gen_e5ai_cs20() {
    const auto o = galileo::e5ai_secondary(); // shared 20-chip CS20 (E5a-I data channel)
    return {o.begin(), o.end()};
}
static std::vector<int8_t> gen_b2ad_cs5() {
    const auto o = beidou::b2ad_secondary(); // shared 5-chip CS5 (B2a data channel)
    return {o.begin(), o.end()};
}
static std::vector<int8_t> gen_b1i_nh20() {
    return {beidou::B1I_NH20.begin(), beidou::B1I_NH20.end()}; // shared 20-chip D1 NH
}
static std::vector<int8_t> gen_b3i_nh20() {
    return {beidou::B3I_NH20.begin(), beidou::B3I_NH20.end()}; // same sequence as B1I_NH20
}
static std::vector<int8_t> gen_e5bi_cs4() {
    const auto o = galileo::e5bi_secondary(); // shared 4-chip CS4 (E5b-I data channel)
    return {o.begin(), o.end()};
}
static std::vector<int8_t> gen_l3oc_bc5() {
    return {glonass::L3OC_BC5.begin(), glonass::L3OC_BC5.end()};
}

// Names are the `secondary_overlay` config strings. The alignment search's noise maximum
// grows as ~sqrt(2 ln length) sigma: ~2.4 for 20 chips.
static const OverlayDescriptor OVERLAY_REGISTRY[] = {
    // GPS L5 I5 data: 10-chip Neuman-Hofman under the CNAV symbol.
    {"L5_NH10", 10, gen_l5_nh10, "GPS_L5_I"},
    // Galileo E5a-I data: shared 20-chip CS20 under the F/NAV symbol (20 ms per symbol).
    {"E5A_CS20", 20, gen_e5ai_cs20, "GAL_E5A_I"},
    // BeiDou-3 B2a data: shared 5-chip secondary under the B-CNAV2 symbol (5 ms per symbol).
    {"B2A_CS5", 5, gen_b2ad_cs5, "BDS_B2A_D"},
    // Galileo E5b-I data: shared 4-chip CS4 under the I/NAV symbol (4 ms per symbol).
    {"E5B_CS4", 4, gen_e5bi_cs4, "GAL_E5B_I"},
    // BeiDou B1I/B3I: the shared 20-chip D1 NH secondary, one chip per 1 ms period under a
    // 20 ms nav symbol. The sequence is the same on both signals; each has its own row so the
    // row names its own signal.
    {"B1I_NH20", 20, gen_b1i_nh20, "BDS_B1I"},
    {"B3I_NH20", 20, gen_b3i_nh20, "BDS_B3I"},
    // B2I: the same D1 NH20.
    {"B2I_NH20", 20, gen_b1i_nh20, "BDS_B2I"},
    // GLONASS L3OC data: shared 5-chip Barker secondary under the 200 sps nav symbol (a
    // different sequence from B2A_CS5).
    {"L3OC_BC5", 5, gen_l3oc_bc5, "GLO_L3OC_D"},
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
