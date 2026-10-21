#include "gnssInterface.hpp"

#include "GnssChanMetadata.hpp" // for GnssChanMetadata
#include "gnssBandPlan.hpp"     // for BandSpan, half_bandwidth_hz, occupied_band
#include "gnssGpuChain.hpp" // for FrameHdr, PrnCtl, MAX_REC, LANES_*, ROWS_*, ROW_*, frame_bytes
#include "gnssRecord.hpp"   // for REC_*, CHAN_*, ELEM_*, CMB_*, CUBE_*, cube_frame_bytes
#include "gnssSignal.hpp"   // for ALL_SIGNALS, Modulation, SignalDescriptor
#include "gnssTelem.hpp"    // for TelemHeader, TELEM_*, telem_frame_bytes
#include "interfaceRegistry.hpp" // for REGISTER_KOTEKAN_INTERFACE, canonical_double

#include "fmt.hpp" // for format, fmt

#include <cstddef>     // for offsetof, size_t
#include <cstdint>     // for int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t
#include <iterator>    // for size
#include <stdexcept>   // for runtime_error
#include <string>      // for string, to_string
#include <type_traits> // for extent_v, is_same_v, remove_all_extents_t
#include <vector>      // for vector

using json = nlohmann::json;
using kotekan::canonical_double;

namespace gnss {

namespace {

// The document's name for the element type of a wire field.
template<typename T>
constexpr const char* type_tag() {
    if constexpr (std::is_same_v<T, char>)
        return "char";
    else if constexpr (std::is_same_v<T, uint8_t>)
        return "u8";
    else if constexpr (std::is_same_v<T, uint16_t>)
        return "u16";
    else if constexpr (std::is_same_v<T, uint32_t>)
        return "u32";
    else if constexpr (std::is_same_v<T, uint64_t>)
        return "u64";
    else if constexpr (std::is_same_v<T, int32_t>)
        return "i32";
    else if constexpr (std::is_same_v<T, int64_t>)
        return "i64";
    else if constexpr (std::is_same_v<T, float>)
        return "f32";
    else if constexpr (std::is_same_v<T, double>)
        return "f64";
    else
        static_assert(sizeof(T) == 0, "add this field type to type_tag()");
}

// One field of a wire struct. count is the array length, 0 for a scalar.
struct Field {
    const char* name;
    size_t offset;
    size_t size;
    const char* type;
    size_t count;
};

#define FIELD(S, f)                                                                                \
    Field {                                                                                        \
        #f, offsetof(S, f), sizeof(S::f), type_tag<std::remove_all_extents_t<decltype(S::f)>>(),   \
            std::extent_v<decltype(S::f)>                                                          \
    }

constexpr Field telem_header_fields[] = {
    FIELD(TelemHeader, magic),    FIELD(TelemHeader, version),
    FIELD(TelemHeader, n_rec),    FIELD(TelemHeader, n_prn),
    FIELD(TelemHeader, n_row),    FIELD(TelemHeader, n_chan),
    FIELD(TelemHeader, n_elem),   FIELD(TelemHeader, hops_per_record),
    FIELD(TelemHeader, fft_len),  FIELD(TelemHeader, win),
    FIELD(TelemHeader, seq),      FIELD(TelemHeader, wstart0),
    FIELD(TelemHeader, utc0),     FIELD(TelemHeader, present),
    FIELD(TelemHeader, max_chan), FIELD(TelemHeader, n_row_total),
    FIELD(TelemHeader, chain),    FIELD(TelemHeader, inst),
    FIELD(TelemHeader, chan_id),
};

constexpr Field frame_hdr_fields[] = {
    FIELD(gnss_gpu::FrameHdr, n_rec),       FIELD(gnss_gpu::FrameHdr, n_prn),
    FIELD(gnss_gpu::FrameHdr, n_chan),      FIELD(gnss_gpu::FrameHdr, n_jobs),
    FIELD(gnss_gpu::FrameHdr, seq0),        FIELD(gnss_gpu::FrameHdr, utc0),
    FIELD(gnss_gpu::FrameHdr, n_rows_spec), FIELD(gnss_gpu::FrameHdr, lanes_per_slot),
    FIELD(gnss_gpu::FrameHdr, _pad1),
};

constexpr Field prn_ctl_fields[] = {
    FIELD(gnss_gpu::PrnCtl, run),         FIELD(gnss_gpu::PrnCtl, reanchored),
    FIELD(gnss_gpu::PrnCtl, prn),         FIELD(gnss_gpu::PrnCtl, job0),
    FIELD(gnss_gpu::PrnCtl, fcar_report), FIELD(gnss_gpu::PrnCtl, n_owned),
    FIELD(gnss_gpu::PrnCtl, cp_seed),     FIELD(gnss_gpu::PrnCtl, f_nco),
    FIELD(gnss_gpu::PrnCtl, chan_mask),   FIELD(gnss_gpu::PrnCtl, ctrim_hz),
    FIELD(gnss_gpu::PrnCtl, ang0),        FIELD(gnss_gpu::PrnCtl, phi_ddop),
    FIELD(gnss_gpu::PrnCtl, fcar),        FIELD(gnss_gpu::PrnCtl, dcyc),
    FIELD(gnss_gpu::PrnCtl, rot),
};

#undef FIELD

// True if the fields cover [0, size) in order, with no gap or overlap: a field added to a struct
// but not to its table fails the static_asserts below.
constexpr bool covers(const Field* fields, size_t n, size_t size) {
    size_t end = 0;
    for (size_t i = 0; i < n; i++) {
        if (fields[i].offset != end)
            return false;
        end += fields[i].size;
    }
    return end == size;
}

static_assert(covers(telem_header_fields, std::size(telem_header_fields), sizeof(TelemHeader)),
              "telem_header_fields must list every TelemHeader field in order");
static_assert(covers(frame_hdr_fields, std::size(frame_hdr_fields), sizeof(gnss_gpu::FrameHdr)),
              "frame_hdr_fields must list every FrameHdr field in order");
static_assert(covers(prn_ctl_fields, std::size(prn_ctl_fields), sizeof(gnss_gpu::PrnCtl)),
              "prn_ctl_fields must list every PrnCtl field in order");

// {"size": bytes, "fields": [[name, offset, size, type], ...]}, type e.g. "u16" or "u16[8]".
json wire_struct(const Field* fields, size_t n, size_t size) {
    json table = json::array();
    for (size_t i = 0; i < n; i++) {
        std::string type = fields[i].type;
        if (fields[i].count > 0)
            type += "[" + std::to_string(fields[i].count) + "]";
        table.push_back(json::array({fields[i].name, fields[i].offset, fields[i].size, type}));
    }
    return {{"size", size}, {"fields", table}};
}

json slot(int index, const std::string& meaning) {
    return {{"slot", index}, {"meaning", meaning}};
}

const char* modulation_name(Modulation mod) {
    switch (mod) {
        case Modulation::BPSK:
            return "BPSK";
        case Modulation::BOC:
            return "BOC";
    }
    throw std::runtime_error("unknown Modulation " + std::to_string(static_cast<int>(mod)));
}

json signals() {
    json out = json::object();
    for (const SignalDescriptor* s : ALL_SIGNALS) {
        // Binds the members by position only to count them: a member added to SignalDescriptor
        // fails to compile here until it is listed below too.
        [[maybe_unused]] const auto& [name, carrier_hz, chip_rate_hz, code_length, code_period_s,
                                      mod, boc_m, boc_n, secondary_length, time_multiplexed,
                                      tdm_phase] = *s;
        const BandSpan band = occupied_band(*s, 0.0);
        out[s->name] = {{"carrier_hz", canonical_double(s->carrier_hz)},
                        {"chip_rate_hz", canonical_double(s->chip_rate_hz)},
                        {"code_length", s->code_length},
                        {"code_period_s", canonical_double(s->code_period_s)},
                        {"mod", modulation_name(s->mod)},
                        {"boc_m", s->boc_m},
                        {"boc_n", s->boc_n},
                        {"secondary_length", s->secondary_length},
                        {"time_multiplexed", s->time_multiplexed},
                        {"tdm_phase", s->tdm_phase},
                        {"half_bandwidth_hz", canonical_double(half_bandwidth_hz(*s))},
                        {"occupied_lo_hz", canonical_double(band.lo_hz)},
                        {"occupied_hi_hz", canonical_double(band.hi_hz)}};
    }
    return out;
}

json telem() {
    json frames = json::array();
    const int samples[][3] = {{1, 1, 0}, {3, 5, 2}, {8, 16, 8}}; // n_rec, n_prn, n_cols
    for (const auto& s : samples)
        frames.push_back(json{{"n_rec", s[0]},
                              {"n_prn", s[1]},
                              {"n_cols", s[2]},
                              {"bytes", telem_frame_bytes(s[0], s[1], s[2])}});
    // Where a reader finds row (r, p) of a frame's payload, and comb column ch of a row, in floats.
    json rows = json::array();
    const int row_samples[][4] = {{0, 1, 3, 0}, {2, 3, 5, 2}}; // r, p, n_prn, n_cols
    for (const auto& s : row_samples)
        rows.push_back(
            json{{"r", s[0]},
                 {"p", s[1]},
                 {"n_prn", s[2]},
                 {"n_cols", s[3]},
                 {"floats", telem_row_offset(s[0], s[1], s[2], telem_row_floats(s[3]))}});
    json cols = json::array();
    for (int ch = 0; ch < TELEM_MAX_CHAN; ch++)
        cols.push_back(telem_chan_offset(ch));
    return {{"TELEM_MAGIC", TELEM_MAGIC},
            {"TELEM_VERSION", TELEM_VERSION},
            {"TELEM_MAX_CHAN", TELEM_MAX_CHAN},
            {"TELEM_MAX_REC", TELEM_MAX_REC},
            {"TELEM_NAME", TELEM_NAME},
            {"TELEM_HEADER_BYTES", TELEM_HEADER_BYTES},
            {"TelemHeader",
             wire_struct(telem_header_fields, std::size(telem_header_fields), sizeof(TelemHeader))},
            {"frame_bytes", frames},
            {"row_offset", rows},
            {"chan_offset", cols}};
}

// Tracker record slots. Combiner records share slots 0-2 and the UTC.
json record_row() {
    const char* prompt = "corr;prompt;rot=exp(-i*REC_PHI0);sign=r2c";
    return {{"RECORD_FLOATS", RECORD_FLOATS},
            {"RECORD_UTC_SLOT", slot(RECORD_UTC_SLOT, "s;f64 over 2 slots;utc of window start")},
            {"REC_PRN", slot(REC_PRN, "prn;integer")},
            {"REC_DOPPLER", slot(REC_DOPPLER, "Hz;doppler;sign=physical")},
            {"REC_CP", slot(REC_CP, "chip;code phase arg at REC_DOPPLER;mod=record_cp_mod_chips")},
            {"REC_P_RE", slot(REC_P_RE, prompt)},
            {"REC_P_IM", slot(REC_P_IM, prompt)},
            {"REC_P_ENERGY", slot(REC_P_ENERGY, "energy;prompt replica")},
            {"REC_NCHAN", slot(REC_NCHAN, "count;covering channels")},
            {"REC_E_ENERGY", slot(REC_E_ENERGY, "energy;early replica")},
            {"REC_L_ENERGY", slot(REC_L_ENERGY, "energy;late replica")},
            {"REC_E_RE", slot(REC_E_RE, "corr;early;rot=none;sign=r2c")},
            {"REC_E_IM", slot(REC_E_IM, "corr;early;rot=none;sign=r2c")},
            {"REC_L_RE", slot(REC_L_RE, "corr;late;rot=none;sign=r2c")},
            {"REC_L_IM", slot(REC_L_IM, "corr;late;rot=none;sign=r2c")},
            {"REC_CPHASE",
             slot(REC_CPHASE, "cycle;increment of fcar*t_abs-nco;sign=physical;0=arc start")},
            {"REC_PH_RE", slot(REC_PH_RE, "corr;prompt head;rot=exp(-i*REC_PHI0);sign=r2c")},
            {"REC_PH_IM", slot(REC_PH_IM, "corr;prompt head;rot=exp(-i*REC_PHI0);sign=r2c")},
            {"REC_PH_ENERGY", slot(REC_PH_ENERGY, "energy;head replica")},
            {"REC_TRIM_INC", slot(REC_TRIM_INC, "cycle;increment of ctrim_hz*dt;0=arc start")},
            {"REC_RES_RE", slot(REC_RES_RE, "reserved;0")},
            {"REC_RES_IM", slot(REC_RES_IM, "reserved;0")},
            {"REC_RES_PH_RE", slot(REC_RES_PH_RE, "reserved;0")},
            {"REC_RES_PH_IM", slot(REC_RES_PH_IM, "reserved;0")},
            {"REC_SKY_RE", slot(REC_SKY_RE, "corr;sky-phase prompt;rot=exp(-i*REC_PHI0);0=absent")},
            {"REC_SKY_IM", slot(REC_SKY_IM, "corr;sky-phase prompt;rot=exp(-i*REC_PHI0);0=absent")},
            {"REC_ANG0", slot(REC_ANG0, "rad;nan=no job")},
            {"REC_PHI_DDOP", slot(REC_PHI_DDOP, "Hz;nan=no phi")},
            {"REC_PHI0", slot(REC_PHI0, "rad;range=[-pi,pi]")},
            {"REC_PROJ_COST", slot(REC_PROJ_COST, "cos2;range=[0,1];-1=none")},
            {"REC_PROJ_K", slot(REC_PROJ_K, "count;0=no basis")},
            {"REC_PROJ_SRC", slot(REC_PROJ_SRC, "bits=REC_PROJ_SRC_*;0=no basis")}};
}

json record_chan() {
    const std::string chan = ";rot=exp(-i*REC_PHI0);sign=r2c;weighted per channel;0=not covering";
    return {{"CHAN_FLOATS", CHAN_FLOATS},
            {"CHAN_RE", slot(CHAN_RE, "corr;prompt" + chan)},
            {"CHAN_IM", slot(CHAN_IM, "corr;prompt" + chan)},
            {"CHAN_ENERGY", slot(CHAN_ENERGY, "energy;prompt replica;weighted per channel")},
            {"CHAN_E_RE", slot(CHAN_E_RE, "corr;early" + chan)},
            {"CHAN_E_IM", slot(CHAN_E_IM, "corr;early" + chan)},
            {"CHAN_E_ENERGY", slot(CHAN_E_ENERGY, "energy;early replica;weighted per channel")},
            {"CHAN_L_RE", slot(CHAN_L_RE, "corr;late" + chan)},
            {"CHAN_L_IM", slot(CHAN_L_IM, "corr;late" + chan)},
            {"CHAN_L_ENERGY", slot(CHAN_L_ENERGY, "energy;late replica;weighted per channel")}};
}

// Tracker element blocks.
json record_elem() {
    return {{"ELEM_FLOATS", ELEM_FLOATS},
            {"ELEM_P_RE", slot(ELEM_P_RE, "corr;prompt;rot=exp(-i*REC_PHI0);sign=r2c")},
            {"ELEM_P_IM", slot(ELEM_P_IM, "corr;prompt;rot=exp(-i*REC_PHI0);sign=r2c")},
            {"ELEM_E_RE", slot(ELEM_E_RE, "corr;early;rot=exp(-i*REC_PHI0);sign=r2c")},
            {"ELEM_E_IM", slot(ELEM_E_IM, "corr;early;rot=exp(-i*REC_PHI0);sign=r2c")},
            {"ELEM_L_RE", slot(ELEM_L_RE, "corr;late;rot=exp(-i*REC_PHI0);sign=r2c")},
            {"ELEM_L_IM", slot(ELEM_L_IM, "corr;late;rot=exp(-i*REC_PHI0);sign=r2c")},
            {"ELEM_PH_RE", slot(ELEM_PH_RE, "corr;prompt head;rot=exp(-i*REC_PHI0);sign=r2c")},
            {"ELEM_PH_IM", slot(ELEM_PH_IM, "corr;prompt head;rot=exp(-i*REC_PHI0);sign=r2c")},
            {"ELEM_RES_RE", slot(ELEM_RES_RE, "reserved;0")},
            {"ELEM_RES_IM", slot(ELEM_RES_IM, "reserved;0")},
            {"ELEM_RES_PH_RE", slot(ELEM_RES_PH_RE, "reserved;0")},
            {"ELEM_RES_PH_IM", slot(ELEM_RES_PH_IM, "reserved;0")}};
}

// Combiner record slots; "elem" holds the combiner's element block slots.
json record_cmb() {
    const char* incoh = "amp;sqrt<|H|^2+|A-H|^2>;A=REC_P/REC_P_ENERGY;"
                        "H=REC_PH/REC_P_ENERGY, or A if REC_PH_ENERGY=0";
    return {{"CMB_AMP_INCOH", slot(CMB_AMP_INCOH, incoh)},
            {"CMB_MEAN_RE", slot(CMB_MEAN_RE, "amp;<A>.re;sign=r2c")},
            {"CMB_MEAN_IM", slot(CMB_MEAN_IM, "amp;<A>.im;sign=r2c")},
            {"CMB_AMP_COH", slot(CMB_AMP_COH, "amp;|<A>|")},
            {"CMB_NCHAN", slot(CMB_NCHAN, "count;mean covering channels")},
            {"CMB_DEEP", slot(CMB_DEEP, "amp;deep |A|")},
            {"CMB_E_POW", slot(CMB_E_POW, "pow;<|REC_E/REC_E_ENERGY|^2>")},
            {"CMB_L_POW", slot(CMB_L_POW, "pow;<|REC_L/REC_L_ENERGY|^2>")},
            {"CMB_DLL_DISC", slot(CMB_DLL_DISC, "ratio;(E-L)/(E+L) of the powers;0=no power")},
            {"CMB_CARRIER_RESID", slot(CMB_CARRIER_RESID, "Hz;carrier residual;sign=r2c")},
            {"CMB_ARC", slot(CMB_ARC, "id;+1 per phase break")},
            {"CMB_HEAD_FRAC", slot(CMB_HEAD_FRAC, "ratio;<PH_energy>/<P_energy>;0=none")},
            {"CMB_PEEL_DEEP", slot(CMB_PEEL_DEEP, "reserved;0")},
            {"CMB_PEEL_INCOH", slot(CMB_PEEL_INCOH, "reserved;0")},
            {"CMB_HOP_SLOT", slot(CMB_HOP_SLOT, "hop;i64 over 2 slots;-1=unset")},
            {"CMB_COH_FRAC", slot(CMB_COH_FRAC, "ratio;|sum A|/sum|A|;0=not computed")},
            {"elem",
             {{"CMB_ELEM_AMP_INCOH",
               slot(CMB_ELEM_AMP_INCOH, "amp;sqrt<|A_e|^2>;A_e=ELEM_P/REC_P_ENERGY")},
              {"CMB_ELEM_MEAN_RE", slot(CMB_ELEM_MEAN_RE, "amp;<A_e>.re;sign=r2c")},
              {"CMB_ELEM_MEAN_IM", slot(CMB_ELEM_MEAN_IM, "amp;<A_e>.im;sign=r2c")},
              {"CMB_ELEM_AMP_COH", slot(CMB_ELEM_AMP_COH, "amp;|<A_e>|")}}}};
}

json cube() {
    json frames = json::array();
    const int samples[][3] = {{1, 1, 1}, {3, 5, 7}, {32, 16, 96}}; // max_prn, max_bins, n_elem
    for (const auto& s : samples)
        frames.push_back(json{{"max_prn", s[0]},
                              {"max_bins", s[1]},
                              {"n_elem", s[2]},
                              {"bytes", cube_frame_bytes(s[0], s[1], s[2])}});
    // The arrays after the header, in the order GnssGpuRecordAssemble writes them:
    // [name, type, length]. Typed from the writer, like the header fields before
    // CUBE_UTC0_OFFSET, which are not listed: only CUBE_VERSION guards them.
    json arrays = json::array({{"phi0", "f64", "max_prn"},
                               {"freq_id_lo", "i32", "max_bins"},
                               {"freq_id_hi", "i32", "max_bins"},
                               {"prn", "i32", "max_prn"},
                               {"n_rec", "i32", "max_prn"},
                               {"n_reanchor", "i32", "max_prn"},
                               {"w", "f32", "max_prn*max_bins"},
                               {"energy", "f32", "max_prn*max_bins"},
                               {"coh_re", "f32", "max_prn*max_bins*n_elem"},
                               {"coh_im", "f32", "max_prn*max_bins*n_elem"},
                               {"incoh", "f32", "max_prn*max_bins*n_elem"}});
    return {{"CUBE_VERSION", CUBE_VERSION},
            {"CUBE_CHAIN_CHARS", CUBE_CHAIN_CHARS},
            {"CUBE_UTC0_OFFSET", CUBE_UTC0_OFFSET},
            {"CUBE_CHAIN_OFFSET", CUBE_CHAIN_OFFSET},
            {"CUBE_HEADER_BYTES", CUBE_HEADER_BYTES},
            {"arrays", arrays},
            {"frame_bytes", frames}};
}

json gpu_chain() {
    json frames = json::array();
    // n_prn, n_chan, rows_spec, n_elem
    const int samples[][4] = {{1, 1, 4, 1}, {3, 5, 6, 7}, {16, 8, 4, 96}};
    for (const auto& s : samples)
        frames.push_back(json{{"n_prn", s[0]},
                              {"n_chan", s[1]},
                              {"rows_spec", s[2]},
                              {"n_elem", s[3]},
                              {"off_corr", gnss_gpu::off_corr(s[0])},
                              {"off_energy", gnss_gpu::off_energy(s[0], s[1], s[2], s[3])},
                              {"bytes", gnss_gpu::frame_bytes(s[0], s[1], s[2], s[3])}});
    return {{"MAX_REC", gnss_gpu::MAX_REC},
            {"MAX_ROT_CHAN", gnss_gpu::MAX_ROT_CHAN},
            {"LANES_FULL", gnss_gpu::LANES_FULL},
            {"LANES_PROMPT", gnss_gpu::LANES_PROMPT},
            {"ROWS_PLAIN", gnss_gpu::ROWS_PLAIN},
            {"ROWS_PEEL", gnss_gpu::ROWS_PEEL},
            {"ROW_E", gnss_gpu::ROW_E},
            {"ROW_P", gnss_gpu::ROW_P},
            {"ROW_L", gnss_gpu::ROW_L},
            {"ROW_PH", gnss_gpu::ROW_PH},
            {"ROW_RES_P", gnss_gpu::ROW_RES_P},
            {"ROW_RES_PH", gnss_gpu::ROW_RES_PH},
            {"FrameHdr", wire_struct(frame_hdr_fields, std::size(frame_hdr_fields),
                                     sizeof(gnss_gpu::FrameHdr))},
            {"PrnCtl",
             wire_struct(prn_ctl_fields, std::size(prn_ctl_fields), sizeof(gnss_gpu::PrnCtl))},
            {"off_winstart", gnss_gpu::off_winstart()},
            {"off_prnctl", gnss_gpu::off_prnctl()},
            {"frame_bytes", frames}};
}

json metadata() {
    GnssChanMetadata m;
    const size_t size = m.get_serialized_size();
    const size_t seq = sizeof(m.sample_seq);
    // GnssChanMetadata::serialize writes sample_seq, then a reserved zero word. "serialized" is
    // what it writes for a known sample_seq, so moving a field or writing a nonzero word changes
    // the document.
    json fields =
        json::array({json::array({"sample_seq", 0, seq, type_tag<decltype(m.sample_seq)>()}),
                     json::array({"reserved", seq, size - seq, "u32"})});
    m.sample_seq = 0x0102030405060708;
    std::vector<char> bytes(size);
    m.serialize(bytes.data());
    std::string hex;
    for (char c : bytes)
        hex += fmt::format(fmt("{:02x}"), (unsigned char)c);
    return {{"GnssChanMetadata",
             {{"size", size},
              {"fields", fields},
              {"serialized", {{"sample_seq", m.sample_seq}, {"bytes", hex}}}}}};
}

json enums() {
    return {{"Modulation",
             {{"BPSK", static_cast<int>(Modulation::BPSK)},
              {"BOC", static_cast<int>(Modulation::BOC)}}},
            {"REC_PROJ_SRC",
             {{"REC_PROJ_SRC_OWN", REC_PROJ_SRC_OWN},
              {"REC_PROJ_SRC_SIBLING", REC_PROJ_SRC_SIBLING},
              {"REC_PROJ_SRC_PROBE", REC_PROJ_SRC_PROBE}}},
            {"PrnCtl.reanchored",
             {{"REANCHORED_NONE", REANCHORED_NONE},
              {"REANCHORED_FRESH", REANCHORED_FRESH},
              {"REANCHORED_REPIN", REANCHORED_REPIN}}}};
}

// The JSON version of each family of REST endpoints: tracker (cudaGnssChordTrack), combiner
// (GnssCoherentCombiner), search (GnssChannelizedSearch), fleet_trim (GnssFleetTrim), assembler
// (GnssGpuRecordAssemble) and tap (GnssChordVoltageTap). Bump a family's version when one of its
// endpoints adds, removes or renames a key, or changes a unit or meaning.
json rest() {
    return {{"tracker", 1},    {"combiner", 1},  {"search", 1},
            {"fleet_trim", 1}, {"assembler", 1}, {"tap", 1}};
}

} // namespace

json build_gnss_interface() {
    return {{"signals", signals()},
            {"telem", telem()},
            {"record.row", record_row()},
            {"record.chan", record_chan()},
            {"record.elem", record_elem()},
            {"record.cmb", record_cmb()},
            {"cube", cube()},
            {"gpu_chain", gpu_chain()},
            {"metadata", metadata()},
            {"enums", enums()},
            {"rest", rest()}};
}

REGISTER_KOTEKAN_INTERFACE("gnss", 1, build_gnss_interface);

} // namespace gnss
