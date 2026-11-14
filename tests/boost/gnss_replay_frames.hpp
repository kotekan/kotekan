#ifndef GNSS_REPLAY_FRAMES_HPP
#define GNSS_REPLAY_FRAMES_HPP

/// @file gnss_replay_frames.hpp
/// @brief The synthetic sky and the frame writers for test_gnss_chain_replay.
///
/// write_ctl() writes the control block as cudaGnssInject does (gnssGpuChain.hpp), and
/// write_tiles() writes the gathered N^2 tiles as the exact inverse of GnssN2RecordAssemble's
/// read, so a planted voltage comes out of the adapter as itself.

#include "gnssElemSteer.hpp"
#include "gnssGpuChain.hpp"

#include "json.hpp"

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace gnss_replay {

/// One chain: its numbers and its four stage blocks (adapter, assembler, combiner, telemetry).
struct ChainSpec {
    std::string pre; ///< e.g. "gnss0_syn_a"; the stages are "/" + pre + "_n2assemble" etc.
    int idx = 0;     ///< position in the chain list, by which the scenario names chains
    int gpu, n_prn, n_chan, n_live, num_elements, num_synth, lanes_per_slot, hops_per_record, n_rec,
        fft_len;
    double sample_rate;
    std::vector<int> prns, channel_ids;
    nlohmann::json adapter, assemble, combine, telem;
};

/// The sky every chain sees: the PRNs carrying a signal, its amplitude in units of the per-record
/// noise, and one complex gain per element.
struct Sky {
    uint64_t seed;
    double amp;
    std::set<int> planted;
    std::vector<std::complex<double>> gain_e;
};

enum class Mutation {
    none,
    rot_sign_A10,
    job0_plus4_C12,
    rename_A,
    tile_word_B10,
    clock_step_50,
    zero_amp
};

/// The events, by frame. Chain A is chains[0], B chains[1] and C chains[2].
struct Scenario {
    int n_frames = 120, swap_frame = 40, swap_slot = 2, proj_live_frame = 45, reanchor_frame = 55,
        reanchor_slot = 1, dark_r0_frame = 70, dark_r0_slot = 3, dark_r0_join = 2,
        ref_elem_frame = 90, shared_ref_frame = 30, gain_frame = 15, geom_every = 24;
    std::vector<int> checkpoints{29, 59, 89, 119};
    Mutation mut = Mutation::none;
    bool refill_a5 = true;
    uint8_t prefill = 0xA5;
    int cpu = -1;
    /// false: every combiner runs without out_buf, and its emits are read from its scratch.
    bool combiner_out_buf = true;
    /// false: the combiners keep a slot's state across a PRN swap (reset_on_prn_change).
    bool combiner_prn_reset = true;
    /// true: beside its own combiner each chain runs two without out_buf, fed copies of the
    /// swap_slot's records around the swap: "cold" from swap_frame on, so the slot starts on the
    /// new PRN with no history, and "scint" from before it with the amplitude x10 in every tenth
    /// record before swap_frame, a history far from the chain's own that the swap must clear.
    bool cold_ref = false;
};

/// Record index of frame 0's first record: a multiple of every window length the chains use
/// (4-record telemetry windows, 24/48/96/100-record spectrum and cube windows).
constexpr int64_t REC_BASE = 96000;
constexpr double UTC0 = 1794262257.0;
/// The boresight GnssGpuRecordAssemble projects about (_bore_az_deg, _bore_el_deg).
constexpr double BORE_AZ = 180.0, BORE_EL = 81.41;
/// The PRN placed at boresight, the live projection's source.
constexpr int BORE_PRN = 1;

inline uint64_t fnv1a64(const void* p, size_t n, uint64_t h = 1469598103934665603ULL) {
    const uint8_t* b = (const uint8_t*)p;
    for (size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/// Replica energy of (slot, channel), the same every record. Slots differ, so a row read with
/// another slot's energy is mis-scaled visibly.
inline double energy(const ChainSpec& s, int slot, int chan) {
    return s.hops_per_record * (1.0 + 0.05 * slot + 0.01 * chan);
}

/// Replica energy of (slot, channel) at record @p rec of @p frame: energy(), except in chain B's
/// dark_r0 frame, where it grows 2% a record, so a record scaled by another record's energy reads
/// visibly wrong there.
inline double energy_at(const ChainSpec& s, const Scenario& sc, int frame, int rec, int slot,
                        int chan) {
    const double e = energy(s, slot, chan);
    return (s.idx == 1 && frame == sc.dark_r0_frame) ? e * (1.0 + 0.02 * rec) : e;
}

/// The PRN in @p slot at @p frame: the configured one, or the swapped one from swap_frame on.
inline int prn_at(const ChainSpec& s, const Scenario& sc, int frame, int slot) {
    const int prn = s.prns[(size_t)slot];
    return (slot == sc.swap_slot && frame >= sc.swap_frame) ? prn + 100 : prn;
}

/// Does @p slot despread record @p rec of @p frame? Planted PRNs, every fourth slot from 2
/// (unplanted, noise only) and slots 0 and dark_r0_slot run; chain B's dark_r0_slot is dark
/// before record dark_r0_join of dark_r0_frame, so later records of that frame shift the job
/// numbering.
inline bool runs(const ChainSpec& s, const Scenario& sc, const Sky& sky, int frame, int rec,
                 int slot) {
    if (s.idx == 1 && frame == sc.dark_r0_frame && rec < sc.dark_r0_join && slot == sc.dark_r0_slot)
        return false;
    return slot == 0 || slot == sc.dark_r0_slot || slot % 4 == 2
           || sky.planted.count(prn_at(s, sc, frame, slot)) > 0;
}

/// Geometry of a PRN: BORE_PRN at boresight, every other PRN at least 10 deg from it.
inline void azel(int prn, double* az, double* el) {
    if (prn == BORE_PRN) {
        *az = BORE_AZ;
        *el = BORE_EL;
        return;
    }
    *az = std::fmod(prn * 137.5, 360.0);
    *el = 20.0 + std::fmod(prn * 23.0, 50.0);
}

/// UTC of record @p rec of @p frame, as the assembler stamps it.
inline double record_utc(const ChainSpec& s, int frame, int rec) {
    const int64_t wstart =
        (REC_BASE + (int64_t)s.n_rec * frame + rec) * s.hops_per_record * s.fft_len;
    return UTC0 + (double)wstart / s.sample_rate;
}

/// The [n_chan][n_elem] phasors the assembler's ElemSteer builds for @p prn, from the chain's own
/// elem_positions_enu (re-referenced to reference_element, as the assembler does), channel_ids
/// and elem_steer_sign. Unit phasors when the chain has no positions.
inline std::vector<gnss::ElemSteer::cf> steer_of(const ChainSpec& s, int prn) {
    const int ne = s.n_live, nc = s.n_chan;
    std::vector<gnss::ElemSteer::cf> tab((size_t)nc * ne, gnss::ElemSteer::cf(1.0f, 0.0f));
    std::vector<double> pos = s.assemble.value("elem_positions_enu", std::vector<double>{});
    if ((int)pos.size() != 3 * ne)
        return tab;
    const int ref = s.assemble.value("reference_element", 0);
    for (int i = 0; i < 3; ++i) {
        const double r0 = pos[(size_t)ref * 3 + i];
        for (int e = 0; e < ne; ++e)
            pos[(size_t)e * 3 + i] -= r0;
    }
    std::vector<double> f_mhz;
    for (int id : s.channel_ids)
        f_mhz.push_back(id * 0.1953125);
    gnss::ElemSteer st(std::move(pos), std::move(f_mhz), 1,
                       s.assemble.value("elem_steer_sign", 1.0), 120.0);
    double az, el;
    azel(prn, &az, &el);
    st.update(0, az, el, 0.0);
    st.copy_slot(0, tab.data());
    return tab;
}

/// NCO slope of @p slot at absolute record @p k, Hz.
inline double nco_hz(int slot, int64_t k) {
    return 350.0 + 60.0 * slot + 25.0 * (double)(k % 5);
}

/// Carrier phase of @p slot's planted signal at absolute record @p k, cycles. write_ctl's f_nco and
/// dcyc advance the assembler's NCO by exactly this phase's step, so the derotated prompt keeps
/// only the arc's starting phase.
inline double carrier_cyc(int slot, int64_t k) {
    return 0.37 * slot + 0.0113 * (slot + 1) * (double)(k % 1000) + 0.05 * (double)((k * k) % 11);
}

/// Writes the control block of @p frame into @p f (@p bytes, the n2ctl frame size), as
/// cudaGnssInject does: the whole block zeroed, every slot's PRN stamped, job0 numbered
/// compactly over the frame's running (record, slot) pairs, and the energy rows after
/// PrnCtl[MAX_REC][n_prn].
inline void write_ctl(uint8_t* f, size_t bytes, const ChainSpec& s, const Sky& sky,
                      const Scenario& sc, int frame) {
    using namespace gnss_gpu;
    std::memset(f, 0, bytes);
    auto* hdr = (FrameHdr*)f;
    auto* win = (int64_t*)(f + off_winstart());
    auto* pctl = (PrnCtl*)(f + off_prnctl());
    auto* en = (double*)(f + off_prnctl() + sizeof(PrnCtl) * MAX_REC * s.n_prn);
    const int64_t rec0 = REC_BASE + (int64_t)s.n_rec * frame;
    const double dt = (double)s.hops_per_record * s.fft_len / s.sample_rate;
    hdr->n_rec = s.n_rec;
    hdr->n_prn = s.n_prn;
    hdr->n_chan = s.n_chan;
    hdr->n_rows_spec = ROWS_PLAIN;
    hdr->lanes_per_slot = s.lanes_per_slot;
    hdr->seq0 = rec0 * s.hops_per_record * s.fft_len;
    hdr->utc0 = UTC0;
    int n_spec = 0;
    for (int r = 0; r < s.n_rec; ++r) {
        win[r] = (rec0 + r) * s.hops_per_record * s.fft_len;
        for (int p = 0; p < s.n_prn; ++p) {
            PrnCtl& c = pctl[(size_t)r * s.n_prn + p];
            c.prn = (uint16_t)prn_at(s, sc, frame, p);
            if (!runs(s, sc, sky, frame, r, p))
                continue;
            const bool was = (r > 0) ? runs(s, sc, sky, frame, r - 1, p)
                                     : (frame > 0 && runs(s, sc, sky, frame - 1, s.n_rec - 1, p));
            const bool fresh = !was || (frame == sc.swap_frame && r == 0 && p == sc.swap_slot)
                               || (frame == sc.reanchor_frame && r == 0 && p == sc.reanchor_slot);
            c.run = 1;
            c.reanchored = fresh ? 1 : 3;
            c.job0 = n_spec * ROWS_PLAIN;
            c.fcar_report = (float)(1000.0 + 37.0 * p);
            c.n_owned = (float)s.n_chan;
            c.cp_seed = 40900.5 + 3.0 * p + 0.25 * ((rec0 + r) % 8);
            const int64_t k = rec0 + r;
            c.f_nco = nco_hz(p, k);
            c.chan_mask = (s.n_chan >= 64) ? ~0ULL : ((1ULL << s.n_chan) - 1ULL);
            c.ctrim_hz = 0.6 + 0.2 * p + 0.15 * (double)(k % 3);
            c.ang0 = 0.1 * p;
            c.phi_ddop = 0.003 * (p + 1) + 0.001 * (double)(k % 4);
            c.fcar = 1.17645e9 + c.fcar_report;
            c.dcyc =
                carrier_cyc(p, k) - carrier_cyc(p, k - 1) - 0.5 * (nco_hz(p, k - 1) + c.f_nco) * dt;
            for (int ch = 0; ch < s.n_chan && ch < MAX_ROT_CHAN; ++ch)
                c.rot[ch] = (float)(0.05 * (ch + 1) + 0.01 * p);
            for (int t = 0; t < ROWS_PLAIN; ++t)
                for (int ch = 0; ch < s.n_chan; ++ch)
                    en[(size_t)(c.job0 + t) * s.n_chan + ch] = energy_at(s, sc, frame, r, p, ch);
            ++n_spec;
        }
    }
    hdr->n_jobs = n_spec * ROWS_PLAIN;
}

/// Mixed tiles per channel in the gather (GnssN2RecordAssemble's _n_tile).
inline int n_tile(const ChainSpec& s) {
    return ((s.num_elements + s.num_synth) / 16 - s.num_elements / 16) * ((s.n_live + 15) / 16);
}

/// int32 index of element @p e's real part in synth lane @p lane, channel @p c, record @p r: the
/// inverse of GnssN2RecordAssemble's tile read.
inline size_t tile_index(const ChainSpec& s, int r, int c, int lane, int e) {
    const int gi = s.num_elements + lane, ihi = gi >> 4, ilo = gi & 15;
    const int mix_k = (ihi - s.num_elements / 16) * ((s.n_live + 15) / 16) + (e >> 4);
    return (((size_t)r * s.n_chan + c) * n_tile(s) + mix_k) * 512 + 32 * ilo + 2 * (e & 15);
}

/// The quantizer scale the pack applies for replica energy @p e (cudaGnssPsiSynth).
inline double pack_scale(const ChainSpec& s, double e) {
    return 7.0 / (3.0 * std::sqrt(e / s.hops_per_record));
}

/// xorshift64*, one stream per (chain, slot, record, channel).
struct Noise {
    uint64_t x;
    Noise(uint64_t seed, const std::string& chain, int slot, int64_t rec, int chan) {
        x = fnv1a64(chain.data(), chain.size(), seed ^ 0x9e3779b97f4a7c15ULL);
        const int64_t k[3] = {slot, rec, chan};
        x = fnv1a64(k, sizeof(k), x) | 1ULL;
    }
    double uniform() {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        return (double)((x * 2685821657736338717ULL) >> 11) * (1.0 / 9007199254740992.0);
    }
    /// Unit variance from four uniforms.
    double normal() {
        return (uniform() + uniform() + uniform() + uniform() - 2.0) * 1.7320508075688772;
    }
};

/// Writes the gathered tiles of @p frame: every running slot's lanes carry V = sqrt(hops)
/// (amp g_e conj(D_ce) f_t + n), quantized as lround(s V) by the slot's energy at its first
/// running record of the frame, as cudaGnssInject's pack scales it. D is the
/// steering phasor the assembler applies, so the steered sum over elements is coherent; f_t is 1
/// for the prompt, 0.5 for its head segment and 0.7 for the E and L lanes of a LANES_FULL chain.
/// The planted term turns with the slot's carrier_cyc(), which the assembler's NCO removes.
inline void write_tiles(int32_t* t, size_t n_i32, const ChainSpec& s, const Sky& sky,
                        const Scenario& sc, int frame) {
    using namespace gnss_gpu;
    std::memset(t, 0, n_i32 * sizeof(int32_t));
    const double sq = std::sqrt((double)s.hops_per_record);
    const int64_t rec0 = REC_BASE + (int64_t)s.n_rec * frame;
    for (int p = 0; p < s.n_prn; ++p) {
        const int prn = prn_at(s, sc, frame, p);
        const bool planted = sky.planted.count(prn) > 0 && sky.amp > 0.0;
        std::vector<gnss::ElemSteer::cf> D;
        if (planted)
            D = steer_of(s, prn);
        int r_scale = -1;
        for (int r = 0; r < s.n_rec; ++r) {
            if (!runs(s, sc, sky, frame, r, p))
                continue;
            if (r_scale < 0)
                r_scale = r;
            const std::complex<double> car = std::polar(1.0, 2.0 * M_PI * carrier_cyc(p, rec0 + r));
            for (int c = 0; c < s.n_chan; ++c) {
                const double qs = pack_scale(s, energy_at(s, sc, frame, r_scale, p, c));
                Noise nz(sky.seed, s.pre, p, rec0 + r, c);
                for (int row = 0; row < ROWS_PLAIN; ++row) {
                    const int lr = lane_of_row(s.lanes_per_slot, row);
                    if (lr < 0)
                        continue;
                    const double f = (row == ROW_P) ? 1.0 : (row == ROW_PH) ? 0.5 : 0.7;
                    const int lane = s.lanes_per_slot * p + lr;
                    for (int e = 0; e < s.n_live; ++e) {
                        std::complex<double> v(nz.normal() * M_SQRT1_2, nz.normal() * M_SQRT1_2);
                        if (planted) {
                            const gnss::ElemSteer::cf d = D[(size_t)c * s.n_live + e];
                            v += sky.amp * f * sky.gain_e[(size_t)e]
                                 * std::conj(std::complex<double>(d.real(), d.imag())) * car;
                        }
                        const size_t i = tile_index(s, r, c, lane, e);
                        t[i] = (int32_t)std::lround(qs * sq * v.real());
                        t[i + 1] = (int32_t)std::lround(qs * sq * v.imag());
                    }
                }
            }
        }
    }
}

} // namespace gnss_replay

#endif // GNSS_REPLAY_FRAMES_HPP
