#include "GnssCudaDespread.hpp"

#include "cudaGnssChordDespread.hpp"
#include "cudaGnssDespreadKernel.hpp"
#include "gnssCarrierNco.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <algorithm>

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("GnssCudaDespread: ") + what + ": "
                                 + cudaGetErrorString(e));
}
} // namespace

struct GnssCudaDespread::Impl {
    gnss::ChannelizedReplicaBank& bank;
    int n_prn, n_chan, n_hops;
    double fs, f_off, refresh_hz;
    long long window_start = 0;
    /// A/B arm for task #52 -- see gnss_cuda::DespreadParams::carrier_phase_from_ref.
    /// ⚠️ TEMPORARY (task #55). Default = the fix.
    ///   0 = absolute sample (pre-86349ac4d), 1 = referenced to the window (#52's fix),
    ///   2 = ACCUMULATED (#71) -- see ang0_acc_for. The KERNEL is identical for 1 and 2;
    ///       only the host's choice of ang0 differs, so arm 2 cannot perturb the arithmetic.
    int carrier_phase_from_ref = 1;

    /// ── #71: THE ACCUMULATOR, i.e. an actual NCO ────────────────────────────────────────
    /// Arms 0 and 1 both evaluate the phase as f * n0 with n0 an ABSOLUTE sample index, so
    /// the replica's whole phase history hangs off the CURRENT frequency estimate. n0/fs is
    /// the uptime (~5.85e5 s at a week), which makes that a lever of catastrophic length: a
    /// Doppler change of 2.7e-7 Hz rotates the phase by a full radian. And the Doppler is
    /// re-propagated EVERY RECORD (cudaGnssChordTrack.cpp: propagate_seed per record, moving
    /// it by dop_rate * 10.5 ms = 1e-4..6e-3 Hz), so every record's replica phase is offset
    /// by a large, essentially arbitrary constant. It is invisible to |A| -- tracking, q, the
    /// DLL and the incoherent C/N0 are all POWER -- and fatal to everything cross-record.
    ///
    /// ⚠️ THIS IS THE SAME BUG THE L1 PATH FIXED ON 2026-07-10. GnssChannelizedTracker.cpp
    /// says it plainly: "retuning its frequency by df mid-stream rotates the whole phase
    /// history by 2*pi*df*t_abs ... this was the root cause of the L1 deep decay". That path
    /// pins the replica frequency between re-anchors and integrates the correction in an NCO
    /// (phi += 2*pi*f*dt). The CHORD path never got the same treatment.
    ///
    /// The cure is to stop asking "what is the phase at absolute sample n0" and start asking
    /// "how much phase has accrued since the last record" -- the lever becomes ONE RECORD
    /// (3.36e7 samples) instead of the uptime, and a frequency update changes the SLOPE
    /// going forward rather than teleporting the value.
    ///
    /// TRAPEZOIDAL IN f, not the last frequency held constant: the Doppler moves linearly
    /// between records (dop_rate is exactly that model), so averaging the interval's two
    /// endpoints integrates it EXACTLY rather than to first order. That is also what makes a
    /// short gap safe -- a dropped record still integrates correctly across the hole.
    std::vector<gnss::CarrierNco> pacc;
    unsigned long long pacc_reanchor = 0;

    // Device buffers (persistent).
    float2* d_data = nullptr;                  // [n_chan][n_hops]
    int8_t* d_code = nullptr;                  // all PRNs' combined-stream codes, concatenated
    std::vector<int> code_offset;              // per PRN slot
    std::vector<int8_t> code_stage;            // live-swap H2D staging (see set_prn)
    std::vector<int> slot_prn;                 // PRN per slot AS THIS ENGINE HAS IT ON DEVICE
    long code_len = 0;                         // combined-stream length (same for all PRNs)
    gnss_cuda::DespreadJob* d_jobs = nullptr;  // [n_prn]      (one quad job per PRN slot)
    double2* d_corr = nullptr;                 // [4*n_prn][n_chan]  (E, P, L, P_HEAD per slot)
    double* d_energy = nullptr;                // [4*n_prn][n_chan]

    // Per-PRN Doppler-bucketed Phi tables on device ([n_chan][Lf+1] each, all channels).
    struct PhiCache {
        bool valid = false;
        double doppler = 0.0;
        int n_chips = 0;
        float2 *d_A = nullptr, *d_B = nullptr; // float32 tables (mixed-precision kernel)
    };
    std::vector<PhiCache> phi;

    /// THE SHARED, DOPPLER-FREE TABLE SET (docs/CHORD_GPU_TODO.md item 2). One (Phi, Psi) pair
    /// for EVERY PRN, built at the band carrier alone; each job carries its own
    /// ddw = wc(prn, doppler) - wc_shared and the gather reconstructs. 14.7 MB against the
    /// 176-235 MB the per-PRN caches hold, which is what moves this kernel: §10.6c measured it
    /// DRAM-footprint-bound, and the shared pair fits the L40S's 96 MB L2 where per-PRN tables
    /// could not. Built once and never rebuilt -- it has no Doppler to go stale.
    struct SharedPhi {
        bool valid = false;
        int n_chips = 0;
        double wc = 0.0; ///< the carrier it was built at, rad/sample
        float2 *d_A = nullptr, *d_B = nullptr, *d_PA = nullptr, *d_PB = nullptr;
    };
    SharedPhi shared;
    bool use_shared = false;

    /// WHAT THE KERNEL ACTUALLY GOT, recorded per PRN slot as build_jobs runs (#72).
    /// ⚠️ RECORDED, NEVER RECOMPUTED. A producer that re-derives ang0 from the same inputs
    /// agrees with itself by construction and would keep agreeing while the kernel diverged --
    /// exactly how carrier_nco_gate passed at 9e-16 rad against a sky that got worse (#71).
    /// NaN means "this PRN built no job this record", which a consumer must read as absent
    /// rather than as zero.
    std::vector<double> last_ang0;      ///< DespreadJob::ang0, radians
    std::vector<double> last_phi_ddop;  ///< doppler_now - the Doppler this PRN's Phi was built at
    std::vector<int> all_chans;
    int Lf = 0;

    std::vector<float2> stage;  // host transpose staging [chan][hop]
    // BENCH: optional CUDA events bracketing the two CHORD kernels, so the synthesis /
    // correlation split can be measured ON THE NODE at the real geometry. Doing it here rather
    // than in a synthetic bench avoids the trap that killed the last attempt: chip_gather's depth
    // (job.n_chips) is set by the PFB span, ~8 chips/hop at the test's 40-sample hop against ~52
    // at CHORD's 16384, so a per-sample rate measured on one geometry does not transfer to the
    // other. Off by default; the events are only created when enabled.
    int max_chips = 0; ///< BENCH: cap chip_gather depth (0 = the filter's true span)
    bool split_timing = false;
    bool split_recorded = false; ///< at least one enqueue has recorded the events
    cudaEvent_t ev_a = nullptr, ev_b = nullptr, ev_c = nullptr;
    cudaStream_t stream = nullptr;
    std::vector<gnss_cuda::DespreadJob> h_jobs;
    std::vector<double2> h_corr;
    std::vector<double> h_energy;
    std::vector<gnss_cuda::PeelJob> h_pjobs; // peel staging (pageable: H2D is host-sync on return)
    std::vector<float2> h_gains;             // [job][head|tail][chan]

    Impl(gnss::ChannelizedReplicaBank& b, int np, int nc, int nh, double fs_, double fo,
         double rh, const std::vector<int>& ids) :
        bank(b), n_prn(np), n_chan(nc), n_hops(nh), fs(fs_), f_off(fo), refresh_hz(rh) {
        // Phi tables index by LOCAL channel ci; each is built at the GLOBAL bin's centre
        // frequency, taken verbatim from `ids`. NO structure is assumed -- the list may be
        // sparse, unsorted, irregularly spaced, or a single channel.
        if (nc > 64)
            throw std::runtime_error("GnssCudaDespread: >64 channels needs a wider chan_mask");
        if (nc <= 0)
            throw std::runtime_error("GnssCudaDespread: empty channel list");
        Lf = bank.fft_len() * 4; // num_taps -- matches the bank's prototype (pfb num_taps)
        // NB the bank doesn't expose num_taps; derive Lf from a probe filter below instead.
        all_chans = ids;
        pacc.assign((size_t)np, gnss::CarrierNco{}); // one NCO per PRN slot (#71)
        // Validate against the spectrum, and reject duplicates: a repeated bin would double
        // that channel's weight in the coherent sum and quietly bias every correlation.
        for (int i = 0; i < nc; ++i) {
            if (all_chans[i] < 0 || all_chans[i] >= bank.spectrum_length())
                throw std::runtime_error("GnssCudaDespread: channel id "
                                         + std::to_string(all_chans[i]) + " outside [0, "
                                         + std::to_string(bank.spectrum_length()) + ")");
            for (int j = 0; j < i; ++j)
                if (all_chans[i] == all_chans[j])
                    throw std::runtime_error("GnssCudaDespread: duplicate channel id "
                                             + std::to_string(all_chans[i]));
        }
        const auto probe = bank.hoprate_filter(all_chans, 0.0);
        Lf = (int)probe.PhiA[0].size() - 1;

        // Code table: concatenate every PRN slot's combined-stream code once.
        code_len = bank.eff_code_length();
        std::vector<int8_t> codes;
        codes.reserve((size_t)np * code_len);
        code_offset.resize(np);
        for (int p = 0; p < np; ++p) {
            code_offset[p] = (int)codes.size();
            const auto& fc = bank.full_code(p);
            codes.insert(codes.end(), fc.begin(), fc.end());
        }
        slot_prn = bank.prns();
        slot_prn.resize((size_t)np, -1);
        ck(cudaMalloc(&d_code, codes.size()), "alloc code");
        ck(cudaMemcpy(d_code, codes.data(), codes.size(), cudaMemcpyHostToDevice), "upload code");

        // Full cross-PRN batch. Jobs and output rows are sized SEPARATELY: one job per PRN slot
        // emits four rows (E, P, L, P_HEAD).
        const size_t maxs = (size_t)np;     // jobs
        const size_t maxb = (size_t)4 * np; // output rows
        ck(cudaMalloc(&d_data, (size_t)nc * nh * sizeof(float2)), "alloc data");
        ck(cudaMalloc(&d_jobs, maxs * sizeof(gnss_cuda::DespreadJob)), "alloc jobs");
        ck(cudaMalloc(&d_corr, maxb * nc * sizeof(double2)), "alloc corr");
        ck(cudaMalloc(&d_energy, maxb * nc * sizeof(double)), "alloc energy");
        ck(cudaStreamCreate(&stream), "stream");
        phi.resize(np);
        // NaN, not 0: "no job built for this PRN this record" must not read as ang0 == 0,
        // which is a perfectly legal value of frac(f*n0/fs).
        last_ang0.assign((size_t)np, std::numeric_limits<double>::quiet_NaN());
        last_phi_ddop.assign((size_t)np, std::numeric_limits<double>::quiet_NaN());
        stage.resize((size_t)nc * nh);
        h_jobs.resize(maxs);
        h_corr.resize(maxb * nc);
        h_energy.resize(maxb * nc);
    }

    ~Impl() {
        if (stream)
            cudaStreamDestroy(stream);
        cudaFree(d_data);
        cudaFree(d_code);
        cudaFree(d_jobs);
        cudaFree(d_corr);
        cudaFree(d_energy);
        for (auto& pc : phi) {
            cudaFree(pc.d_A);
            cudaFree(pc.d_B);
        }
    }

    /// THE code-period boundary hop for a window: where the PROMPT's absolute code phase
    /// C = cp0 + n_m*cps crosses the next multiple of code_len. Head is [0, m_head), tail the
    /// rest, and the secondary overlay / nav symbol flips sign exactly there.
    ///
    /// SINGLE SOURCE, deliberately. The despread cuts its P_HEAD row here and the peel switches
    /// gains here; if the two ever computed it separately and drifted by a hop, the analytic
    /// add-back would stop being exact in precisely the windows that matter (the ones with a
    /// flip) and the error would look like a small unexplained depth loss.
    /// Hop granularity is exact enough: the filter span mixes ~1 hop at the seam, versus the
    /// 12/25-records-nulled failure that segmenting repairs at all.
    int m_head_for(double cp0, double cps, long long window_start_sample) const {
        const double n0 = (double)(window_start_sample + bank.fft_len() - 1);
        const double C0 = cp0 + n0 * cps;
        const double Lc = (double)code_len;
        const double next_boundary = (std::floor(C0 / Lc) + 1.0) * Lc;
        const double hops_to_b = (next_boundary - C0) / ((double)bank.fft_len() * cps);
        int m = (int)std::ceil(hops_to_b);
        if (m < 0)
            m = 0;
        if (m > n_hops)
            m = n_hops;
        return m;
    }

    /// chips per sample including the code Doppler, and the carrier angular rate -- the two
    /// derived quantities every job (despread or peel) needs from a Doppler.
    double cps_for(double doppler_hz) const {
        return bank.eff_chip_rate() / fs
               * (1.0 + bank.code_doppler_sign * doppler_hz / bank.carrier_hz());
    }
    // ★ PER-PRN carrier. f_off is the BAND offset; under FDMA (GLONASS L1OF/L2OF) each
    // satellite sits on its own carrier, so the phasor has to come from the bank's per-PRN
    // total. Identical to f_off for every CDMA signal, where no offsets are set.
    // ctrim is the CARRIER trim, carrier-only -- cps_for deliberately does not take it, because
    // code and carrier are separate control paths. Defaulted to 0 for the peel path, which never
    // carried one. ensure_phi stays keyed on doppler alone: it caches the channelizer response,
    // and a few tens of Hz is nothing against a MHz-wide channel.
    double wc_for(int p, double doppler_hz, double ctrim_hz = 0.0) const {
        return 2.0 * M_PI * (bank.carrier_offset(p) + doppler_hz + ctrim_hz) / fs;
    }

    /// Carrier phase at absolute sample n0, radians in [0, 2*pi). Task #52 -- see
    /// gnss_cuda::DespreadJob::ang0 for why this has to exist at all.
    ///
    /// LONG DOUBLE IS LOAD-BEARING. The quantity is frac(f * n0 / fs) with f ~ 1.18e9 and
    /// n0 ~ 1.9e15, i.e. 6.9e14 whole cycles to be discarded before a fraction survives. In
    /// double that leaves nothing (the ulp at 6.9e14 is 0.125 cycles); x87 long double's 64-bit
    /// mantissa leaves ~3.7e-5 cycles = 2.3e-4 rad, three orders below the 0.2 rad this
    /// removes. CUDA maps long double to double, which is exactly why the reduction is done
    /// HERE and shipped to the kernel rather than computed there.
    ///
    /// ⚠️ CALLERS MUST PASS THE SAME n0 THE KERNEL USES (par.n0 = window_start + fft_len - 1,
    /// the hop's LAST sample). A first/last mismatch is a constant rotation of the whole
    /// record, and that convention has already cost 52 chips once on the code side.
    /// Prompt code phase at absolute sample n0, combined-stream chips, reduced mod code_len.
    /// Task #54 -- the code twin of ang0_for, and long double is load-bearing for the same
    /// reason: cp0 + n0*cps is ~6e12 chips at CHORD uptime and a double keeps ~1e-3 of a chip
    /// there, re-rolled every hop. See gnss_cuda::DespreadJob::cp_ref.
    ///
    /// Reduced mod code_len so the kernel's gather sees a small argument; the code is periodic
    /// so this is exact, and it keeps cp_ref from being a large number again by the back door.
    double cp_ref_for(double cp0, double cps, long long n0) const {
        long double c = (long double)cp0 + (long double)n0 * (long double)cps;
        const long double L = (long double)code_len;
        c = fmodl(c, L);
        if (c < 0.0L)
            c += L;
        return (double)c;
    }

    double ang0_for(int p, double doppler_hz, double ctrim_hz, long long n0) const {
        constexpr long double TWO_PI_L = 6.283185307179586476925286766559005768L;
        const long double f = (long double)bank.carrier_offset(p) + (long double)doppler_hz
                              + (long double)ctrim_hz;
        long double fr = fmodl(f / (long double)fs * (long double)n0, 1.0L);
        if (fr < 0.0L)
            fr += 1.0L;
        return (double)(TWO_PI_L * fr);
    }

    /// Arm 2's ang0: the phase ACCRUED since this PRN's previous record, not the phase at an
    /// absolute sample. See PhaseAcc for why. Stateful and therefore ORDER-DEPENDENT -- it
    /// must be called once per PRN per record, in record order, which is how build_jobs runs.
    ///
    /// ⚠️ THE ABSOLUTE VALUE IS MEANINGLESS AND THAT IS FINE. Only phase DIFFERENCES between
    /// records are observable in a correlation; the constant of integration cancels. What
    /// this buys is that the difference is now the physical one instead of
    /// 2*pi*df*uptime of bookkeeping.
    double ang0_acc_for(int p, double doppler_hz, double ctrim_hz, long long n0) {
        const double f_now = bank.carrier_offset(p) + doppler_hz + ctrim_hz;
        if ((size_t)p >= pacc.size())
            return ang0_for(p, doppler_hz, ctrim_hz, n0);
        // A gap longer than this cannot be integrated honestly: only the endpoint frequencies
        // are known and the Doppler may have been re-seeded inside the hole, so re-anchor and
        // SAY SO rather than extrapolate. 64 records ~ 0.67 s; the trapezoid's own error over
        // that is < 1e-3 cycles at a 0.5 Hz/s dop_rate, so the bound is about not trusting the
        // model through a re-seed, not about arithmetic.
        const long long GAP_MAX = 64LL * 2048LL * 16384LL;
        // Seed a re-anchor from the absolute form: any value is admissible (only phase
        // DIFFERENCES are observable), and this one keeps arm 2 numerically comparable to
        // arm 1 on the first record of a track.
        return gnss::carrier_nco_advance(pacc[(size_t)p], f_now, n0, fs,
                                         ang0_for(p, doppler_hz, ctrim_hz, n0), GAP_MAX,
                                         &pacc_reanchor);
    }

    /// Build the shared, Doppler-free (Phi, Psi) set once. Returns false if this signal is
    /// FDMA, where the scheme does not apply: there the satellite identity lives in the CARRIER
    /// (_prn_df), so different PRNs have genuinely different tables and there is nothing to
    /// share. Detected by asking the bank, never assumed -- ChannelizedReplicaBank::swap_prn
    /// already refuses FDMA for the same reason.
    bool ensure_shared() {
        if (shared.valid)
            return true;
        for (int p = 1; p < n_prn; ++p)
            if (bank.carrier_offset(p) != bank.carrier_offset(0)) {
                std::fprintf(stderr, "WARNING: GnssCudaDespread: shared Phi REFUSED -- "
                                     "per-PRN carrier offsets differ (FDMA); falling back to "
                                     "per-PRN tables.\n");
                return false;
            }
        const auto f = bank.hoprate_filter(all_chans, 0.0, -1, /*want_psi=*/true);
        const size_t n = (size_t)n_chan * (size_t)(Lf + 1);
        std::vector<float2> hA(n), hB(n), hPA(n), hPB(n);
        for (int c = 0; c < n_chan; ++c)
            for (int k = 0; k <= Lf; ++k) {
                const size_t o = (size_t)c * (size_t)(Lf + 1) + (size_t)k;
                hA[o] = make_float2((float)f.PhiA[c][k].real(), (float)f.PhiA[c][k].imag());
                hB[o] = make_float2((float)f.PhiB[c][k].real(), (float)f.PhiB[c][k].imag());
                hPA[o] = make_float2((float)f.PsiA[c][k].real(), (float)f.PsiA[c][k].imag());
                hPB[o] = make_float2((float)f.PsiB[c][k].real(), (float)f.PsiB[c][k].imag());
            }
        ck(cudaMalloc(&shared.d_A, n * sizeof(float2)), "alloc shared PhiA");
        ck(cudaMalloc(&shared.d_B, n * sizeof(float2)), "alloc shared PhiB");
        ck(cudaMalloc(&shared.d_PA, n * sizeof(float2)), "alloc shared PsiA");
        ck(cudaMalloc(&shared.d_PB, n * sizeof(float2)), "alloc shared PsiB");
        ck(cudaMemcpy(shared.d_A, hA.data(), n * sizeof(float2), cudaMemcpyHostToDevice), "sPhiA");
        ck(cudaMemcpy(shared.d_B, hB.data(), n * sizeof(float2), cudaMemcpyHostToDevice), "sPhiB");
        ck(cudaMemcpy(shared.d_PA, hPA.data(), n * sizeof(float2), cudaMemcpyHostToDevice), "sPsiA");
        ck(cudaMemcpy(shared.d_PB, hPB.data(), n * sizeof(float2), cudaMemcpyHostToDevice), "sPsiB");
        shared.n_chips = f.n_chips;
        shared.wc = f.wc_built;
        shared.valid = true;
        std::fprintf(stderr,
                     "INFO: GnssCudaDespread: SHARED Phi/Psi built at wc %.6f rad/sample, "
                     "%d chips, %.1f MB over %d channels -- serving all %d PRNs "
                     "(per-PRN tables would be %.1f MB)\n",
                     shared.wc, shared.n_chips, 4.0 * (double)n * (double)sizeof(float2) / 1e6,
                     n_chan, n_prn,
                     2.0 * (double)n * (double)sizeof(float2) * (double)n_prn / 1e6);
        return true;
    }

    // (Re)build PRN p's Phi bucket at this Doppler if it moved more than refresh_hz.
    PhiCache& ensure_phi(int p, double doppler) {
        PhiCache& pc = phi[(size_t)p];
        if (pc.valid && std::fabs(doppler - pc.doppler) <= refresh_hz)
            return pc;
        // phi[] is already indexed per PRN; build this PRN's carrier into it too.
        const auto f = bank.hoprate_filter(all_chans, doppler, p);
        const size_t n = (size_t)n_chan * (Lf + 1);
        std::vector<float2> hA(n), hB(n);
        for (int c = 0; c < n_chan; ++c)
            for (int k = 0; k <= Lf; ++k) {
                hA[(size_t)c * (Lf + 1) + k] =
                    make_float2((float)f.PhiA[c][k].real(), (float)f.PhiA[c][k].imag());
                hB[(size_t)c * (Lf + 1) + k] =
                    make_float2((float)f.PhiB[c][k].real(), (float)f.PhiB[c][k].imag());
            }
        if (!pc.d_A) {
            ck(cudaMalloc(&pc.d_A, n * sizeof(float2)), "alloc PhiA");
            ck(cudaMalloc(&pc.d_B, n * sizeof(float2)), "alloc PhiB");
        }
        ck(cudaMemcpy(pc.d_A, hA.data(), n * sizeof(float2), cudaMemcpyHostToDevice), "PhiA up");
        ck(cudaMemcpy(pc.d_B, hB.data(), n * sizeof(float2), cudaMemcpyHostToDevice), "PhiB up");
        pc.valid = true;
        pc.doppler = doppler;
        pc.n_chips = f.n_chips;
        // BENCH: cap the gather depth. n_chips = ceil((Lf-1)*cps)+2 is the chips the PFB filter
        // spans -- 210 at CHORD (65536-sample span, 312.8 samples/chip) against the 13 the
        // kernel's own comment describes for airspy L5. If synthesis time is proportional to
        // this, the gather loop IS the cost and a layout fix is worth the risk; if it is not,
        // the bottleneck is elsewhere and the layout change would be wasted. TRUNCATES THE
        // FILTER, so a capped run is for TIMING ONLY -- the despread output is wrong.
        if (max_chips > 0 && pc.n_chips > max_chips)
            pc.n_chips = max_chips;
        return pc;
    }
};

GnssCudaDespread::GnssCudaDespread(gnss::ChannelizedReplicaBank& bank, int n_prn,
                                   const std::vector<int>& chan_ids, int n_hops,
                                   double sample_rate, double f_offset, double refresh_hz) :
    _impl(new Impl(bank, n_prn, (int)chan_ids.size(), n_hops, sample_rate, f_offset, refresh_hz,
                   chan_ids)) {
    // FDMA (GLONASS L1OF/L2OF) IS supported: the Phi cache was already indexed per PRN, so it
    // only needed the per-PRN carrier passed into the filter and the phasor (see wc_for). A
    // guard here first caught that this path -- reached INTERNALLY from cudaGnssTrack, which is
    // why grepping the band config for "GnssCudaDespread" found nothing -- was quietly building
    // every satellite's filter at the band centre.
    //
    // ⚠️ `chan_ids`, not (n_chan, chan_offset). CHORD's covering set is a STRIDED comb
    // (5972, 5988, ... 6068), and a contiguous range is what put every replica at DC -- the
    // lock blocker. A contiguous band is expressible as ids, so this signature is the superset.
}

GnssCudaDespread::~GnssCudaDespread() = default;

int GnssCudaDespread::prn_at(int p) const {
    // THIS ENGINE's device view, not the shared bank's -- see set_prn.
    return (p >= 0 && (size_t)p < _impl->slot_prn.size()) ? _impl->slot_prn[(size_t)p] : -1;
}

bool GnssCudaDespread::set_prn(int p, int prn, void* stream) {
    Impl& im = *_impl;
    if (p < 0 || p >= im.n_prn)
        return false;
    // ⚠️ COMPARE AGAINST THIS ENGINE'S OWN VIEW, NOT THE BANK'S. Several engines can share
    // one bank (the search builds one GnssCudaDespread per <=64-channel refine group over a
    // single ChannelizedReplicaBank), and each keeps its OWN device copy of the code table.
    // Short-circuiting on the bank would let the first engine update the shared bank and every
    // other engine then conclude it had nothing to do -- leaving them correlating the new
    // satellite's model against the old satellite's code, on the GPU, with no symptom but a
    // refine that never finds a peak.
    if (im.slot_prn[(size_t)p] == prn)
        return true; // this engine is already there
    if (im.bank.prn_at(p) != prn && !im.bank.set_prn(p, prn))
        return false;
    // Re-upload just this slot's row of the concatenated code table. The staging buffer is a
    // MEMBER, not a local: the copy is stream-ordered and therefore outlives this call, and a
    // local vector would be freed while the DMA still had its pages.
    if (im.code_stage.size() != (size_t)im.n_prn * (size_t)im.code_len)
        im.code_stage.assign((size_t)im.n_prn * (size_t)im.code_len, 0);
    const auto& fc = im.bank.full_code(p);
    std::copy(fc.begin(), fc.end(), im.code_stage.begin() + (size_t)im.code_offset[(size_t)p]);
    // nullptr = this engine's OWN stream, which is what despread_batch enqueues on. Callers
    // that drive the engine from an external stream (the tracker) pass theirs; callers that
    // only ever use the batch API (the search) can pass nothing and still be ordered.
    const cudaStream_t st = stream ? (cudaStream_t)stream : im.stream;
    ck(cudaMemcpyAsync(im.d_code + im.code_offset[(size_t)p],
                       im.code_stage.data() + im.code_offset[(size_t)p], (size_t)im.code_len,
                       cudaMemcpyHostToDevice, st),
       "code slot re-upload");
    // Per-slot state HELD HERE. The Phi cache is carrier-only (so for CDMA it would survive
    // correctly), but it is invalidated anyway: it is rebuilt on the next record for a few
    // hundred microseconds, and a cache keyed to a satellite that has left is precisely the
    // kind of thing that stays right until the day it doesn't.
    im.phi[(size_t)p].valid = false;
    // The accumulated carrier NCO (#71 arm 2) MUST reset -- its whole content is the phase
    // history of a satellite this slot no longer holds. Clearing it makes the next record a
    // fresh anchor, which is the honest description of what just happened.
    if ((size_t)p < im.pacc.size())
        im.pacc[(size_t)p] = gnss::CarrierNco{};
    im.last_ang0[(size_t)p] = std::numeric_limits<double>::quiet_NaN();
    im.last_phi_ddop[(size_t)p] = std::numeric_limits<double>::quiet_NaN();
    im.slot_prn[(size_t)p] = prn;
    return true;
}

double GnssCudaDespread::last_ang0(int p) const {
    return (p >= 0 && (size_t)p < _impl->last_ang0.size())
               ? _impl->last_ang0[(size_t)p] : std::numeric_limits<double>::quiet_NaN();
}

double GnssCudaDespread::last_phi_ddop(int p) const {
    return (p >= 0 && (size_t)p < _impl->last_phi_ddop.size())
               ? _impl->last_phi_ddop[(size_t)p] : std::numeric_limits<double>::quiet_NaN();
}

void GnssCudaDespread::upload_window(const std::complex<float>* window,
                                     long long window_start_sample) {
    Impl& im = *_impl;
    // Tracker holds [hop][chan]; kernel wants [chan][hop] -- transpose through the staging buffer.
    for (int m = 0; m < im.n_hops; ++m)
        for (int c = 0; c < im.n_chan; ++c) {
            const std::complex<float>& v = window[(size_t)m * im.n_chan + c];
            im.stage[(size_t)c * im.n_hops + m] = make_float2(v.real(), v.imag());
        }
    ck(cudaMemcpyAsync(im.d_data, im.stage.data(), im.stage.size() * sizeof(float2),
                       cudaMemcpyHostToDevice, im.stream),
       "window upload");
    im.window_start = window_start_sample;
}

void GnssCudaDespread::set_carrier_phase_from_ref(bool on) {
    _impl->carrier_phase_from_ref = on ? 1 : 0;
}

void GnssCudaDespread::set_carrier_phase_mode(int mode) {
    // Clamped rather than trusted: an out-of-range arm would silently fall through to the
    // absolute-sample path, which is the one arm nobody wants and the hardest to notice.
    _impl->carrier_phase_from_ref = (mode < 0) ? 0 : ((mode > 2) ? 2 : mode);
}

unsigned long long GnssCudaDespread::carrier_phase_reanchors() const {
    return _impl->pacc_reanchor;
}

void GnssCudaDespread::enable_split_timing(bool on) {
    Impl& im = *_impl;
    if (on && !im.ev_a) {
        ck(cudaEventCreate(&im.ev_a), "ev_a");
        ck(cudaEventCreate(&im.ev_b), "ev_b");
        ck(cudaEventCreate(&im.ev_c), "ev_c");
    }
    im.split_timing = on;
}

bool GnssCudaDespread::split_timing_ms(double& synthesis_ms, double& correlation_ms) const {
    Impl& im = *_impl;
    // split_recorded matters: the tracker queries this near the TOP of execute, so on the very
    // first frame the events exist but have never been recorded. cudaEventElapsedTime on those
    // is an error rather than a crash, but relying on that is not a guarantee worth taking.
    if (!im.split_timing || !im.ev_c || !im.split_recorded)
        return false;
    // The caller reads this after the frame has completed (gpuProcess logs profiling from
    // results_thread, post finalize), so the events are already resolved; synchronize anyway
    // rather than assume it.
    if (cudaEventSynchronize(im.ev_c) != cudaSuccess)
        return false;
    float a = 0.f, b = 0.f;
    if (cudaEventElapsedTime(&a, im.ev_a, im.ev_b) != cudaSuccess
        || cudaEventElapsedTime(&b, im.ev_b, im.ev_c) != cudaSuccess)
        return false;
    synthesis_ms = a;
    correlation_ms = b;
    return true;
}

bool GnssCudaDespread::set_shared_phi(bool on) {
    if (!on) {
        _impl->use_shared = false;
        return false;
    }
    _impl->use_shared = _impl->ensure_shared();
    return _impl->use_shared;
}

void GnssCudaDespread::set_max_chips(int n) {
    _impl->max_chips = n;
    for (auto& pc : _impl->phi) // force a rebuild so the cap takes effect on cached buckets
        pc.valid = false;
}

int GnssCudaDespread::max_batch_specs() const {
    // THIS path's arena is `maxs = np` (see the constructor) -- ONE job per PRN slot, on the
    // tracker's assumption of one spec per PRN. It is NOT gnss_gpu::max_specs(n_prn) =
    // n_prn * MAX_REC; that larger bound belongs to the DEVICE path (enqueue_batch_device), whose
    // per-frame arena slice holds every record in flight. Confusing the two over-batches by
    // MAX_REC and the only symptom is "jobs upload: invalid argument" from the H2D, an opaque
    // CUDA error a long way from the cause. A caller batching many specs against ONE PRN (the A5
    // refine scan) must chunk against this.
    return _impl->n_prn;
}

std::vector<std::array<gnss::DespreadResult, 3>>
GnssCudaDespread::despread_batch(const std::vector<Spec>& specs) {
    Impl& im = *_impl;
    std::vector<std::array<gnss::DespreadResult, 3>> out(specs.size());
    if (specs.empty())
        return out;
    const int n_spec = (int)specs.size();
    const int n_out = 4 * n_spec; // the kernel always emits the full quad

    // Build the job array: one quad job per spec, each carrying its PRN's Doppler-bucketed Phi
    // tables (rebuilt+uploaded only when that sat's Doppler moved > refresh_hz). m_head = 0 --
    // this path's contract is the plain E/P/L triple, so the head row comes back all-zero and is
    // dropped below (the segmented despread lives on the device path, enqueue_batch_device).
    for (size_t i = 0; i < specs.size(); ++i) {
        const Spec& sp = specs[i];
        auto& pc = im.ensure_phi(sp.p, sp.doppler_hz);
        const double cps =
            im.bank.eff_chip_rate() / im.fs
            * (1.0 + im.bank.code_doppler_sign * sp.doppler_hz / im.bank.carrier_hz());
        // + ctrim: the CARRIER trim, carrier-only (cps above deliberately does not take
        // it -- code and carrier are separate control paths). ensure_phi stays keyed on
        // doppler alone: it caches the channelizer response, and a few tens of Hz is
        // nothing against a MHz-wide channel.
        const double wc = im.wc_for(sp.p, sp.doppler_hz, sp.ctrim_hz);
        uint64_t mask = 0;
        for (int c : sp.covering)
            if (c >= 0 && c < im.n_chan)
                mask |= (1ULL << c);
        const double cp0_i = (double)im.bank.comb_mult() * sp.cp_seed;
        im.h_jobs[i] = {cp0_i,
                        (double)im.bank.comb_mult() * sp.spacing_chips,
                        // Same n0 as ang0 and par.n0 -- the hop's LAST sample (task #54).
                        im.cp_ref_for(cp0_i, cps, im.window_start + im.bank.fft_len() - 1),
                        cps,
                        1.0 / cps,
                        wc,
                        // Same n0 the kernel is handed below (par.n0) -- hop's LAST sample.
                        (im.carrier_phase_from_ref == 2
                             ? im.ang0_acc_for(sp.p, sp.doppler_hz, sp.ctrim_hz,
                                               im.window_start + im.bank.fft_len() - 1)
                             : im.ang0_for(sp.p, sp.doppler_hz, sp.ctrim_hz,
                                           im.window_start + im.bank.fft_len() - 1)),
                        im.code_offset[(size_t)sp.p],
                        (int)im.code_len,
                        mask,
                        im.use_shared ? im.shared.d_A : pc.d_A,
                        im.use_shared ? im.shared.d_B : pc.d_B,
                        im.use_shared ? im.shared.n_chips : pc.n_chips,
                        0,
                        // SHARED-TABLE MODE: the Doppler this PRN needs, MINUS the carrier the
                        // shared table was built at. Per-PRN mode leaves these null/0 and the
                        // gather takes its original, bit-identical path.
                        im.use_shared ? im.shared.d_PA : nullptr,
                        im.use_shared ? im.shared.d_PB : nullptr,
                        im.use_shared
                            ? (float)(2.0 * M_PI
                                          * (im.bank.carrier_offset(sp.p) + sp.doppler_hz)
                                          / im.fs
                                      - im.shared.wc)
                            : 0.0f};
        // #72: record WHAT THE KERNEL IS GETTING -- read back OUT OF THE JOB, never re-derived.
        // A producer-side re-derivation agrees with itself by construction and would keep
        // agreeing while the kernel diverged (how carrier_nco_gate passed at 9e-16 rad while
        // the sky got worse, #71).
        im.last_ang0[(size_t)sp.p] = im.h_jobs[i].ang0;
        im.last_phi_ddop[(size_t)sp.p] = pc.valid ? (sp.doppler_hz - pc.doppler)
                                                  : std::numeric_limits<double>::quiet_NaN();
    }
    ck(cudaMemcpyAsync(im.d_jobs, im.h_jobs.data(),
                       (size_t)n_spec * sizeof(gnss_cuda::DespreadJob), cudaMemcpyHostToDevice,
                       im.stream),
       "jobs upload");

    gnss_cuda::DespreadParams par;
    par.shared = im.use_shared; // item 2: picks the shared-table kernel
    par.n0 = im.window_start + im.bank.fft_len() - 1; // hoprate_stream's per-hop reference
    par.carrier_phase_from_ref = im.carrier_phase_from_ref;
    par.fft_len = im.bank.fft_len();
    par.n_hops = im.n_hops;
    par.Lf = im.Lf;
    par.data_stride = im.n_hops; // packed per-record staging buffer
    ck(gnss_cuda::launch_despread(im.d_data, im.d_code, im.d_jobs, n_spec, im.n_chan, par,
                                  im.d_corr, im.d_energy, im.stream),
       "launch");
    ck(cudaMemcpyAsync(im.h_corr.data(), im.d_corr, (size_t)n_out * im.n_chan * sizeof(double2),
                       cudaMemcpyDeviceToHost, im.stream),
       "corr down");
    ck(cudaMemcpyAsync(im.h_energy.data(), im.d_energy, (size_t)n_out * im.n_chan * sizeof(double),
                       cudaMemcpyDeviceToHost, im.stream),
       "energy down");
    ck(cudaStreamSynchronize(im.stream), "sync"); // ONE sync per record for the whole batch

    for (size_t i = 0; i < specs.size(); ++i)
        for (int t = 0; t < 3; ++t) { // rows 4i+0..2 = E/P/L; row 4i+3 (head) unused here
            const size_t row = (size_t)(4 * i + t) * im.n_chan;
            std::complex<double> g(0.0, 0.0);
            double e = 0.0;
            for (int c = 0; c < im.n_chan; ++c) {
                g += std::complex<double>(im.h_corr[row + c].x, im.h_corr[row + c].y);
                e += im.h_energy[row + c];
            }
            out[i][t].correlation = g;
            out[i][t].replica_energy = e;
            out[i][t].amplitude = (e > 0.0) ? g / e : std::complex<double>(0.0, 0.0);
        }
    return out;
}

// Build and upload the per-PRN jobs. Shared verbatim by the fused path and the CHORD N x M
// path: the two differ ONLY in which kernels consume the jobs, and the whole point of the split
// is that the replicas stay identical, so the job construction must not fork.
void GnssCudaDespread::build_jobs(const std::vector<Spec>& specs, void* d_jobs_slot,
                                  long long window_start_sample, void* stream) {
    Impl& im = *_impl;
    const int n_spec = (int)specs.size();
    const cudaStream_t st = (cudaStream_t)stream;

    // h_jobs staging is pageable, so the async H2D below is host-synchronous on return and the
    // buffer is safely reusable per call.
    for (size_t i = 0; i < specs.size(); ++i) {
        const Spec& sp = specs[i];
        auto& pc = im.ensure_phi(sp.p, sp.doppler_hz);
        const double cps =
            im.bank.eff_chip_rate() / im.fs
            * (1.0 + im.bank.code_doppler_sign * sp.doppler_hz / im.bank.carrier_hz());
        // + ctrim: the CARRIER trim, carrier-only (cps above deliberately does not take
        // it -- code and carrier are separate control paths). ensure_phi stays keyed on
        // doppler alone: it caches the channelizer response, and a few tens of Hz is
        // nothing against a MHz-wide channel.
        const double wc = im.wc_for(sp.p, sp.doppler_hz, sp.ctrim_hz);
        uint64_t mask = 0;
        for (int c : sp.covering)
            if (c >= 0 && c < im.n_chan)
                mask |= (1ULL << c);
        const double cp0 = (double)im.bank.comb_mult() * sp.cp_seed;

        // Output row 3 = P_HEAD: the prompt cut at the code-period boundary (Impl::m_head_for --
        // shared with the peel, which switches gains at the same hop).
        const int m_head = im.m_head_for(cp0, cps, window_start_sample);

        im.h_jobs[i] = {cp0,
                        (double)im.bank.comb_mult() * sp.spacing_chips,
                        im.cp_ref_for(cp0, cps, window_start_sample + im.bank.fft_len() - 1),
                        cps,
                        1.0 / cps,
                        wc,
                        (im.carrier_phase_from_ref == 2
                             ? im.ang0_acc_for(sp.p, sp.doppler_hz, sp.ctrim_hz,
                                               window_start_sample + im.bank.fft_len() - 1)
                             : im.ang0_for(sp.p, sp.doppler_hz, sp.ctrim_hz,
                                           window_start_sample + im.bank.fft_len() - 1)),
                        im.code_offset[(size_t)sp.p],
                        (int)im.code_len,
                        mask,
                        im.use_shared ? im.shared.d_A : pc.d_A,
                        im.use_shared ? im.shared.d_B : pc.d_B,
                        im.use_shared ? im.shared.n_chips : pc.n_chips,
                        m_head,
                        // SHARED-TABLE MODE (item 2): psi companions + this PRN's Doppler
                        // offset from the shared table's own carrier. Per-PRN mode leaves these
                        // null/0 and the gather takes its original, bit-identical path.
                        im.use_shared ? im.shared.d_PA : nullptr,
                        im.use_shared ? im.shared.d_PB : nullptr,
                        im.use_shared
                            ? (float)(2.0 * M_PI
                                          * (im.bank.carrier_offset(sp.p) + sp.doppler_hz)
                                          / im.fs
                                      - im.shared.wc)
                            : 0.0f};
        // #72: record WHAT THE KERNEL IS GETTING -- read back OUT OF THE JOB, never re-derived.
        // A producer-side re-derivation agrees with itself by construction and would keep
        // agreeing while the kernel diverged (how carrier_nco_gate passed at 9e-16 rad while
        // the sky got worse, #71).
        im.last_ang0[(size_t)sp.p] = im.h_jobs[i].ang0;
        im.last_phi_ddop[(size_t)sp.p] = pc.valid ? (sp.doppler_hz - pc.doppler)
                                                  : std::numeric_limits<double>::quiet_NaN();
    }
    ck(cudaMemcpyAsync(d_jobs_slot, im.h_jobs.data(),
                       (size_t)n_spec * sizeof(gnss_cuda::DespreadJob), cudaMemcpyHostToDevice,
                       st),
       "jobs upload (device batch)");
}

int GnssCudaDespread::enqueue_batch_device(const void* d_window, int data_stride,
                                           long long window_start_sample,
                                           const std::vector<Spec>& specs, void* d_jobs_slot,
                                           void* d_corr_out, void* d_energy_out, void* stream,
                                           const void* d_chan_scale, void* d_xcorr_out,
                                           int rows_spec) {
    Impl& im = *_impl;
    if (specs.empty())
        return 0;
    const int n_spec = (int)specs.size();
    const cudaStream_t st = (cudaStream_t)stream;
    build_jobs(specs, d_jobs_slot, window_start_sample, stream);

    gnss_cuda::DespreadParams par;
    par.shared = im.use_shared; // item 2: picks the shared-table kernel
    par.n0 = window_start_sample + im.bank.fft_len() - 1; // hoprate_stream's per-hop reference
    par.carrier_phase_from_ref = im.carrier_phase_from_ref;
    par.fft_len = im.bank.fft_len();
    par.n_hops = im.n_hops;
    par.Lf = im.Lf;
    par.data_stride = data_stride;
    par.out_rows_spec = rows_spec; // 6 when the frame also carries the peel residual rows
    // d_chan_scale selects the ring's sample type: 4+4b bytes (decode with the scales the
    // quantizer encoded with) vs fp32. Same kernel, only the voltage load differs.
    if (d_chan_scale)
        ck(gnss_cuda::launch_despread_q((const unsigned char*)d_window,
                                        (const float*)d_chan_scale, im.d_code,
                                        (gnss_cuda::DespreadJob*)d_jobs_slot, n_spec, im.n_chan,
                                        par, (double2*)d_corr_out, (double*)d_energy_out, st,
                                        (double2*)d_xcorr_out),
           "launch q (device batch)");
    else
        ck(gnss_cuda::launch_despread((const float2*)d_window, im.d_code,
                                      (gnss_cuda::DespreadJob*)d_jobs_slot, n_spec, im.n_chan,
                                      par, (double2*)d_corr_out, (double*)d_energy_out, st,
                                      (double2*)d_xcorr_out),
           "launch (device batch)");
    // OUTPUT rows RESERVED per spec -- not the job count, and not the rows actually written
    // (with rows_spec 6 the despread fills 0-3 and the add-back fills 4-5).
    return rows_spec * n_spec;
}

int GnssCudaDespread::enqueue_batch_nm(const void* d_frame, const void* d_chan_scale,
                                       const int* d_chan_ids, void* d_wave, int n_elem,
                                       int elem_stride, int frame_chan_stride,
                                       long long window_start_sample,
                                       const std::vector<Spec>& specs, void* d_jobs_slot,
                                       void* d_corr_out, void* d_energy_out, void* stream,
                                       bool conjugate) {
    Impl& im = *_impl;
    if (specs.empty())
        return 0;
    const int n_spec = (int)specs.size();
    const cudaStream_t st = (cudaStream_t)stream;
    build_jobs(specs, d_jobs_slot, window_start_sample, stream);

    gnss_cuda::DespreadParams par;
    par.shared = im.use_shared; // item 2: picks the shared-table kernel
    par.n0 = window_start_sample + im.bank.fft_len() - 1; // hoprate_stream's per-hop reference
    par.carrier_phase_from_ref = im.carrier_phase_from_ref;
    par.fft_len = im.bank.fft_len();
    par.n_hops = im.n_hops;
    par.Lf = im.Lf;
    // The CHORD correlator reads the frame in its native [hop][chan][elem] order, so there is no
    // channel-major ring and no data_stride to speak of; the geometry is carried by
    // frame_chan_stride/elem_stride instead.
    par.data_stride = im.n_hops;
    par.out_rows_spec = 4;
    par.conj_data = conjugate ? 1 : 0;

    // GENERATE ONCE. The replicas are materialised here and reused across all n_elem antennas by
    // the correlator below -- that reuse is the entire reason the fused kernel is split for
    // CHORD. Both launches go on the SAME stream, so the correlator cannot start before the
    // waveform it consumes is complete.
    if (im.split_timing)
        ck(cudaEventRecord(im.ev_a, st), "split ev_a");
    ck(gnss_cuda::launch_waveform(im.d_code, (gnss_cuda::DespreadJob*)d_jobs_slot, n_spec,
                                  im.n_chan, par, (float2*)d_wave, (double*)d_energy_out, st),
       "launch waveform (NxM)");
    if (im.split_timing)
        ck(cudaEventRecord(im.ev_b, st), "split ev_b");
    ck(gnss_cuda::launch_correlate_nm((const unsigned char*)d_frame, (const float*)d_chan_scale,
                                      d_chan_ids, (const float2*)d_wave,
                                      (gnss_cuda::DespreadJob*)d_jobs_slot, n_spec, im.n_chan,
                                      n_elem, elem_stride, frame_chan_stride, par,
                                      (double2*)d_corr_out, st),
       "launch correlate NxM");
    if (im.split_timing) {
        ck(cudaEventRecord(im.ev_c, st), "split ev_c");
        im.split_recorded = true;
    }
    return 4 * n_spec;
}

int GnssCudaDespread::enqueue_waveform(long long window_start_sample,
                                       const std::vector<Spec>& specs, void* d_jobs_slot,
                                       void* d_wave, void* d_energy_out, void* stream) {
    Impl& im = *_impl;
    if (specs.empty())
        return 0;
    const int n_spec = (int)specs.size();
    const cudaStream_t st = (cudaStream_t)stream;
    build_jobs(specs, d_jobs_slot, window_start_sample, stream);

    gnss_cuda::DespreadParams par;
    par.shared = im.use_shared; // item 2: picks the shared-table kernel
    par.n0 = window_start_sample + im.bank.fft_len() - 1; // hoprate_stream's per-hop reference
    par.carrier_phase_from_ref = im.carrier_phase_from_ref;
    par.fft_len = im.bank.fft_len();
    par.n_hops = im.n_hops;
    par.Lf = im.Lf;
    par.data_stride = im.n_hops;
    par.out_rows_spec = 4;

    ck(gnss_cuda::launch_waveform(im.d_code, (gnss_cuda::DespreadJob*)d_jobs_slot, n_spec,
                                  im.n_chan, par, (float2*)d_wave, (double*)d_energy_out, st),
       "launch waveform (inject)");
    return n_spec;
}

int GnssCudaDespread::enqueue_peel_device(const void* d_window, int data_stride,
                                          long long window_start_sample,
                                          const std::vector<PeelSpec>& specs, void* d_pjobs_slot,
                                          void* d_gain_slot, void* d_resid_out, void* stream,
                                          const void* d_chan_scale) {
    Impl& im = *_impl;
    const cudaStream_t st = (cudaStream_t)stream;
    const int n_job = (int)specs.size();

    // Zero jobs still has to run: the residual buffer must be FILLED (a straight copy of the
    // voltage), not left holding the previous record. The despread downstream reads it either
    // way, so "no PRN converged yet" must mean "un-peeled voltage", never "stale window".
    im.h_pjobs.resize((size_t)n_job);
    im.h_gains.assign((size_t)2 * n_job * im.n_chan, make_float2(0.f, 0.f));
    for (int i = 0; i < n_job; ++i) {
        const PeelSpec& sp = specs[(size_t)i];
        auto& pc = im.ensure_phi(sp.p, sp.doppler_hz);
        const double cps = im.cps_for(sp.doppler_hz);
        const double cp0 = (double)im.bank.comb_mult() * sp.cp_seed;
        uint64_t mask = 0;
        for (int c : sp.covering)
            if (c >= 0 && c < im.n_chan)
                mask |= (1ULL << c);
        // Gains land in the arena as [job][head|tail][chan]; the job points at its two rows.
        float2* const g_head = (float2*)d_gain_slot + (size_t)(2 * i) * im.n_chan;
        float2* const g_tail = (float2*)d_gain_slot + (size_t)(2 * i + 1) * im.n_chan;
        for (int c = 0; c < im.n_chan; ++c) {
            const bool own = (mask >> c) & 1ULL;
            const auto ah = (own && c < (int)sp.a_head.size()) ? sp.a_head[(size_t)c]
                                                               : std::complex<float>(0.f, 0.f);
            const auto at = (own && c < (int)sp.a_tail.size()) ? sp.a_tail[(size_t)c]
                                                               : std::complex<float>(0.f, 0.f);
            im.h_gains[(size_t)(2 * i) * im.n_chan + c] = make_float2(ah.real(), ah.imag());
            im.h_gains[(size_t)(2 * i + 1) * im.n_chan + c] = make_float2(at.real(), at.imag());
        }
        im.h_pjobs[(size_t)i] = {cp0,
                                 // BIT-IDENTICAL to the despread's (see PeelJob::cp_ref).
                                 im.cp_ref_for(cp0, cps,
                                               window_start_sample + im.bank.fft_len() - 1),
                                 cps,
                                 1.0 / cps,
                                 im.wc_for(sp.p, sp.doppler_hz),
                                 // BIT-IDENTICAL to the despread's, or the analytic add-back
                                 // stops being exact (see PeelJob::ang0). Note the peel takes
                                 // no ctrim, exactly as its wc_for call does not.
                                 im.ang0_for(sp.p, sp.doppler_hz, 0.0,
                                             window_start_sample + im.bank.fft_len() - 1),
                                 im.code_offset[(size_t)sp.p],
                                 (int)im.code_len,
                                 mask,
                                 im.use_shared ? im.shared.d_A : pc.d_A,
                                 im.use_shared ? im.shared.d_B : pc.d_B,
                                 im.use_shared ? im.shared.n_chips : pc.n_chips,
                                 im.m_head_for(cp0, cps, window_start_sample), // SHARED with P_HEAD
                                 g_head,
                                 g_tail,
                                 // THE PEEL MUST MATCH THE DESPREAD EXACTLY -- same tables, same
                                 // ddw -- or the analytic add-back stops being exact.
                                 im.use_shared ? im.shared.d_PA : nullptr,
                                 im.use_shared ? im.shared.d_PB : nullptr,
                                 im.use_shared
                                     ? (float)(2.0 * M_PI
                                                   * (im.bank.carrier_offset(sp.p)
                                                      + sp.doppler_hz)
                                                   / im.fs
                                               - im.shared.wc)
                                     : 0.0f};
    }
    if (n_job > 0) {
        ck(cudaMemcpyAsync(d_gain_slot, im.h_gains.data(), im.h_gains.size() * sizeof(float2),
                           cudaMemcpyHostToDevice, st),
           "gains upload (peel)");
        ck(cudaMemcpyAsync(d_pjobs_slot, im.h_pjobs.data(),
                           (size_t)n_job * sizeof(gnss_cuda::PeelJob), cudaMemcpyHostToDevice,
                           st),
           "peel jobs upload");
    }

    gnss_cuda::DespreadParams par;
    par.shared = im.use_shared; // item 2: picks the shared-table kernel
    par.n0 = window_start_sample + im.bank.fft_len() - 1; // same per-hop reference as the despread
    par.carrier_phase_from_ref = im.carrier_phase_from_ref;
    par.fft_len = im.bank.fft_len();
    par.n_hops = im.n_hops;
    par.Lf = im.Lf;
    par.data_stride = data_stride;
    if (d_chan_scale)
        ck(gnss_cuda::launch_peel_q((const unsigned char*)d_window, (const float*)d_chan_scale,
                                    im.d_code, (gnss_cuda::PeelJob*)d_pjobs_slot, n_job,
                                    im.n_chan, par, (float2*)d_resid_out, st),
           "launch peel q");
    else
        ck(gnss_cuda::launch_peel((const float2*)d_window, im.d_code,
                                  (gnss_cuda::PeelJob*)d_pjobs_slot, n_job, im.n_chan, par,
                                  (float2*)d_resid_out, st),
           "launch peel");
    return n_job;
}

std::array<gnss::DespreadResult, 3>
GnssCudaDespread::despread3(int p, double cp_seed, double spacing_chips, double doppler_hz,
                            const std::vector<int>& covering) {
    return despread_batch({Spec{p, cp_seed, spacing_chips, doppler_hz, covering}})[0];
}
