#include "GnssCudaDespread.hpp"

#include "cudaGnssChordDespread.hpp"
#include "cudaGnssDespreadKernel.hpp"
#include "kotekanNvtx.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <immintrin.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("GnssCudaDespread: ") + what + ": "
                                 + cudaGetErrorString(e));
}

// n float pairs -> n __half2 (x in the low half), rounded to nearest even: bit for bit what
// gnss_cuda::phi_to_half gives. That one converts with __floats2half2_rn, which on the host is
// CUDA's slow software emulation; F16C converts eight floats per instruction. The halves are
// written as raw bits, so no CUDA half type enters this g++-compiled file.
void pairs_to_half(const float2* src, void* dst, size_t n) {
#if defined(__F16C__) && defined(__AVX__)
    const float* s = &src[0].x;
    auto* d = static_cast<uint16_t*>(dst);
    const size_t m = 2 * n;
    size_t i = 0;
    for (; i + 8 <= m; i += 8)
        _mm_storeu_si128(reinterpret_cast<__m128i*>(d + i),
                         _mm256_cvtps_ph(_mm256_loadu_ps(s + i), _MM_FROUND_TO_NEAREST_INT));
    for (; i < m; ++i)
        d[i] = _cvtss_sh(s[i], _MM_FROUND_TO_NEAREST_INT);
#else
    gnss_cuda::phi_to_half(src, dst, n);
#endif
}
} // namespace

struct GnssCudaDespread::Impl {
    gnss::ChannelizedReplicaBank& bank;
    int n_prn, n_chan, n_hops;
    double fs, f_off, refresh_hz;
    long long window_start = 0;
    // Device buffers (persistent).
    float2* d_data = nullptr;                 // [n_chan][n_hops]
    int8_t* d_code = nullptr;                 // all PRNs' combined-stream codes, concatenated
    std::vector<int> code_offset;             // per PRN slot
    std::vector<int8_t> code_stage;           // set_prn's H2D staging
    std::vector<int> slot_prn;                // PRN per slot as this engine has it on the device
    long code_len = 0;                        // combined-stream length (same for all PRNs)
    gnss_cuda::DespreadJob* d_jobs = nullptr; // [n_prn], one quad job per PRN slot
    double2* d_corr = nullptr;                // [4*n_prn][n_chan]  (E, P, L, P_HEAD per slot)
    double* d_energy = nullptr;               // [4*n_prn][n_chan]

    // Per-PRN Doppler-bucketed Phi tables on the device ([n_chan][Lf+1] each, all channels).
    struct PhiCache {
        bool valid = false;
        double doppler = 0.0;
        int n_chips = 0;
        int d_first = 0; // first gather chip (centered truncation); 0 = full span or one-sided
        float2 *d_A = nullptr, *d_B = nullptr; // float2 tables, or __half2 storage when `half`
                                               // (the kernel template reinterprets the pointer)
        bool half = false;                     // what the device buffers currently hold
    };
    std::vector<PhiCache> phi;

    /// Pinned upload buffers for ensure_phi on a caller's stream, used in turn. A slot is
    /// rewritten only after the copy out of it has completed (`done`).
    struct Staging {
        unsigned char* host = nullptr;
        size_t bytes = 0;
        cudaEvent_t done = nullptr;
    };
    Staging staging[2];
    int staging_next = 0;
    /// One table's taps as float pairs, [n_chan][Lf+1], kept across builds: the bank writes
    /// into them and the upload converts out of them, so a build allocates nothing.
    std::vector<float> taps_a, taps_b;

    /// fp16 Phi: ensure_phi stores __half2 tables, halving their device footprint, and
    /// launch_waveform takes the __half2 gather. launch_despread reads Phi as float2, so
    /// despread_batch throws while this is set rather than despread garbage.
    bool use_fp16 = false;

    /// What the kernel was given, per PRN slot, read back from each job as it is built. A
    /// recomputation from the same inputs would agree with itself by construction and could
    /// not show the kernel diverging. Values persist until the slot's next job; NaN until its
    /// first (and after set_prn repoints the slot), which a consumer must read as absent, not
    /// as zero.
    std::vector<double> last_ang0;     ///< DespreadJob::ang0, radians
    std::vector<double> last_phi_ddop; ///< doppler_now - the Doppler this PRN's Phi was built at
    std::vector<int> all_chans;
    int Lf = 0;

    std::vector<float2> stage;   // host transpose staging [chan][hop]
    int max_chips = 0;           ///< cap on the chip gather depth (0 = the filter's full span)
    bool chips_centered = false; ///< centre the max_chips window on the span
    bool split_timing = false;
    bool split_recorded = false; ///< at least one enqueue has recorded the events
    cudaEvent_t ev_a = nullptr, ev_b = nullptr, ev_c = nullptr; ///< created when timing is on
    cudaStream_t stream = nullptr;
    std::vector<gnss_cuda::DespreadJob> h_jobs;
    std::vector<double2> h_corr;
    std::vector<double> h_energy;

    Impl(gnss::ChannelizedReplicaBank& b, int np, int nc, int nh, double fs_, double fo, double rh,
         const std::vector<int>& ids) :
        bank(b), n_prn(np), n_chan(nc), n_hops(nh), fs(fs_), f_off(fo), refresh_hz(rh) {
        // Phi tables index by local channel ci; each is built at the centre frequency of global
        // bin ids[ci]. No structure is assumed: the list may be sparse, unsorted, irregularly
        // spaced, or a single channel.
        if (nc > 64)
            throw std::runtime_error("GnssCudaDespread: >64 channels needs a wider chan_mask");
        if (nc <= 0)
            throw std::runtime_error("GnssCudaDespread: empty channel list");
        Lf = bank.fft_len() * 4; // overwritten below from the probe filter
        all_chans = ids;
        // Validate against the spectrum, and reject duplicates: a repeated bin would double
        // that channel's weight in the coherent sum and bias every correlation.
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
        // The bank does not expose num_taps, so Lf comes from a probe filter's length.
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

        // despread_batch's arena: one job per PRN slot, each emitting four output rows
        // (E, P, L, P_HEAD).
        const size_t maxs = (size_t)np;     // jobs
        const size_t maxb = (size_t)4 * np; // output rows
        ck(cudaMalloc(&d_data, (size_t)nc * nh * sizeof(float2)), "alloc data");
        ck(cudaMalloc(&d_jobs, maxs * sizeof(gnss_cuda::DespreadJob)), "alloc jobs");
        ck(cudaMalloc(&d_corr, maxb * nc * sizeof(double2)), "alloc corr");
        ck(cudaMalloc(&d_energy, maxb * nc * sizeof(double)), "alloc energy");
        ck(cudaStreamCreate(&stream), "stream");
        phi.resize(np);
        // NaN, not 0: "no job built for this PRN" must not read as ang0 == 0, a legal value of
        // 2*pi*frac(f*n0/fs).
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
        for (auto& sg : staging) {
            if (sg.done)
                cudaEventDestroy(sg.done);
            cudaFreeHost(sg.host);
        }
    }

    /// The next pinned upload slot, at least `bytes` long, once the copy last made from it
    /// has completed.
    Staging& next_staging(size_t bytes) {
        Staging& sg = staging[staging_next];
        staging_next = (staging_next + 1) % 2;
        if (sg.done)
            ck(cudaEventSynchronize(sg.done), "Phi staging reuse");
        else
            ck(cudaEventCreateWithFlags(&sg.done, cudaEventDisableTiming), "Phi staging event");
        if (sg.bytes < bytes) {
            ck(cudaFreeHost(sg.host), "free Phi staging");
            ck(cudaMallocHost((void**)&sg.host, bytes), "alloc Phi staging");
            sg.bytes = bytes;
        }
        return sg;
    }

    /// The code-period boundary hop for a window: the first hop m at which the prompt's
    /// absolute code phase C0 + m*fft_len*cps (C0 at the hop-0 reference sample) reaches the
    /// next multiple of code_len. Head is hops [0, m_head), tail the rest; the secondary overlay
    /// and nav symbol flip sign at the boundary. n_hops if the window holds no boundary.
    ///
    /// The P_HEAD row comes from here. Hop granularity is enough: the filter span already mixes
    /// about one hop at the seam.
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

    /// Carrier angular rate, rad/sample, at the PRN's own carrier offset (per PRN under FDMA,
    /// the band offset for CDMA). The carrier trim enters here and in ang0_for but never the
    /// code rate: carrier and code are separate control paths. The Phi cache stays keyed on the
    /// Doppler alone, since a trim of tens of Hz barely changes the channel filter response.
    double wc_for(int p, double doppler_hz, double ctrim_hz = 0.0) const {
        return 2.0 * M_PI * (bank.carrier_offset(p) + doppler_hz + ctrim_hz) / fs;
    }

    /// Prompt code phase at absolute sample n0, combined-stream chips, reduced mod code_len
    /// (DespreadJob::cp_ref). Long double is required, as in ang0_for: cp0 + n0*cps is ~6e12
    /// chips at n0 ~ 1.9e15, where a double resolves only ~1e-3 chip. The code is periodic, so
    /// the reduction is exact, and it hands the kernel's gather a small argument.
    double cp_ref_for(double cp0, double cps, long long n0) const {
        long double c = (long double)cp0 + (long double)n0 * (long double)cps;
        const long double L = (long double)code_len;
        c = fmodl(c, L);
        if (c < 0.0L)
            c += L;
        return (double)c;
    }

    /// Carrier phase at absolute sample n0, radians in [0, 2*pi) (DespreadJob::ang0).
    ///
    /// Long double is required. The quantity is frac(f * n0 / fs) with f ~ 1.18e9 and
    /// n0 ~ 1.9e15: ~6.9e14 whole cycles are discarded before the fraction survives. A double's
    /// ulp there is 0.125 cycle; x87 long double's 64-bit mantissa leaves ~3.7e-5 cycle
    /// (2.3e-4 rad). CUDA maps long double to double, so the reduction is done here and passed
    /// to the kernel.
    ///
    /// Callers must pass the n0 the kernel uses (par.n0 = window_start + fft_len - 1, the hop's
    /// last sample); any other n0 rotates the whole record by a constant.
    double ang0_for(int p, double doppler_hz, double ctrim_hz, long long n0) const {
        constexpr long double TWO_PI_L = 6.283185307179586476925286766559005768L;
        const long double f =
            (long double)bank.carrier_offset(p) + (long double)doppler_hz + (long double)ctrim_hz;
        long double fr = fmodl(f / (long double)fs * (long double)n0, 1.0L);
        if (fr < 0.0L)
            fr += 1.0L;
        return (double)(TWO_PI_L * fr);
    }

    // (Re)build PRN p's Phi bucket at this Doppler if it moved more than refresh_hz
    // (or if the storage precision no longer matches use_fp16). The upload is ordered on `st`,
    // after any earlier kernel there that reads this bucket. `pinned` copies through a staging
    // slot and never waits for `st`; without it the copy is from pageable memory, which first
    // waits for `st` to drain, so only a caller whose stream is otherwise idle passes false.
    PhiCache& ensure_phi(int p, double doppler, cudaStream_t st, bool pinned) {
        PhiCache& pc = phi[(size_t)p];
        if (pc.valid && pc.half == use_fp16 && std::fabs(doppler - pc.doppler) <= refresh_hz)
            return pc;
        KOTEKAN_NVTX_RANGE("phi build");
        KOTEKAN_NVTX_PUSH("phi taps");
        // At this PRN's carrier (per PRN under FDMA), as float pairs [channel][Lf+1]: the bank
        // rounds its double prefix sums once, directly into taps_a / taps_b.
        const size_t n = (size_t)n_chan * (Lf + 1);
        taps_a.resize(2 * n);
        taps_b.resize(2 * n);
        const auto f = bank.hoprate_filter_f32(all_chans, doppler, p, taps_a.data(), taps_b.data());
        const float2* hA = reinterpret_cast<const float2*>(taps_a.data());
        const float2* hB = reinterpret_cast<const float2*>(taps_b.data());
        KOTEKAN_NVTX_POP();
        // __half2 is 4 bytes, written as sizeof(float2)/2 so that no CUDA half type enters this
        // g++-compiled file.
        const size_t esz = use_fp16 ? sizeof(float2) / 2 : sizeof(float2);
        if (pc.d_A && pc.half != use_fp16) {
            ck(cudaFree(pc.d_A), "free PhiA (precision flip)");
            ck(cudaFree(pc.d_B), "free PhiB (precision flip)");
            pc.d_A = pc.d_B = nullptr;
        }
        if (!pc.d_A) {
            KOTEKAN_NVTX_RANGE("phi alloc");
            ck(cudaMalloc(&pc.d_A, n * esz), "alloc PhiA");
            ck(cudaMalloc(&pc.d_B, n * esz), "alloc PhiB");
        }
        // fp16 converts on the host, so the upload is half the size too.
        const size_t bytes = n * esz;
        const void* srcA = hA;
        const void* srcB = hB;
        std::vector<unsigned> sA, sB; // 4 B per __half2
        Staging* sg = nullptr;
        if (pinned) {
            {
                // Waits for this slot's previous upload, which is queued behind whatever the
                // stream held when it was issued.
                KOTEKAN_NVTX_RANGE("phi staging wait");
                sg = &next_staging(2 * bytes);
            }
            if (use_fp16) {
                KOTEKAN_NVTX_RANGE("phi to half");
                pairs_to_half(hA, sg->host, n);
                pairs_to_half(hB, sg->host + bytes, n);
            } else {
                std::memcpy(sg->host, hA, bytes);
                std::memcpy(sg->host + bytes, hB, bytes);
            }
            srcA = sg->host;
            srcB = sg->host + bytes;
        } else if (use_fp16) {
            sA.resize(n);
            sB.resize(n);
            pairs_to_half(hA, sA.data(), n);
            pairs_to_half(hB, sB.data(), n);
            srcA = sA.data();
            srcB = sB.data();
        }
        ck(cudaMemcpyAsync(pc.d_A, srcA, bytes, cudaMemcpyHostToDevice, st), "PhiA up");
        ck(cudaMemcpyAsync(pc.d_B, srcB, bytes, cudaMemcpyHostToDevice, st), "PhiB up");
        if (sg)
            ck(cudaEventRecord(sg->done, st), "Phi staging record");
        pc.half = use_fp16;
        pc.valid = true;
        pc.doppler = doppler;
        pc.n_chips = f.n_chips;
        pc.d_first = f.d_first; // the bank's own centered truncation, if set
        // The engine's cap, on top of any cap set on the bank. n_chips = ceil((Lf-1)*cps)+2 is
        // the PFB filter's span in chips (~210 at CHORD: 65536 samples at 312.8 samples/chip).
        // Capping it truncates the filter, so it changes the replica as well as the cost.
        if (max_chips > 0 && pc.n_chips - pc.d_first > max_chips) {
            if (chips_centered) {
                // Keep the central max_chips: the prototype peaks mid-span, and a one-sided
                // cap shorter than about half the span cuts through the peak.
                pc.d_first += (pc.n_chips - pc.d_first - max_chips) / 2;
                pc.n_chips = pc.d_first + max_chips;
            } else {
                pc.n_chips = pc.d_first + max_chips;
            }
        }
        return pc;
    }
};

GnssCudaDespread::GnssCudaDespread(gnss::ChannelizedReplicaBank& bank, int n_prn,
                                   const std::vector<int>& chan_ids, int n_hops, double sample_rate,
                                   double f_offset, double refresh_hz) :
    _impl(new Impl(bank, n_prn, (int)chan_ids.size(), n_hops, sample_rate, f_offset, refresh_hz,
                   chan_ids)) {
    // FDMA (GLONASS L1OF/L2OF) is supported: the Phi cache is per PRN, and each PRN's filter and
    // carrier phasor use its own offset (see wc_for).
}

GnssCudaDespread::~GnssCudaDespread() = default;

int GnssCudaDespread::prn_at(int p) const {
    // This engine's device view, not the shared bank's; see set_prn.
    return (p >= 0 && (size_t)p < _impl->slot_prn.size()) ? _impl->slot_prn[(size_t)p] : -1;
}

bool GnssCudaDespread::set_prn(int p, int prn, void* stream) {
    Impl& im = *_impl;
    if (p < 0 || p >= im.n_prn)
        return false;
    // Compare against this engine's own view, not the bank's. Several engines can share one
    // bank (the search builds one per <=64-channel refine group), each with its own device copy
    // of the code table; after the first engine updates the bank, the others must still upload.
    if (im.slot_prn[(size_t)p] == prn)
        return true; // this engine is already there
    if (im.bank.prn_at(p) != prn && !im.bank.set_prn(p, prn))
        return false;
    // Re-upload this slot's row of the concatenated code table. The staging buffer is a member
    // so that it outlives the stream-ordered copy.
    if (im.code_stage.size() != (size_t)im.n_prn * (size_t)im.code_len)
        im.code_stage.assign((size_t)im.n_prn * (size_t)im.code_len, 0);
    const auto& fc = im.bank.full_code(p);
    std::copy(fc.begin(), fc.end(), im.code_stage.begin() + (size_t)im.code_offset[(size_t)p]);
    // nullptr = this engine's own stream, the one despread_batch uses, so a caller of the batch
    // API alone (the search) is still ordered.
    const cudaStream_t st = stream ? (cudaStream_t)stream : im.stream;
    ck(cudaMemcpyAsync(im.d_code + im.code_offset[(size_t)p],
                       im.code_stage.data() + im.code_offset[(size_t)p], (size_t)im.code_len,
                       cudaMemcpyHostToDevice, st),
       "code slot re-upload");
    // Per-slot state held here. The Phi cache depends only on the carrier, so under CDMA it
    // would still be valid, but it is dropped with the rest so that nothing keyed to the old
    // satellite survives; it is rebuilt on the slot's next job.
    im.phi[(size_t)p].valid = false;
    im.last_ang0[(size_t)p] = std::numeric_limits<double>::quiet_NaN();
    im.last_phi_ddop[(size_t)p] = std::numeric_limits<double>::quiet_NaN();
    im.slot_prn[(size_t)p] = prn;
    return true;
}

double GnssCudaDespread::last_ang0(int p) const {
    return (p >= 0 && (size_t)p < _impl->last_ang0.size())
               ? _impl->last_ang0[(size_t)p]
               : std::numeric_limits<double>::quiet_NaN();
}

double GnssCudaDespread::last_phi_ddop(int p) const {
    return (p >= 0 && (size_t)p < _impl->last_phi_ddop.size())
               ? _impl->last_phi_ddop[(size_t)p]
               : std::numeric_limits<double>::quiet_NaN();
}

void GnssCudaDespread::upload_window(const std::complex<float>* window,
                                     long long window_start_sample) {
    Impl& im = *_impl;
    // Callers hold [hop][chan]; the kernel reads [chan][hop]. Transpose through the staging buffer.
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
    // Before the first enqueue_batch_nm the events exist but have never been recorded, and
    // cudaEventElapsedTime on them is an error.
    if (!im.split_timing || !im.ev_c || !im.split_recorded)
        return false;
    // Synchronize in case the call that recorded the events is still in flight.
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

bool GnssCudaDespread::set_phi_fp16(bool on) {
    Impl& im = *_impl;
    if (!on) {
        im.use_fp16 = false; // ensure_phi's precision check rebuilds any half buckets lazily
        return false;
    }
    im.use_fp16 = true;
    const size_t n = (size_t)im.n_chan * (size_t)(im.Lf + 1);
    std::fprintf(stderr,
                 "INFO: GnssCudaDespread: fp16 Phi ARMED -- %.1f MB over %d channels per PRN "
                 "bucket (fp32 was %.1f), storage error ~3.3e-4 (phibits); despread_batch will "
                 "REFUSE while this is set.\n",
                 2.0 * (double)n * (sizeof(float2) / 2) / 1e6, im.n_chan,
                 2.0 * (double)n * sizeof(float2) / 1e6);
    return true;
}

void GnssCudaDespread::set_max_chips(int n) {
    _impl->max_chips = n;
    for (auto& pc : _impl->phi) // force a rebuild so the cap takes effect on cached buckets
        pc.valid = false;
}

void GnssCudaDespread::set_chips_centered(bool c) {
    _impl->chips_centered = c;
    for (auto& pc : _impl->phi) // force a rebuild so the window placement takes effect
        pc.valid = false;
}

int GnssCudaDespread::max_batch_specs() const {
    // Every Spec-taking call stages its jobs in h_jobs, which holds n_prn (one per PRN slot), and
    // despread_batch's device arena (maxs in the constructor) is the same size. This is not
    // gnss_gpu::max_specs(n_prn) = n_prn * MAX_REC, which sizes a whole frame's device arena
    // across records; each enqueue fills one record's slice of it.
    return _impl->n_prn;
}

std::vector<std::array<gnss::DespreadResult, 3>>
GnssCudaDespread::despread_batch(const std::vector<Spec>& specs) {
    Impl& im = *_impl;
    if (im.use_fp16)
        throw std::runtime_error("GnssCudaDespread: despread_batch with fp16 Phi armed -- "
                                 "launch_despread reads Phi as raw float2 and would despread "
                                 "garbage; phi_fp16 is only for the waveform (inject) path");
    std::vector<std::array<gnss::DespreadResult, 3>> out(specs.size());
    if (specs.empty())
        return out;
    const int n_spec = (int)specs.size();
    const int n_out = 4 * n_spec; // the kernel always emits the full quad

    // One quad job per spec, as build_jobs builds them but with m_head = 0: this path returns
    // the plain E/P/L triple, so the head row comes back all-zero and is dropped below.
    for (size_t i = 0; i < specs.size(); ++i) {
        const Spec& sp = specs[i];
        auto& pc = im.ensure_phi(sp.p, sp.doppler_hz, im.stream, false);
        const double cps =
            im.bank.eff_chip_rate() / im.fs
            * (1.0 + im.bank.code_doppler_sign * sp.doppler_hz / im.bank.carrier_hz());
        // The carrier trim enters wc and ang0 only (see wc_for).
        const double wc = im.wc_for(sp.p, sp.doppler_hz, sp.ctrim_hz);
        uint64_t mask = 0;
        for (int c : sp.covering)
            if (c >= 0 && c < im.n_chan)
                mask |= (1ULL << c);
        const double cp0_i = (double)im.bank.comb_mult() * sp.cp_seed;
        im.h_jobs[i] = {
            (double)im.bank.comb_mult() * sp.spacing_chips,
            // cp_ref and ang0 at par.n0, the hop's last sample.
            im.cp_ref_for(cp0_i, cps, im.window_start + im.bank.fft_len() - 1), cps, 1.0 / cps, wc,
            im.ang0_for(sp.p, sp.doppler_hz, sp.ctrim_hz, im.window_start + im.bank.fft_len() - 1),
            im.code_offset[(size_t)sp.p], (int)im.code_len, mask, pc.d_A, pc.d_B, pc.n_chips, 0,
            pc.d_first};
        // Record what the kernel is given, read back from the job (see Impl::last_ang0).
        im.last_ang0[(size_t)sp.p] = im.h_jobs[i].ang0;
        im.last_phi_ddop[(size_t)sp.p] =
            pc.valid ? (sp.doppler_hz - pc.doppler) : std::numeric_limits<double>::quiet_NaN();
    }
    ck(cudaMemcpyAsync(im.d_jobs, im.h_jobs.data(), (size_t)n_spec * sizeof(gnss_cuda::DespreadJob),
                       cudaMemcpyHostToDevice, im.stream),
       "jobs upload");

    gnss_cuda::DespreadParams par;
    par.phi_half = im.use_fp16 ? 1 : 0;               // __half2 gather (waveform kernels only)
    par.n0 = im.window_start + im.bank.fft_len() - 1; // hoprate_stream's per-hop reference
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
    ck(cudaStreamSynchronize(im.stream), "sync"); // one sync for the whole batch

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

// Every enqueue_* call that takes Specs builds its jobs here, so their replicas cannot differ;
// despread_batch builds the same jobs inline with m_head = 0.
void GnssCudaDespread::build_jobs(const std::vector<Spec>& specs, void* d_jobs_slot,
                                  long long window_start_sample, void* stream) {
    Impl& im = *_impl;
    const int n_spec = (int)specs.size();
    const cudaStream_t st = (cudaStream_t)stream;

    // h_jobs staging is pageable, so the async H2D below is host-synchronous on return and the
    // buffer is safely reusable per call.
    for (size_t i = 0; i < specs.size(); ++i) {
        const Spec& sp = specs[i];
        auto& pc = im.ensure_phi(sp.p, sp.doppler_hz, st, true);
        const double cps =
            im.bank.eff_chip_rate() / im.fs
            * (1.0 + im.bank.code_doppler_sign * sp.doppler_hz / im.bank.carrier_hz());
        // The carrier trim enters wc and ang0 only (see wc_for).
        const double wc = im.wc_for(sp.p, sp.doppler_hz, sp.ctrim_hz);
        uint64_t mask = 0;
        for (int c : sp.covering)
            if (c >= 0 && c < im.n_chan)
                mask |= (1ULL << c);
        const double cp0 = (double)im.bank.comb_mult() * sp.cp_seed;

        // Output row 3 = P_HEAD: the prompt up to the code-period boundary (Impl::m_head_for).
        const int m_head = im.m_head_for(cp0, cps, window_start_sample);

        im.h_jobs[i] = {(double)im.bank.comb_mult() * sp.spacing_chips,
                        im.cp_ref_for(cp0, cps, window_start_sample + im.bank.fft_len() - 1),
                        cps,
                        1.0 / cps,
                        wc,
                        im.ang0_for(sp.p, sp.doppler_hz, sp.ctrim_hz,
                                    window_start_sample + im.bank.fft_len() - 1),
                        im.code_offset[(size_t)sp.p],
                        (int)im.code_len,
                        mask,
                        pc.d_A,
                        pc.d_B,
                        pc.n_chips,
                        m_head,
                        pc.d_first};
        // Record what the kernel is given, read back from the job (see Impl::last_ang0).
        im.last_ang0[(size_t)sp.p] = im.h_jobs[i].ang0;
        im.last_phi_ddop[(size_t)sp.p] =
            pc.valid ? (sp.doppler_hz - pc.doppler) : std::numeric_limits<double>::quiet_NaN();
    }
    ck(cudaMemcpyAsync(d_jobs_slot, im.h_jobs.data(),
                       (size_t)n_spec * sizeof(gnss_cuda::DespreadJob), cudaMemcpyHostToDevice, st),
       "jobs upload (device batch)");
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
    par.phi_half = im.use_fp16 ? 1 : 0;                   // __half2 gather (waveform kernels only)
    par.n0 = window_start_sample + im.bank.fft_len() - 1; // hoprate_stream's per-hop reference
    par.fft_len = im.bank.fft_len();
    par.n_hops = im.n_hops;
    par.Lf = im.Lf;
    // The correlator reads the frame in its native [hop][chan][elem] order through
    // frame_chan_stride and elem_stride; data_stride does not apply.
    par.data_stride = im.n_hops;
    par.conj_data = conjugate ? 1 : 0;

    // The replicas are generated once and reused for all n_elem antennas, which is why this
    // path splits the fused kernel. Both launches are on one stream, so the correlator starts
    // after the waveform it reads is complete.
    if (im.split_timing)
        ck(cudaEventRecord(im.ev_a, st), "split ev_a");
    ck(gnss_cuda::launch_waveform(im.d_code, (gnss_cuda::DespreadJob*)d_jobs_slot, n_spec,
                                  im.n_chan, par, (float2*)d_wave, (double*)d_energy_out, st),
       "launch waveform (NxM)");
    if (im.split_timing)
        ck(cudaEventRecord(im.ev_b, st), "split ev_b");
    ck(gnss_cuda::launch_correlate_nm(
           (const unsigned char*)d_frame, (const float*)d_chan_scale, d_chan_ids,
           (const float2*)d_wave, (gnss_cuda::DespreadJob*)d_jobs_slot, n_spec, im.n_chan, n_elem,
           elem_stride, frame_chan_stride, par, (double2*)d_corr_out, st),
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
    par.phi_half = im.use_fp16 ? 1 : 0;                   // __half2 gather (waveform kernels only)
    par.n0 = window_start_sample + im.bank.fft_len() - 1; // hoprate_stream's per-hop reference
    par.fft_len = im.bank.fft_len();
    par.n_hops = im.n_hops;
    par.Lf = im.Lf;
    par.data_stride = im.n_hops;

    ck(gnss_cuda::launch_waveform(im.d_code, (gnss_cuda::DespreadJob*)d_jobs_slot, n_spec,
                                  im.n_chan, par, (float2*)d_wave, (double*)d_energy_out, st),
       "launch waveform (inject)");
    return n_spec;
}

std::array<gnss::DespreadResult, 3> GnssCudaDespread::despread3(int p, double cp_seed,
                                                                double spacing_chips,
                                                                double doppler_hz,
                                                                const std::vector<int>& covering) {
    return despread_batch({Spec{p, cp_seed, spacing_chips, doppler_hz, covering}})[0];
}
