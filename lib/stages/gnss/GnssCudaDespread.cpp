#include "GnssCudaDespread.hpp"

#include "cudaGnssChordDespread.hpp"
#include "cudaGnssDespreadKernel.hpp"
#include "cudaGnssPsiSynth.hpp"
#include "kotekanLogging.hpp"
#include "kotekanNvtx.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <deque>
#include <immintrin.h>
#include <limits>
#include <mutex>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <thread>

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
    double fs, refresh_hz;
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
        int d_first = 0; // first gather chip (centered truncation); 0 = from chip 0
        float2 *d_A = nullptr, *d_B = nullptr; // float2 tables, or __half2 storage under
                                               // use_fp16 (the kernel template reinterprets
                                               // the pointer)
    };
    std::vector<PhiCache> phi;

    /// Per-PRN Doppler-bucketed Psi code-phase tables on the device (set_psi_synth): one
    /// primary period plus the previous-period lookback and the period signs, the PsiJob
    /// layout of cudaGnssPsiSynth.hpp, over all channels like Phi. Written only by the
    /// enqueue thread (psi_upload); the builder thread never touches it.
    struct PsiCache {
        bool valid = false;
        double doppler = 0.0;
        int P = 0; ///< primary period of the slot's code (from the build; set_prn resets)
        int G = 1, Ns = 1, n_prev = 0, n_chips = 0;
        void* d_psi = nullptr;   // [n_chan][G*P] float2, or __half2 under psi_half
        void* d_prev = nullptr;  // [n_chan][n_prev]; nullptr for a plain code
        int8_t* d_sec = nullptr; // [Ns]; nullptr for a plain code
        size_t psi_bytes = 0, prev_bytes = 0, sec_bytes = 0; // allocated
        int n_builds = 0;
    };
    std::vector<PsiCache> psi;
    bool psi_on = false;
    bool psi_half = false;
    double psi_refresh_hz = 30.0;
    std::vector<gnss_cuda::PsiJob> h_pjobs; ///< [n_prn] staging, pageable like h_jobs
    std::vector<float> h_theta;             ///< [n_prn][n_chan] staging

    // ---- The Psi table builder: one thread per engine, started by the first request, joined
    // by the destructor. A build (build_psi_split: 12-25 ms per PRN by the direct sum, a few ms
    // by FFT, at L5 geometry) must never run on the enqueue thread: on 10-23 the first seeds of
    // ~11 chains x 12 PRNs built their tables synchronously there (~2.6 s), the DPDK voltage
    // ring (8 frames, ~0.3 s) overflowed and the node died. The enqueue thread only queues
    // requests (psi_request), drains finished builds (psi_drain) and uploads them (psi_upload,
    // at most one per record, pinned, async on its stream); psi_ready is the one entry point
    // the stage calls, before it builds a spec, and a slot it refuses is left out of the record.
    //
    // The bank is shared with the enqueue thread. Of what a build reads, hoprate_filter is
    // const and allocates its own buffers, the geometry is fixed at construction, and the code
    // table is the one thing set_prn mutates: the builder copies the slot's split code under
    // code_mtx (psi_split_code, O(L)) and builds from the copy, and set_prn mutates the bank
    // under the same lock after bumping the slot's generation, so a build queued or running
    // for the old satellite is dropped when it lands (its generation no longer matches).
    struct PsiReq {
        int p = -1;
        double doppler = 0.0;
        uint32_t gen = 0;
        std::chrono::steady_clock::time_point t_req;
    };
    struct PsiDone {
        int p = -1;
        double doppler = 0.0;
        uint32_t gen = 0;
        int P = 0;
        gnss_cuda::PsiSplit S;
        std::string error; ///< a build that threw: rethrown on the enqueue thread (psi_ready)
        double wait_ms = 0.0, build_ms = 0.0;
        std::chrono::steady_clock::time_point t_done;
    };
    std::thread psi_thread;
    std::mutex psi_mtx; ///< psi_reqs, psi_done, psi_stop, the thread start
    std::condition_variable psi_cv;
    std::deque<PsiReq> psi_reqs;
    std::vector<PsiDone> psi_done; ///< finished builds, drained by the enqueue thread
    bool psi_stop = false;
    std::mutex code_mtx;            ///< the bank's code table (set_prn vs the builder's copy)
    std::vector<uint32_t> slot_gen; ///< [n_prn] bumped by set_prn, under code_mtx
    std::vector<int> psi_P_builder; ///< [n_prn] the builder's primary period per slot; 0 =
                                    ///< search afresh (set_prn resets), under code_mtx
    // Enqueue-thread-owned: a slot with a request queued, building, or built and waiting for
    // its upload (one request per slot at a time), and the finished builds awaiting upload.
    std::vector<uint8_t> psi_pending;
    std::deque<PsiDone> psi_ready_q;
    int psi_uploads_left = 1; ///< per-record upload budget, reset by enqueue_psi_pack

    /// Pinned upload buffers for ensure_phi on a caller's stream, used in turn. A slot is
    /// rewritten only after the copy out of it has completed (`done`).
    struct Staging {
        unsigned char* host = nullptr;
        size_t bytes = 0;
        cudaEvent_t done = nullptr;
    };
    Staging staging[2];
    int staging_next = 0;
    /// Pinned staging ring for the Psi uploads (psi_upload). Unlike next_staging it is never
    /// waited on: a slot is taken only when its previous upload has completed
    /// (cudaEventQuery), otherwise the upload waits for the next record, so the enqueue thread
    /// never blocks on its own stream here. Four slots at one upload per record is a frame of
    /// lag before that happens.
    static constexpr int N_PSI_STAGE = 4;
    Staging psi_staging[N_PSI_STAGE];
    /// One table's taps as float pairs, [n_chan][Lf+1], kept across builds: the bank writes
    /// into them and the upload converts out of them, so a build allocates nothing.
    std::vector<float> taps_a, taps_b;

    /// fp16 Phi: ensure_phi stores __half2 tables, halving their device footprint, and
    /// launch_waveform takes the __half2 gather. launch_despread reads Phi as float2, so
    /// despread_batch throws while this is set rather than despread garbage.
    bool use_fp16 = false;
    bool epl_rot = false;     ///< set_epl_from_prompt: enqueue_waveform passes d_rot
    std::vector<float> h_rot; ///< [n_prn][n_chan] staging for d_rot
    float* d_rot = nullptr;

    /// What the kernel was given, per PRN slot, read back from each job as it is built. A
    /// recomputation from the same inputs would agree with itself by construction and could
    /// not show the kernel diverging. Values persist until the slot's next job; NaN until its
    /// first (and after set_prn repoints the slot), which a consumer must read as absent, not
    /// as zero.
    std::vector<double> last_ang0;     ///< DespreadJob::ang0, radians
    std::vector<double> last_phi_ddop; ///< doppler_now - the Doppler this PRN's Phi was built at
    std::vector<float> last_rot;       ///< [n_prn][n_chan] the rot each slot's last job got
    std::vector<uint8_t> last_rot_ok;  ///< [n_prn] 1 once the slot has built a rotated job
    std::vector<int> all_chans;
    int Lf = 0;

    std::vector<float2> stage; // host transpose staging [chan][hop]
    int max_chips = 0; ///< centred cap on the chip gather depth (0 = the filter's full span)
    cudaStream_t stream = nullptr;
    std::vector<gnss_cuda::DespreadJob> h_jobs;
    std::vector<double2> h_corr;
    std::vector<double> h_energy;

    Impl(gnss::ChannelizedReplicaBank& b, int np, int nc, int nh, double fs_, double rh,
         const std::vector<int>& ids) :
        bank(b), n_prn(np), n_chan(nc), n_hops(nh), fs(fs_), refresh_hz(rh) {
        // Phi tables index by local channel ci; each is built at the centre frequency of global
        // bin ids[ci]. No structure is assumed: the list may be sparse, unsorted, irregularly
        // spaced, or a single channel.
        if (nc > 64)
            throw std::runtime_error("GnssCudaDespread: >64 channels needs a wider chan_mask");
        if (nc <= 0)
            throw std::runtime_error("GnssCudaDespread: empty channel list");
        Lf = bank.fft_len() * bank.num_taps(); // the Phi tables hold Lf+1 entries
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
        ck(cudaMalloc(&d_rot, maxs * nc * sizeof(float)), "alloc rot");
        h_rot.assign(maxs * nc, 0.0f);
        ck(cudaStreamCreate(&stream), "stream");
        phi.resize(np);
        psi.resize(np);
        slot_gen.assign((size_t)np, 0);
        psi_P_builder.assign((size_t)np, 0);
        psi_pending.assign((size_t)np, 0);
        h_pjobs.resize(maxs);
        h_theta.assign(maxs * nc, 0.0f);
        // NaN, not 0: "no job built for this PRN" must not read as ang0 == 0, a legal value of
        // 2*pi*frac(f*n0/fs).
        last_ang0.assign((size_t)np, std::numeric_limits<double>::quiet_NaN());
        last_phi_ddop.assign((size_t)np, std::numeric_limits<double>::quiet_NaN());
        last_rot.assign((size_t)np * n_chan, 0.0f);
        last_rot_ok.assign((size_t)np, 0);
        stage.resize((size_t)nc * nh);
        h_jobs.resize(maxs);
        h_corr.resize(maxb * nc);
        h_energy.resize(maxb * nc);
    }

    ~Impl() {
        // The builder reads the bank and this object: stop it before anything is torn down.
        {
            std::lock_guard<std::mutex> lk(psi_mtx);
            psi_stop = true;
        }
        psi_cv.notify_all();
        if (psi_thread.joinable())
            psi_thread.join();
        if (stream)
            cudaStreamDestroy(stream);
        cudaFree(d_data);
        cudaFree(d_code);
        cudaFree(d_jobs);
        cudaFree(d_corr);
        cudaFree(d_energy);
        cudaFree(d_rot);
        for (auto& pc : phi) {
            cudaFree(pc.d_A);
            cudaFree(pc.d_B);
        }
        for (auto& pc : psi) {
            cudaFree(pc.d_psi);
            cudaFree(pc.d_prev);
            cudaFree(pc.d_sec);
        }
        for (auto& sg : staging) {
            if (sg.done)
                cudaEventDestroy(sg.done);
            cudaFreeHost(sg.host);
        }
        for (auto& sg : psi_staging) {
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

    // (Re)build PRN p's Phi bucket at this Doppler if it moved more than refresh_hz. The upload
    // is ordered on `st`, after any earlier kernel there that reads this bucket. `pinned` copies
    // through a staging slot and never waits for `st`; without it the copy is from pageable
    // memory, which first waits for `st` to drain, so only a caller whose stream is otherwise
    // idle passes false. Unpinned tables go up as float2, so `pinned` is required under use_fp16
    // (despread_batch, the only unpinned caller, refuses fp16 first).
    PhiCache& ensure_phi(int p, double doppler, cudaStream_t st, bool pinned) {
        PhiCache& pc = phi[(size_t)p];
        if (pc.valid && std::fabs(doppler - pc.doppler) <= refresh_hz)
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
        // g++-compiled file. A bucket allocated before set_phi_fp16 keeps its float2-sized
        // buffers, which hold the half tables too.
        const size_t esz = use_fp16 ? sizeof(float2) / 2 : sizeof(float2);
        if (!pc.d_A) {
            KOTEKAN_NVTX_RANGE("phi alloc");
            ck(cudaMalloc(&pc.d_A, n * esz), "alloc PhiA");
            ck(cudaMalloc(&pc.d_B, n * esz), "alloc PhiB");
        }
        // fp16 converts on the host, so the upload is half the size too.
        const size_t bytes = n * esz;
        const void* srcA = hA;
        const void* srcB = hB;
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
        }
        ck(cudaMemcpyAsync(pc.d_A, srcA, bytes, cudaMemcpyHostToDevice, st), "PhiA up");
        ck(cudaMemcpyAsync(pc.d_B, srcB, bytes, cudaMemcpyHostToDevice, st), "PhiB up");
        if (sg)
            ck(cudaEventRecord(sg->done, st), "Phi staging record");
        pc.valid = true;
        pc.doppler = doppler;
        pc.n_chips = f.n_chips;
        pc.d_first = f.d_first; // the bank's own centered truncation, if set
        // The engine's cap, on top of any cap set on the bank. n_chips = ceil((Lf-1)*cps)+2 is
        // the PFB filter's span in chips (~210 at CHORD: 65536 samples at 312.8 samples/chip).
        // Capping it truncates the filter, so it changes the replica as well as the cost.
        if (max_chips > 0 && pc.n_chips - pc.d_first > max_chips) {
            // Keep the central max_chips: the prototype peaks mid-span, and a one-sided cap
            // shorter than about half the span cuts through the peak.
            pc.d_first += (pc.n_chips - pc.d_first - max_chips) / 2;
            pc.n_chips = pc.d_first + max_chips;
        }
        return pc;
    }

    /// Queue a build of slot p's tables at this Doppler unless one is already queued, building
    /// or waiting for its upload (psi_pending). Starts the builder thread on the first call.
    void psi_request(int p, double doppler) {
        if (psi_pending[(size_t)p])
            return;
        psi_pending[(size_t)p] = 1;
        PsiReq r;
        r.p = p;
        r.doppler = doppler;
        r.t_req = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lk(psi_mtx);
            r.gen = slot_gen[(size_t)p];
            psi_reqs.push_back(r);
            if (!psi_thread.joinable())
                psi_thread = std::thread(&Impl::psi_builder_loop, this);
        }
        psi_cv.notify_one();
    }

    /// The builder thread: one request at a time, in order. Niced below the enqueue thread it
    /// inherits its affinity from (the gpuProcess's cpu_affinity), so it takes that core's idle
    /// time and not the enqueue thread's; the CUDA context is never touched here.
    void psi_builder_loop() {
        pthread_setname_np(pthread_self(), "psi_build");
        setpriority(PRIO_PROCESS, 0, 19); // this thread only (Linux: nice is per thread)
        for (;;) {
            PsiReq r;
            {
                std::unique_lock<std::mutex> lk(psi_mtx);
                psi_cv.wait(lk, [&] { return psi_stop || !psi_reqs.empty(); });
                if (psi_stop)
                    return;
                r = psi_reqs.front();
                psi_reqs.pop_front();
            }
            PsiDone d;
            d.p = r.p;
            d.doppler = r.doppler;
            d.gen = r.gen;
            const auto t0 = std::chrono::steady_clock::now();
            d.wait_ms = std::chrono::duration<double, std::milli>(t0 - r.t_req).count();
            try {
                const double cps = bank.eff_chip_rate() / fs
                                   * (1.0 + bank.code_doppler_sign * r.doppler / bank.carrier_hz());
                const int n_full = (int)std::ceil((double)(Lf - 1) * cps) + 2; // the bank's span
                gnss_cuda::PsiCodeSplit cs;
                bool stale = false;
                {
                    // The only read of the bank's code table: a copy of the slot's split code,
                    // under the lock set_prn mutates it under.
                    std::lock_guard<std::mutex> lk(code_mtx);
                    if (slot_gen[(size_t)r.p] != r.gen) {
                        stale = true; // swapped since the request: nothing to build
                    } else {
                        const auto& code = bank.full_code(r.p);
                        if (psi_P_builder[(size_t)r.p] == 0) {
                            psi_P_builder[(size_t)r.p] =
                                gnss_cuda::psi_primary_period(code.data(), code_len, n_full);
                            if (psi_P_builder[(size_t)r.p] <= 0)
                                throw std::runtime_error(
                                    "GnssCudaDespread: PRN slot " + std::to_string(r.p)
                                    + ": no primary period for the Psi split (code length "
                                    + std::to_string(code_len) + ", PFB span "
                                    + std::to_string(n_full) + " chips)");
                        }
                        d.P = psi_P_builder[(size_t)r.p];
                        cs = gnss_cuda::psi_split_code(code.data(), code_len, d.P);
                    }
                }
                if (!stale) {
                    const int G = gnss_cuda::psi_grid_for(1.0 / cps);
                    d.S =
                        gnss_cuda::build_psi_split_from(bank, r.p, all_chans, r.doppler, fs, G, cs);
                    // The table holds the near carrier image only. Refuse a channel whose far
                    // image is the nearer one (a carrier above fs/2) or is not far enough to be
                    // negligible.
                    for (int ci = 0; ci < n_chan; ++ci) {
                        double far = 0.0;
                        const double thB = gnss_cuda::psi_theta(all_chans[(size_t)ci],
                                                                bank.fft_len(), d.S.wc, &far);
                        if (far <= std::fabs(thB) || far * d.S.inv_cps < 50.0)
                            throw std::runtime_error(
                                "GnssCudaDespread: channel " + std::to_string(all_chans[(size_t)ci])
                                + ": far carrier image at |theta| " + std::to_string(far)
                                + " rad/sample (" + std::to_string(far * d.S.inv_cps)
                                + " rad/chip, near image " + std::to_string(std::fabs(thB))
                                + ") -- the Psi table carries the near image only");
                    }
                }
            } catch (const std::exception& e) {
                d.error = e.what();
            }
            d.t_done = std::chrono::steady_clock::now();
            d.build_ms = std::chrono::duration<double, std::milli>(d.t_done - t0).count();
            DEBUG_NON_OO("GnssCudaDespread: Psi build slot {:d} at Doppler {:.1f} Hz: {:.1f} ms "
                         "(queued {:.1f} ms){:s}",
                         r.p, r.doppler, d.build_ms, d.wait_ms,
                         d.error.empty() ? std::string() : " FAILED: " + d.error);
            {
                std::lock_guard<std::mutex> lk(psi_mtx);
                psi_done.push_back(std::move(d));
            }
        }
    }

    /// Move the builder's finished builds to the enqueue thread's queue.
    void psi_drain() {
        std::vector<PsiDone> got;
        {
            std::lock_guard<std::mutex> lk(psi_mtx);
            if (psi_done.empty())
                return;
            got.swap(psi_done);
        }
        for (auto& d : got)
            psi_ready_q.push_back(std::move(d));
    }

    /// Upload a finished build to slot p's device tables on `st` through a free pinned staging
    /// slot, and mark the cache valid. False, with nothing done, when every staging slot still
    /// has an upload in flight (the caller tries again next record).
    bool psi_upload(PsiDone& d, cudaStream_t st) {
        Staging* sg = nullptr;
        for (auto& cand : psi_staging) {
            if (!cand.done) {
                ck(cudaEventCreateWithFlags(&cand.done, cudaEventDisableTiming),
                   "Psi staging event");
                sg = &cand;
                break;
            }
            const cudaError_t q = cudaEventQuery(cand.done);
            if (q == cudaSuccess) {
                sg = &cand;
                break;
            }
            if (q != cudaErrorNotReady)
                ck(q, "Psi staging query");
        }
        if (!sg)
            return false;
        PsiCache& pc = psi[(size_t)d.p];
        const gnss_cuda::PsiSplit& S = d.S;
        const size_t esz = psi_half ? sizeof(float2) / 2 : sizeof(float2);
        const size_t n_psi = S.psi_plain.size();
        const size_t n_prev = S.Ns > 1 ? S.psi_prev.size() : 0; // a plain code has no lookback
        const size_t b_psi = n_psi * esz, b_prev = n_prev * esz;
        const size_t b_sec = S.Ns > 1 ? S.sec.size() : 0;
        // G and P are fixed by the code and the chip rate; n_chips moves by at most a chip
        // with the Doppler, so the lookback is allocated with two grid cells per channel of
        // headroom and never regrows (cudaFree would synchronise the device).
        if (pc.psi_bytes < b_psi) {
            ck(cudaFree(pc.d_psi), "free psi");
            ck(cudaMalloc(&pc.d_psi, b_psi), "alloc psi");
            pc.psi_bytes = b_psi;
        }
        if (pc.prev_bytes < b_prev) {
            const size_t want = b_prev + (size_t)2 * S.G * n_chan * esz;
            ck(cudaFree(pc.d_prev), "free psi_prev");
            ck(cudaMalloc(&pc.d_prev, want), "alloc psi_prev");
            pc.prev_bytes = want;
        }
        if (pc.sec_bytes < b_sec) {
            ck(cudaFree(pc.d_sec), "free psi sec");
            ck(cudaMalloc(&pc.d_sec, b_sec), "alloc psi sec");
            pc.sec_bytes = b_sec;
        }
        const size_t bytes = b_psi + b_prev + b_sec;
        if (sg->bytes < bytes) {
            ck(cudaFreeHost(sg->host), "free Psi staging");
            ck(cudaMallocHost((void**)&sg->host, bytes), "alloc Psi staging");
            sg->bytes = bytes;
        }
        if (psi_half) {
            pairs_to_half(S.psi_plain.data(), sg->host, n_psi);
            if (n_prev)
                pairs_to_half(S.psi_prev.data(), sg->host + b_psi, n_prev);
        } else {
            std::memcpy(sg->host, S.psi_plain.data(), b_psi);
            std::memcpy(sg->host + b_psi, S.psi_prev.data(), b_prev);
        }
        std::memcpy(sg->host + b_psi + b_prev, S.sec.data(), b_sec);
        ck(cudaMemcpyAsync(pc.d_psi, sg->host, b_psi, cudaMemcpyHostToDevice, st), "psi up");
        if (b_prev)
            ck(cudaMemcpyAsync(pc.d_prev, sg->host + b_psi, b_prev, cudaMemcpyHostToDevice, st),
               "psi_prev up");
        if (b_sec)
            ck(cudaMemcpyAsync(pc.d_sec, sg->host + b_psi + b_prev, b_sec, cudaMemcpyHostToDevice,
                               st),
               "psi sec up");
        ck(cudaEventRecord(sg->done, st), "psi staging record");
        pc.valid = true;
        pc.doppler = d.doppler;
        pc.P = d.P;
        pc.G = S.G;
        pc.Ns = S.Ns;
        pc.n_prev = S.Ns > 1 ? S.n_prev : 0; // per channel (PsiJob::n_prev)
        pc.n_chips = S.n_chips;
        const double age_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - d.t_done)
                .count();
        if (pc.n_builds++ == 0)
            INFO_NON_OO("GnssCudaDespread: Psi tables for slot {:d} (PRN {:d}): G {:d}, period "
                        "{:d} x {:d}, span {:d} chips, {:.1f} KB/channel {:s}, built in {:.1f} "
                        "ms (queued {:.1f} ms, uploaded {:.1f} ms after the build) at Doppler "
                        "{:.1f} Hz",
                        d.p, slot_prn[(size_t)d.p], S.G, pc.P, S.Ns, S.n_chips,
                        (double)(b_psi + b_prev) / n_chan / 1e3, psi_half ? "fp16" : "fp32",
                        d.build_ms, d.wait_ms, age_ms, d.doppler);
        else
            DEBUG_NON_OO("GnssCudaDespread: Psi rebuild slot {:d} at Doppler {:.1f} Hz: {:.1f} ms "
                         "(queued {:.1f}, uploaded {:.1f} ms after)",
                         d.p, d.doppler, d.build_ms, d.wait_ms, age_ms);
        return true;
    }

    /// GnssCudaDespread::psi_ready.
    bool psi_ready(int p, double doppler, cudaStream_t st) {
        psi_drain();
        // This slot's finished build, if it has one: upload it (one per record), drop it if
        // the slot was swapped since, rethrow if it failed.
        for (auto it = psi_ready_q.begin(); it != psi_ready_q.end(); ++it) {
            if (it->p != p)
                continue;
            if (it->gen != slot_gen[(size_t)p]) {
                psi_ready_q.erase(it);
                psi_pending[(size_t)p] = 0;
            } else if (!it->error.empty()) {
                const std::string e = it->error;
                psi_ready_q.erase(it);
                psi_pending[(size_t)p] = 0;
                throw std::runtime_error(e);
            } else if (psi_uploads_left > 0 && psi_upload(*it, st)) {
                --psi_uploads_left;
                psi_ready_q.erase(it);
                psi_pending[(size_t)p] = 0;
            }
            break;
        }
        PsiCache& pc = psi[(size_t)p];
        const double dd =
            pc.valid ? std::fabs(doppler - pc.doppler) : std::numeric_limits<double>::infinity();
        if (dd > psi_refresh_hz)
            psi_request(p, doppler);
        // A table within the refresh band is current; one past it keeps serving while its
        // replacement builds (the filter response barely moves over tens of Hz, and going dark
        // would break the slot's phase arc at every refresh), up to twice the band, beyond
        // which the slot is dark until the rebuild lands.
        return pc.valid && dd <= 2.0 * psi_refresh_hz;
    }

    // Build the per-PRN jobs for `specs` against the window starting at window_start, building
    // any Phi bucket that needs it on `st` (`pinned` as for ensure_phi), and upload them to
    // d_jobs_slot on `st`. Every Spec-taking call builds its jobs here, so their replicas cannot
    // differ. h_jobs staging is pageable, so the async H2D below is host-synchronous on return
    // and the buffer is safely reusable per call. `psi_path` takes the slot's Psi tables
    // instead of Phi: the job's Phi fields are then null and its span the table's (the Psi
    // kernels ignore both), and last_phi_ddop records the Psi table's staleness.
    void upload_jobs(const std::vector<Spec>& specs, void* d_jobs_slot, long long window_start,
                     cudaStream_t st, bool pinned, bool psi_path = false) {
        for (size_t i = 0; i < specs.size(); ++i) {
            const Spec& sp = specs[i];
            const float2 *dA = nullptr, *dB = nullptr;
            int n_chips = 0, d_first = 0;
            double table_doppler = 0.0;
            if (psi_path) {
                // Never built here: psi_ready admits a slot to a record only once its tables
                // are on the device, so a spec without them is a caller's error, not a stall.
                const PsiCache& pc = psi[(size_t)sp.p];
                if (!pc.valid)
                    throw std::runtime_error("GnssCudaDespread: PRN slot " + std::to_string(sp.p)
                                             + " has no Psi tables on the device -- psi_ready "
                                               "must admit a slot before it is despread");
                n_chips = pc.n_chips;
                table_doppler = pc.doppler;
            } else {
                const PhiCache& pc = ensure_phi(sp.p, sp.doppler_hz, st, pinned);
                dA = pc.d_A;
                dB = pc.d_B;
                n_chips = pc.n_chips;
                d_first = pc.d_first;
                table_doppler = pc.doppler;
            }
            const double cps = bank.eff_chip_rate() / fs
                               * (1.0 + bank.code_doppler_sign * sp.doppler_hz / bank.carrier_hz());
            // The carrier trim enters wc and ang0 only (see wc_for).
            const double wc = wc_for(sp.p, sp.doppler_hz, sp.ctrim_hz);
            uint64_t mask = 0;
            for (int c : sp.covering)
                if (c >= 0 && c < n_chan)
                    mask |= (1ULL << c);
            const double cp0 = (double)bank.comb_mult() * sp.cp_seed;

            // Output row 3 = P_HEAD: the prompt up to the code-period boundary (m_head_for).
            const int m_head = m_head_for(cp0, cps, window_start);

            h_jobs[i] = {
                (double)bank.comb_mult() * sp.spacing_chips,
                cp_ref_for(cp0, cps, window_start + bank.fft_len() - 1),
                cps,
                1.0 / cps,
                wc,
                ang0_for(sp.p, sp.doppler_hz, sp.ctrim_hz, window_start + bank.fft_len() - 1),
                code_offset[(size_t)sp.p],
                (int)code_len,
                mask,
                dA,
                dB,
                n_chips,
                m_head,
                d_first};
            // Record what the kernel is given, read back from the job (see last_ang0).
            last_ang0[(size_t)sp.p] = h_jobs[i].ang0;
            last_phi_ddop[(size_t)sp.p] = sp.doppler_hz - table_doppler;
            if (epl_rot) {
                // Row 0 (the prompt - ds trial) is the prompt rotated by
                // +2 pi (f_c - f_car) ds/cps/fs on the channel centred at f_c, with f_car the
                // carrier the kernel applies (wc); row 2 takes the opposite sign. The sign is
                // pinned by cudaGnssDespreadTest against the fused kernel's own E row.
                const double delay_s = h_jobs[i].ds / cps / fs;
                const double f_car = wc * fs / (2.0 * M_PI);
                for (int ci = 0; ci < n_chan; ++ci) {
                    const double f_c = (double)all_chans[(size_t)ci] * fs / bank.fft_len();
                    h_rot[i * (size_t)n_chan + ci] = (float)(2.0 * M_PI * (f_c - f_car) * delay_s);
                    last_rot[(size_t)sp.p * n_chan + ci] = h_rot[i * (size_t)n_chan + ci];
                }
                last_rot_ok[(size_t)sp.p] = 1;
            }
        }
        ck(cudaMemcpyAsync(d_jobs_slot, h_jobs.data(),
                           specs.size() * sizeof(gnss_cuda::DespreadJob), cudaMemcpyHostToDevice,
                           st),
           "jobs upload");
        if (epl_rot)
            ck(cudaMemcpyAsync(d_rot, h_rot.data(), specs.size() * (size_t)n_chan * sizeof(float),
                               cudaMemcpyHostToDevice, st),
               "rot upload");
    }
};

GnssCudaDespread::GnssCudaDespread(gnss::ChannelizedReplicaBank& bank, int n_prn,
                                   const std::vector<int>& chan_ids, int n_hops, double sample_rate,
                                   double refresh_hz) :
    _impl(new Impl(bank, n_prn, (int)chan_ids.size(), n_hops, sample_rate, refresh_hz, chan_ids)) {
    // FDMA (GLONASS L1OF/L2OF) is supported: the Phi cache is per PRN, and each PRN's filter and
    // carrier phasor use its own offset (see wc_for).
}

GnssCudaDespread::~GnssCudaDespread() = default;

bool GnssCudaDespread::set_prn(int p, int prn, void* stream) {
    Impl& im = *_impl;
    if (p < 0 || p >= im.n_prn)
        return false;
    // Compare against this engine's own view, not the bank's. Several engines can share one
    // bank (the search builds one per <=64-channel refine group), each with its own device copy
    // of the code table; after the first engine updates the bank, the others must still upload.
    if (im.slot_prn[(size_t)p] == prn)
        return true; // this engine is already there
    {
        // The Psi builder copies a slot's code under this lock: the generation bump before the
        // bank changes retires any build it has queued or running for the old satellite.
        std::lock_guard<std::mutex> lk(im.code_mtx);
        if (im.bank.prn_at(p) != prn && !im.bank.set_prn(p, prn))
            return false;
        ++im.slot_gen[(size_t)p];
        im.psi_P_builder[(size_t)p] = 0; // the new code's primary period is searched afresh
        // Re-upload this slot's row of the concatenated code table. The staging buffer is a
        // member so that it outlives the stream-ordered copy.
        if (im.code_stage.size() != (size_t)im.n_prn * (size_t)im.code_len)
            im.code_stage.assign((size_t)im.n_prn * (size_t)im.code_len, 0);
        const auto& fc = im.bank.full_code(p);
        std::copy(fc.begin(), fc.end(), im.code_stage.begin() + (size_t)im.code_offset[(size_t)p]);
    }
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
    im.psi[(size_t)p].valid = false; // rebuilt by the builder on the slot's next psi_ready
    im.psi[(size_t)p].P = 0;
    im.last_ang0[(size_t)p] = std::numeric_limits<double>::quiet_NaN();
    im.last_phi_ddop[(size_t)p] = std::numeric_limits<double>::quiet_NaN();
    im.slot_prn[(size_t)p] = prn;
    return true;
}

const float* GnssCudaDespread::last_rot(int p) const {
    if (!_impl->epl_rot || p < 0 || (size_t)p >= _impl->last_rot_ok.size()
        || !_impl->last_rot_ok[(size_t)p])
        return nullptr;
    return _impl->last_rot.data() + (size_t)p * _impl->n_chan;
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

void GnssCudaDespread::set_phi_fp16() {
    Impl& im = *_impl;
    im.use_fp16 = true;
    for (auto& pc : im.phi) // rebuild any float2 bucket as half before a kernel reads it
        pc.valid = false;
    const size_t n = (size_t)im.n_chan * (size_t)(im.Lf + 1);
    std::fprintf(stderr,
                 "INFO: GnssCudaDespread: fp16 Phi ARMED -- %.1f MB over %d channels per PRN "
                 "bucket (fp32 was %.1f), storage error ~3.3e-4; despread_batch will "
                 "REFUSE while this is set.\n",
                 2.0 * (double)n * (sizeof(float2) / 2) / 1e6, im.n_chan,
                 2.0 * (double)n * sizeof(float2) / 1e6);
}

void GnssCudaDespread::set_epl_from_prompt() {
    _impl->epl_rot = true;
}

void GnssCudaDespread::set_psi_synth(double refresh_hz) {
    Impl& im = *_impl;
    im.psi_on = true;
    im.psi_refresh_hz = refresh_hz;
    INFO_NON_OO("GnssCudaDespread: Psi synthesis ARMED (table refresh {:g} Hz, {:s} tables, "
                "built on the engine's builder thread: a slot is dark until its tables land), "
                "full PFB span{:s}",
                refresh_hz, im.psi_half ? "fp16" : "fp32",
                im.max_chips > 0 ? ": the chip cap is the Phi path's and does not apply" : "");
}

bool GnssCudaDespread::psi_ready(int p, double doppler_hz, void* stream) {
    Impl& im = *_impl;
    if (!im.psi_on)
        throw std::runtime_error("GnssCudaDespread: psi_ready without set_psi_synth");
    if (p < 0 || p >= im.n_prn)
        return false;
    return im.psi_ready(p, doppler_hz, (cudaStream_t)stream);
}

void GnssCudaDespread::set_psi_fp16() {
    Impl& im = *_impl;
    im.psi_half = true;
    for (auto& pc : im.psi) // rebuild any float2 table as half before a kernel reads it
        pc.valid = false;
}

void GnssCudaDespread::set_max_chips(int n) {
    _impl->max_chips = n;
    for (auto& pc : _impl->phi) // force a rebuild so the cap takes effect on cached buckets
        pc.valid = false;
    if (_impl->psi_on)
        INFO_NON_OO("GnssCudaDespread: max_chips {:d} applies to the Phi path only; the Psi "
                    "tables keep the full PFB span",
                    n);
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

    // This engine's stream is idle between batches, so the Phi uploads need no staging.
    im.upload_jobs(specs, im.d_jobs, im.window_start, im.stream, false);

    gnss_cuda::DespreadParams par;
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

void GnssCudaDespread::build_jobs(const std::vector<Spec>& specs, void* d_jobs_slot,
                                  long long window_start_sample, void* stream) {
    _impl->upload_jobs(specs, d_jobs_slot, window_start_sample, (cudaStream_t)stream, true);
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
    ck(gnss_cuda::launch_waveform(im.d_code, (gnss_cuda::DespreadJob*)d_jobs_slot, n_spec,
                                  im.n_chan, par, nullptr, (float2*)d_wave, (double*)d_energy_out,
                                  st),
       "launch waveform (NxM)");
    ck(gnss_cuda::launch_correlate_nm(
           (const unsigned char*)d_frame, (const float*)d_chan_scale, d_chan_ids,
           (const float2*)d_wave, (gnss_cuda::DespreadJob*)d_jobs_slot, n_spec, im.n_chan, n_elem,
           elem_stride, frame_chan_stride, par, (double2*)d_corr_out, st),
       "launch correlate NxM");
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
                                  im.n_chan, par, im.epl_rot ? im.d_rot : nullptr, (float2*)d_wave,
                                  (double*)d_energy_out, st),
       "launch waveform (inject)");
    return n_spec;
}

int GnssCudaDespread::enqueue_psi_pack(long long window_start_sample,
                                       const std::vector<Spec>& specs, void* d_jobs_slot,
                                       void* d_pjobs_slot, void* d_theta_slot,
                                       const int* slot2spec_dev, int n_slot, int lanes_per_slot,
                                       const int* chan_map_dev, int frame_chan_stride, int n_lanes,
                                       int lane_base, int lane_pitch, const double* energy_scale,
                                       void* d_energy_out, unsigned char* d_synth_rec,
                                       void* stream) {
    Impl& im = *_impl;
    if (!im.psi_on)
        throw std::runtime_error("GnssCudaDespread: enqueue_psi_pack without set_psi_synth");
    const int n_spec = (int)specs.size();
    const cudaStream_t st = (cudaStream_t)stream;
    if (n_spec > 0) {
        im.upload_jobs(specs, d_jobs_slot, window_start_sample, st, true, true);
        for (int i = 0; i < n_spec; ++i) {
            const Impl::PsiCache& pc = im.psi[(size_t)specs[(size_t)i].p];
            const gnss_cuda::DespreadJob& j = im.h_jobs[(size_t)i];
            gnss_cuda::PsiJob& pj = im.h_pjobs[(size_t)i];
            pj = gnss_cuda::PsiJob{};
            pj.psi = pc.d_psi;
            pj.psi_prev = pc.d_prev;
            pj.sec = pc.d_sec;
            pj.P = pc.P;
            pj.Ns = pc.Ns;
            pj.G = pc.G;
            pj.n_prev = pc.n_prev;
            gnss_cuda::psi_fixed_setup(j, pj, im.bank.fft_len(), im.n_hops);
            // The interpolation ramp per channel at this job's carrier (trim included), in
            // the kernel's grid units: theta_B * inv_cps / G.
            for (int ci = 0; ci < im.n_chan; ++ci)
                im.h_theta[(size_t)i * im.n_chan + ci] =
                    (float)(gnss_cuda::psi_theta(im.all_chans[(size_t)ci], im.bank.fft_len(), j.wc)
                            * j.inv_cps / pc.G);
        }
        ck(cudaMemcpyAsync(d_pjobs_slot, im.h_pjobs.data(),
                           (size_t)n_spec * sizeof(gnss_cuda::PsiJob), cudaMemcpyHostToDevice, st),
           "psi jobs upload");
        ck(cudaMemcpyAsync(d_theta_slot, im.h_theta.data(),
                           (size_t)n_spec * im.n_chan * sizeof(float), cudaMemcpyHostToDevice, st),
           "psi theta upload");
    }

    gnss_cuda::DespreadParams par;
    par.phi_half = im.psi_half ? 1 : 0; // selects the __half2 TABLE kernel; Phi is not read
    par.n0 = window_start_sample + im.bank.fft_len() - 1;
    par.fft_len = im.bank.fft_len();
    par.n_hops = im.n_hops;
    par.Lf = im.Lf;
    par.data_stride = im.n_hops;
    // Unconditional, as launch_pack44's is: with no spec the record's lanes still go 0x88.
    ck(gnss_cuda::launch_psi_pack(
           (const gnss_cuda::DespreadJob*)d_jobs_slot, (const gnss_cuda::PsiJob*)d_pjobs_slot,
           (const float*)d_theta_slot, im.epl_rot ? im.d_rot : nullptr, /*fixed=*/1, slot2spec_dev,
           n_slot, lanes_per_slot, im.n_chan, par, chan_map_dev, frame_chan_stride, n_lanes,
           lane_base, lane_pitch, energy_scale, (double*)d_energy_out, d_synth_rec, st),
       "launch psi pack (inject)");
    // One launch per record: the next record's psi_ready calls may upload one more table.
    im.psi_uploads_left = 1;
    return n_spec;
}
