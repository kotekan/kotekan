#include "GnssCudaDespread.hpp"

#include "cudaGnssChordDespread.hpp"
#include "cudaGnssDespreadKernel.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

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

    // Device buffers (persistent).
    float2* d_data = nullptr;                  // [n_chan][n_hops]
    int8_t* d_code = nullptr;                  // all PRNs' combined-stream codes, concatenated
    std::vector<int> code_offset;              // per PRN slot
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
    std::vector<int> all_chans;
    int Lf = 0;

    std::vector<float2> stage;  // host transpose staging [chan][hop]
    cudaStream_t stream = nullptr;
    std::vector<gnss_cuda::DespreadJob> h_jobs;
    std::vector<double2> h_corr;
    std::vector<double> h_energy;
    std::vector<gnss_cuda::PeelJob> h_pjobs; // peel staging (pageable: H2D is host-sync on return)
    std::vector<float2> h_gains;             // [job][head|tail][chan]

    Impl(gnss::ChannelizedReplicaBank& b, int np, int nc, int coff, int nh, double fs_, double fo,
         double rh) :
        bank(b), n_prn(np), n_chan(nc), n_hops(nh), fs(fs_), f_off(fo), refresh_hz(rh) {
        // Phi tables index by LOCAL channel ci but must be built at the GLOBAL channel
        // frequency (the subband's absolute position in the band) -- chan_offset maps them.
        (void)coff;
        if (nc > 64)
            throw std::runtime_error("GnssCudaDespread: >64 channels needs a wider chan_mask");
        Lf = bank.fft_len() * 4; // num_taps -- matches the bank's prototype (pfb num_taps)
        // NB the bank doesn't expose num_taps; derive Lf from a probe filter below instead.
        all_chans.resize(nc);
        for (int c = 0; c < nc; ++c)
            all_chans[c] = coff + c;
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
    double wc_for(double doppler_hz) const {
        return 2.0 * M_PI * (f_off + doppler_hz) / fs;
    }

    // (Re)build PRN p's Phi bucket at this Doppler if it moved more than refresh_hz.
    PhiCache& ensure_phi(int p, double doppler) {
        PhiCache& pc = phi[(size_t)p];
        if (pc.valid && std::fabs(doppler - pc.doppler) <= refresh_hz)
            return pc;
        const auto f = bank.hoprate_filter(all_chans, doppler);
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
        return pc;
    }
};

GnssCudaDespread::GnssCudaDespread(gnss::ChannelizedReplicaBank& bank, int n_prn, int n_chan,
                                   int chan_offset, int n_hops, double sample_rate,
                                   double f_offset, double refresh_hz) :
    _impl(new Impl(bank, n_prn, n_chan, chan_offset, n_hops, sample_rate, f_offset, refresh_hz)) {}

GnssCudaDespread::~GnssCudaDespread() = default;

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
        const double wc = 2.0 * M_PI * (im.f_off + sp.doppler_hz) / im.fs;
        uint64_t mask = 0;
        for (int c : sp.covering)
            if (c >= 0 && c < im.n_chan)
                mask |= (1ULL << c);
        im.h_jobs[i] = {(double)im.bank.comb_mult() * sp.cp_seed,
                        (double)im.bank.comb_mult() * sp.spacing_chips,
                        cps,
                        1.0 / cps,
                        wc,
                        im.code_offset[(size_t)sp.p],
                        (int)im.code_len,
                        mask,
                        pc.d_A,
                        pc.d_B,
                        pc.n_chips,
                        0};
    }
    ck(cudaMemcpyAsync(im.d_jobs, im.h_jobs.data(),
                       (size_t)n_spec * sizeof(gnss_cuda::DespreadJob), cudaMemcpyHostToDevice,
                       im.stream),
       "jobs upload");

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
        const double wc = 2.0 * M_PI * (im.f_off + sp.doppler_hz) / im.fs;
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
                        cps,
                        1.0 / cps,
                        wc,
                        im.code_offset[(size_t)sp.p],
                        (int)im.code_len,
                        mask,
                        pc.d_A,
                        pc.d_B,
                        pc.n_chips,
                        m_head};
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
    par.n0 = window_start_sample + im.bank.fft_len() - 1; // hoprate_stream's per-hop reference
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
    par.n0 = window_start_sample + im.bank.fft_len() - 1; // hoprate_stream's per-hop reference
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
    ck(gnss_cuda::launch_waveform(im.d_code, (gnss_cuda::DespreadJob*)d_jobs_slot, n_spec,
                                  im.n_chan, par, (float2*)d_wave, (double*)d_energy_out, st),
       "launch waveform (NxM)");
    ck(gnss_cuda::launch_correlate_nm((const unsigned char*)d_frame, (const float*)d_chan_scale,
                                      d_chan_ids, (const float2*)d_wave,
                                      (gnss_cuda::DespreadJob*)d_jobs_slot, n_spec, im.n_chan,
                                      n_elem, elem_stride, frame_chan_stride, par,
                                      (double2*)d_corr_out, st),
       "launch correlate NxM");
    return 4 * n_spec;
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
                                 cps,
                                 1.0 / cps,
                                 im.wc_for(sp.doppler_hz),
                                 im.code_offset[(size_t)sp.p],
                                 (int)im.code_len,
                                 mask,
                                 pc.d_A,
                                 pc.d_B,
                                 pc.n_chips,
                                 im.m_head_for(cp0, cps, window_start_sample), // SHARED with P_HEAD
                                 g_head,
                                 g_tail};
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
    par.n0 = window_start_sample + im.bank.fft_len() - 1; // same per-hop reference as the despread
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
