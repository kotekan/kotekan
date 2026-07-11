#include "GnssCudaDespread.hpp"

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
    gnss_cuda::DespreadJob* d_jobs = nullptr;  // [3]
    double2* d_corr = nullptr;                 // [3][n_chan]
    double* d_energy = nullptr;                // [3][n_chan]

    // Per-PRN Doppler-bucketed Phi tables on device ([n_chan][Lf+1] each, all channels).
    struct PhiCache {
        bool valid = false;
        double doppler = 0.0;
        int n_chips = 0;
        double2 *d_A = nullptr, *d_B = nullptr;
    };
    std::vector<PhiCache> phi;
    std::vector<int> all_chans;
    int Lf = 0;

    std::vector<float2> stage;  // host transpose staging [chan][hop]

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

        ck(cudaMalloc(&d_data, (size_t)nc * nh * sizeof(float2)), "alloc data");
        ck(cudaMalloc(&d_jobs, 3 * sizeof(gnss_cuda::DespreadJob)), "alloc jobs");
        ck(cudaMalloc(&d_corr, (size_t)3 * nc * sizeof(double2)), "alloc corr");
        ck(cudaMalloc(&d_energy, (size_t)3 * nc * sizeof(double)), "alloc energy");
        phi.resize(np);
        stage.resize((size_t)nc * nh);
    }

    ~Impl() {
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

    // (Re)build PRN p's Phi bucket at this Doppler if it moved more than refresh_hz.
    PhiCache& ensure_phi(int p, double doppler) {
        PhiCache& pc = phi[(size_t)p];
        if (pc.valid && std::fabs(doppler - pc.doppler) <= refresh_hz)
            return pc;
        const auto f = bank.hoprate_filter(all_chans, doppler);
        const size_t n = (size_t)n_chan * (Lf + 1);
        std::vector<double2> hA(n), hB(n);
        for (int c = 0; c < n_chan; ++c)
            for (int k = 0; k <= Lf; ++k) {
                hA[(size_t)c * (Lf + 1) + k] =
                    make_double2(f.PhiA[c][k].real(), f.PhiA[c][k].imag());
                hB[(size_t)c * (Lf + 1) + k] =
                    make_double2(f.PhiB[c][k].real(), f.PhiB[c][k].imag());
            }
        if (!pc.d_A) {
            ck(cudaMalloc(&pc.d_A, n * sizeof(double2)), "alloc PhiA");
            ck(cudaMalloc(&pc.d_B, n * sizeof(double2)), "alloc PhiB");
        }
        ck(cudaMemcpy(pc.d_A, hA.data(), n * sizeof(double2), cudaMemcpyHostToDevice), "PhiA up");
        ck(cudaMemcpy(pc.d_B, hB.data(), n * sizeof(double2), cudaMemcpyHostToDevice), "PhiB up");
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
    ck(cudaMemcpy(im.d_data, im.stage.data(), im.stage.size() * sizeof(float2),
                  cudaMemcpyHostToDevice),
       "window upload");
    im.window_start = window_start_sample;
}

std::array<gnss::DespreadResult, 3>
GnssCudaDespread::despread3(int p, double cp_seed, double spacing_chips, double doppler_hz,
                            const std::vector<int>& covering) {
    Impl& im = *_impl;
    auto& pc = im.ensure_phi(p, doppler_hz);

    // Job parameters, exactly as hoprate_stream computes them.
    const double cps = im.bank.eff_chip_rate() / im.fs
                       * (1.0 + im.bank.code_doppler_sign * doppler_hz / im.bank.carrier_hz());
    const double wc = 2.0 * M_PI * (im.f_off + doppler_hz) / im.fs;
    uint64_t mask = 0;
    for (int c : covering)
        if (c >= 0 && c < im.n_chan)
            mask |= (1ULL << c);

    gnss_cuda::DespreadJob jobs[3];
    const double trials[3] = {cp_seed - spacing_chips, cp_seed, cp_seed + spacing_chips};
    for (int t = 0; t < 3; ++t)
        jobs[t] = {(double)im.bank.comb_mult() * trials[t], cps, wc,
                   im.code_offset[(size_t)p], (int)im.code_len, mask};
    ck(cudaMemcpy(im.d_jobs, jobs, sizeof(jobs), cudaMemcpyHostToDevice), "jobs upload");

    gnss_cuda::DespreadParams par;
    par.n0 = im.window_start + im.bank.fft_len() - 1; // hoprate_stream's per-hop reference
    par.fft_len = im.bank.fft_len();
    par.n_hops = im.n_hops;
    par.Lf = im.Lf;
    par.n_chips = pc.n_chips;
    ck(gnss_cuda::launch_despread(im.d_data, im.d_code, pc.d_A, pc.d_B, im.d_jobs, 3, im.n_chan,
                                  par, im.d_corr, im.d_energy, 0),
       "launch");

    std::vector<double2> corr((size_t)3 * im.n_chan);
    std::vector<double> energy((size_t)3 * im.n_chan);
    ck(cudaMemcpy(corr.data(), im.d_corr, corr.size() * sizeof(double2), cudaMemcpyDeviceToHost),
       "corr down");
    ck(cudaMemcpy(energy.data(), im.d_energy, energy.size() * sizeof(double),
                  cudaMemcpyDeviceToHost),
       "energy down");

    std::array<gnss::DespreadResult, 3> out;
    for (int t = 0; t < 3; ++t) {
        std::complex<double> g(0.0, 0.0);
        double e = 0.0;
        for (int c = 0; c < im.n_chan; ++c) {
            g += std::complex<double>(corr[(size_t)t * im.n_chan + c].x,
                                      corr[(size_t)t * im.n_chan + c].y);
            e += energy[(size_t)t * im.n_chan + c];
        }
        out[t].correlation = g;
        out[t].replica_energy = e;
        out[t].amplitude = (e > 0.0) ? g / e : std::complex<double>(0.0, 0.0);
    }
    return out;
}
