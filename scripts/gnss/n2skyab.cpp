/**
 * @file
 * @brief THE ON-SKY A/B: path A (launch_correlate_nm) vs path B (n2k_dual mixed tiles) over
 *        the SAME real CHORD voltage frame, per element. GPU required.
 *
 * n2dualxval proved the two agree on SYNTHETIC data (+0.036 dB / 2.29 mrad). This closes the
 * loop on real sky, and it exists because path A exports only the elem_sum-calibrated header
 * value -- there is no per-element correlation on the wire. peek_hold on the tap buffer gives
 * us the actual bytes instead, so both paths can be run offline over identical input and
 * compared element by element. Any difference is then purely what path B changes: the replica
 * quantized to 4+4b synthetic lanes and correlated by the tensor-core kernel.
 *
 * Input: n2sky_fetch.py's <prefix>.bin + <prefix>.json (real voltage + tracked cp/doppler).
 *
 * Seeds are propagated from the combiner's (cp, doppler) at its ref_hop to each record window
 * with gnss::propagate_seed -- the SAME function the tracker uses, so the replicas cannot fork.
 *
 * Geometry note: the tap frame is [hop][n_chan][n_elem] with n_chan=7, n_elem=32. n2k_dual
 * needs [T][F][128] stations and a compiled (NS,NF); we pad to F=8, NSA=128 (elements 0..31
 * live, 32..127 = 0x88) and use the dual_kernel_256_8 instantiation.
 *
 * usage: n2skyab [prefix] [record_index]
 */

#include "GnssCudaDespread.hpp"
#include "cudaGnssChordDespread.hpp"
#include "cudaGnssDespreadKernel.hpp"
#include "gnssChannelizedReplica.hpp"
#include "gnssSeedTransport.hpp"
#include "gnssSignal.hpp"

#include <n2k_dual/DualCorrelator.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using cd = std::complex<double>;

#define CK(x)                                                                                     \
    do {                                                                                          \
        cudaError_t e_ = (x);                                                                     \
        if (e_ != cudaSuccess) {                                                                  \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);       \
            return 1;                                                                             \
        }                                                                                         \
    } while (0)

// Minimal JSON scalar/array scrapes -- the file is ours and flat, so this stays honest without
// dragging a parser into the tool build.
static std::string slurp(const std::string& p) {
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
static double jnum(const std::string& s, const std::string& key, size_t from = 0) {
    size_t k = s.find("\"" + key + "\"", from);
    if (k == std::string::npos)
        return NAN;
    k = s.find(':', k);
    return atof(s.c_str() + k + 1);
}
static std::vector<int> jints(const std::string& s, const std::string& key) {
    std::vector<int> out;
    size_t k = s.find("\"" + key + "\"");
    if (k == std::string::npos)
        return out;
    size_t a = s.find('[', k), b = s.find(']', a);
    std::string body = s.substr(a + 1, b - a - 1);
    for (size_t i = 0; i < body.size();) {
        while (i < body.size() && !(isdigit(body[i]) || body[i] == '-'))
            ++i;
        if (i >= body.size())
            break;
        out.push_back(atoi(body.c_str() + i));
        while (i < body.size() && (isdigit(body[i]) || body[i] == '-'))
            ++i;
    }
    return out;
}

int main(int argc, char** argv) {
    const std::string pre = argc > 1 ? argv[1] : "/tmp/n2sky";
    const int rec_idx = argc > 2 ? atoi(argv[2]) : 0;

    const std::string js = slurp(pre + ".json");
    if (js.empty()) {
        printf("cannot read %s.json\n", pre.c_str());
        return 1;
    }
    const int n_hops_frame = (int)jnum(js, "n_hops");
    const int n_chan = (int)jnum(js, "n_chan");
    const int n_elem = (int)jnum(js, "n_elem");
    const int fft_len = (int)jnum(js, "fft_len");
    const int hops_per_record = (int)jnum(js, "hops_per_record");
    const long long hop0_frame = (long long)jnum(js, "hop0");
    const double f_off = jnum(js, "f_offset_hz");
    const double fs = jnum(js, "sample_rate");
    const double dll_spacing = jnum(js, "dll_spacing");
    const bool conj = js.find("\"conjugate\": true") != std::string::npos;
    const std::vector<int> chan_ids = jints(js, "channel_ids");
    std::string signame = "GPS_L5_Q_NH";
    {
        size_t k = js.find("\"signal\"");
        if (k != std::string::npos) {
            size_t a = js.find('"', js.find(':', k)) + 1, b = js.find('"', a);
            signame = js.substr(a, b - a);
        }
    }

    std::vector<uint8_t> volt;
    {
        std::ifstream f(pre + ".bin", std::ios::binary);
        volt.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    const size_t want = (size_t)n_hops_frame * n_chan * n_elem;
    if (volt.size() != want) {
        printf("voltage size %zu != expected %zu\n", volt.size(), want);
        return 1;
    }
    printf("n2skyab: %d hops x %d chan x %d elem, hop0 %lld, signal %s, conjugate %s\n",
           n_hops_frame, n_chan, n_elem, hop0_frame, signame.c_str(), conj ? "ON" : "off");

    // ---- satellites -----------------------------------------------------------------------
    struct Sat {
        int prn;
        double cp, dop;
        long long ref_hop;
    };
    std::vector<Sat> sats;
    for (size_t k = js.find("\"prn\""); k != std::string::npos; k = js.find("\"prn\"", k + 1)) {
        Sat s;
        s.prn = (int)jnum(js, "prn", k - 1);
        s.cp = jnum(js, "cp_chips", k);
        s.dop = jnum(js, "doppler_hz", k);
        s.ref_hop = (long long)jnum(js, "ref_hop", k);
        if (!std::isnan(s.cp) && !std::isnan(s.dop))
            sats.push_back(s);
    }
    if (sats.empty()) {
        printf("no sats in json\n");
        return 1;
    }

    const long long wstart_hop = hop0_frame + (long long)rec_idx * hops_per_record;
    const long long wstart = wstart_hop * (long long)fft_len;
    printf("  record %d: hops [%lld, %lld), %zu sats\n\n", rec_idx, wstart_hop,
           wstart_hop + hops_per_record, sats.size());

    // ---- path A machinery (the shipped despread) --------------------------------------------
    const auto* sig = gnss::signal_by_name(signame.c_str());
    if (!sig) {
        printf("unknown signal %s\n", signame.c_str());
        return 1;
    }
    std::vector<int> prns;
    for (auto& s : sats)
        prns.push_back(s.prn);
    gnss::ChannelizedReplicaBank bank(*sig, fs, f_off, fft_len / 2, 4, dsp::Window::Hamming, prns);
    GnssCudaDespread despr(bank, (int)sats.size(), chan_ids, hops_per_record, fs, f_off);

    std::vector<int> covering(n_chan);
    for (int c = 0; c < n_chan; c++)
        covering[c] = c;

    std::vector<GnssCudaDespread::Spec> specs;
    for (size_t i = 0; i < sats.size(); i++) {
        gnss::SeedState ss;
        ss.cp_chips = sats[i].cp;
        ss.phase_ref_chips = -1.0;
        ss.doppler_hz = sats[i].dop;
        ss.dop_rate = 0.0;
        ss.cp_rate = 0.0;
        ss.ref_hop = sats[i].ref_hop;
        const gnss::SeedPropagation pr =
            gnss::propagate_seed(bank, ss, wstart_hop, fs, f_off, 0.0);
        GnssCudaDespread::Spec sp;
        sp.p = (int)i;
        sp.doppler_hz = pr.doppler_hz;
        sp.cp_seed = pr.cp;
        sp.spacing_chips = dll_spacing;
        sp.covering = covering;
        specs.push_back(sp);
    }

    // ---- device: the record's voltage, native layout ---------------------------------------
    const size_t rec_bytes = (size_t)hops_per_record * n_chan * n_elem;
    const uint8_t* h_rec = volt.data() + (size_t)rec_idx * rec_bytes;
    unsigned char* d_frame;
    CK(cudaMalloc(&d_frame, rec_bytes));
    CK(cudaMemcpy(d_frame, h_rec, rec_bytes, cudaMemcpyHostToDevice));

    const int n_spec = (int)specs.size();
    const size_t wave_n = (size_t)3 * n_spec * n_chan * hops_per_record;
    float2* d_wave;
    double2* d_corr;
    double* d_energy;
    gnss_cuda::DespreadJob* d_jobs;
    float* d_scale;
    int* d_cids;
    CK(cudaMalloc(&d_wave, wave_n * sizeof(float2)));
    CK(cudaMalloc(&d_corr, (size_t)4 * n_spec * n_chan * n_elem * sizeof(double2)));
    CK(cudaMalloc(&d_energy, (size_t)4 * n_spec * n_chan * sizeof(double)));
    CK(cudaMalloc(&d_jobs, n_spec * sizeof(gnss_cuda::DespreadJob)));
    CK(cudaMalloc(&d_scale, n_chan * sizeof(float)));
    CK(cudaMalloc(&d_cids, n_chan * sizeof(int)));
    std::vector<float> h_scale(n_chan, 1.0f);
    CK(cudaMemcpy(d_scale, h_scale.data(), n_chan * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_cids, covering.data(), n_chan * 4, cudaMemcpyHostToDevice));

    // PATH A, exactly as the tracker runs it (conjugate flag included).
    despr.enqueue_batch_nm(d_frame, d_scale, d_cids, d_wave, n_elem, n_elem, n_chan, wstart,
                           specs, d_jobs, d_corr, d_energy, nullptr, conj);
    CK(cudaDeviceSynchronize());
    std::vector<double2> corrA((size_t)4 * n_spec * n_chan * n_elem);
    CK(cudaMemcpy(corrA.data(), d_corr, corrA.size() * sizeof(double2), cudaMemcpyDeviceToHost));
    auto ca = [&](int spec, int row, int c, int e) {
        const double2& v = corrA[((size_t)(4 * spec + row) * n_chan + c) * n_elem + e];
        return cd(v.x, v.y);
    };

    // ---- PATH B: pack the SAME replicas into 4+4b lanes, run the dual correlator ------------
    constexpr int NSA = 128, NSB = 128, NF = 8; // padded; dual_kernel_256_8
    const int NS = NSA + NSB;
    // A input: the same voltage, elements 0..31 live, rest 0x88; channel axis padded to NF.
    std::vector<uint8_t> hA((size_t)hops_per_record * NF * NSA, 0x88);
    for (int m = 0; m < hops_per_record; m++)
        for (int c = 0; c < n_chan; c++)
            memcpy(&hA[((size_t)m * NF + c) * NSA], &h_rec[((size_t)m * n_chan + c) * n_elem],
                   n_elem);
    int8_t *dA, *dB;
    CK(cudaMalloc(&dA, hA.size()));
    CK(cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dB, (size_t)hops_per_record * NF * NSB));
    CK(cudaMemset(dB, 0x88, (size_t)hops_per_record * NF * NSB));

    // slot2spec / chan_map for the production pack kernel -- the SAME launch_pack44 the
    // injector uses, so the quantization cannot fork from what runs on the node.
    std::vector<int> h_s2s(n_spec), h_cmap(n_chan);
    for (int i = 0; i < n_spec; i++)
        h_s2s[i] = i;
    for (int c = 0; c < n_chan; c++)
        h_cmap[c] = c;
    int *d_s2s, *d_cmap;
    CK(cudaMalloc(&d_s2s, n_spec * sizeof(int)));
    CK(cudaMalloc(&d_cmap, n_chan * sizeof(int)));
    CK(cudaMemcpy(d_s2s, h_s2s.data(), n_spec * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_cmap, h_cmap.data(), n_chan * 4, cudaMemcpyHostToDevice));
    CK(gnss_cuda::launch_pack44(d_wave, d_energy, d_jobs, d_s2s, n_spec, n_chan, hops_per_record,
                                d_cmap, NF, NSB, conj, (unsigned char*)dB, nullptr));
    CK(cudaDeviceSynchronize());

    auto ntiles = [](int ns) { return (ns / 16) * (ns / 16 + 1) / 2; };
    const size_t nvis = (size_t)NF * ntiles(NS) * 512;
    int* d_vis;
    CK(cudaMalloc(&d_vis, nvis * 4));
    CK(cudaMemset(d_vis, 0, nvis * 4));
    std::vector<uint32_t> h_rm((size_t)NF * (hops_per_record / 32), 0xffffffffu);
    uint* d_rm;
    CK(cudaMalloc(&d_rm, h_rm.size() * 4));
    CK(cudaMemcpy(d_rm, h_rm.data(), h_rm.size() * 4, cudaMemcpyHostToDevice));

    n2k_dual::DualCorrelator dual(NSA, NSB, NF);
    dual.launch(d_vis, dA, dB, d_rm, 1, hops_per_record, nullptr, true);
    std::vector<int> vis(nvis);
    CK(cudaMemcpy(vis.data(), d_vis, nvis * 4, cudaMemcpyDeviceToHost));

    auto vb = [&](int lane, int c, int e) {
        int gi = NSA + lane, ihi = gi >> 4, ilo = gi & 15, jhi = e >> 4, jlo = e & 15;
        size_t o = (size_t)c * ntiles(NS) * 512 + 512 * ((size_t)ihi * (ihi + 1) / 2 + jhi)
                   + 32 * ilo + 2 * jlo;
        return cd(vis[o], vis[o + 1]);
    };
    auto m2 = [&](int lane, int c) {
        int gi = NSA + lane, ihi = gi >> 4, ilo = gi & 15;
        size_t o = (size_t)c * ntiles(NS) * 512 + 512 * ((size_t)ihi * (ihi + 1) / 2 + ihi)
                   + 32 * ilo + 2 * ilo;
        return (double)vis[o];
    };

    // ---- compare, per element ---------------------------------------------------------------
    // With conj_replica the orientation is path A == V (NOT conj V) -- measured in n2dualxval.
    // Normalize path B by the M^2 diagonal (the quantized replica's own energy) and path A by
    // its replica energy, so both are amplitudes and the pack's scale cancels.
    std::vector<double> h_en((size_t)4 * n_spec * n_chan);
    CK(cudaMemcpy(h_en.data(), d_energy, h_en.size() * sizeof(double), cudaMemcpyDeviceToHost));

    // SCALE-FREE COMPARISON. V/M^2 and corr/energy differ by the pack's scale 1/s (V ~ s,
    // M^2 ~ s^2), and s varies per (lane, channel) -- so a raw ratio measures the quantizer,
    // not the agreement. Instead compare the 32-ELEMENT VECTORS: if path B reproduces the
    // despread, B_e = k * A_e for a single complex k per (lane, channel), so the normalized
    // complex correlation |<B,A>|/(|B||A|) is 1 whatever k is. This also does NOT require the
    // seed to be right -- identical inputs must agree even on noise.
    printf("%-5s %-4s %9s %10s %10s %9s\n", "PRN", "row", "|corr|", "med|k|", "dB(k)",
           "phase_mr");
    const char* rn[3] = {"E", "P", "L"};
    for (int i = 0; i < n_spec; i++) {
        for (int t = 0; t < 3; t++) {
            std::vector<double> cors, mags, phs;
            for (int c = 0; c < n_chan; c++) {
                const double eA = h_en[(size_t)(4 * i + t) * n_chan + c];
                const double eB = m2(4 * i + t, c);
                if (eA <= 0 || eB <= 0)
                    continue;
                cd num(0, 0);
                double na = 0, nb = 0;
                for (int e = 0; e < n_elem; e++) {
                    const cd A = ca(i, t, c, e), B = vb(4 * i + t, c, e);
                    num += B * std::conj(A);
                    na += std::norm(A);
                    nb += std::norm(B);
                }
                if (na > 0 && nb > 0) {
                    cors.push_back(std::abs(num) / std::sqrt(na * nb));
                    const cd k = num / na; // least-squares B = k A
                    mags.push_back(std::abs(k));
                    phs.push_back(std::arg(k) * 1e3);
                }
            }
            if (cors.empty())
                continue;
            std::sort(cors.begin(), cors.end());
            std::sort(mags.begin(), mags.end());
            std::sort(phs.begin(), phs.end());
            printf("%-5d %-4s %9.4f %10.4f %+10.4f %9.1f\n", sats[i].prn, rn[t],
                   cors[cors.size() / 2], mags[mags.size() / 2],
                   20.0 * log10(mags[mags.size() / 2]), phs[phs.size() / 2]);
        }
    }
    printf("\n|corr| over the %d-element vector, median across %d channels.\n"
           "  |corr| ~ 1.000 => path B reproduces the shipped despread on real sky (k is the\n"
           "  pack's scale and is expected to vary); |corr| ~ 1/sqrt(%d) = %.3f => unrelated.\n",
           n_elem, n_chan, n_elem, 1.0 / std::sqrt((double)n_elem));
    return 0;
}
