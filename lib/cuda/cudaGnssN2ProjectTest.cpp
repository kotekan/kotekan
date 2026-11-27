// Standalone GPU check of the cudaGnssN2Project kernels against GnssN2Project's CPU path
// (gnssN2ProjectMath.hpp: extract, project, writeback, extract_cross, project_cross,
// writeback_cross) on synthetic n2k frames:
//  - every word of every projected channel equal (FP64, the record's complex<float> basis);
//  - the channels outside the slots, a slot with k 0, the other inputs' own products and the
//    upper halves of the diagonal blocks untouched;
//  - a station with no lane component, k from 0 to k_slot, k_slot 40 and 64 (the update's
//    shared memory above 48 KB);
// then the time of one frame at production shape, and, for reference, how far the CPU path
// with the float basis lies from the same with the double basis it was rounded from.
// Exits 0 when every check passes.   ./cuda_gnss_n2_project_test [device]

#include "cudaGnssN2ProjectKernel.hpp"
#include "gnssN2ProjectMath.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x)                                                                                      \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__,          \
                    __LINE__);                                                                     \
            exit(2);                                                                               \
        }                                                                                          \
    } while (0)

using cd = std::complex<double>;
using gnss_cuda::N2P_INPUTS;
using gnss_cuda::N2P_WORDS;

namespace {

int fails = 0;

void check(bool ok, const char* what) {
    printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok)
        ++fails;
}

struct Case {
    int n_freq, k_slot;
    std::vector<int> stations, slots, k; ///< k per slot
    std::vector<int32_t> frame;          ///< [n_freq][N2P_WORDS]
    std::vector<uint8_t> rec;            ///< the basis record
    std::vector<std::vector<cd>> q_dbl;  ///< per slot, the basis before rounding [k][n]
};

// A Hermitian frame: noise, three bright rank-1 sources and real autos (the correlator's are: a
// nonzero imaginary part on the diagonal makes the block non-Hermitian, which both paths assume
// it is not); poison in the upper halves of the diagonal blocks.
void make_frame(Case& c, std::mt19937& rng) {
    std::normal_distribution<double> g(0.0, 1.0);
    gnss_n2proj::Layout L;
    L.n_elem = N2P_INPUTS;
    c.frame.assign((size_t)c.n_freq * N2P_WORDS, 0x7f7f7f7f);
    for (int f = 0; f < c.n_freq; ++f) {
        int32_t* fr = &c.frame[(size_t)f * N2P_WORDS];
        std::vector<cd> a(3 * N2P_INPUTS);
        for (cd& x : a)
            x = cd(g(rng), g(rng));
        for (int i = 0; i < N2P_INPUTS; ++i)
            for (int j = 0; j <= i; ++j) {
                cd v =
                    i == j ? cd(4.0e6 + 1.0e5 * g(rng), 0.0) : cd(3.0e3 * g(rng), 3.0e3 * g(rng));
                for (int s = 0; s < 3; ++s)
                    v += 1.0e6 * (s + 1) * a[s * N2P_INPUTS + i] * std::conj(a[s * N2P_INPUTS + j]);
                if (i == j)
                    v.imag(0.0);
                const size_t w = L.idx(i, j);
                fr[w] = (int32_t)std::lround(v.real());
                fr[w + 1] = (int32_t)std::lround(v.imag());
            }
    }
}

// The record: per slot k orthonormal directions on the stations axis (one station left out of
// every direction), stored as complex<float> and kept in double for the reference.
void make_record(Case& c, std::mt19937& rng) {
    std::normal_distribution<double> g(0.0, 1.0);
    const int n = (int)c.stations.size(), S = (int)c.slots.size();
    const size_t head = gnss_n2proj::basis_head_bytes(n);
    const size_t slot = gnss_n2proj::basis_slot_bytes(c.k_slot, n);
    c.rec.assign(head + S * slot, 0);
    const int32_t h[7] = {1, n, 1, S, c.k_slot, (int32_t)slot, gnss_n2proj::BASIS_MAGIC};
    std::memcpy(c.rec.data() + 8, h, sizeof(h));
    std::memcpy(c.rec.data() + 64, c.stations.data(), 4 * (size_t)n);
    const int dark = n / 3; // no lane component on this station
    c.q_dbl.assign(S, {});
    for (int s = 0; s < S; ++s) {
        const int k = c.k[s];
        std::vector<cd>& q = c.q_dbl[s];
        q.assign((size_t)k * n, cd(0.0, 0.0));
        for (int j = 0; j < k; ++j) {
            cd* r = &q[(size_t)j * n];
            for (int i = 0; i < n; ++i)
                r[i] = i == dark ? cd(0.0, 0.0) : cd(g(rng), g(rng));
            for (int pass = 0; pass < 2; ++pass)
                for (int l = 0; l < j; ++l) {
                    cd p(0.0, 0.0);
                    for (int i = 0; i < n; ++i)
                        p += std::conj(q[(size_t)l * n + i]) * r[i];
                    for (int i = 0; i < n; ++i)
                        r[i] -= q[(size_t)l * n + i] * p;
                }
            double e = 0.0;
            for (int i = 0; i < n; ++i)
                e += std::norm(r[i]);
            for (int i = 0; i < n; ++i)
                r[i] /= std::sqrt(e);
        }
        uint8_t* p = c.rec.data() + head + s * slot;
        const int32_t sh[2] = {1000 + c.slots[s], k};
        std::memcpy(p, sh, sizeof(sh));
        auto* qf = (std::complex<float>*)(p + 16);
        for (size_t x = 0; x < q.size(); ++x)
            qf[x] = std::complex<float>(q[x]);
    }
}

// GnssN2Project's CPU path on slot s's channel of @p fr, with basis @p q ([k][n]).
void cpu_project(const Case& c, int32_t* fr, int k, const cd* q) {
    if (k <= 0)
        return;
    gnss_n2proj::Layout L;
    L.n_elem = N2P_INPUTS;
    const int n = (int)c.stations.size();
    std::vector<int> others;
    for (int i = 0; i < N2P_INPUTS; ++i)
        if (!std::binary_search(c.stations.begin(), c.stations.end(), i))
            others.push_back(i);
    const int n_o = (int)others.size();
    std::vector<cd> V((size_t)n * n), W, M, X((size_t)n * n_o), Y;
    const std::vector<int32_t> fin(fr, fr + N2P_WORDS);
    gnss_n2proj::extract(L, fin.data(), c.stations, V.data());
    gnss_n2proj::project(n, k, q, V.data(), W, M);
    gnss_n2proj::writeback(L, fr, c.stations, V.data());
    gnss_n2proj::extract_cross(L, fin.data(), c.stations, others, X.data());
    gnss_n2proj::project_cross(n, n_o, k, q, X.data(), Y);
    gnss_n2proj::writeback_cross(L, fr, c.stations, others, X.data());
}

std::vector<cd> float_basis(const Case& c, int s) {
    const int n = (int)c.stations.size();
    const auto* qf =
        (const std::complex<float>*)(c.rec.data() + gnss_n2proj::basis_head_bytes(n)
                                     + s * gnss_n2proj::basis_slot_bytes(c.k_slot, n) + 16);
    std::vector<cd> q((size_t)c.k[s] * n);
    for (size_t x = 0; x < q.size(); ++x)
        q[x] = cd(qf[x]);
    return q;
}

struct Dev {
    int32_t* frame = nullptr;
    int* chan = nullptr;
    uint8_t* rec = nullptr;
    double2 *v = nullptr, *q = nullptr, *w = nullptr, *m = nullptr;
    int32_t* live = nullptr;
    cublasHandle_t blas = nullptr;
    cudaStream_t stream = nullptr;
};

void dev_alloc(Dev& d, const Case& c) {
    const int S = (int)c.slots.size();
    CK(cudaMalloc(&d.frame, c.frame.size() * 4));
    CK(cudaMalloc(&d.chan, S * sizeof(int)));
    CK(cudaMalloc(&d.rec, c.rec.size()));
    CK(cudaMalloc(&d.v, (size_t)S * N2P_INPUTS * N2P_INPUTS * sizeof(double2)));
    CK(cudaMalloc(&d.q, (size_t)S * N2P_INPUTS * c.k_slot * sizeof(double2)));
    CK(cudaMalloc(&d.w, (size_t)S * N2P_INPUTS * c.k_slot * sizeof(double2)));
    CK(cudaMalloc(&d.m, (size_t)S * c.k_slot * c.k_slot * sizeof(double2)));
    CK(cudaMalloc(&d.live, 2 * (size_t)S * N2P_WORDS * 4));
    CK(cudaMemcpy(d.chan, c.slots.data(), S * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.rec, c.rec.data(), c.rec.size(), cudaMemcpyHostToDevice));
    CK(cudaStreamCreate(&d.stream));
    if (cublasCreate(&d.blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(d.blas, d.stream))
        exit(2);
}

void dev_free(Dev& d) {
    cublasDestroy(d.blas);
    cudaStreamDestroy(d.stream);
    for (void* p : {(void*)d.frame, (void*)d.chan, (void*)d.rec, (void*)d.v, (void*)d.q, (void*)d.w,
                    (void*)d.m, (void*)d.live})
        cudaFree(p);
}

// The command's sequence: gather raw, project, gather projected.
void dev_run(Dev& d, const Case& c) {
    const int n = (int)c.stations.size(), S = (int)c.slots.size();
    const int k_max = *std::max_element(c.k.begin(), c.k.end());
    gnss_cuda::launch_n2proj_gather(d.frame, d.chan, S, d.live, d.stream);
    if (k_max > 0) {
        gnss_cuda::launch_n2proj_dense(d.frame, d.chan, d.rec, n, c.k_slot, S, c.k_slot, d.v, d.q,
                                       d.stream);
        if (gnss_cuda::n2proj_wz(d.blas, S, k_max, c.k_slot, d.v, d.q, d.w, d.m)
            != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cuBLAS failed\n");
            exit(2);
        }
        gnss_cuda::launch_n2proj_update(d.frame, d.chan, d.rec, n, c.k_slot, S, k_max, c.k_slot,
                                        d.q, d.w, d.stream);
    }
    gnss_cuda::launch_n2proj_gather(d.frame, d.chan, S, d.live + (size_t)S * N2P_WORDS, d.stream);
    CK(cudaGetLastError());
}

void run_case(const char* name, Case& c, std::mt19937& rng) {
    printf("%s: %d channels, %d slots, %d stations, k_slot %d\n", name, c.n_freq,
           (int)c.slots.size(), (int)c.stations.size(), c.k_slot);
    make_frame(c, rng);
    make_record(c, rng);
    const int S = (int)c.slots.size();
    std::vector<int32_t> ref = c.frame;
    for (int s = 0; s < S; ++s) {
        const std::vector<cd> q = float_basis(c, s);
        cpu_project(c, &ref[(size_t)c.slots[s] * N2P_WORDS], c.k[s], q.data());
    }
    Dev d;
    dev_alloc(d, c);
    CK(cudaMemcpy(d.frame, c.frame.data(), c.frame.size() * 4, cudaMemcpyHostToDevice));
    dev_run(d, c);
    std::vector<int32_t> got(c.frame.size()), live(2 * (size_t)S * N2P_WORDS);
    CK(cudaMemcpy(got.data(), d.frame, got.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(live.data(), d.live, live.size() * 4, cudaMemcpyDeviceToHost));
    dev_free(d);

    long diff = 0, maxd = 0, slot_words = 0, changed = 0;
    for (size_t x = 0; x < got.size(); ++x) {
        const long dd = std::labs((long)got[x] - (long)ref[x]);
        diff += dd != 0;
        maxd = std::max(maxd, dd);
    }
    for (int s = 0; s < S; ++s)
        for (int x = 0; x < N2P_WORDS; ++x) {
            const size_t w = (size_t)c.slots[s] * N2P_WORDS + x;
            slot_words += 1;
            changed += got[w] != c.frame[w];
        }
    char msg[256];
    snprintf(msg, sizeof msg,
             "every word of the frame equals the CPU path (%ld of %zu differ, max |d| %ld; %ld of "
             "the slots' %ld words projected)",
             diff, got.size(), maxd, changed, slot_words);
    check(diff == 0, msg);
    bool raw_ok = true, proj_ok = true;
    for (int s = 0; s < S; ++s) {
        const int32_t* a = &c.frame[(size_t)c.slots[s] * N2P_WORDS];
        const int32_t* b = &ref[(size_t)c.slots[s] * N2P_WORDS];
        raw_ok &= std::memcmp(&live[(size_t)s * N2P_WORDS], a, N2P_WORDS * 4) == 0;
        proj_ok &= std::memcmp(&live[(size_t)(S + s) * N2P_WORDS], b, N2P_WORDS * 4) == 0;
    }
    check(raw_ok && proj_ok, "the live blocks: the raw channel, then the projected one");

    // The float basis against the double it was rounded from, on the CPU path (for reference:
    // what moving the CPU path to the recorded basis changes).
    std::vector<int32_t> dbl = c.frame;
    for (int s = 0; s < S; ++s)
        cpu_project(c, &dbl[(size_t)c.slots[s] * N2P_WORDS], c.k[s], c.q_dbl[s].data());
    long fd = 0, fmax = 0;
    for (size_t x = 0; x < dbl.size(); ++x) {
        const long dd = std::labs((long)dbl[x] - (long)ref[x]);
        fd += dd != 0;
        fmax = std::max(fmax, dd);
    }
    printf("  CPU path, float basis against double: %ld of %ld projected words differ, max |d| "
           "%ld (values up to ~1e7)\n",
           fd, changed, fmax);
}

Case base_case(int n_freq, int k_slot, const std::vector<int>& slots, const std::vector<int>& k) {
    Case c;
    c.n_freq = n_freq;
    c.k_slot = k_slot;
    for (int i = 0; i < N2P_INPUTS; ++i)
        if (i != 7 && i % 13 != 4 && i < 122) // 111 stations; others inside and at the end
            c.stations.push_back(i);
    c.slots = slots;
    c.k = k;
    return c;
}

} // namespace

int main(int argc, char** argv) {
    CK(cudaSetDevice(argc > 1 ? atoi(argv[1]) : 0));
    std::mt19937 rng(20261127);

    Case a = base_case(24, 40, {1, 2, 5, 9, 14, 20, 23}, {27, 0, 40, 1, 33, 12, 40});
    run_case("k_slot 40, k 0..40", a, rng);
    Case b = base_case(12, 64, {0, 3, 4, 11}, {64, 50, 2, 17});
    run_case("k_slot 64", b, rng);

    // Production shape: 464 channels, 43 slots, 109 stations, 40 directions each.
    std::vector<int> slots, k;
    for (int s = 0; s < 43; ++s) {
        slots.push_back(3 + 10 * s);
        k.push_back(40);
    }
    Case p = base_case(464, 40, slots, k);
    printf("timing: %d channels, %d slots, k %d\n", p.n_freq, (int)p.slots.size(), p.k_slot);
    make_frame(p, rng);
    make_record(p, rng);
    Dev d;
    dev_alloc(d, p);
    CK(cudaMemcpy(d.frame, p.frame.data(), p.frame.size() * 4, cudaMemcpyHostToDevice));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    float total = 0.0f;
    const int reps = 50;
    for (int r = 0; r < reps + 5; ++r) {
        CK(cudaEventRecord(e0, d.stream));
        dev_run(d, p);
        CK(cudaEventRecord(e1, d.stream));
        CK(cudaEventSynchronize(e1));
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        if (r >= 5)
            total += ms;
    }
    dev_free(d);
    printf("  gather, project and gather: %.3f ms per frame (k 40 on every slot)\n", total / reps);

    printf(fails ? "\n%d FAILED\n" : "\nall passed\n", fails);
    return fails ? 1 : 0;
}
