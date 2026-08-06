/**
 * @file
 * @brief M1/M2 gate for n2k_dual (the two-input N^2 clone). GPU required.
 *
 * What is proven, in order:
 *
 *   [1] REFERENCE     n2k(128,NF) == CPU reference           (validates the reference itself)
 *   [2] CLONE         dual(128+0)  == n2k(128) bitwise       (NSB=0 regression gate)
 *   [3] PREFIX        dual(128+128) first-36-tiles == n2k(128) bitwise, per (t,f)
 *                     (the "verbatim prefix" property the path-B split relies on)
 *   [4] DUAL          dual(128+128) == CPU reference over the concatenated 256 stations
 *                     (covers AA + mixed + BB tiles, with a random RFI mask)
 *   [5] CLASS MASK    dual(mask=MIXED|BB) matches the full run on mixed+BB tiles bitwise
 *
 * All outputs are int32 -- every comparison is EXACT equality. Comparisons cover only the
 * defined entries: stored tiles (ihi >= jhi) and, within them, global i >= j (the kernel
 * writes deterministic garbage above the diagonal of diagonal tiles; not part of the
 * contract, not compared).
 *
 * Geometry: NF=8, nt_outer=2, nt_inner=1024 (T=2048, exercising touter and both rfimask
 * coarse-time blocks). Data: random 4+4 offset-encoded, values in [-7,7] (never -8).
 */

#include <n2k/Correlator.hpp>
#include <n2k_dual/DualCorrelator.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x)                                                                                     \
    do {                                                                                          \
        cudaError_t e_ = (x);                                                                     \
        if (e_ != cudaSuccess) {                                                                  \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);       \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

static constexpr int NF = 8;
static constexpr int NT_OUTER = 2;
static constexpr int NT_INNER = 1024;
static constexpr int T = NT_OUTER * NT_INNER;
static constexpr int NSA = 128;
static constexpr int NSB = 128;
static constexpr int NS = NSA + NSB;

static int ntiles(int ns) {
    int m = ns / 16;
    return m * (m + 1) / 2;
}

// RFI mask bit for (f, t), decoding the RING layout exactly as the kernel's access_rfimask
// and gpuSimulateN2kCorr do: uint index (t%1024)/32 + f*32 + (t/1024)*32*NF, bit t%32.
static bool mask_bit(const std::vector<uint32_t>& rm, int f, int t) {
    int idx = ((t & 1023) >> 5) + f * 32 + (t >> 10) * 32 * NF;
    return (rm[idx] >> (t & 31)) & 1;
}

// CPU reference over 'ns' stations: data is int8[T][NF][ns] offset-encoded 4+4
// (imag low nibble, real high nibble). Output layout as n2k: per (tout, f), tiles
// ihi >= jhi at int32 offset 512*(ihi(ihi+1)/2 + jhi) + 32*ilo + 2*jlo, [re, im].
static std::vector<int> cpu_reference(const std::vector<uint8_t>& data,
                                      const std::vector<uint32_t>& rm, int ns) {
    const int nt_tiles = ntiles(ns);
    const size_t fstride = (size_t)nt_tiles * 512;
    std::vector<int> out((size_t)NT_OUTER * NF * fstride, 0);

    // Precompute decoded values, masked to 0 (matching the kernel: masked samples are
    // zeroed after offset removal, so they contribute nothing).
    std::vector<int8_t> re((size_t)T * NF * ns), im((size_t)T * NF * ns);
#pragma omp parallel for
    for (int t = 0; t < T; t++)
        for (int f = 0; f < NF; f++) {
            bool ok = mask_bit(rm, f, t);
            const uint8_t* row = &data[((size_t)t * NF + f) * ns];
            int8_t* rr = &re[((size_t)t * NF + f) * ns];
            int8_t* ri = &im[((size_t)t * NF + f) * ns];
            for (int s = 0; s < ns; s++) {
                rr[s] = ok ? (int8_t)((row[s] >> 4) - 8) : 0;
                ri[s] = ok ? (int8_t)((row[s] & 0xf) - 8) : 0;
            }
        }

#pragma omp parallel for collapse(2)
    for (int tout = 0; tout < NT_OUTER; tout++)
        for (int f = 0; f < NF; f++) {
            int* vp = &out[((size_t)tout * NF + f) * fstride];
            for (int i = 0; i < ns; i++)
                for (int j = 0; j <= i; j++) {
                    long acc_re = 0, acc_im = 0;
                    for (int tin = 0; tin < NT_INNER; tin++) {
                        int t = tout * NT_INNER + tin;
                        size_t base = ((size_t)t * NF + f) * ns;
                        int xr = re[base + i], xi = im[base + i];
                        int yr = re[base + j], yi = im[base + j];
                        acc_re += xr * yr + xi * yi; // V_ij = E_i * conj(E_j)
                        acc_im += xi * yr - xr * yi;
                    }
                    int ihi = i >> 4, ilo = i & 15, jhi = j >> 4, jlo = j & 15;
                    size_t o = 512 * ((size_t)ihi * (ihi + 1) / 2 + jhi) + 32 * ilo + 2 * jlo;
                    vp[o + 0] = (int)acc_re;
                    vp[o + 1] = (int)acc_im;
                }
        }
    return out;
}

struct CmpStats {
    long checked = 0, bad = 0;
    int first_bad_printed = 0;
};

// Compare defined entries of two vis buffers with per-buffer station counts and a tile
// predicate (global tile coords of the LARGER-or-equal geometry; both buffers must contain
// the tile). ns_x/ns_y give each buffer's own fstride; tiles are addressed by (ihi, jhi).
static void cmp_tiles(CmpStats& st, const char* label, const int* X, int ns_x, const int* Y,
                      int ns_y, bool (*want)(int ihi, int jhi)) {
    const size_t fx = (size_t)ntiles(ns_x) * 512, fy = (size_t)ntiles(ns_y) * 512;
    const int mtx = ns_x / 16, mty = ns_y / 16;
    const int mt = mtx < mty ? mtx : mty;
    for (int tout = 0; tout < NT_OUTER; tout++)
        for (int f = 0; f < NF; f++) {
            const int* vx = X + ((size_t)tout * NF + f) * fx;
            const int* vy = Y + ((size_t)tout * NF + f) * fy;
            for (int ihi = 0; ihi < mt; ihi++)
                for (int jhi = 0; jhi <= ihi; jhi++) {
                    if (!want(ihi, jhi))
                        continue;
                    size_t to = 512 * ((size_t)ihi * (ihi + 1) / 2 + jhi);
                    for (int ilo = 0; ilo < 16; ilo++)
                        for (int jlo = 0; jlo < 16; jlo++) {
                            int gi = 16 * ihi + ilo, gj = 16 * jhi + jlo;
                            if (gi < gj)
                                continue; // above-diagonal entries of diagonal tiles: garbage
                            size_t o = to + 32 * ilo + 2 * jlo;
                            st.checked += 2;
                            for (int c = 0; c < 2; c++) {
                                if (vx[o + c] != vy[o + c]) {
                                    st.bad++;
                                    if (st.first_bad_printed < 5) {
                                        printf("  %s MISMATCH tout=%d f=%d i=%d j=%d %s: %d vs "
                                               "%d\n",
                                               label, tout, f, gi, gj, c ? "im" : "re", vx[o + c],
                                               vy[o + c]);
                                        st.first_bad_printed++;
                                    }
                                }
                            }
                        }
                }
        }
}

static bool tiles_all(int, int) {
    return true;
}
static bool tiles_aa(int ihi, int jhi) {
    return ihi < NSA / 16 && jhi < NSA / 16;
}
static bool tiles_mixed_bb(int ihi, int jhi) {
    return !tiles_aa(ihi, jhi);
}

static int report(const char* name, const CmpStats& st) {
    printf("[%s] %-42s : %ld entries, %ld mismatches\n", st.bad ? "FAIL" : " OK ", name,
           st.checked, st.bad);
    return st.bad ? 1 : 0;
}

int main(int argc, char** argv) {
    const unsigned seed = argc > 1 ? (unsigned)atoi(argv[1]) : 12345u;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> nib(1, 15); // offset-encoded [-7,7]; NEVER 0 (= -8)
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    printf("n2dualtest: NF=%d nt_outer=%d nt_inner=%d NSA=%d NSB=%d seed=%u\n", NF, NT_OUTER,
           NT_INNER, NSA, NSB, seed);

    // ---- host data -----------------------------------------------------------------------
    std::vector<uint8_t> ha((size_t)T * NF * NSA), hb((size_t)T * NF * NSB);
    for (auto& v : ha)
        v = (uint8_t)((nib(rng) << 4) | nib(rng));
    for (auto& v : hb)
        v = (uint8_t)((nib(rng) << 4) | nib(rng));

    // Concatenated copy, for the CPU reference of the dual run.
    std::vector<uint8_t> hab((size_t)T * NF * NS);
    for (int t = 0; t < T; t++)
        for (int f = 0; f < NF; f++) {
            memcpy(&hab[((size_t)t * NF + f) * NS], &ha[((size_t)t * NF + f) * NSA], NSA);
            memcpy(&hab[((size_t)t * NF + f) * NS + NSA], &hb[((size_t)t * NF + f) * NSB], NSB);
        }

    // RFI mask, ring layout, ~15% of (f, 32-sample words) masked.
    std::vector<uint32_t> hrm((size_t)(T / 1024) * NF * 32);
    for (auto& w : hrm)
        w = (uni(rng) < 0.15) ? (uint32_t)rng() : 0xffffffffu;

    // ---- device buffers ------------------------------------------------------------------
    auto up = [&](const void* p, size_t n) {
        void* d;
        CK(cudaMalloc(&d, n));
        CK(cudaMemcpy(d, p, n, cudaMemcpyHostToDevice));
        return d;
    };
    int8_t* da = (int8_t*)up(ha.data(), ha.size());
    int8_t* db = (int8_t*)up(hb.data(), hb.size());
    uint* drm = (uint*)up(hrm.data(), hrm.size() * 4);

    const size_t nvis_a = (size_t)NT_OUTER * NF * ntiles(NSA) * 512;
    const size_t nvis_d = (size_t)NT_OUTER * NF * ntiles(NS) * 512;
    int *va, *vd0, *vd, *vm;
    CK(cudaMalloc(&va, nvis_a * 4));
    CK(cudaMalloc(&vd0, nvis_a * 4));
    CK(cudaMalloc(&vd, nvis_d * 4));
    CK(cudaMalloc(&vm, nvis_d * 4));
    CK(cudaMemset(va, 0xa5, nvis_a * 4)); // poison: unwritten entries differ unless written
    CK(cudaMemset(vd0, 0xa5, nvis_a * 4));
    CK(cudaMemset(vd, 0xa5, nvis_d * 4));
    CK(cudaMemset(vm, 0xa5, nvis_d * 4));

    // ---- runs ----------------------------------------------------------------------------
    n2k::Correlator ref(NSA, NF);
    ref.launch(va, da, drm, NT_OUTER, NT_INNER, nullptr, true);

    n2k_dual::DualCorrelator d0(NSA, 0, NF);
    d0.launch(vd0, da, nullptr, drm, NT_OUTER, NT_INNER, nullptr, true);

    n2k_dual::DualCorrelator dual(NSA, NSB, NF);
    dual.launch(vd, da, db, drm, NT_OUTER, NT_INNER, nullptr, true);

    n2k_dual::DualCorrelatorParams pm(NSA, NSB, NF,
                                      n2k_dual::BLOCK_MASK_MIXED | n2k_dual::BLOCK_MASK_BB);
    n2k_dual::DualCorrelator dmask(pm);
    dmask.launch(vm, da, db, drm, NT_OUTER, NT_INNER, nullptr, true);

    std::vector<int> hva(nvis_a), hvd0(nvis_a), hvd(nvis_d), hvm(nvis_d);
    CK(cudaMemcpy(hva.data(), va, nvis_a * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hvd0.data(), vd0, nvis_a * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hvd.data(), vd, nvis_d * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hvm.data(), vm, nvis_d * 4, cudaMemcpyDeviceToHost));

    // ---- references ----------------------------------------------------------------------
    std::vector<int> ref_a = cpu_reference(ha, hrm, NSA);
    std::vector<int> ref_ab = cpu_reference(hab, hrm, NS);

    // ---- comparisons ---------------------------------------------------------------------
    int fails = 0;
    {
        CmpStats st;
        cmp_tiles(st, "ref", hva.data(), NSA, ref_a.data(), NSA, tiles_all);
        fails += report("[1] n2k(128) vs CPU reference", st);
    }
    {
        CmpStats st;
        cmp_tiles(st, "clone", hvd0.data(), NSA, hva.data(), NSA, tiles_all);
        fails += report("[2] dual(128+0) vs n2k(128), bitwise", st);
    }
    {
        CmpStats st;
        cmp_tiles(st, "prefix", hvd.data(), NS, hva.data(), NSA, tiles_aa);
        fails += report("[3] dual(128+128) AA prefix vs n2k(128)", st);
    }
    {
        CmpStats st;
        cmp_tiles(st, "dual", hvd.data(), NS, ref_ab.data(), NS, tiles_all);
        fails += report("[4] dual(128+128) vs CPU reference (256)", st);
    }
    {
        CmpStats st;
        cmp_tiles(st, "mask", hvm.data(), NS, hvd.data(), NS, tiles_mixed_bb);
        fails += report("[5] class-mask MIXED|BB vs full, bitwise", st);
    }

    printf(fails ? "*** %d comparison(s) FAILED\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
