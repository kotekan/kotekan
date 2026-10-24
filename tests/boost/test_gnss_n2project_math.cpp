#define BOOST_TEST_MODULE "test_gnss_n2project_math"
#include "gnssN2ProjectMath.hpp" // for project, project_cross, extract, writeback, Layout
#include "gnssProjSubspace.hpp"  // for ProjBasis

#include <boost/test/included/unit_test.hpp>
#include <complex> // for complex, conj, norm
#include <cstdint> // for int32_t
#include <random>  // for mt19937, normal_distribution
#include <vector>
using cd = std::complex<double>;

namespace {
std::mt19937 rng(7);
std::normal_distribution<double> g(0.0, 1.0);

// Random Hermitian positive definite N x N: A A^H + I.
std::vector<cd> hermitian(int N) {
    std::vector<cd> A((size_t)N * N), V((size_t)N * N);
    for (auto& a : A)
        a = cd(g(rng), g(rng));
    for (int i = 0; i < N; ++i)
        for (int c = 0; c < N; ++c) {
            cd s(i == c ? 1.0 : 0.0, 0.0);
            for (int l = 0; l < N; ++l)
                s += A[(size_t)i * N + l] * std::conj(A[(size_t)c * N + l]);
            V[(size_t)i * N + c] = s;
        }
    return V;
}

// k orthonormal directions over n stations, row-wise q[j * n + i].
gnss::ProjBasis basis(int n, int k) {
    gnss::ProjBasis B;
    B.init(n, k);
    std::vector<cd> v(n);
    while (B.k < k) {
        for (auto& x : v)
            x = cd(g(rng), g(rng));
        B.add(v.data(), 0.01);
    }
    return B;
}

// Brute force M V M^H, M = I - Q Q^H on the first n of N inputs and the identity on the rest.
std::vector<cd> reference(int N, int n, int k, const std::vector<cd>& q, const std::vector<cd>& V) {
    std::vector<cd> Mm((size_t)N * N, cd(0, 0)), T((size_t)N * N), R((size_t)N * N);
    for (int i = 0; i < N; ++i)
        for (int c = 0; c < N; ++c) {
            cd s(i == c ? 1.0 : 0.0, 0.0);
            if (i < n && c < n)
                for (int j = 0; j < k; ++j)
                    s -= q[(size_t)j * n + i] * std::conj(q[(size_t)j * n + c]);
            Mm[(size_t)i * N + c] = s;
        }
    for (int i = 0; i < N; ++i)
        for (int c = 0; c < N; ++c) {
            cd s(0, 0);
            for (int l = 0; l < N; ++l)
                s += Mm[(size_t)i * N + l] * V[(size_t)l * N + c];
            T[(size_t)i * N + c] = s;
        }
    for (int i = 0; i < N; ++i)
        for (int c = 0; c < N; ++c) {
            cd s(0, 0);
            for (int l = 0; l < N; ++l)
                s += T[(size_t)i * N + l] * std::conj(Mm[(size_t)c * N + l]);
            R[(size_t)i * N + c] = s;
        }
    return R;
}
} // namespace

BOOST_AUTO_TEST_CASE(project_matches_brute_force) {
    for (int n : {47, 96, 127})
        for (int k : {1, 8, 33}) {
            const std::vector<cd> V0 = hermitian(n);
            const gnss::ProjBasis B = basis(n, k);
            const std::vector<cd> R = reference(n, n, k, B.q, V0);
            std::vector<cd> V = V0, W, M;
            gnss_n2proj::project(n, k, B.q.data(), V.data(), W, M);
            double err = 0.0, ref = 0.0, herm = 0.0;
            for (int i = 0; i < n; ++i)
                for (int c = 0; c < n; ++c) {
                    err += std::norm(V[(size_t)i * n + c] - R[(size_t)i * n + c]);
                    ref += std::norm(R[(size_t)i * n + c]);
                    herm += std::norm(V[(size_t)i * n + c] - std::conj(V[(size_t)c * n + i]));
                }
            BOOST_TEST_INFO("n " << n << " k " << k);
            BOOST_CHECK_LT(std::sqrt(err / ref), 1e-10);
            BOOST_CHECK_LT(std::sqrt(herm / ref), 1e-10);
        }
}

// The frame written back (station block and the cross products to the other inputs) must be the
// projection of the station voltages alone, read through the n2k layout both ways round.
BOOST_AUTO_TEST_CASE(frame_is_station_voltage_projection) {
    gnss_n2proj::Layout L;
    const int N = L.n_elem;
    std::vector<int> st, ot;
    for (int i = 0; i < N; ++i)
        ((i == 7 || (i >= 48 && i < 64) || i >= 112) ? ot : st).push_back(i);
    const int n = (int)st.size(), n_o = (int)ot.size(), k = 12;
    const std::vector<cd> V0 = hermitian(N);
    // integer frame of 1000 V0, lower triangle in the n2k layout
    const double sc = 1000.0;
    std::vector<int32_t> fr(L.per_freq(), 0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j <= i; ++j) {
            const cd v = sc * V0[(size_t)i * N + j];
            fr[L.idx(i, j)] = gnss_n2proj::to_i32(v.real());
            fr[L.idx(i, j) + 1] = i == j ? 0 : gnss_n2proj::to_i32(v.imag());
        }
    const gnss::ProjBasis B = basis(n, k);
    std::vector<cd> V((size_t)n * n), X((size_t)n * n_o), W, M, Y;
    gnss_n2proj::extract(L, fr.data(), st, V.data());
    gnss_n2proj::extract_cross(L, fr.data(), st, ot, X.data());
    gnss_n2proj::project(n, k, B.q.data(), V.data(), W, M);
    gnss_n2proj::project_cross(n, n_o, k, B.q.data(), X.data(), Y);
    gnss_n2proj::writeback(L, fr.data(), st, V.data());
    gnss_n2proj::writeback_cross(L, fr.data(), st, ot, X.data());
    // reference on the reordered inputs: stations first, then the others
    std::vector<int> ord(st);
    ord.insert(ord.end(), ot.begin(), ot.end());
    std::vector<cd> Vr((size_t)N * N);
    for (int a = 0; a < N; ++a)
        for (int b = 0; b < N; ++b)
            Vr[(size_t)a * N + b] = V0[(size_t)ord[a] * N + ord[b]];
    const std::vector<cd> R = reference(N, n, k, B.q, Vr);
    double err = 0.0, ref = 0.0;
    for (int a = 0; a < N; ++a)
        for (int b = 0; b < N; ++b) {
            const int i = ord[a], j = ord[b];
            const bool lower = i >= j;
            const size_t x = lower ? L.idx(i, j) : L.idx(j, i);
            cd v((double)fr[x], (double)fr[x + 1]);
            if (!lower)
                v = std::conj(v);
            err += std::norm(v / sc - R[(size_t)a * N + b]);
            ref += std::norm(R[(size_t)a * N + b]);
        }
    // int32 rounding at scale 1000 bounds the agreement
    BOOST_CHECK_LT(std::sqrt(err / ref), 1e-4);
}
