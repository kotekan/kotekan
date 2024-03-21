// CHORD FRB beamformer
// See ../docs/CHORD_FRB_beamformer.pdf in this repository.

#include <array>
#include <cassert>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <sys/time.h>

using namespace std::complex_literals;

using std::norm;
using std::polar;

// Kernel parameters

constexpr int C = 2;  // number of complex components
constexpr int T = 48; // TODO 2064; // number of times
constexpr int M = 4;  // TODO 24;  // number of beams
constexpr int N = 4;  // TODO 24;  // number of beams
constexpr int D = 10; // TODO 512; // number of dishes
constexpr int P = 1;  // TODO 2;   // number of polarizations
constexpr int F = 1;  // TODO 256; // frequency channels per GPU

constexpr int Tds = 40; // time downsampling factor

double gettime() {
  struct timeval tp;
  gettimeofday(&tp, nullptr);
  return tp.tv_sec + tp.tv_usec / 1.0e+6;
}

// Integer divide, rounding up (towards positive infinity)
constexpr int cld(const int x, const int y) { return (x + y - 1) / y; }

// 4-bit integers

using int4x2_t = uint8_t;

constexpr int4x2_t set4(const int8_t lo, const int8_t hi) {
  return (uint8_t(lo) & 0x0f) | ((uint8_t(hi) << 4) & 0xf0);
}

constexpr std::array<int8_t, 2> get4(const int4x2_t i) {
  return {int8_t(int8_t((i + 0x08) & 0x0f) - 0x08),
          int8_t(int8_t(((i >> 4) + 0x08) & 0x0f) - 0x08)};
}

constexpr bool test_get4_set4() {
  for (int hi = -8; hi <= 7; ++hi) {
    for (int lo = -8; lo <= 7; ++lo) {
      if (get4(set4(lo, hi))[0] != lo)
        return false;
      if (get4(set4(lo, hi))[1] != hi)
        return false;
    }
  }
  return true;
}

static_assert(test_get4_set4());

template <typename T> inline std::complex<T> cispi(const T x) {
  return polar(T(1), T(M_PI) * x);
}

using float16_t = _Float16;

// - name: "S"
//   intent: in
//   type: Int32
//   indices: [D]
//   shape: [$(M*N)]
//   strides: [1]
// - name: "W"
//   intent: in
//   type: Float16
//   indices: [C, dishM, dishN, F, P]
//   shape: [$C, $M, $N, $F, $P]
//   strides: [1, $C, $(C*M), $(C*M*N), $(C*M*N*F), $(C*M*N*F*P)]
// - name: "E"
//   intent: in
//   type: Int4
//   indices: [C, D, F, P, T]
//   shape: [$C, $D, $F, $P, $T]
//   strides: [1, $C, $(C*D), $(C*D*F), $(C*D*F*P)]
// - name: "I"
//   intent: out
//   type: Float16
//   indices: [beamP, beamQ, F, Tds]
//   shape: [$(2*M), $(2*N), $(cld(T, Tds)), $F]
//   strides: [1, $(2*M), $(2*M*2*N), $(2*M*2*N*cld(T,Tds))]
void frb_simple(const int32_t *__restrict__ const S,
                const float16_t *__restrict__ const W,
                const int4x2_t *__restrict__ const E,
                float16_t *__restrict__ const I) {
  // Check consistency of `S`
  {
    std::vector<bool> E1(M * N, false);
    for (int d = 0; d < M * N; ++d)
      E1.at(S[d]) = true;
    for (int d = 0; d < M * N; ++d)
      assert(E1[d]);
  }

#pragma omp parallel for
  for (int freq = 0; freq < F; ++freq) {

    float I1[(2 * M) * (2 * N)];
    int tds = 0;
    int t_running = 0;
    for (int q = 0; q < 2 * N; ++q)
      for (int p = 0; p < 2 * M; ++p)
        I1[p + 2 * M * q] = 0;

    auto time0 = gettime();
    for (int time = 0; time < T; ++time) {
      auto time1 = gettime();
      if (time1 >= time0 + 1) {
        time0 = time1;
        std::cout << "  time=" << time << "...\n";
      }

      for (int polr = 0; polr < P; ++polr) {

        // grid the dishes
        std::complex<float> E1[M * N];
        for (int d = 0; d < D; ++d)
          E1[S[d]] = std::complex<float>(
              get4(E[d + D * freq + D * F * polr + D * F * P * time])[1],
              get4(E[d + D * freq + D * F * polr + D * F * P * time])[0]);
        for (int d = D; d < M * N; ++d)
          E1[S[d]] = 0;

        // FT in n direction
        std::complex<float> G[M * (2 * N)];
        for (int m = 0; m < M; ++m) {
          for (int q = 0; q < 2 * N; ++q) {
            std::complex<float> s = 0;
            for (int n = 0; n < N; ++n) {
              const std::complex<float> w(
                  W[0 + C * m + C * M * n + C * M * N * freq +
                    C * M * N * F * polr],
                  W[1 + C * m + C * M * n + C * M * N * freq +
                    C * M * N * F * polr]);
              const std::complex<float> e1 = E1[n * M + m];
              s += w * e1 * cispi(float(2 * n * q) / float(2 * N));
            }
            G[m + M * q] = s;
          }
        }

        // FT in m direction
        std::complex<float> Et[2 * M * 2 * N];
        for (int q = 0; q < 2 * N; ++q) {
          for (int p = 0; p < 2 * M; ++p) {
            std::complex<float> s = 0;
            for (int m = 0; m < M; ++m) {
              const std::complex<float> g = G[m + M * q];
              s += g * cispi(float(2 * m * p) / float(2 * M));
            }
            Et[p + 2 * M * q] = s;
          }
        }

        for (int q = 0; q < 2 * N; ++q)
          for (int p = 0; p < 2 * M; ++p)
            I1[p + 2 * M * q] += norm(Et[p + 2 * M * q]);

      } // for polr

      t_running += 1;
      if (t_running == Tds) {
        for (int q = 0; q < 2 * N; ++q)
          for (int p = 0; p < 2 * M; ++p)
            I[p + 2 * M * q + 2 * M * 2 * N * tds +
              2 * M * 2 * N * cld(T, Tds) * freq] = I1[p + 2 * M * q];
        tds += 1;
        t_running = 0;
        for (int q = 0; q < 2 * N; ++q)
          for (int p = 0; p < 2 * M; ++p)
            I1[p + 2 * M * q] = 0;
      }

    } // for time

    if (t_running != 0) {
      for (int q = 0; q < 2 * N; ++q)
        for (int p = 0; p < 2 * M; ++p)
          I[p + 2 * M * q + 2 * M * 2 * N * tds +
            2 * M * 2 * N * cld(T, Tds) * freq] = I1[p + 2 * M * q];
    }

  } // for freq
}

int main(int argc, char **argv) {
  std::cout << "CHORD FRB beamformer\n";

  std::cout << "Initializing input...\n";
  srand(0);
  const int time = std::int64_t(rand()) * T / (std::int64_t(RAND_MAX) + 1);
  const int dish =
      5; // TODO std::int64_t(rand()) * D / (std::int64_t(RAND_MAX) + 1);
  const int polr = std::int64_t(rand()) * P / (std::int64_t(RAND_MAX) + 1);
  const int freq = std::int64_t(rand()) * F / (std::int64_t(RAND_MAX) + 1);
  std::cout << "  Input: time=" << time << " dish=" << dish << " polr=" << polr
            << " freq=" << freq << "\n";
  std::vector<int32_t> S(M * N);
  std::vector<float16_t> W(C * M * N * F * P);
  std::vector<int4x2_t> E(C * D * F * P * T / 2);
  std::vector<float16_t> I((2 * M) * (2 * N) * cld(T, Tds) * F);
  // Dishes
  for (int d = 0; d < D; ++d)
    S.at(d) = d;
  // Empty dish locations
  for (int d = D; d < M * N; ++d)
    S.at(d) = d;
  // Input gains
  for (int p = 0; p < P; ++p) {
    for (int f = 0; f < F; ++f) {
      for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
          std::complex<float16_t> Wval = 1;
          W.at(0 + C * m + (C * M) * n + (C * M * N) * f +
               (C * M * N * F) * p) = Wval.real();
          W.at(1 + C * m + (C * M) * n + (C * M * N) * f +
               (C * M * N * F) * p) = Wval.imag();
        }
      }
    }
  }
  // Electric field
  for (int t = 0; t < T; ++t) {
    for (int p = 0; p < P; ++p) {
      for (int f = 0; f < F; ++f) {
        for (int d = 0; d < D; ++d) {
          std::complex<int8_t> Eval = 0;
          if (t == time && p == polr && f == freq && d == dish)
            Eval = 1;
          E.at(d + D * f + D * F * p + D * F * P * t) =
              set4(Eval.imag(), Eval.real());
        }
      }
    }
  }

  std::cout << "Calling kernel...\n";
  frb_simple(S.data(), W.data(), E.data(), I.data());

  std::cout << "Checking output...\n";
  int errorcount = 0;
  std::vector<bool> I1(I.size(), false);
  for (int f = 0; f < F; ++f) {
    for (int tds = 0; tds < cld(T, Tds); ++tds) {
      for (int q = 0; q < 2 * N; ++q) {
        for (int p = 0; p < 2 * M; ++p) {
          const int iidx = p + 2 * M * q + 2 * M * 2 * N * tds +
                           2 * M * 2 * N * cld(T, Tds) * f;
          assert(!I1.at(iidx));
          I1.at(iidx) = true;
          const float Ival = I.at(iidx);
          if (Ival != 0) {
            if (errorcount == 100) {
              std::cout << "  (More nonzero outputs omitted)\n";
              goto done;
            }
            std::cout << "  Nonzero output: p=" << p << " q=" << q
                      << " tds=" << tds << " f=" << f << "   I=" << Ival
                      << "\n";
            ++errorcount;
          }
        }
      }
    }
  }
  for (bool i1 : I1)
    assert(i1);
done:;

  std::cout << "Done.\n";
  return 0;
}
