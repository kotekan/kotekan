// CHORD upchannelization kernel

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <utility>
#include <vector>

constexpr int C = 2;     // number of complex components
constexpr int T = 32768; // number of times
constexpr int D = 1;     // TODO 512;   // number of dishes
constexpr int P = 1;     // TODO 2;     // number of polarizations
constexpr int F = 1;     // TODO 16;    // frequency channels per GPU
constexpr int U = 16;    // upchannelization factor
constexpr int M = 4;     // number of taps

// 4-bit integers

using int4x2_t = uint8_t;

constexpr int4x2_t set4(const int8_t lo, const int8_t hi) {
  return (uint8_t(lo) & 0x0f) | ((uint8_t(hi) << 4) & 0xf0);
}
constexpr int4x2_t set4(const std::array<int8_t, 2> a) {
  return set4(a[0], a[1]);
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

// 16-bit floats

using float16_t = _Float16;

// Storage management

template <typename T, typename I> constexpr T convert(const I i) {
  return T(i);
}
template <typename T, typename I>
constexpr std::complex<T> convert(const std::complex<I> i) {
  return std::complex<T>(convert<I>(i.real()), convert<I>(i.imag()));
}

#if 1
// Use 4-bit integers for E and Ebar

using storage_t = int4x2_t;
using value_t = int8_t;

constexpr float maxabserr = 0.8f;

constexpr storage_t set_storage(const int8_t lo, const int8_t hi) {
  return set4(lo, hi);
}
constexpr storage_t set_storage(const std::array<int8_t, 2> x) {
  return set_storage(x[0], x[1]);
}
constexpr std::array<int8_t, 2> get_storage(const storage_t x) {
  return get4(x);
}

template <typename I, typename T>
constexpr I quantize(const T x, const I imax) {
  using std::floor;
  const I itmp = I(floor(x + T(0.5)));
  using std::max, std::min;
  const I i = min(imax, max(I(-imax), itmp));
  return i;
}

#else
// Use 32-bit floats for E and Ebar

using storage_t = std::complex<float>;
using value_t = float;

constexpr float maxabserr = 0.0f;

constexpr storage_t set_storage(const float lo, const float hi) {
  return {lo, hi};
}
constexpr storage_t set_storage(const std::array<float, 2> x) {
  return set_storage(x[0], x[1]);
}
constexpr std::array<float, 2> get_storage(const storage_t x) {
  return {real(x), imag(x)};
}

template <typename I, typename T>
constexpr I quantize(const T x, const I imax) {
  return I(x);
}

#endif

template <typename I, typename T>
constexpr std::complex<I> quantize(const std::complex<T> x, const I imax) {
  return std::complex<I>(quantize<I>(x.real(), imax),
                         quantize<I>(x.imag(), imax));
}

// complex numbers

template <typename T>
constexpr std::complex<T> to_complex(const std::array<T, 2> a) {
  return std::complex<T>(a[0], a[1]);
}
template <typename T>
constexpr std::array<T, 2> to_array(const std::complex<T> c) {
  return std::array<T, 2>{c.real(), c.imag()};
}

// functions

template <typename T>
constexpr T linterp(const T x1, const T y1, const T x2, const T y2, const T x) {
  return (x - x2) * y1 / (x1 - x2) + (x - x1) * y2 / (x2 - x1);
}
static_assert(linterp(1.0f, 2.0f, 3.0f, 4.0f, 1.0f) == 2.0f);
static_assert(linterp(1.0f, 2.0f, 3.0f, 4.0f, 3.0f) == 4.0f);
static_assert(linterp(1.0f, 2.0f, 3.0f, 4.0f, 2.0f) == 3.0f);

template <typename T, typename U, std::size_t N>
constexpr U interp(const std::array<std::pair<T, U>, N> &table, const T x) {
  static_assert(N > 0);
  assert(x >= table.front().first);
  assert(x <= table.back().first);
  for (std::size_t n = 0; n < table.size() - 1; ++n)
    if (x <= table[n + 1].first)
      return linterp(table[n].first, table[n].second, table[n + 1].first,
                     table[n + 1].second, x);
  assert(false);
}
namespace {
constexpr std::array<std::pair<float, float>, 3> table{{
    {1.0f, +1.0f},
    {2.0f, -1.0f},
    {3.0f, +3.0f},
}};
static_assert(interp(table, 1.0f) == +1.0f);
static_assert(interp(table, 1.5f) == +0.0f);
static_assert(interp(table, 2.0f) == -1.0f);
static_assert(interp(table, 2.5f) == +1.0f);
static_assert(interp(table, 3.0f) == +3.0f);
} // namespace

template <typename T> constexpr T sinc(const T x) {
  using std::abs;
  assert(x == T(0) || abs(x) > T(1.0e-10));
  return x == T(0) ? T(1) : sin(x) / x;
}

// array indexing

constexpr int Eidx(int c, int d, int f, int p, int t) {
  assert(c >= 0 && c < C);
  assert(d >= 0 && d < D);
  assert(f >= 0 && f < F);
  assert(p >= 0 && p < P);
  assert(t >= 0 && t < T + M * U - 1);
  return d + D * f + D * F * p + D * F * P * t;
}

constexpr int Ebaridx(int c, int d, int fbar, int p, int tbar) {
  assert(c >= 0 && c < C);
  assert(d >= 0 && d < D);
  assert(fbar >= 0 && fbar < F * U);
  assert(p >= 0 && p < P);
  assert(tbar >= 0 && tbar < T / U);
  return d + D * fbar + D * F * U * p + D * F * U * P * tbar;
}

// kernel

void upchan_simple(const float16_t *__restrict__ const W,
                   const float16_t *__restrict__ const G,
                   const storage_t *__restrict__ const E,
                   storage_t *__restrict__ const Ebar) {
#pragma omp parallel for collapse(5)
  for (int f = 0; f < F; ++f) {
    for (int p = 0; p < P; ++p) {
      for (int d = 0; d < D; ++d) {
        for (int u = 0; u < U; ++u) {
          for (int tbar = 0; tbar < T / U; ++tbar) {

            const int fbar = u + U * f;

            std::complex<float> Ebar1 = 0.0f;

            for (int s = 0; s < M * U; ++s) {
#error"look into the past instead"
              const int t = s + U * tbar;

              const float W1 = W[s];

              using std::polar;
              const std::complex<float> phase =
                  polar(1.0f, -2 * float(M_PI) * (u - (U - 1) / 2.0f) / U * s);

              const std::complex<float> E1 = convert<float>(
                  to_complex(get_storage(E[Eidx(0, d, f, p, t)])));

              Ebar1 += W1 * phase * E1;

            } // s

            const float G1 = G[u];
            Ebar1 *= G1;
            Ebar[Ebaridx(0, d, fbar, p, tbar)] =
                set_storage(to_array(quantize<value_t>(Ebar1, 7)));

          } // tbar
        }   // u
      }     // d
    }       // p
  }         // f
}

// driver

void driver(const float amp, const int bin, const float delta) {
  std::cout << "Initializing input...\n";
  std::cout << "  amp=" << amp << " bin=" << bin << " delta=" << delta << "\n";

  std::vector<float16_t> W(M * U); // PFB weight function
  std::vector<float16_t> G(U);     // output gains
  std::vector<storage_t> E(C * D * F * P * (T + M * U - 1) /
                           2); // electric field
  std::vector<storage_t> Ebar(C * D * F * P * T /
                              2); // upchannelized electric field

  // Set up window function
  using std::cos, std::pow, std::sin;
  float sumW = 0;
  for (int s = 0; s < M * U; ++s) {
    // sinc-Hanning window function, eqn. (11), with `N=U`
    W.at(s) =
        pow(cos(float(M_PI) * (s - (M * U - 1) / 2.0f) / (M * U + 1)), 2) *
        sinc((s - (M * U - 1) / 2.0f) / U);
    sumW += W.at(s);
    // std::cout << "  s=" << s << " W=" << float(W.at(s)) << "\n";
  }
  // Normalize the window function
  for (int s = 0; s < M * U; ++s)
    W.at(s) /= sumW;

  // Input gains
  for (int u = 0; u < U; ++u)
    G.at(u) = 1.0f;

  // Test input
  // constexpr float amp = 7.5f;
  // constexpr int bin = 0;         // [0, U-1]
  // constexpr float delta = +0.5f; // [-1/2, +1/2]
  const float freq = bin - (U - 1) / 2.0f + delta;
  // These are measured attenuation factors, including the float16 quantization
  // of W, for M=4 and U=16.
  //     delta     att
  //     0         1.00007
  //     0.0001    1.00007
  //     0.001     1.00005
  //     0.01      0.999116
  //     0.1       0.910357
  //     0.2       0.680212
  //     0.3       0.402912
  //     0.4       0.172467
  //     0.5       0.0374226
  //     1.0       0.000714811
  constexpr std::array<std::pair<float, float>, 11> attenuation_factors{{
      {0, 1.00007},
      {0.0001, 1.00007},
      {0.001, 1.00005},
      {0.01, 0.999116},
      {0.1, 0.910357},
      {0.2, 0.680212},
      {0.3, 0.402912},
      {0.4, 0.172467},
      {0.5, 0.0374226},
      {1.0, 0.000714811},
      {2.0, 0}, // not measured, down in the noise
  }};
  using std::abs;
  const float att = interp(attenuation_factors, abs(delta));

  for (int t = 0; t < T + M * U - 1; ++t) {
    for (int p = 0; p < P; ++p) {
      for (int f = 0; f < F; ++f) {
        for (int d = 0; d < D; ++d) {
          using std::fmod, std::polar;
          const std::complex<float> E1 =
              polar(amp, 2 * float(M_PI) * fmod(1.0f * t / U * freq, 1.0f));
          E.at(Eidx(0, d, f, p, t)) =
              set_storage(to_array(quantize<value_t>(E1, 7)));
        }
      }
    }
  }

  // Poison output
  for (int tbar = 0; tbar < T / U; ++tbar)
    for (int p = 0; p < P; ++p)
      for (int fbar = 0; fbar < F * U; ++fbar)
        for (int d = 0; d < D; ++d)
          Ebar.at(Ebaridx(0, d, fbar, p, tbar)) = set_storage(-8, -8);

  std::cout << "Calling kernel...\n";
  upchan_simple(W.data(), G.data(), E.data(), Ebar.data());

  std::cout << "Checking output...\n";
  int num_errors = 0;
  for (int fbar = 0; fbar < F * U; ++fbar) {
    for (int p = 0; p < P; ++p) {
      for (int d = 0; d < D; ++d) {
        float I = 0;
        for (int tbar = 0; tbar < T / U; ++tbar) {
          const std::complex<float> Ebar1 = convert<float>(
              to_complex(get_storage(Ebar.at(Ebaridx(0, d, fbar, p, tbar)))));
          using std::fmod, std::polar;
          const std::complex<float> Ebar1_wanted =
              fbar == bin
                  ? polar(att * amp,
                          2 * float(M_PI) *
                              fmod((tbar + M / 2.0f) * (0.5f + delta), 1.0f))
                  : 0;
          using std::abs, std::max;
          const float abserr = abs(Ebar1 - Ebar1_wanted);
          const float relerr =
              abserr / max({abs(att * amp), abs(Ebar1), abs(Ebar1_wanted)});
          float maxrelerr;
          if (delta == 0) {
            // Exact frequency
            if (fbar == bin) {
              maxrelerr = 1.0e-4f;
            } else if (abs(fbar - bin) == 1 || abs(fbar - bin) == U - 1) {
              // Allow a larger error in the neighbouring bin due to aliasing
              maxrelerr = 1.0e-3f;
            } else {
              maxrelerr = 1.0e-4f;
            }
          } else if (abs(delta) < 0.1f) {
            // Close frequencies
            if (fbar == bin) {
              maxrelerr = 1.0e-2f;
            } else if (abs(fbar - bin) == 1 || abs(fbar - bin) == U - 1) {
              // Allow a larger error in the neighbouring bin due to aliasing
              maxrelerr = 1.0e-2f;
            } else {
              maxrelerr = 1.0e-3f;
            }
          } else if (abs(delta) < 0.5f) {
            // Close frequencies
            if (fbar == bin) {
              maxrelerr = 1.0e-1f;
            } else if (abs(fbar - bin) == 1 || abs(fbar - bin) == U - 1) {
              // Allow a larger error in the neighbouring bin due to aliasing
              maxrelerr = 1.0e-1f;
            } else {
              maxrelerr = 1.0e-2f;
            }
          } else {
            // Close frequencies
            if (fbar == bin) {
              maxrelerr = 1.0e-1f;
            } else if (abs(fbar - bin) == 1 || abs(fbar - bin) == U - 1) {
              // Allow a larger error in the neighbouring bin due to aliasing
              maxrelerr = 1.0e-0f;
            } else {
              maxrelerr = 1.0e-2f;
            }
          }
          if (relerr > maxrelerr && abserr > maxabserr) {
            ++num_errors;
            if (num_errors <= 100)
              std::cout << "  d=" << d << " fbar=" << fbar << " p=" << p
                        << " tbar=" << tbar << " Ebar=" << Ebar1
                        << " Ebar_wanted=" << Ebar1_wanted
                        << " diff=" << abs(Ebar1 - Ebar1_wanted)
                        << " abs=" << (abs(Ebar1) - abs(Ebar1_wanted))
                        << " phase=" << (arg(Ebar1) - arg(Ebar1_wanted))
                        << "\n";
          }
          I += norm(Ebar1);
        }
        I /= T / U;
        std::cout << "  d=" << d << " fbar=" << fbar << " p=" << p << " I=" << I
                  << " att=" << sqrt(I) / amp << "\n";
      }
    }
  }
  if (num_errors > 0)
    std::cout << "ERRORS FOUND: " << num_errors << "\n";

  assert(num_errors == 0);
}

int main(int argc, char **argv) {
  std::cout << "CHORD upchannelization kernel\n";

  driver(7.5f, 0, 0.0f);
  driver(7.0f, 1, 0.0001f);
  driver(6.5f, 2, 0.001f);
  driver(6.0f, 3, 0.01f);
  driver(7.4f, 4, 0.1f);
  driver(6.9f, 5, 0.2f);
  driver(6.4f, 6, 0.3f);
  driver(7.3f, 7, 0.4f);
  driver(6.8f, 8, 0.5f);

  std::cout << "Done.\n";
  return 0;
}
