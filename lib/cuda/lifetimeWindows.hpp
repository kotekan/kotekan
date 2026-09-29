#ifndef LIFETIME_WINDOWS_HPP
#define LIFETIME_WINDOWS_HPP

#include <cassert> // for assert
#include <cstdint> // for int64_t

namespace kotekan {

/**
 * @class LifetimeWindows
 * @brief Choose upchannelizer windows that end exactly on the lifetime boundaries of a slowly
 *        varying input, such as the gains.
 *
 * A window reads `n` granules of `granularity` input samples and, because of the PFB overlap,
 * produces `n * g - o` output samples, where `g = granularity / U` and `o = overlap / U` (i.e.
 * `M - 1` for `M` taps). The overlap is re-read by the next window. A window may use only one
 * element of the slowly varying input, so the windows have to tile each lifetime exactly: the
 * last window before a boundary must end on it.
 *
 * Not every window size is possible (`p = n * g - o` is always `-o` modulo `g`), so a window
 * must also leave a remainder that whole windows can still fill. A remainder `x > 0` can be
 * filled by `m` windows exactly when
 *
 *     x + m * o == N * g   with   m * n_min <= N <= m * n_max,
 *
 * i.e. when `x + m * o` is a multiple of `g` and `m * p_min <= x <= m * p_max`. (`N` granules
 * split into `m` windows of `n_min...n_max` granules each whenever that inequality holds.)
 * `reachable` evaluates this in constant time, and `num_granules` picks the largest window
 * that fits the available data and leaves a reachable remainder. That choice does not depend
 * on any history, so it is the same for every instance of a command, and a short (starved)
 * window never paints the schedule into a corner: the remainder after any window it returns
 * is reachable again.
 *
 * With enough data the largest window is the stateless closed form
 *
 *     m = smallest m >= 1 with (r + m * o) % g == 0 and r <= m * p_max
 *     p = min(p_max, r - (m - 1) * p_min),
 *
 * i.e. the windows reach the boundary in the smallest possible number of steps. The boost test
 * `test_lifetimeWindows` checks both formulations against a brute-force search.
 *
 * All quantities are integers; input quantities are in input samples, and the remaining
 * distance to a boundary is in output samples. The class knows nothing about Kotekan, so the
 * generated upchannelizer wrappers and `gpuSimulateCudaUpchannelizer` can share it, and must:
 * the two have to claim the same data.
 */
class LifetimeWindows {
    std::int64_t g;     // output samples per granule
    std::int64_t o;     // overlap in output samples
    std::int64_t n_min; // smallest number of granules that produces output
    std::int64_t n_max; // largest number of granules per window
    std::int64_t q;     // g / gcd(o, g), the period of the congruence in `m`
    std::int64_t d;     // gcd(o, g)
    std::int64_t o_inv; // inverse of o / d modulo q

    static std::int64_t gcd(std::int64_t a, std::int64_t b) {
        while (b != 0) {
            const std::int64_t t = a % b;
            a = b;
            b = t;
        }
        return a;
    }
    // Remainder in [0, y) for y > 0
    static std::int64_t pmod(const std::int64_t x, const std::int64_t y) {
        const std::int64_t r = x % y;
        return r < 0 ? r + y : r;
    }

public:
    /// Whether these parameters describe a usable upchannelizer: whole output samples per
    /// granule and per overlap, and at least one window size that produces output.
    static constexpr bool valid(const std::int64_t granularity, const std::int64_t overlap,
                                const std::int64_t upchan_factor, const std::int64_t max_granules) {
        if (!(granularity > 0 && overlap >= 0 && upchan_factor > 0 && max_granules > 0))
            return false;
        if (granularity % upchan_factor != 0 || overlap % upchan_factor != 0)
            return false;
        // The largest window must produce at least one output sample
        return max_granules * granularity > overlap;
    }

    /// @param granularity    Input samples per granule; a window reads a whole number of them
    /// @param overlap        Input samples each window shares with the next, `(M - 1) * U`
    /// @param upchan_factor  Upchannelization factor `U`
    /// @param max_granules   Largest number of granules one window may read
    LifetimeWindows(const std::int64_t granularity, const std::int64_t overlap,
                    const std::int64_t upchan_factor, const std::int64_t max_granules) :
        g(granularity / upchan_factor), o(overlap / upchan_factor), n_min(o / g + 1),
        n_max(max_granules), q(0), d(0), o_inv(0) {
        assert(valid(granularity, overlap, upchan_factor, max_granules));
        assert(n_min * g - o > 0 && (n_min - 1) * g - o <= 0);
        assert(n_min <= n_max);
        d = gcd(o, g); // gcd(0, g) == g
        q = g / d;
        // `o / d` and `q` are coprime, so the inverse exists; `q` is at most a few hundred
        for (o_inv = 0; o_inv < q; ++o_inv)
            if (pmod((o / d) * o_inv, q) == pmod(1, q))
                break;
        assert(o_inv < q);
    }

    std::int64_t min_granules() const {
        return n_min;
    }
    std::int64_t max_granules() const {
        return n_max;
    }
    /// Output samples produced by a window of `n` granules
    std::int64_t produced(const std::int64_t n) const {
        return n * g - o;
    }
    std::int64_t min_produced() const {
        return produced(n_min);
    }
    std::int64_t max_produced() const {
        return produced(n_max);
    }

    /// The smallest number of windows that produce exactly `x > 0` output samples, or 0 if no
    /// number of windows does.
    std::int64_t min_num_windows(const std::int64_t x) const {
        assert(x > 0);
        // `x + m * o` can only be a multiple of `g` if `d` divides `x`
        if (x % d != 0)
            return 0;
        // All solutions of `(x + m * o) % g == 0`: `m == m0 (mod q)`
        const std::int64_t m0 = pmod(-(x / d) * o_inv, q);
        // The fewest windows that can hold `x`, rounded up to a solution
        const std::int64_t m_lo = (x + max_produced() - 1) / max_produced();
        const std::int64_t m = m_lo + pmod(m0 - m_lo, q);
        // ... and they must not be too many to be filled
        if (m * min_produced() > x)
            return 0;
        return m;
    }

    /// Can whole windows produce exactly `x >= 0` output samples?
    bool reachable(const std::int64_t x) const {
        if (x < 0)
            return false;
        if (x == 0)
            return true;
        return min_num_windows(x) > 0;
    }

    /// The number of granules to read next.
    ///
    /// @param remaining           Output samples left until the next lifetime boundary; must
    ///                            be positive and `reachable`
    /// @param available_granules  How many granules could be read right now, already capped by
    ///                            the caller's own read limit
    /// @returns The largest `n <= min(available_granules, max_granules())` whose window fits
    ///          before the boundary and leaves a `reachable` remainder, or 0 if there is none,
    ///          in which case the caller has to wait for more data. It is never 0 when
    ///          `available_granules >= max_granules()`.
    std::int64_t num_granules(const std::int64_t remaining,
                              const std::int64_t available_granules) const {
        assert(remaining > 0);
        assert(reachable(remaining));
        std::int64_t n = available_granules < n_max ? available_granules : n_max;
        // A window must not produce more than `remaining`
        const std::int64_t n_fit = (remaining + o) / g;
        if (n_fit < n)
            n = n_fit;
        for (; n >= n_min; --n)
            if (reachable(remaining - produced(n)))
                return n;
        return 0;
    }
};

} // namespace kotekan

#endif // #ifndef LIFETIME_WINDOWS_HPP
