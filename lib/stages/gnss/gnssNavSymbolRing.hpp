/**
 * @file
 * @brief Streamed data symbols of one satellite, keyed by absolute symbol index, for the
 *        replica's per-period data sign (PsiJob::nav).
 *  - NavSymbolRing
 */
#ifndef GNSS_NAV_SYMBOL_RING_HPP
#define GNSS_NAV_SYMBOL_RING_HPP

#include <algorithm> // for fill
#include <cstddef>   // for size_t
#include <cstdint>   // for int64_t, int8_t, uint32_t
#include <vector>    // for vector

namespace gnss {

/**
 * @class NavSymbolRing
 * @brief The last N symbols received for one PRN of one signal.
 *
 * A symbol's index is its transmit time on the satellite's clock divided by the symbol period,
 * counted from the GPS epoch in GPS time (BeiDou: BDT + 14 s), so every receiver labels a
 * symbol alike. Symbols may arrive out of order or twice. A new stream_id drops what is held:
 * the sender restarts its demodulator there, and a restarted stream's sign need not match.
 */
class NavSymbolRing {
public:
    static constexpr int N = 4096;

    /// Hold the @p n signs (+-1) of @p signs, the first at index @p sym0, from @p stream_id.
    void put(int64_t sym0, int n, const int8_t* signs, uint32_t stream_id) {
        if (have_stream && stream_id != stream)
            clear();
        stream = stream_id;
        have_stream = true;
        for (int k = 0; k < n; ++k) {
            const size_t s = slot(sym0 + k);
            tag[s] = sym0 + k;
            sign[s] = signs[k];
        }
    }

    /// The signs of indices [first, first + n) into @p out. False if any is not held, and then
    /// @p out is undefined.
    bool get(int64_t first, int n, int8_t* out) const {
        for (int k = 0; k < n; ++k) {
            const size_t s = slot(first + k);
            if (tag[s] != first + k)
                return false;
            out[k] = sign[s];
        }
        return true;
    }

    void clear() {
        std::fill(tag.begin(), tag.end(), -1);
        have_stream = false;
    }

private:
    static size_t slot(int64_t i) {
        return (size_t)(((i % N) + N) % N);
    }
    std::vector<int64_t> tag = std::vector<int64_t>(N, -1);
    std::vector<int8_t> sign = std::vector<int8_t>(N, 0);
    uint32_t stream = 0;
    bool have_stream = false;
};

} // namespace gnss

#endif
