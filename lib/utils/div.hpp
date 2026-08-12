#ifndef DIV_HPP
#define DIV_HPP

#include <cassert>
#include <stdexcept>

namespace kotekan {

// Round down `x` to the next lower multiple of `y`
template<typename T, typename U>
auto round_down(T x, U y) {
    assert(x >= 0);
    assert(y > 0);
    auto r = x / y * y;
    assert(r % y == 0);
    assert(0 <= r && r <= x && r + y > x);
    return r;
}

// Round up `x` to the next higher multiple of `y`
template<typename T, typename U>
auto round_up(T x, U y) {
    assert(x >= 0);
    assert(y > 0);
    auto r = (x + y - 1) / y * y;
    assert(r % y == 0);
    assert(0 <= r && r - y < x && r >= x);
    return r;
}

// Calculate `x div y`
template<typename T, typename U>
auto div_noremainder(T x, U y) {
    assert(x >= 0);
    assert(y > 0);
    assert(x % y == 0);
    auto r = x / y;
    return r;
}

// Calculate `ceil(x / y)`, i.e. the number of size-`y` blocks needed to tile `x`
template<typename T, typename U>
constexpr auto div_ceil(T x, U y) {
    if (y == 0)
        throw std::invalid_argument("div_ceil: y must be nonzero");
    assert(x >= 0);
    assert(y > 0);
    // Unlike `(x - 1) / y + 1` this is correct for x == 0 (no unsigned
    // wraparound), and unlike `(x + y - 1) / y` it cannot overflow.
    auto r = x / y + (x % y != 0 ? 1 : 0);
    assert(r * y >= x && (r == 0 || (r - 1) * y < x));
    return r;
}

// Number of blocks in a blocked triangular (e.g. visibility) matrix:
// an `n` x `n` matrix tiled by `block` x `block` blocks, keeping only the
// blocks on one side of the diagonal, has nb * (nb + 1) / 2 blocks for
// nb = div_ceil(n, block) blocks per side.
template<typename T, typename U>
constexpr auto num_triangle_blocks(T n, U block) {
    auto nb = div_ceil(n, block);
    return nb * (nb + 1) / 2;
}

// Calculate `x div y`
template<typename T, typename U>
auto div(T x, U y) {
    assert(y > 0);
    auto r = (x >= 0 ? x : x - y + 1) / y;
    assert(x < (r + 1) * y && r * y <= x);
    return r;
}

// Calculate `x mod y`, returning `x` with `0 <= x < y`
template<typename T, typename U>
auto mod(T x, U y) {
    assert(y > 0);
    auto r = (x >= 0 ? x : x - y + 1) % y;
    assert(0 <= r && r < y);
    assert(div(x, y) * y + r == x);
    return r;
}

} // namespace kotekan

#endif // #ifndef DIV_HPP
