#ifndef DIV_HPP
#define DIV_HPP

#include <cassert>

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

// Calculate `x div y`
template<typename T, typename U>
auto div_noremainder(T x, U y) {
    assert(x >= 0);
    assert(y > 0);
    assert(x % y == 0);
    auto r = x / y;
    return r;
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
