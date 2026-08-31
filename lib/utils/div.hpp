#ifndef DIV_HPP
#define DIV_HPP

#include <cassert>
#include <stdexcept>
#include <type_traits>

// Integer division helpers.
//
// All of these take two independent integer types and do their arithmetic in the
// common type the usual arithmetic conversions would pick, which is also the
// return type. That type is not `T`: for the small integer types it is the
// promoted type, so `round_up(uint8_t(250), uint8_t(8))` returns `int` 256, not
// `uint8_t` 0. The promotion is what keeps the narrow types safe here, so do not
// assign a result back to a narrow type without checking that it fits.
//
// Mixing a signed and an unsigned type is fine as long as the values are
// non-negative, which the preconditions require anyway. `div` and `mod` also
// take a negative `x`, but only when the common type is signed - see there.
//
// Only `div_ceil`, `round_up` and `num_triangle_blocks` reject `y == 0` with an
// exception; the others assert it, so under NDEBUG they divide by zero instead.

namespace kotekan {

namespace div_detail {

// `bool` is integral but never a sensible argument here.
template<typename T>
constexpr bool is_index_type = std::is_integral_v<T> && !std::is_same_v<std::remove_cv_t<T>, bool>;

// The type the arithmetic below happens in. Converting both arguments to it up
// front keeps every comparison same-signed, which the mixed `size_t`/`int` call
// sites would otherwise trip `-Wsign-compare` on.
template<typename T, typename U>
using result_t = decltype(std::declval<T>() / std::declval<U>());

// A bare `x >= 0` precondition is tautologically true when `T` is unsigned, i.e.
// dead code that reads as if it still checks something. (Neither GCC nor clang
// warns about it here - they suppress tautological-comparison diagnostics inside
// templates.) This states the same precondition and applies it only where it can
// actually fail.
template<typename T>
constexpr bool is_nonnegative([[maybe_unused]] T x) {
    if constexpr (std::is_signed_v<T>)
        return x >= 0;
    else
        return true;
}

} // namespace div_detail

// Round down `x` to the next lower multiple of `y`
template<typename T, typename U>
auto round_down(T x, U y) {
    static_assert(div_detail::is_index_type<T> && div_detail::is_index_type<U>,
                  "round_down: arguments must be integers");
    assert(div_detail::is_nonnegative(x));
    assert(y > 0);
    using R = div_detail::result_t<T, U>;
    const R rx = x, ry = y;
    const R r = rx / ry * ry;
    assert(r % ry == 0);
    // `rx - r` rather than `r + ry > rx`: the sum overflows for `x` near the type
    // maximum. `r <= rx` holds, so the difference cannot.
    assert(div_detail::is_nonnegative(r) && r <= rx && rx - r < ry);
    return r;
}

// Calculate `ceil(x / y)`, i.e. the number of size-`y` blocks needed to tile `x`
template<typename T, typename U>
constexpr auto div_ceil(T x, U y) {
    static_assert(div_detail::is_index_type<T> && div_detail::is_index_type<U>,
                  "div_ceil: arguments must be integers");
    if (y == 0)
        throw std::invalid_argument("div_ceil: y must be nonzero");
    assert(div_detail::is_nonnegative(x));
    assert(y > 0);
    using R = div_detail::result_t<T, U>;
    const R rx = x, ry = y;
    // Unlike `(x - 1) / y + 1` this is correct for x == 0 (no unsigned
    // wraparound), and unlike `(x + y - 1) / y` it cannot overflow.
    const R r = rx / ry + (rx % ry != 0 ? 1 : 0);
    // No `r * y >= x` postcondition: that product overflows for `x` near the
    // type maximum even though the division above does not, so the check itself
    // was undefined behaviour (UBSan flags `div_ceil(INT_MAX, 3)`).
    assert((r == 0) == (rx == 0));
    return r;
}

// Round up `x` to the next higher multiple of `y`
template<typename T, typename U>
auto round_up(T x, U y) {
    static_assert(div_detail::is_index_type<T> && div_detail::is_index_type<U>,
                  "round_up: arguments must be integers");
    assert(div_detail::is_nonnegative(x));
    assert(y > 0);
    using R = div_detail::result_t<T, U>;
    const R rx = x, ry = y;
    // Via `div_ceil` rather than `(x + y - 1) / y * y`: that addition is signed
    // overflow (i.e. undefined) for `x` within `y` of the type maximum, and wraps
    // for unsigned types - `round_up(4294967290u, 8u)` returned 0 under NDEBUG.
    const R r = div_ceil(rx, ry) * ry;
    assert(r % ry == 0);
    // `r - rx` rather than the old `r - y < x`, which underflowed whenever
    // `r < y`: that made `round_up(0u, 8u)` abort in every debug build. The
    // multiplication above can still overflow when the rounded-up value is not
    // representable at all, which `r >= rx` catches.
    assert(div_detail::is_nonnegative(r) && r >= rx && r - rx < ry);
    return r;
}

// Calculate `x div y`
template<typename T, typename U>
auto div_noremainder(T x, U y) {
    static_assert(div_detail::is_index_type<T> && div_detail::is_index_type<U>,
                  "div_noremainder: arguments must be integers");
    assert(div_detail::is_nonnegative(x));
    assert(y > 0);
    using R = div_detail::result_t<T, U>;
    const R rx = x, ry = y;
    assert(rx % ry == 0);
    const R r = rx / ry;
    return r;
}

// Number of blocks in a blocked triangular (e.g. visibility) matrix:
// an `n` x `n` matrix tiled by `block` x `block` blocks, keeping only the
// blocks on one side of the diagonal, has nb * (nb + 1) / 2 blocks for
// nb = div_ceil(n, block) blocks per side.
template<typename T, typename U>
constexpr auto num_triangle_blocks(T n, U block) {
    static_assert(div_detail::is_index_type<T> && div_detail::is_index_type<U>,
                  "num_triangle_blocks: arguments must be integers");
    using R = div_detail::result_t<T, U>;
    const R nb = div_ceil(n, block);
    // Exactly one of `nb` and `nb + 1` is even, so halving before multiplying is
    // exact and the product is the final result - there is no intermediate left
    // to overflow. Plain `nb * (nb + 1) / 2` needed twice the bits of `nb`: in
    // 32-bit arithmetic it wrapped from nb == 46341 upwards, where this form
    // stays exact all the way to nb == 65535 (past which the count itself does
    // not fit).
    const R r = nb % 2 == 0 ? nb / 2 * (nb + 1) : nb * ((nb + 1) / 2);
    assert(div_detail::is_nonnegative(r) && r >= nb);
    return r;
}

// Calculate `x div y`, rounding towards negative infinity
template<typename T, typename U>
auto div(T x, U y) {
    static_assert(div_detail::is_index_type<T> && div_detail::is_index_type<U>,
                  "div: arguments must be integers");
    assert(y > 0);
    using R = div_detail::result_t<T, U>;
    // A negative `x` cannot survive the conversion to an unsigned common type,
    // as in `div(ptrdiff_t, size_t)`: `div(-4, 3u)` used to return 1431655763.
    if constexpr (!std::is_signed_v<R>)
        assert(div_detail::is_nonnegative(x));
    const R rx = x, ry = y;
    // Truncated division rounds towards zero; step the quotient down by one when
    // it left a negative remainder. Unlike `(x - y + 1) / y` this does not
    // underflow for `x` near the type minimum.
    const R q = rx / ry;
    const R rem = rx % ry;
    const R r = div_detail::is_nonnegative(rem) ? q : q - 1;
    return r;
}

// Calculate `x mod y`, returning `r` with `0 <= r < y`
template<typename T, typename U>
auto mod(T x, U y) {
    static_assert(div_detail::is_index_type<T> && div_detail::is_index_type<U>,
                  "mod: arguments must be integers");
    assert(y > 0);
    using R = div_detail::result_t<T, U>;
    // See `div`.
    if constexpr (!std::is_signed_v<R>)
        assert(div_detail::is_nonnegative(x));
    const R rx = x, ry = y;
    // `%` yields a remainder with the sign of `x`, so shift a negative one up
    // into `[0, y)`. The old `(x - y + 1) % y` was the floored-division trick
    // misapplied to the remainder: it landed in `(-y, 0]`, so `mod(-2, 3)`
    // returned -1 and `mod(-1, 3)` returned 0.
    R r = rx % ry;
    if (!div_detail::is_nonnegative(r))
        r += ry;
    // No `div(x, y) * y + r == x` cross-check: that product underflows for `x`
    // near the type minimum, which is the defect this file removes elsewhere.
    // `0 <= r < y` is the contract that matters.
    assert(div_detail::is_nonnegative(r) && r < ry);
    return r;
}

} // namespace kotekan

#endif // #ifndef DIV_HPP
