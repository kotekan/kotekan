#ifndef GNSS_CLOCK_HPP
#define GNSS_CLOCK_HPP

#include <chrono>

namespace gnss {

/// Test only: when set, steady_now_s() returns this instead of the steady clock. Set from test
/// code before any stage is built, never from a config key; null in production.
inline double (*steady_now_hook)() = nullptr;

/// The steady clock in seconds, the clock the record assembler's freshness and cadence decisions
/// use.
inline double steady_now_s() {
    return steady_now_hook
               ? steady_now_hook()
               : std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
                     .count();
}

} // namespace gnss

#endif // GNSS_CLOCK_HPP
