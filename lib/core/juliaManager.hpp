#ifndef JULIA_MANAGER_HPP
#define JULIA_MANAGER_HPP

#ifndef WITH_JULIA
#error "Julia is not supported"
#endif

#include <any>
#include <functional>
#include <tuple>
#include <type_traits>

namespace kotekan {

///
/// Start up the Julia run-time system
void juliaStartup();
///
/// Shut down the Julia run-time system
void juliaShutdown();

/// Execute some code that may call the Julia run-time. It is NOT
/// possible to call functions that are defined via Julia code
/// directly, but such functions can be called via the Julia run-time
/// system. See [Embedding
/// Julia](https://docs.julialang.org/en/v1/manual/embedding/).
///
/// The main reason for this wrapper is that Julia's run-time is
/// running in its own thread, and this mechanism implements
/// thread-safe cross-thread calls.
///
/// We use `std::any` to be both generic and type-safe.
std::any juliaCallAny(const std::function<std::any()>& fun);

/// Wrapper to execute a function in the thread that runs the Julia
/// run-time (for functions returning `void`)
template<typename F, typename R = std::result_of_t<F()>>
std::enable_if_t<std::is_void_v<R>> juliaCall(const F& fun) {
    juliaCallAny([&]() { return fun(), std::any(std::make_tuple<>()); });
}

/// Wrapper to execute a function in the thread that runs the Julia
/// run-time (for functions not returning `void`)
template<typename F, typename R = std::result_of_t<F()>>
std::enable_if_t<!std::is_void_v<R>, R> juliaCall(const F& fun) {
    return std::any_cast<R>(juliaCallAny([&]() { return std::any(fun()); }));
}

} // namespace kotekan

#endif // #ifdef JULIA_MANAGER_HPP
