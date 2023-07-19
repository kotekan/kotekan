#ifndef JULIA_MANAGER_HPP
#define JULIA_MANAGER_HPP

#include <any>
#include <functional>
#include <tuple>
#include <type_traits>

std::any juliaCallAny(const std::function<std::any()>& fun);

void juliaShutdown();

template<typename F, typename R = std::result_of_t<F()>>
std::enable_if_t<std::is_void_v<R>> juliaCall(const F& fun) {
    juliaCallAny([&]() { return fun(), std::any(std::make_tuple<>()); });
}

template<typename F, typename R = std::result_of_t<F()>>
std::enable_if_t<!std::is_void_v<R>, R> juliaCall(const F& fun) {
    return std::any_cast<R>(juliaCallAny([&]() { return std::any(fun()); }));
}

#endif // #ifdef JULIA_MANAGER_HPP
