#ifndef KOTEKAN_TEST_LOGGING_HPP
#define KOTEKAN_TEST_LOGGING_HPP

#include "errors.h"

#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <signal.h> // for sigaction, SIGTERM

namespace kotekan_test_logging {

inline void ignore_signal_handler(int /*sig*/) {}

/**
 * @brief Installs a no-op SIGTERM handler for as long as it exists.
 *
 * FATAL_ERROR (and FATAL_ERROR_NON_OO) call exit_kotekan(), which raises
 * SIGTERM before throwing FatalError. Nothing handles that signal in a test
 * binary, so the process dies before the throw can be checked. A test that
 * means to catch a fatal path declares one of these at file scope:
 *
 *     static kotekan_test_logging::SigtermGuard g_sigterm_guard;
 *
 * Not to be combined with configure(), which installs a handler of its own
 * that reports the error and _Exits; whichever runs last wins.
 */
struct SigtermGuard {
    struct sigaction old_action;
    SigtermGuard() {
        struct sigaction new_action;
        new_action.sa_handler = ignore_signal_handler;
        sigemptyset(&new_action.sa_mask);
        new_action.sa_flags = 0;
        sigaction(SIGTERM, &new_action, &old_action);
    }
    ~SigtermGuard() {
        sigaction(SIGTERM, &old_action, nullptr);
    }
};

inline void fatal_signal_handler(int sig) {
    const char* msg = get_error_message();
    if (msg && msg[0] != '\0')
        std::cerr << "\n[kotekan] Fatal error: " << msg << std::endl;
    std::_Exit(128 + sig);
}

inline void configure() {
    static bool configured = false;
    if (configured)
        return;
    configured = true;
    auto prev = std::signal(SIGTERM, fatal_signal_handler);
    if (prev == SIG_ERR) {
        std::cerr << "[kotekan] Failed to install SIGTERM handler: " << std::strerror(errno)
                  << std::endl;
    }
    __enable_syslog = 0;
    if (_global_log_level < 3)
        _global_log_level = 3;
}

} // namespace kotekan_test_logging

#endif // KOTEKAN_TEST_LOGGING_HPP
