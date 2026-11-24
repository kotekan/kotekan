#ifndef KOTEKAN_TEST_LOGGING_HPP
#define KOTEKAN_TEST_LOGGING_HPP

#include "errors.h"
#include "kotekanLogging.hpp"

#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace kotekan_test_logging {

// Not async-signal-safe: meant for the SIGTERM that exit_kotekan raises on the failing thread,
// which holds no lock at that point.
inline void fatal_signal_handler(int sig) {
    const std::string msg = kotekan::kotekanLogging::get_error_message();
    if (!msg.empty())
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
