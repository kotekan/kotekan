#ifndef KOTEKAN_TEST_LOGGING_HPP
#define KOTEKAN_TEST_LOGGING_HPP

#include "errors.h"

#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace kotekan_test_logging {

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
