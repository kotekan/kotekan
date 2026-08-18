#ifndef KOTEKAN_TEST_LOGGING_HPP
#define KOTEKAN_TEST_LOGGING_HPP

#include "kotekanLogging.hpp" // for log_event, log_event_handler, log_event_hook

#include <boost/test/included/unit_test.hpp>
#include <csignal>   // for signal, SIGTERM, SIG_IGN
#include <stdexcept> // for runtime_error
#include <string>    // for string, to_string

/// Boost fixture that makes an error logged by kotekan fail the test.
///
/// Install it by adding
///
///     BOOST_GLOBAL_FIXTURE(kotekan_logging_fixture);
///
/// to a test. An ERROR (including the one FATAL_ERROR logs first) then throws
/// std::runtime_error, and a WARN registers a boost warning. Note that FatalError
/// derives from std::runtime_error, so a BOOST_CHECK_THROW on std::runtime_error
/// matches either exception.
///
/// This replaces the compile-time KTK_BOOST_ERR/KTK_BOOST_WARN switch in
/// kotekanLogging.hpp, which was an ODR violation for every function defined in a
/// header that logs; see the comment on kotekan::log_event_handler.
///
/// SIGTERM is ignored as well. FATAL_ERROR calls exit_kotekan(), which raises
/// SIGTERM, before throwing FatalError; the handler below throws first so that is
/// normally not reached, but a path that reaches exit_kotekan() by another route
/// should not take the test process down with it.
struct kotekan_logging_fixture {
    kotekan_logging_fixture() {
        std::signal(SIGTERM, SIG_IGN);
        kotekan::log_event_hook.store(&handle);
    }

    ~kotekan_logging_fixture() {
        kotekan::log_event_hook.store(nullptr);
    }

    static void handle(const kotekan::log_event kind, const char* const file, const int line,
                       const std::string& message) {
        const std::string described =
            std::string(file) + ":" + std::to_string(line) + ": " + message;
        switch (kind) {
            case kotekan::log_event::warning:
                BOOST_WARN_MESSAGE(false, described);
                break;
            case kotekan::log_event::error:
                throw std::runtime_error(described);
        }
    }
};

#endif // KOTEKAN_TEST_LOGGING_HPP
