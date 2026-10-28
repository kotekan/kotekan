#ifndef KOTEKAN_LOGGING_FIXTURE_HPP
#define KOTEKAN_LOGGING_FIXTURE_HPP

#include "kotekanLogging.hpp" // for logLevel, log_event_handler, log_event_hook, FORMAT

#include <boost/test/included/unit_test.hpp>
#include <csignal>  // for sigaction, SIGTERM, SIG_IGN
#include <cstdlib>  // for _Exit
#include <iostream> // for cerr, cout, flush
#include <mutex>    // for mutex, lock_guard
#include <string>   // for string
#include <thread>   // for thread::id, this_thread::get_id

/// Boost fixture that makes an error logged by kotekan fail the test.
///
/// Install it by adding
///
///     BOOST_GLOBAL_FIXTURE(kotekan_logging_fixture);
///
/// to a test. An ERROR (including the one FATAL_ERROR logs first) then throws
/// FatalError, and a WARN registers a boost warning. FatalError is the narrowest
/// type that satisfies every BOOST_CHECK_THROW in tests/boost, including those
/// naming FatalError itself.
///
/// Only on the thread that installed the fixture. Boost.Test assertions are not
/// safe to call from more than one thread, and restClient and restServer log from
/// libevent callbacks that cannot carry an exception out. Events from any other
/// thread are printed and counted instead.
///
/// SIGTERM is ignored for the lifetime of the fixture and the previous
/// disposition put back, since test_logging.hpp's configure() installs a handler
/// of its own and a test may use both.
struct kotekan_logging_fixture {
    kotekan_logging_fixture() {
        struct sigaction ignore;
        ignore.sa_handler = SIG_IGN;
        sigemptyset(&ignore.sa_mask);
        ignore.sa_flags = 0;
        sigaction(SIGTERM, &ignore, &old_sigterm);

        state().test_thread = std::this_thread::get_id();
        kotekan::log_event_hook.store(&handle);
    }

    ~kotekan_logging_fixture() {
        kotekan::log_event_hook.store(nullptr);
        sigaction(SIGTERM, &old_sigterm, nullptr);

        int errors, warnings;
        {
            const std::lock_guard<std::mutex> lock(state().mutex);
            errors = state().off_thread_errors;
            warnings = state().off_thread_warnings;
        }
        if (errors == 0 && warnings == 0)
            return;

        // Report directly: a Boost.Test assertion raised from here, during the
        // framework's teardown, is swallowed as an empty "Test setup error".
        std::cerr << "\n[kotekan] " << (errors + warnings)
                  << " log event(s) reported off the test thread (" << errors << " error(s), "
                  << warnings << " warning(s)); see the lines above." << std::endl;
        if (errors > 0) {
            std::cout << std::flush;
            std::_Exit(boost::exit_test_failure);
        }
    }

    static void handle(const kotekan::logLevel level, const char* const file, const int line,
                       const std::string& message) {
        const bool is_error = level == kotekan::logLevel::ERROR;
        const std::string described = FORMAT("{}:{}: {}", file, line, message);

        if (std::this_thread::get_id() != state().test_thread) {
            // Neither throw nor touch Boost.Test from here; see the comment above.
            const std::lock_guard<std::mutex> lock(state().mutex);
            if (is_error)
                state().off_thread_errors++;
            else
                state().off_thread_warnings++;
            std::cerr << "[kotekan] " + std::string(is_error ? "error" : "warning")
                             + " logged off the test thread: " + described + "\n"
                      << std::flush;
            return;
        }

        if (is_error)
            throw FatalError(described);
        BOOST_WARN_MESSAGE(false, described);
    }

private:
    /// The SIGTERM disposition from before the fixture ignored it.
    struct sigaction old_sigterm;

    struct shared_state {
        std::thread::id test_thread;
        std::mutex mutex;
        int off_thread_errors = 0;
        int off_thread_warnings = 0;
    };

    /// The hook is a plain function pointer, so handle() has no instance to reach
    /// the fixture's state through.
    static shared_state& state() {
        static shared_state shared;
        return shared;
    }
};

#endif // KOTEKAN_LOGGING_FIXTURE_HPP
