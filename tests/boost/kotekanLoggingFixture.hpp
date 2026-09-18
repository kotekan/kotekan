#ifndef KOTEKAN_LOGGING_FIXTURE_HPP
#define KOTEKAN_LOGGING_FIXTURE_HPP

#include "kotekanLogging.hpp" // for log_event, log_event_handler, log_event_hook, FORMAT

#include <boost/test/included/unit_test.hpp>
#include <csignal>  // for signal, SIGTERM, SIG_IGN
#include <cstdlib>  // for _Exit
#include <iostream> // for cerr, cout, flush
#include <mutex>    // for mutex, lock_guard
#include <string>   // for string, to_string
#include <thread>   // for thread::id, this_thread::get_id

/// Boost fixture that makes an error logged by kotekan fail the test.
///
/// Install it by adding
///
///     BOOST_GLOBAL_FIXTURE(kotekan_logging_fixture);
///
/// to a test. An ERROR (including the one FATAL_ERROR logs first) then throws
/// FatalError, and a WARN registers a boost warning.
///
/// FatalError rather than a plain std::runtime_error because it is the narrower
/// of the two: it derives from std::runtime_error, so a BOOST_CHECK_THROW on
/// std::runtime_error or std::exception still matches, and one on FatalError --
/// which is what a test asserting a FATAL_ERROR path naturally writes, see
/// test_Telescope's _get_eop_out_of_range_fatal -- matches as well. Throwing the
/// base class would fail those.
///
/// That happens only on the thread that installed the fixture. Boost.Test
/// assertions are not safe to call from more than one thread, and kotekan logs
/// from threads that cannot carry an exception either: restClient and restServer
/// both log from libevent callbacks, and restServer's ERROR_NON_OO calls sit
/// inside the very catch blocks that turn an exception into a 500 reply, so
/// throwing there would unwind into libevent's C frames and terminate the
/// process. An event from any other thread is printed instead, and counted so
/// that the run still fails if it was an error.
///
/// SIGTERM is ignored for the lifetime of the fixture, and the previous
/// disposition put back afterwards. FATAL_ERROR calls exit_kotekan(), which
/// raises SIGTERM, before throwing FatalError; the handler below throws first so
/// that is normally not reached, but a path that reaches exit_kotekan() by
/// another route should not take the test process down with it. Restoring
/// matters because test_logging.hpp's configure() installs a SIGTERM handler of
/// its own, and a test may use both.
///
/// Note: test_logging.hpp in this directory is a separate, opt-in helper
/// (kotekan_test_logging::configure()) that raises the log level and prints
/// kotekan's stored error message on SIGTERM. The two are independent; a test may
/// use either or both.
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

        // No Boost.Test assertion can be raised from here: BOOST_GLOBAL_FIXTURE
        // destroys the fixture during the framework's teardown, after the report
        // has been written, where an assertion is reported as an empty "Test setup
        // error" instead (Boost 1.83). So report this directly, and end the process
        // with boost's own failure status if an error was among the events -- an
        // error on the test thread fails the run, and one on another thread must
        // not be quieter just because it could not be thrown.
        std::cerr << "\n[kotekan] " << (errors + warnings)
                  << " log event(s) reported off the test thread (" << errors << " error(s), "
                  << warnings << " warning(s)); see the lines above." << std::endl;
        if (errors > 0) {
            std::cerr << "[kotekan] failing the run: an error logged off the test thread cannot "
                         "be turned into an exception there."
                      << std::endl;
            std::cout << std::flush;
            std::_Exit(boost::exit_test_failure);
        }
    }

    static void handle(const kotekan::log_event kind, const char* const file, const int line,
                       const std::string& message) {
        const std::string described = FORMAT("{}:{}: {}", file, line, message);

        if (std::this_thread::get_id() != state().test_thread) {
            // Neither throw nor touch Boost.Test from here; see the comment above.
            const std::lock_guard<std::mutex> lock(state().mutex);
            if (kind == kotekan::log_event::error)
                state().off_thread_errors++;
            else
                state().off_thread_warnings++;
            std::cerr << "[kotekan] "
                             + std::string(kind == kotekan::log_event::error ? "error" : "warning")
                             + " logged off the test thread: " + described + "\n"
                      << std::flush;
            return;
        }

        switch (kind) {
            case kotekan::log_event::warning:
                BOOST_WARN_MESSAGE(false, described);
                break;
            case kotekan::log_event::error:
                throw FatalError(described);
        }
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
