// TOOD: Look at
// <https://stackoverflow.com/questions/77005/how-to-generate-a-stacktrace-when-my-gcc-c-app-crashes>,
// and see whether we can improve the code below

#if defined(__clang__)
#pragma clang optimize off
#elif defined(__GNUC__)
#pragma GCC optimize("O0")
#endif

// needed for dladdr, best at the top to avoid inconsistent includes
#define _GNU_SOURCE 1

#include "backtrace.hpp"

#include <cxxabi.h>    // for __cxa_demangle, abi
#include <dlfcn.h>     // for dladdr, Dl_info
#include <execinfo.h>  // for backtrace, backtrace_symbols
#include <signal.h>    // for sighandler_t, signal, kill, SIGABRT, SIGBUS, SIGFPE, SIGILL, SIGQUIT
#include <unistd.h>    // for getpid, NULL, pid_t
#include <fmt/core.h>  // for format, format_string
#include <stdlib.h>    // for free
#include <fstream>     // for basic_ostream, operator<<, basic_ostream::operator<<, basic_ofstream
#include <iostream>    // for cerr
#include <sstream>     // for basic_stringstream
#include <string>      // for char_traits, operator==, allocator, basic_string, operator<<, string
#include <vector>      // for vector
#include <algorithm>   // for find
#include <stdexcept>   // for runtime_error

#include "fmt.hpp"     // for join, join_view

using namespace std;

namespace kotekan {

// Generate a stack backtrace and send it to the given output
// stream
void generate_backtrace(ostream& stacktrace) {
    const int MAXSTACK = 100;
    static void* addresses[MAXSTACK];

    stacktrace << "Backtrace from pid " << getpid() << ":" << endl;

    int n = 0;
    n = backtrace(addresses, MAXSTACK);
    if (n < 2) {
        stacktrace << "Backtrace not available!\n";
    } else {
        auto oldflags = stacktrace.flags();
        stacktrace.setf(ios::hex);
        char** names = backtrace_symbols(addresses, n);
        for (int i = 2; i < n; i++) {
            char* demangled = NULL;
            // Attempt to demangle this if possible
            // Get the nearest symbol to feed to demangler
            Dl_info info;

            if (dladdr(addresses[i], &info) != 0) {
                int stat;
                // __cxa_demangle is a naughty obscure backend and no
                // self-respecting person would ever call it directly. ;-)
                // However it is a convenient glibc way to demangle syms.
                demangled = abi::__cxa_demangle(info.dli_sname, 0, 0, &stat);
            }

            if (demangled != NULL) {

                stacktrace << i - 1 << ". " << demangled << "   [" << names[i] << "]" << '\n';
                free(demangled);
            } else { // Just output the raw symbol
                stacktrace << i - 1 << ". " << names[i] << '\n';
            }
        }
        free(names);
        stacktrace.flags(oldflags);
    }
}

// Output a stack backtrace file backtrace.<rank>.txt
void write_backtrace_file(void) {
    ofstream myfile;
    stringstream ss;

    ss << "backtrace." << getpid() << ".txt";
    string filename = ss.str();

    cerr << "Writing backtrace to " << filename << endl;
    myfile.open(filename.c_str());
    generate_backtrace(myfile);
    myfile << "\n"
           << "The hexadecimal addresses in this backtrace can also be interpreted\n"
           << "with a debugger (e.g. gdb), or with the 'addr2line' (or "
              "'gaddr2line')\n"
           << "command line tool: 'addr2line -e cactus_sim <address>'.\n";
    myfile.close();
}

//////////////////////////////////////////////////////////////////////////////

template<int signum_>
struct signal_handler_t {
    signal_handler_t() {
        old_handler = signal(signum_, signal_handler);
    }

    static sighandler_t old_handler;

    static void signal_handler(int const signum) {
        pid_t const pid = getpid();

        cerr << "PID " << pid << " " << "received signal " << signum << endl;
        // Restore the default signal handler
        signal(signum, old_handler);

        write_backtrace_file();

        // Re-raise the signal to be caught by the default handler
        kill(pid, signum);
    }
};
template<int signum_>
sighandler_t signal_handler_t<signum_>::old_handler = (sighandler_t)SIG_DFL;

void request_backtraces(std::vector<std::string> signals) {
    if (auto it = std::find(signals.begin(), signals.end(), "SIGQUIT"); it != signals.end()) {
        static signal_handler_t<SIGQUIT> quit_signal;
        signals.erase(it);
    }
    if (auto it = std::find(signals.begin(), signals.end(), "SIGILL"); it != signals.end()) {
        static signal_handler_t<SIGILL> ill_signal;
        signals.erase(it);
    }
    if (auto it = std::find(signals.begin(), signals.end(), "SIGABRT"); it != signals.end()) {
        static signal_handler_t<SIGABRT> abrt_signal;
        signals.erase(it);
    }
    if (auto it = std::find(signals.begin(), signals.end(), "SIGFPE"); it != signals.end()) {
        static signal_handler_t<SIGFPE> fpe_signal;
        signals.erase(it);
    }
    if (auto it = std::find(signals.begin(), signals.end(), "SIGBUS"); it != signals.end()) {
        static signal_handler_t<SIGBUS> bus_signal;
        signals.erase(it);
    }
    if (auto it = std::find(signals.begin(), signals.end(), "SIGSEGV"); it != signals.end()) {
        static signal_handler_t<SIGSEGV> segv_signal;
        signals.erase(it);
    }

    if (!signals.empty()) {
        throw std::runtime_error(
            fmt::format("backtrace: unknown signal names: {}", fmt::join(signals, ", ")));
    }
}

} // namespace kotekan
