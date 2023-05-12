#include "juliaManager.hpp"

#include <cstdlib>
#include <julia.h>
#include <kotekanLogging.hpp>
#include <mutex>

// Only define this once, in an executable (not in a shared library) if you want fast code.
// JULIA_DEFINE_FAST_TLS

void juliaInit() {
    static std::once_flag init_once_flag;
    std::call_once(init_once_flag, [&] {
        INFO_F("Starting Julia run-time");
        // Required: setup the Julia context.
        jl_init();
        atexit([]() {
            // Strongly recommended: notify Julia that the program is about to terminate. This
            // allows Julia time to cleanup pending write requests and run all finalizers.
            jl_atexit_hook(0);
        });
    });
}
