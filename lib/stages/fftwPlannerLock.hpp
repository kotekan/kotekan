#ifndef FFTW_PLANNER_LOCK_HPP
#define FFTW_PLANNER_LOCK_HPP

#include <mutex>

/**
 * @brief Process-wide lock serializing FFTW planner calls.
 *
 * FFTW's planner routines (``fftwf_plan_*`` and ``fftwf_destroy_plan``)
 * mutate shared global state and are explicitly NOT thread-safe; only
 * ``fftwf_execute`` may run concurrently. Stage construction is serial so
 * ctor plan creation is already safe, but plan *destruction* (stage dtors
 * at shutdown) and AirspyAlign's per-REST-callback plan create/destroy run
 * on other threads and can race each other and any concurrent stage dtor.
 *
 * Every fftwf plan create/destroy across all stages must take this one
 * lock. It's a Meyers singleton in an inline function so all translation
 * units share a single mutex without a separate .cpp.
 */
inline std::mutex& fftw_planner_mutex() {
    static std::mutex m;
    return m;
}

#endif // FFTW_PLANNER_LOCK_HPP
