#include "juliaManager.hpp"

#include <any>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <future>
#include <julia.h>
#include <kotekanLogging.hpp>
#include <mutex>
#include <queue>
#include <thread>
#include <tuple>

// Only define this once, in an executable (not in a shared library) if you want fast code.
JULIA_DEFINE_FAST_TLS

namespace kotekan {

// The thread in which the Julia run-time runs
std::thread julia_thread;

// We use a task queue to communicate between the Kotekan and the
// Julia threads. Each task is just a `std::function<void()>`.
std::mutex julia_task_queue_mutex;
std::queue<std::function<void()>> julia_task_queue;

// The Julia task server. This initializes the Julia run-time, waits
// for a task in the task queue, and then executes the task. An
// empty task shuts down the task server.
void runJulia() {
    using namespace std::chrono_literals;

    INFO_F("[JM] Starting Julia run-time system");
    // Required: setup the Julia context.
    jl_init();

    atexit([]() {
        // Strongly recommended: notify Julia that the program is about to terminate. This
        // allows Julia time to cleanup pending write requests and run all finalizers.
        jl_atexit_hook(0);
    });

    INFO_F("[JM] Julia run-time system is running");
    while (true) {
        std::function<void()> task;
        while (!task) {
            std::unique_lock lk(julia_task_queue_mutex);
            if (!julia_task_queue.empty()) {
                // INFO_F("[JM] Found new task");
                task = std::move(julia_task_queue.front());
                julia_task_queue.pop();
                if (!task)
                    goto done;
            }
        }
        if (task) {
            // INFO_F("[JM] Running task...");
            task();
            // INFO_F("[JM] Done running task.");
        } else {
            // Oh no, we are using busy-waiting instead of condition
            // variables! Truth be told, condition variables in C++ are
            // awkward to use, difficult to get correct, and we don't
            // expect Julia code to be run in production. Busy waiting
            // is an excellent design choice here.
            std::this_thread::sleep_for(1ms);
        }
    }

done:
    INFO_F("[JM] Stopped Julia run-time system");
}

// Start the Julia run-time
void juliaStartup() {
    INFO_F("juliaManager: Starting Julia run-time");
    // Do nothing if the Julia run-time is already running
    if (julia_thread.joinable())
        return;
    julia_thread = std::thread(runJulia);
}

// Stop the Julia run-time
void juliaShutdown() {
    INFO_F("juliaManager: Stopping Julia run-time system...");
    // Check whether the Julia run-time is actually running
    assert(julia_thread.joinable());
    {
        // Send an empty task as indicator to shut down Julia
        std::unique_lock lk(julia_task_queue_mutex);
        julia_task_queue.push(std::function<void()>());
    }

    INFO_F("juliaManager: Waiting for Julia run-time system to stop...");
    julia_thread.join();
    INFO_F("juliaManager: Done.");
}

std::any juliaCallAny(const std::function<std::any()>& fun) {
    // INFO_F("juliaManager: Sending Julia task...");

    // We set up a promise and send it together with the function `fun`
    // that runs in the Julia thread. `fun` will fulful the promise with
    // its result. Below, we get a future from the promise to wait for
    // the result.
    std::promise<std::any> send_result;
    {
        std::unique_lock lk(julia_task_queue_mutex);
        julia_task_queue.push([&]() { send_result.set_value(fun()); });
    }

    // INFO_F("juliaManager: Waiting for Julia result...");
    std::future<std::any> recv_result = send_result.get_future();
    const std::any res = recv_result.get();

    // INFO_F("juliaManager: Returning result.");
    return res;
}

} // namespace kotekan
