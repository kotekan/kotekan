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

static std::once_flag julia_thread_once_flag;
static std::thread julia_thread;

static std::queue<std::function<void()>> julia_task_queue;
static std::mutex julia_task_queue_mutex;
static std::condition_variable julia_task_queue_condition_variable;

static void initJulia() {
    // Required: setup the Julia context.
    // INFO_F("[JM] Initializing Julia...");
    jl_init();
    atexit([]() {
        // Strongly recommended: notify Julia that the program is about to terminate. This
        // allows Julia time to cleanup pending write requests and run all finalizers.
        jl_atexit_hook(0);
    });
    // INFO_F("[JM] Done initializing Julia.");
}

static void runJulia() {
    using namespace std::chrono_literals;

    INFO_F("[JM] Starting Julia run-time system");
    static std::once_flag init_julia_once_flag;
    std::call_once(init_julia_once_flag, initJulia);

    while (true) {
        std::unique_lock lk(julia_task_queue_mutex);
        while (!julia_task_queue.empty()) {
            // INFO_F("[JM] Running Julia task...");
            const std::function<void()>& task = julia_task_queue.front();
            if (!task)
                goto done;
            task();
            julia_task_queue.pop();
            // INFO_F("[JM] Done.");
        }
        julia_task_queue_condition_variable.wait_for(lk, 1ms);
    }

done:
    INFO_F("[JM] Stopping Julia run-time system");
}

void juliaShutdown() {
    INFO_F("juliaManager: Stopping Julia run-time system...");
    {
        std::unique_lock lk(julia_task_queue_mutex);
        julia_task_queue.push(std::function<void()>());
    }
    julia_task_queue_condition_variable.notify_one();

    INFO_F("juliaManager: Waiting for Julia run-time system to stop...");
    julia_thread.join();
    INFO_F("juliaManager: Done.");
}

std::any juliaCallAny(const std::function<std::any()>& fun) {
    std::call_once(julia_thread_once_flag, [&] {
        INFO_F("juliaManager: Starting Julia run-time");
        julia_thread = std::thread(runJulia);
    });

    // INFO_F("juliaManager: Sending Julia task...");
    std::promise<std::any> send_result;
    {
        std::unique_lock lk(julia_task_queue_mutex);
        julia_task_queue.push([&]() { send_result.set_value(fun()); });
    }
    julia_task_queue_condition_variable.notify_one();

    // INFO_F("juliaManager: Waiting for Julia result...");
    std::future<std::any> recv_result = send_result.get_future();
    const std::any res = recv_result.get();

    // INFO_F("juliaManager: Done.");
    return res;
}
