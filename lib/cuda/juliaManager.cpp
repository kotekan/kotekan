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

static std::queue<std::function<void()>> julia_task_queue;
static std::mutex julia_task_queue_mutex;
static std::condition_variable julia_task_queue_condition_variable;

static std::tuple<> initJulia() {
    // Required: setup the Julia context.
    INFO_F("[JM] Initializing Julia...");
    jl_init();
    atexit([]() {
        // Strongly recommended: notify Julia that the program is about to terminate. This
        // allows Julia time to cleanup pending write requests and run all finalizers.
        jl_atexit_hook(0);
    });
    INFO_F("[JM] Done initializing Julia.");
    return std::tuple<>();
}

static void runJulia() {
    using namespace std::chrono_literals;

    static std::tuple<> dummy __attribute__((__unused__)) = initJulia();

    while (true) {
        std::unique_lock lk(julia_task_queue_mutex);
        while (!julia_task_queue.empty()) {
            INFO_F("[JM] Running Julia task...");
            const std::function<void()>& task = julia_task_queue.front();
            task();
            julia_task_queue.pop();
            INFO_F("[JM] Done.");
        }
        julia_task_queue_condition_variable.wait_for(lk, 1ms);
    }
}

std::any juliaCallAny(const std::function<std::any()>& fun) {
    static std::once_flag init_once_flag;
    static std::thread julia_thread;
    std::call_once(init_once_flag, [&] {
        INFO_F("Starting Julia run-time");
        julia_thread = std::thread(runJulia);
        julia_thread.detach();
    });

    INFO_F("Sending Julia task...");
    std::promise<std::any> send_result;
    {
        std::unique_lock lk(julia_task_queue_mutex);
        julia_task_queue.push([&]() { send_result.set_value(fun()); });
    }
    julia_task_queue_condition_variable.notify_one();

    INFO_F("Waiting for Julia result...");
    std::future<std::any> recv_result = send_result.get_future();
    const std::any res = recv_result.get();

    INFO_F("Done.");
    return res;
}
