#define BOOST_TEST_MODULE "test_cuda_stream_assignment"

#include "cudaStreamAssignment.hpp" // for resolve_cuda_stream, unique_ascending_streams, lock_...
#include "gpuCommand.hpp"           // for gpuCommandType

#include <algorithm>
#include <atomic>
#include <boost/test/included/unit_test.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>
#include <iterator>
#include <mutex>
#include <vector>

// Three properties are pinned here.
//
// 1. Stream assignment by role within the pipeline's own triple, selected by the index in
//    cuda_stream_base: base 0 is 0/1/2, which is what makes the key safe to add to existing
//    configs, and every other base is a wholly disjoint triple.
//
// 2. That the queuing locks are ACQUIRED in ascending stream order, whatever order a pipeline
//    supplies its set in. cudaProcess takes one mutex per stream, several at a time, and
//    multiple locks can only deadlock if two acquirers disagree about the order; ascending
//    stream id is that global order, and lock_streams_ascending is the one place it is
//    imposed. The concurrent case below drives it on host mutexes from competing threads.
//
// 3. That a command whose stream is outside its own stage's num_cuda_streams is refused, by
//    the same function cudaProcess::collect_stream_ids calls, and that an instance which never
//    enqueues (a do_once cudaInputData past the first) is skipped rather than refused.

// ---------------------------------------------------------------- role -> stream

BOOST_AUTO_TEST_CASE(base_zero_is_the_plain_triple) {
    BOOST_CHECK_EQUAL(resolve_cuda_stream(gpuCommandType::COPY_IN, 0), 0);
    BOOST_CHECK_EQUAL(resolve_cuda_stream(gpuCommandType::COPY_OUT, 0), 1);
    BOOST_CHECK_EQUAL(resolve_cuda_stream(gpuCommandType::KERNEL, 0), 2);
}

// The base is an INDEX: it scales by three, so consecutive bases give adjacent triples.
BOOST_AUTO_TEST_CASE(a_base_selects_a_whole_triple_and_keeps_the_roles_adjacent) {
    for (std::int32_t base : {1, 2, 3, 8}) {
        BOOST_CHECK_EQUAL(resolve_cuda_stream(gpuCommandType::COPY_IN, base), 3 * base + 0);
        BOOST_CHECK_EQUAL(resolve_cuda_stream(gpuCommandType::COPY_OUT, base), 3 * base + 1);
        BOOST_CHECK_EQUAL(resolve_cuda_stream(gpuCommandType::KERNEL, base), 3 * base + 2);
    }
}

// The property the index buys, and the reason Andre asked for it: no two bases can produce
// triples that partially overlap, so a copy can never land on another pipeline's kernel
// stream by a careless config. A device has one real copy queue in each direction, so such a
// copy would stall that pipeline's kernels behind unrelated traffic.
BOOST_AUTO_TEST_CASE(distinct_bases_can_never_partially_overlap) {
    const gpuCommandType roles[] = {gpuCommandType::COPY_IN, gpuCommandType::COPY_OUT,
                                    gpuCommandType::KERNEL};
    for (std::int32_t a = 0; a < 8; a++) {
        for (std::int32_t b = 0; b < 8; b++) {
            std::vector<std::int32_t> sa, sb;
            for (gpuCommandType r : roles) {
                sa.push_back(resolve_cuda_stream(r, a));
                sb.push_back(resolve_cuda_stream(r, b));
            }
            std::vector<std::int32_t> both;
            std::set_intersection(sa.begin(), sa.end(), sb.begin(), sb.end(),
                                  std::back_inserter(both));
            BOOST_CHECK_MESSAGE(both.size() == (a == b ? 3u : 0u),
                                "bases " << a << " and " << b << " share " << both.size()
                                         << " stream(s); a triple must be all or nothing");
        }
    }
}

// A BARRIER has no role to derive a stream from; the caller must reject it.
BOOST_AUTO_TEST_CASE(a_barrier_without_an_explicit_stream_is_refused) {
    BOOST_CHECK_EQUAL(resolve_cuda_stream(gpuCommandType::BARRIER, 0), CUDA_STREAM_NEEDS_EXPLICIT);
    BOOST_CHECK_EQUAL(resolve_cuda_stream(gpuCommandType::BARRIER, 2), CUDA_STREAM_NEEDS_EXPLICIT);
}

// The negative-base refusal here is the one cudaCommand::set_command_type, cudaSyncInput and
// cudaSyncOutput all rely on; none of them checks the base themselves.
BOOST_AUTO_TEST_CASE(a_negative_base_and_an_unset_type_both_throw) {
    BOOST_CHECK_THROW(resolve_cuda_stream(gpuCommandType::KERNEL, -1), std::runtime_error);
    BOOST_CHECK_THROW(resolve_cuda_stream(gpuCommandType::COPY_IN, -3), std::runtime_error);
    BOOST_CHECK_THROW(resolve_cuda_stream(gpuCommandType::NOT_SET, 0), std::runtime_error);
}

// ---------------------------------------------------------------- lock order

// The deadlock-freedom property, tested where it is enforced: locks are taken in ascending
// stream order no matter what order the set arrives in.
BOOST_AUTO_TEST_CASE(locks_are_acquired_in_ascending_stream_order) {
    std::recursive_mutex mu[16];
    std::vector<std::int32_t> order;
    auto recording = [&](std::int32_t sid) -> std::recursive_mutex& {
        order.push_back(sid);
        return mu[sid];
    };
    {
        auto locks = lock_streams_ascending({8, 6, 7}, recording);
        BOOST_CHECK_EQUAL(locks.size(), 3u);
        BOOST_CHECK(std::all_of(
            locks.begin(), locks.end(),
            [](const std::unique_lock<std::recursive_mutex>& l) { return l.owns_lock(); }));
    }
    BOOST_CHECK((order == std::vector<std::int32_t>{6, 7, 8}));
    order.clear();
    { auto locks = lock_streams_ascending({1, 2}, recording); }
    BOOST_CHECK((order == std::vector<std::int32_t>{1, 2}));
    // Once the locks are released the mutexes are free again.
    for (int s = 0; s < 16; ++s) {
        BOOST_CHECK(mu[s].try_lock());
        mu[s].unlock();
    }
}

// Three pipelines that all share streams 1 and 2, each handing its set in a different order,
// locking concurrently thousands of times: with the order imposed inside the helper this must
// finish (it takes milliseconds). Locking in the supplied order instead is the classic AB/BA
// deadlock, and a deadlocked std::async cannot be unwound past -- its future's destructor
// joins -- so on a missed deadline the process reports and exits nonzero directly.
BOOST_AUTO_TEST_CASE(overlapping_pipelines_locking_concurrently_cannot_deadlock) {
    std::recursive_mutex mu[8];
    auto mutex_of = [&](std::int32_t sid) -> std::recursive_mutex& { return mu[sid]; };
    std::atomic<int> held_both{0}, max_held{0};
    auto pipeline = [&](std::vector<std::int32_t> set) {
        for (int i = 0; i < 4000; ++i) {
            auto locks = lock_streams_ascending(set, mutex_of);
            int h = ++held_both;
            int m = max_held.load();
            while (h > m && !max_held.compare_exchange_weak(m, h)) {
            }
            --held_both;
        }
    };
    auto a = std::async(std::launch::async, pipeline, std::vector<std::int32_t>{2, 1});
    auto b = std::async(std::launch::async, pipeline, std::vector<std::int32_t>{0, 2, 1});
    auto c = std::async(std::launch::async, pipeline, std::vector<std::int32_t>{1, 2, 4});
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    for (auto* f : {&a, &b, &c})
        if (f->wait_until(deadline) != std::future_status::ready) {
            std::cerr << "overlapping_pipelines_locking_concurrently_cannot_deadlock: "
                         "DEADLOCKED (lock order not enforced)\n";
            std::_Exit(1);
        }
    // Sets that overlap are mutually exclusive: at most one of them holds its locks at a time.
    BOOST_CHECK_EQUAL(max_held.load(), 1);
}

// ---------------------------------------------------------------- pipeline stream set

// The bound is the owning stage's, enforced by the function cudaProcess::collect_stream_ids
// calls for every command once they are all built.
BOOST_AUTO_TEST_CASE(a_stream_beyond_its_stages_num_cuda_streams_is_refused) {
    BOOST_CHECK_NO_THROW(check_stream_within("/p/commands/0", 5, "/p", 6));
    BOOST_CHECK_NO_THROW(check_stream_within("/p/commands/0", 0, "/p", 1));
    BOOST_CHECK_THROW(check_stream_within("/p/commands/0", 6, "/p", 6), std::runtime_error);
    BOOST_CHECK_THROW(check_stream_within("/p/commands/0", 5, "/p", 3), std::runtime_error);
    try {
        check_stream_within("/gpuB/gpu_0/commands/2", 5, "/gpuB/gpu_0", 3);
        BOOST_FAIL("expected a throw");
    } catch (const std::runtime_error& e) {
        const std::string m = e.what();
        BOOST_CHECK(m.find("/gpuB/gpu_0/commands/2") != std::string::npos);
        BOOST_CHECK(m.find("stream 5") != std::string::npos);
        BOOST_CHECK(m.find("num_cuda_streams 3") != std::string::npos);
    }
}

// cudaInputData with do_once returns from its constructor before set_command_type for every
// instance but the first, so those instances stay at -1 and never enqueue. The bound check
// accepts them and the set builder drops them; config/ci-tests/gpu_batch/test_xpose2048.j2
// starts only because of this.
BOOST_AUTO_TEST_CASE(a_do_once_instance_without_a_stream_is_skipped) {
    BOOST_CHECK_NO_THROW(check_stream_within("/p/commands/0", -1, "/p", 3));
    const auto s = unique_ascending_streams({3, -1, 5, -1});
    BOOST_CHECK((s == std::vector<std::int32_t>{3, 5}));
}

// The set is unique and ascending, and a pipeline with no commands that enqueue has no
// streams: it is cudaProcess's job to refuse that, not this function's to invent a stream.
BOOST_AUTO_TEST_CASE(the_stream_set_is_unique_ascending_and_empty_for_no_commands) {
    const auto s = unique_ascending_streams({2, 1, 2, 1, 0});
    BOOST_CHECK((s == std::vector<std::int32_t>{0, 1, 2}));
    BOOST_CHECK(unique_ascending_streams({}).empty());
    BOOST_CHECK(unique_ascending_streams({-1, -1}).empty());
}

// The realistic case: a pipeline at base 2, i.e. streams 6/7/8, declared kernel-first.
BOOST_AUTO_TEST_CASE(a_shifted_pipelines_set_is_its_own_triple_in_order) {
    std::vector<std::int32_t> ids;
    for (auto t : {gpuCommandType::KERNEL, gpuCommandType::KERNEL, gpuCommandType::COPY_OUT,
                   gpuCommandType::COPY_OUT})
        ids.push_back(resolve_cuda_stream(t, 2));
    const auto s = unique_ascending_streams(ids);
    BOOST_REQUIRE_EQUAL(s.size(), 2u);
    BOOST_CHECK_EQUAL(s[0], 7); // COPY_OUT
    BOOST_CHECK_EQUAL(s[1], 8); // KERNEL
}
