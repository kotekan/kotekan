#define BOOST_TEST_MODULE "test_cuda_frame_join"

#include "cudaFrameJoin.hpp" // for cudaFrameJoin, plan_cuda_frame_join

#include <algorithm>
#include <boost/test/included/unit_test.hpp>
#include <cstdint>
#include <vector>

// A cudaProcess spreads one frame's commands across several CUDA streams, which are
// independent in-order queues, so the final command completing does not mean the frame is done.
// Frame completion releases host frames (cudaInputData -> mark_frame_empty, cudaOutputData ->
// mark_frame_full), so the end-of-frame join must wait on every other stream with work, and must
// sit on the stream the chain ends on.
//
// The streams a command lands on by type, at cuda_stream_base B: COPY_IN B+0, COPY_OUT B+1,
// KERNEL B+2.

namespace {

std::vector<bool> streams(std::size_t n, const std::vector<std::int32_t>& with_events) {
    std::vector<bool> v(n, false);
    for (auto s : with_events)
        v[s] = true;
    return v;
}

bool waits_on(const cudaFrameJoin& j, std::int32_t s) {
    return std::find(j.wait_streams.begin(), j.wait_streams.end(), s) != j.wait_streams.end();
}

} // namespace

// The join goes on the stream the chain ends on, never min(streams). For the default triple the
// lowest stream is the copy-in, and a join there would queue the next frame's host-to-device
// copy behind this frame's compute and output.
BOOST_AUTO_TEST_CASE(join_stream_is_the_final_stream_never_the_lowest) {
    // The canonical pipeline: cudaInputData(0), cudaSyncInput(2), kernel(2),
    // cudaSyncOutput(1), cudaOutputData(1). The last command to return an event is on 1.
    const auto j = plan_cuda_frame_join(streams(3, {0, 1, 2}), /*final_stream=*/1);
    BOOST_CHECK_EQUAL(j.join_stream, 1);
    BOOST_CHECK_NE(j.join_stream, 0); // min({0,1,2}) == 0 is the copy-in stream
    BOOST_CHECK(j.needed);
}

// The same shape shifted by cuda_stream_base: nothing about the choice may depend on the
// absolute stream numbers, only on which one the final command ran on.
BOOST_AUTO_TEST_CASE(join_stream_follows_the_stream_base) {
    const auto j = plan_cuda_frame_join(streams(12, {6, 7, 8}), /*final_stream=*/7);
    BOOST_CHECK_EQUAL(j.join_stream, 7);
    BOOST_CHECK(j.needed);
    BOOST_CHECK(waits_on(j, 6));
    BOOST_CHECK(waits_on(j, 8));
}

// A lone non-final event on stream 0 still forces a join: at base 0 that is the copy-in, whose
// DMA must finish before mark_frame_empty() hands the buffer back to the producer.
BOOST_AUTO_TEST_CASE(a_lone_event_on_stream_zero_still_forces_a_join) {
    const auto j = plan_cuda_frame_join(streams(3, {0, 2}), /*final_stream=*/2);
    BOOST_REQUIRE(j.needed);
    BOOST_CHECK(waits_on(j, 0));
    BOOST_CHECK_EQUAL(j.wait_streams.size(), 1u);
}

// When the final command's stream carries the only event there is nothing to join, and the
// join must cost nothing: no wait, no per-frame event create/destroy.
BOOST_AUTO_TEST_CASE(no_join_when_the_final_stream_carries_everything) {
    const auto j = plan_cuda_frame_join(streams(3, {1}), /*final_stream=*/1);
    BOOST_CHECK(!j.needed);
    BOOST_CHECK(j.wait_streams.empty());
}

// A join stream waiting on its own event is at best a no-op and at worst a self-dependency.
BOOST_AUTO_TEST_CASE(the_join_never_waits_on_itself) {
    for (std::int32_t fs : {0, 1, 2}) {
        const auto j = plan_cuda_frame_join(streams(3, {0, 1, 2}), fs);
        BOOST_CHECK(!waits_on(j, fs));
        BOOST_CHECK_EQUAL(j.wait_streams.size(), 2u);
    }
}

// The parallel-chain shape the join exists for: three kernel+output pairs on their own
// streams, the last command on stream 3. Streams 1 and 2 may still be running when it
// finishes, so both must be waited on.
BOOST_AUTO_TEST_CASE(parallel_per_stream_chains_wait_on_every_sibling) {
    const auto j = plan_cuda_frame_join(streams(4, {1, 2, 3}), /*final_stream=*/3);
    BOOST_REQUIRE(j.needed);
    BOOST_CHECK_EQUAL(j.join_stream, 3);
    BOOST_CHECK_EQUAL(j.wait_streams.size(), 2u);
    BOOST_CHECK(waits_on(j, 1));
    BOOST_CHECK(waits_on(j, 2));
}

// A pipeline that produced no events at all (every command skipped by a required_flag, say)
// must not try to join.
BOOST_AUTO_TEST_CASE(no_events_no_join) {
    const auto j = plan_cuda_frame_join(streams(3, {}), /*final_stream=*/0);
    BOOST_CHECK(!j.needed);
    BOOST_CHECK(j.wait_streams.empty());
}

// The wait list is used to index events[], so it must stay inside the vector it was built
// from, and ascending order keeps the enqueue deterministic frame to frame.
BOOST_AUTO_TEST_CASE(wait_streams_are_in_range_and_ascending) {
    const auto j = plan_cuda_frame_join(streams(8, {0, 3, 5, 7}), /*final_stream=*/5);
    BOOST_REQUIRE(j.needed);
    for (auto s : j.wait_streams) {
        BOOST_CHECK_GE(s, 0);
        BOOST_CHECK_LT(s, 8);
    }
    BOOST_CHECK(std::is_sorted(j.wait_streams.begin(), j.wait_streams.end()));
}
