#ifndef CUDA_STREAM_ASSIGNMENT_HPP
#define CUDA_STREAM_ASSIGNMENT_HPP

#include "gpuCommand.hpp" // for gpuCommandType

#include "fmt.hpp" // for format

#include <algorithm> // for sort
#include <cstdint>   // for int32_t, uint32_t
#include <mutex>     // for recursive_mutex, unique_lock
#include <set>       // for set
#include <stdexcept> // for runtime_error
#include <string>    // for string
#include <vector>    // for vector

/**
 * @file
 * @brief Which CUDA stream a command runs on, and which streams a pipeline owns.
 *
 * Split out of cudaCommand::set_command_type and cudaProcess::collect_stream_ids so both can
 * be unit tested without a GPU: these are pure, and every CUDA call and config read stays at
 * the call site. See tests/boost/test_cuda_stream_assignment.cpp.
 */

/// Returned by resolve_cuda_stream for a BARRIER with no explicit `cuda_stream`; the caller
/// turns it into the "cuda_stream required for barrier type command object" error.
constexpr std::int32_t CUDA_STREAM_NEEDS_EXPLICIT = -1;

/**
 * @brief The stream a command lands on.
 *
 * @param type  the command's gpuCommandType.
 * @param base  the owning pipeline's `cuda_stream_base`, an INDEX not a stream id.
 *
 * Default assignment is by ROLE within the pipeline's own triple: COPY_IN 3*base+0, COPY_OUT
 * 3*base+1, KERNEL 3*base+2. Those are indices into a stream vector shared by every
 * cudaProcess on the device, so two pipelines at the same base share streams -- and since a
 * CUDA stream is an in-order queue, their kernels serialize. Distinct bases are what give
 * pipelines genuinely independent queues.
 *
 * THE FACTOR OF THREE IS NOT COSMETIC: it makes a triple that overlaps another pipeline's
 * inexpressible. A device has effectively one host-to-device and one device-to-host copy
 * queue however many streams feed them, so a copy landing on a stream that also carries
 * kernels blocks those kernels until an unrelated copy elsewhere clears. Partial overlap is
 * also strictly worse than either extreme: it costs the extra streams while still sharing
 * queuing locks. An index admits neither.
 *
 * This resolves the DEFAULT only. A per-command `cuda_stream` is absolute and ignores the
 * base, so cudaCommand::set_command_type short-circuits on one before calling this; a BARRIER
 * has no role to derive a stream from and must be given one that way. cudaSyncInput and
 * cudaSyncOutput call this too, so the streams they wait on are the streams the commands they
 * wait for are assigned by the same rule.
 *
 * @throws std::runtime_error for a negative base, and for NOT_SET, which is a programming
 *         error in the command.
 */
inline std::int32_t resolve_cuda_stream(gpuCommandType type, std::int32_t base) {
    if (base < 0)
        throw std::runtime_error("cuda_stream_base must be >= 0");
    switch (type) {
        case gpuCommandType::COPY_IN:
            return 3 * base + 0;
        case gpuCommandType::COPY_OUT:
            return 3 * base + 1;
        case gpuCommandType::KERNEL:
            return 3 * base + 2;
        case gpuCommandType::BARRIER:
            return CUDA_STREAM_NEEDS_EXPLICIT;
        case gpuCommandType::NOT_SET:
        default:
            throw std::runtime_error("No command type set");
    }
}

/**
 * @brief Refuse a stream that is not one its own stage asked for.
 *
 * @param command           the command's unique name, for the message.
 * @param stream            its resolved or explicit cuda_stream_id, or a source stream.
 * @param stage             the owning cudaProcess, for the message.
 * @param num_cuda_streams  the count that stage passed to prepareStreams.
 *
 * The bound is the owning stage's, not the device's: prepareStreams only grows the device's
 * stream vector, so a sibling stage that asked for more would otherwise let a command land on
 * a stream its own stage never declared.
 *
 * A negative id is an instance that never enqueues and is accepted: cudaInputData with
 * `do_once` returns from its constructor before set_command_type for every instance but the
 * first, and its execute() returns no event for those, so queue_commands never indexes the
 * per-frame events vector with the -1. unique_ascending_streams drops them.
 *
 * @throws std::runtime_error naming the command, the stream and the stage.
 */
inline void check_stream_within(const std::string& command, std::int32_t stream,
                                const std::string& stage, std::uint32_t num_cuda_streams) {
    if (stream >= 0 && (std::uint32_t)stream >= num_cuda_streams)
        throw std::runtime_error(fmt::format("{:s} is on CUDA stream {:d}, but {:s} uses "
                                             "num_cuda_streams {:d} -- raise it above {:d}",
                                             command, stream, stage, num_cuda_streams, stream));
}

/**
 * @brief The distinct streams a pipeline enqueues onto, ascending and unique.
 *
 * Every id must already have passed check_stream_within. Negative ids are instances that never
 * enqueue (see there) and are dropped. The result is empty only for a pipeline with no commands
 * that enqueue, which cudaProcess refuses.
 */
inline std::vector<std::int32_t> unique_ascending_streams(const std::vector<std::int32_t>& ids) {
    std::set<std::int32_t> uniq;
    for (const std::int32_t id : ids)
        if (id >= 0)
            uniq.insert(id);
    return std::vector<std::int32_t>(uniq.begin(), uniq.end());
}

/**
 * @brief Take one queuing lock per stream, in ascending stream order, and hand them back.
 *
 * @param streams   the streams a pipeline enqueues onto, in any order; duplicates tolerated.
 * @param mutex_of  maps a stream id to its `std::recursive_mutex&`.
 * @return the held locks; releasing them is the caller's scope ending.
 *
 * This is the one place the lock order is imposed. Several cudaProcess stages on one device
 * each hold several of these at once, and multiple locks can only deadlock when two acquirers
 * disagree about the order, so every acquirer takes them ascending by stream id. The caller's
 * order is deliberately not trusted: the set is sorted here on every call (a handful of ints),
 * so nothing upstream of this function can reintroduce a cycle.
 *
 * Pure with respect to CUDA: the test drives it with host mutexes from concurrent threads.
 */
template<class MutexOf>
inline std::vector<std::unique_lock<std::recursive_mutex>>
lock_streams_ascending(const std::vector<std::int32_t>& streams, MutexOf&& mutex_of) {
    std::vector<std::int32_t> sorted(streams);
    std::sort(sorted.begin(), sorted.end());
    std::vector<std::unique_lock<std::recursive_mutex>> locks;
    locks.reserve(sorted.size());
    for (const std::int32_t sid : sorted)
        locks.emplace_back(mutex_of(sid));
    return locks;
}

#endif // CUDA_STREAM_ASSIGNMENT_HPP
