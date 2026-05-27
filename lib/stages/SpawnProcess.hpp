/**
 * @file
 * @brief Stage that launches and supervises an external child process.
 *  - SpawnProcess : public kotekan::Stage
 */

#ifndef SPAWN_PROCESS_HPP
#define SPAWN_PROCESS_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <string>      // for string
#include <sys/types.h> // for pid_t

/**
 * @class SpawnProcess
 * @brief Kotekan stage that runs an external command for the lifetime of the pipeline.
 *
 * On startup the stage @c fork s and @c execv s @c "/bin/sh -c <exec>" so the
 * child can be an inline shell line. The child closes every inherited file
 * descriptor above stdio before exec, so it cannot pin kotekan's resources (in
 * particular the airspy USB device fd) open for its lifetime. The stage then
 * drains @c in_buf (the buffer it is registered as a consumer of) so the
 * upstream pipeline never blocks -- and keeps draining even if the spawn
 * failed, since @c in_buf may be multi-consumer.
 *
 * At shutdown the stage supervises the child to completion: it reaps it if it
 * already exited, otherwise sends @c SIGINT, waits ~5s, then escalates to
 * @c SIGKILL and blocks until reaped, so no zombie or runaway child survives
 * teardown. The destructor repeats this as an idempotent backstop.
 *
 * The buffer consumption side-effect is what keeps the stage alive while the
 * child runs; @c SpawnProcess does no actual processing of the frames it sees.
 *
 * @par Buffers
 * @buffer in_buf Buffer whose frames are drained while the child runs.
 *     @buffer_format any (frames are not inspected)
 *     @buffer_metadata none
 *
 * @conf   exec   String (default ""). Shell command to run.
 *
 * @author Keith Vanderlinde
 */
class SpawnProcess : public kotekan::Stage {
public:
    SpawnProcess(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);
    ~SpawnProcess() override;

    void main_thread() override;

private:
    /// Reap/terminate the child if one is running. Idempotent: clears
    /// @c child_pid up front so the dtor backstop is a no-op after the
    /// thread already ran it.
    void terminate_child();

    Buffer* buf;
    std::string exec_cmd;
    pid_t child_pid = -1;
};

#endif
