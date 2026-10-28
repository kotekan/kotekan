#include "SpawnProcess.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for INFO, ERROR, WARN

#include <errno.h>    // for errno, ECHILD
#include <functional> // for bind
#include <signal.h>   // for kill, SIGINT, SIGKILL
#include <string.h>   // for strerror
#include <sys/wait.h> // for waitpid, WNOHANG
#include <unistd.h>   // for fork, execv, close, _exit, sysconf, usleep, setpgid, getpgid

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(SpawnProcess);

SpawnProcess::SpawnProcess(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&SpawnProcess::main_thread, this)) {

    buf = get_buffer("in_buf");
    buf->register_consumer(unique_name);

    exec_cmd = config.get_default<std::string>(unique_name, "exec", "");
    INFO("SpawnProcess: {:s}", exec_cmd);
}

SpawnProcess::~SpawnProcess() {
    // Backstop in case main_thread() never reached its own terminate_child()
    // (e.g. the object is torn down without a clean run). The thread is joined
    // before the stage is destroyed, so this is sequential with main_thread's
    // call; terminate_child() is idempotent regardless.
    terminate_child();
}

void SpawnProcess::main_thread() {
    INFO("Spawning: {:s}", exec_cmd);

    // We fork+exec rather than posix_spawn so the child can close every
    // inherited file descriptor before exec. Otherwise the child inherits *all*
    // of kotekan's open fds -- including the airspy USB device fd -- and a
    // long-lived helper (the python viewer) keeps the device pinned open for
    // its whole lifetime, which outlives airspy_close() and can wedge the
    // dongle until a physical replug.
    //
    // Build argv for `/bin/sh -c <cmd>` so `exec` can be an inline shell line.
    // execv wants `char* const[]`, so drop const-ness on the string literals.
    char arg0[] = "/bin/sh";
    char arg1[] = "-c";
    char* argv[] = {arg0, arg1, const_cast<char*>(exec_cmd.c_str()), nullptr};

    long maxfd = sysconf(_SC_OPEN_MAX);
    if (maxfd <= 0 || maxfd > 65536)
        maxfd = 65536; // fall back to a sane cap if the limit is unknown/absurd

    pid_t pid = fork();
    if (pid == 0) {
        // CHILD. Between fork and exec only async-signal-safe calls are legal
        // (no logging, no allocation). Close every fd above stdio so none of
        // the parent's handles (USB device, sockets, buffer mmaps) leak into
        // the child, then exec. execv keeps the current environment. _exit(127)
        // mirrors the shell's "command not found" status if execv fails.
        //
        // Put the child in its own process group so terminate_child() can signal
        // the whole tree. `sh -c` only exec-replaces itself when the command is
        // simple enough; an exec line needing expansion (as the airspy configs'
        // `${VIEWER_PYTHON:-python3} ...` does) makes it fork instead, and dash
        // defers SIGINT while waiting on a foreground child. Signalling the sh pid
        // alone then does nothing at all: the grace period elapses in full, SIGKILL
        // reaps the shell only, and the real command is orphaned still holding its
        // ports. setpgid is async-signal-safe, so it is legal here.
        setpgid(0, 0);
        for (int fd = 3; fd < static_cast<int>(maxfd); fd++)
            close(fd);
        execv("/bin/sh", argv);
        _exit(127);
    }

    if (pid < 0) {
        // Fork failed. Do NOT return: in_buf may be multi-consumer (the
        // crosscorr post_corr_buf also feeds networkPowerStream), and a frame
        // recycles only once *every* consumer marks it empty. Bailing here
        // would back up and stall the upstream pipeline. So log and fall through
        // to the drain loop with no child to supervise.
        ERROR("SpawnProcess: fork failed: {:s}; pipeline continues without the child process",
              strerror(errno));
    } else {
        // Also set the group from this side: whichever of parent and child runs
        // first wins and the other's call is a harmless no-op (or EACCES once the
        // child has exec'd). Doing it in both places removes the race where we
        // could signal before the child's own setpgid had taken effect.
        setpgid(pid, pid);
        child_pid = pid;
        INFO("Spawned PID {:d}", pid);
    }

    int frame_id = 0;
    while (!stop_thread) {
        // Drain the buffer; we don't inspect frames.
        void* frame = buf->wait_for_full_frame(unique_name, frame_id);
        if (frame == nullptr)
            break;
        buf->mark_frame_empty(unique_name, frame_id);
        frame_id = (frame_id + 1) % buf->num_frames;
    }

    terminate_child();
}

void SpawnProcess::terminate_child() {
    if (child_pid <= 0)
        return;
    // Take ownership of the pid and clear the member up front so the dtor
    // backstop can't act on it a second time.
    pid_t pid = child_pid;
    child_pid = -1;

    int status;
    // Already exited on its own? Reap it without signalling. ECHILD means it was
    // never ours / already reaped -- nothing to do either way.
    pid_t r = waitpid(pid, &status, WNOHANG);
    if (r == pid || (r == -1 && errno == ECHILD))
        return;

    // Signal the child's whole process group, so a `sh -c` that forked rather than
    // exec-replaced itself takes the real command down with it. Only do that once
    // the child is confirmed to lead its own group: if setpgid failed on both
    // sides, `-pid` would name no group and the signal would go nowhere, leaving
    // the final waitpid() below to block forever. Fall back to the bare pid there.
    const pid_t pgid = getpgid(pid);
    const pid_t target = (pgid == pid) ? -pid : pid;

    // Ask politely, then wait up to ~5s for it to go.
    kill(target, SIGINT);
    for (int i = 0; i < 50; i++) {
        r = waitpid(pid, &status, WNOHANG);
        if (r == pid || (r == -1 && errno == ECHILD))
            return;
        usleep(100000); // 100 ms
    }

    // Still alive: escalate and block until reaped so we never leak a zombie or
    // leave a runaway child behind after teardown.
    WARN("SpawnProcess: PID {:d} did not exit on SIGINT; sending SIGKILL", pid);
    kill(target, SIGKILL);
    waitpid(pid, &status, 0);
}
