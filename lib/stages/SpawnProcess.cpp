#include "SpawnProcess.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"          // for Buffer, mark_frame_empty, mark_frame_full, register_consumer
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for DEBUG

#include <atomic>      // for atomic_bool
#include <exception>   // for exception
#include <functional>  // for _Bind_helper<>::type, bind, function
#include <regex>       // for match_results<>::_Base_type
#include <stdexcept>   // for runtime_error
#include <stdint.h>    // for uint32_t
#include <stdlib.h>    // for calloc, free
#include <string.h>    // for memset
#include <sys/types.h> // for uint
#include <vector>      // for vector
#include <spawn.h>     // for spawn
#include <signal.h>    // for kill

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(SpawnProcess);

SpawnProcess::SpawnProcess(Config& config, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&SpawnProcess::main_thread, this)) {

    buf = get_buffer("in_buf");
    register_consumer(buf, unique_name.c_str());

//    exec_cmd = std::string("\"cd ")+config.get_default<std::string>(unique_name, "path", "./");
//    exec_cmd += std::string(" && ");
    exec_cmd = config.get_default<std::string>(unique_name, "exec", "");// + "\"";
    INFO("SpawnProcess: {}",exec_cmd);
}

SpawnProcess::~SpawnProcess() {
}

void SpawnProcess::main_thread() {
    pid_t pid;
    extern char **environ;
    char *cmd_cstr = const_cast<char*>(exec_cmd.c_str() );
    INFO("Executable: {}",cmd_cstr);
    char *v[] = {"/bin/sh", "-c", cmd_cstr, NULL};
    int status = posix_spawn(&pid, "/bin/sh", NULL, NULL, v, environ);
    INFO("Status: {}",status);
    if (status > 0) perror("ERROR:");

    int frame_id = 0;

    while (true){
        // This call is blocking.
        void *frame = wait_for_full_frame(buf, unique_name.c_str(), frame_id);
        if (frame == nullptr)
            break;
        mark_frame_empty(buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % buf->num_frames;
    }

    kill(pid,SIGINT);
}
