#include <Config.hpp>
#include <cudaCommand.hpp>
#include <julia.h>
#include <juliaManager.hpp>
#include <thread>

class juliaHelloWorld : public cudaCommand {
public:
    juliaHelloWorld(kotekan::Config& config, const std::string& unique_name,
                    kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device);
    ~juliaHelloWorld();
    cudaEvent_t execute(int gpu_frame_id, const std::vector<cudaEvent_t>& pre_events,
                        bool* quit) override;
};

REGISTER_CUDA_COMMAND(juliaHelloWorld);

juliaHelloWorld::juliaHelloWorld(kotekan::Config& config, const std::string& unique_name,
                                 kotekan::bufferContainer& host_buffers,
                                 cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "juliaHelloWorld") {
    set_command_type(gpuCommandType::KERNEL);
}

juliaHelloWorld::~juliaHelloWorld() {}

cudaEvent_t juliaHelloWorld::execute(const int gpu_frame_id,
                                     const std::vector<cudaEvent_t>& /*pre_events*/,
                                     bool* const /*quit*/) {
    pre_execute(gpu_frame_id);
    record_start_event(gpu_frame_id);

    juliaInit();
    static const std::thread::id julia_thread = std::this_thread::get_id();
    const std::thread::id current_thread = std::this_thread::get_id();
    if (current_thread != julia_thread)
        FATAL_ERROR("Trying to call Julia from the wrong thread");
    INFO("Calling Julia...");
    (void)jl_eval_string("println(\"Hello, World!\")");
    INFO("Done.");

    return record_end_event(gpu_frame_id);
}
