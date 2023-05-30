#include <Config.hpp>
#include <cassert>
#include <cmath>
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
    static int count = 0;
    if (++count == 10)
        std::terminate();

    pre_execute(gpu_frame_id);
    record_start_event(gpu_frame_id);

    void* const mem = device.get_gpu_memory_array("voltage", gpu_frame_id, 32768);
    assert(mem);

    INFO("[HelloWorld: gpu_frame_id={:d}: Calling Julia...", gpu_frame_id);
    const double retval = juliaCall([&]() {
        (void)jl_eval_string("println(\"Hello, World!\")");
        jl_function_t* const func = jl_get_function(jl_base_module, "sqrt");
        jl_value_t* const arg = jl_box_float64(2.0);
        // jl_value_t* ret = jl_call1(func, argument);
        jl_value_t* args[] = {arg};
        jl_value_t* const ret = jl_call(func, args, sizeof args / sizeof *args);
        assert(jl_typeis(ret, jl_float64_type));
        const double retval = jl_unbox_float64(ret);
        return retval;
    });
    using std::sqrt;
    assert(retval == sqrt(2.0));
    INFO("Done.");

    return record_end_event(gpu_frame_id);
}
