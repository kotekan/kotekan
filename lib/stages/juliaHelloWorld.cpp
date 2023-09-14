#include <Config.hpp>
#include <Stage.hpp>
#include <StageFactory.hpp>
#include <cassert>
#include <julia.h>
#include <juliaManager.hpp>
#include <string>

class juliaHelloWorld : public kotekan::Stage {
public:
    juliaHelloWorld(kotekan::Config& config, const std::string& unique_name,
                    kotekan::bufferContainer& buffer_container);
    ~juliaHelloWorld();
    void main_thread() override;
};

REGISTER_KOTEKAN_STAGE(juliaHelloWorld);

juliaHelloWorld::juliaHelloWorld(kotekan::Config& config, const std::string& unique_name,
                                 kotekan::bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, [](const kotekan::Stage& stage) {
        return const_cast<kotekan::Stage&>(stage).main_thread();
    }) {
    INFO("juliaHelloWorld: Starting Julia...");
    juliaStartup();
}

juliaHelloWorld::~juliaHelloWorld() {
    INFO("juliaHelloWorld: Shutting down Julia...");
    juliaShutdown();
    INFO("juliaHelloWorld: Done.");
}

void juliaHelloWorld::main_thread() {
    INFO("juliaHelloWorld: Calling Julia...");
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
    INFO("juliaHelloWorld: Done calling Julia.");

    INFO("juliaHelloWorld: Exiting.");
    exit_kotekan(CLEAN_EXIT);
}
