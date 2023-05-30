#include <Config.hpp>
#include <Stage.hpp>
#include <StageFactory.hpp>
#include <Telescope.hpp>
#include <buffer.h>
#include <bufferContainer.hpp>
#include <cassert>
#include <chimeMetadata.hpp>
#include <chrono>
#include <cstdint>
#include <cudaCommand.hpp>
#include <julia.h>
#include <juliaManager.hpp>
#include <string>
#include <thread>

class juliaDataGen : public kotekan::Stage {
    const std::string unique_name;

    int samples_per_data_set;
    int num_frames;
    std::uint64_t first_frame_index;

    stream_t stream_id;

    Buffer* buf;

public:
    juliaDataGen(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_conainer);
    ~juliaDataGen();
    void main_thread() override;
};

REGISTER_KOTEKAN_STAGE(juliaDataGen);

juliaDataGen::juliaDataGen(kotekan::Config& config, const std::string& unique_name,
                           kotekan::bufferContainer& buffer_container) :
    Stage(
        config, unique_name, buffer_container,
        // std::function<void(const Stage&)> main_thread_ref
        [](const Stage& stage) {
            // The passed-in `Stage` object isn't really `const`
            const_cast<juliaDataGen*>(reinterpret_cast<const juliaDataGen*>(&stage))->main_thread();
        }),
    unique_name(unique_name) {

    INFO("Defining Julia code...");
    juliaCall([&]() {
        INFO("This is thread {:d}", std::hash<std::thread::id>()(std::this_thread::get_id()));
        (void)jl_eval_string("# module DataGen\n"
                             "\n"
                             "function fill_buffer(ptr::Ptr{UInt8}, sz::Int64)\n"
                             "    for i in 1:sz\n"
                             "        unsafe_store!(p, i % UInt8, i)\n"
                             "    end\n"
                             "end\n"
                             "\n"
                             "println(\"DataGen\")"
                             "\n"
                             "# end\n");
    });
    INFO("Done defining Julia code.");

    samples_per_data_set = config.get_default<int>(unique_name, "samples_per_data_set", 32768);
    num_frames = config.get_default<int>(unique_name, "num_frames", 1);
    first_frame_index = config.get_default<std::uint64_t>(unique_name, "first_frame_index", 0);

    stream_id.id = config.get_default<uint64_t>(unique_name, "stream_id", 0);

    buf = get_buffer("out_buf");
    assert(buf);
    register_producer(buf, unique_name.c_str());
}

juliaDataGen::~juliaDataGen() {}

void juliaDataGen::main_thread() {
    using namespace std::chrono_literals;

    for (int frame_id = 0; frame_id < num_frames; ++frame_id) {
        INFO("DataGen: frame_id={:d}", frame_id);
        const std::uint64_t abs_frame_index = first_frame_index + frame_id;
        const std::uint64_t seq_num = samples_per_data_set * abs_frame_index;

        std::uint8_t* const frame =
            static_cast<uint8_t*>(wait_for_empty_frame(buf, unique_name.c_str(), frame_id));
        if (!frame)
            break;

        allocate_new_metadata_object(buf, frame_id);
        set_fpga_seq_num(buf, frame_id, seq_num);
        set_stream_id(buf, frame_id, stream_id);

        const std::size_t frame_size = buf->frame_size; // bytes
        const std::size_t num_elements = frame_size / sizeof *frame;

        INFO("Calling Julia...");
        juliaCall([&]() {
            INFO("This is thread {:d}", std::hash<std::thread::id>()(std::this_thread::get_id()));
            (void)jl_eval_string("println(\"DataGen.2\")");
            (void)jl_eval_string("println(DataGen)");
            (void)jl_eval_string("println(DataGen.fill_buffer)");

            // jl_function_t* const func = jl_get_function(jl_main_module, "DataGen.fill_buffer");
            jl_function_t* const func = jl_get_function(jl_main_module, "fill_buffer");
            assert(func);
            jl_value_t* const arg_ptr = jl_box_uint8pointer(frame);
            assert(arg_ptr);
            jl_value_t* const arg_sz = jl_box_int64(num_elements);
            assert(arg_sz);
            jl_value_t* args[] = {arg_ptr, arg_sz};
            (void)jl_call(func, args, sizeof args / sizeof *args);
        });
        INFO("Done calling Julia.");

        mark_frame_full(buf, unique_name.c_str(), frame_id);

        // Why wait...
        // std::this_thread::sleep_for(1000us);
    }
}
