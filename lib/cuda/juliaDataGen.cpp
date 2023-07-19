#include <Config.hpp>
#include <Stage.hpp>
#include <StageFactory.hpp>
#include <Telescope.hpp>
#include <buffer.h>
#include <bufferContainer.hpp>
#include <cassert>
#include <chimeMetadata.hpp>
#include <cstdint>
#include <cudaCommand.hpp>
#include <fstream>
#include <iostream>
#include <julia.h>
#include <juliaManager.hpp>
#include <string>
#include <vector>

class juliaDataGen : public kotekan::Stage {
    const std::string unique_name;

    int samples_per_data_set;
    int num_frames;
    std::uint64_t first_frame_index;

    int buffer_depth;
    stream_t stream_id;

    Buffer* buf;

public:
    juliaDataGen(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_conainer);
    virtual ~juliaDataGen();
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

    {
        INFO("juliaDataGen: Defining Julia code...");
        std::ifstream file("lib/cuda/juliaDataGen.jl");
        file.seekg(0, std::ios_base::end);
        const auto julia_source_length = file.tellg();
        file.seekg(0);
        std::vector<char> julia_source(julia_source_length);
        file.read(julia_source.data(), julia_source_length);
        file.close();
        juliaCall([&]() { (void)jl_eval_string(julia_source.data()); });
        INFO("juliaDataGen: Done.");
    }

    samples_per_data_set = config.get_default<int>(unique_name, "samples_per_data_set", 32768);
    num_frames = config.get_default<int>(unique_name, "num_frames", 1);
    first_frame_index = config.get_default<std::uint64_t>(unique_name, "first_frame_index", 0);

    buffer_depth = config.get_default<uint64_t>(unique_name, "buffer_depth", 1);
    stream_id.id = config.get_default<uint64_t>(unique_name, "stream_id", 0);

    buf = get_buffer("out_buf");
    assert(buf);
    register_producer(buf, unique_name.c_str());
}

juliaDataGen::~juliaDataGen() {
    INFO("juliaDataGen: Shutting down Julia...");
    juliaShutdown();
    INFO("juliaDataGen: Done.");
}

void juliaDataGen::main_thread() {
    for (std::uint64_t frame_index = first_frame_index;
         frame_index < first_frame_index + num_frames; ++frame_index) {
        INFO("juliaDataGen: frame_index={:d}", frame_index);
        const int frame_id = frame_index % buffer_depth;
        const std::uint64_t seq_num = samples_per_data_set * frame_index;

        std::uint8_t* const frame =
            static_cast<uint8_t*>(wait_for_empty_frame(buf, unique_name.c_str(), frame_id));
        if (!frame)
            break;

        allocate_new_metadata_object(buf, frame_id);
        set_fpga_seq_num(buf, frame_id, seq_num);
        set_stream_id(buf, frame_id, stream_id);

        const std::size_t frame_size = buf->frame_size; // bytes
        const std::size_t num_elements = frame_size / sizeof *frame;

        INFO("juliaDataGen: frame_index={:d} Filling buffer...", frame_index);
        juliaCall([&]() {
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
        INFO("juliaDataGen: frame_index={:d} Done.", frame_index);

        mark_frame_full(buf, unique_name.c_str(), frame_id);
    }

    INFO("juliaDataGen: Exiting.");
}
