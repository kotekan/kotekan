#include <Config.hpp>
#include <Stage.hpp>
#include <StageFactory.hpp>
#include <cassert>
#include <chimeMetadata.hpp>
#include <cstdint>
#include <fstream>
#include <julia.h>
#include <juliaManager.hpp>
#include <signal.h>
#include <string>

class juliaDataGen : public kotekan::Stage {
    const std::string unique_name;

    const int num_frames;
    const int samples_per_data_set;

    Buffer* const buf;

public:
    juliaDataGen(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_conainer);
    virtual ~juliaDataGen();
    void main_thread() override;
};

REGISTER_KOTEKAN_STAGE(juliaDataGen);

juliaDataGen::juliaDataGen(kotekan::Config& config, const std::string& unique_name,
                           kotekan::bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          [](const kotekan::Stage& stage) {
              return const_cast<kotekan::Stage&>(stage).main_thread();
          }),
    unique_name(unique_name), num_frames(config.get_default<int>(unique_name, "num_frames", 1)),
    samples_per_data_set(config.get_default<int>(unique_name, "samples_per_data_set", 32768)),
    buf(get_buffer("out_buf")) {
    assert(buf);
    register_producer(buf, unique_name.c_str());

    INFO("Starting Julia...");
    juliaStartup();

    INFO("Defining Julia code...");
    {
        std::ifstream file("lib/stages/juliaDataGen.jl");
        file.seekg(0, std::ios_base::end);
        const auto julia_source_length = file.tellg();
        file.seekg(0);
        std::vector<char> julia_source(julia_source_length);
        file.read(julia_source.data(), julia_source_length);
        file.close();
        juliaCall([&]() {
            jl_value_t* const res = jl_eval_string(julia_source.data());
            assert(res);
        });
    }
}

juliaDataGen::~juliaDataGen() {
    INFO("Shutting down Julia...");
    juliaShutdown();
    INFO("Done.");
}

void juliaDataGen::main_thread() {
    const int buffer_depth = buf->num_frames;

    for (int frame_index = 0; frame_index < num_frames; ++frame_index) {
        INFO("[{:d}] Getting buffer...", frame_index);
        const int frame_id = frame_index % buffer_depth;
        const std::uint64_t seq_num = std::uint64_t(1) * samples_per_data_set * frame_index;

        std::uint8_t* const frame =
            static_cast<uint8_t*>(wait_for_empty_frame(buf, unique_name.c_str(), frame_id));
        if (!frame)
            break;
        const std::size_t frame_size = buf->frame_size; // bytes
        assert(frame_size % sizeof *frame == 0);
        const std::size_t num_elements = frame_size / sizeof *frame;
        assert(std::ptrdiff_t(num_elements) == samples_per_data_set);

        allocate_new_metadata_object(buf, frame_id);
        set_fpga_seq_num(buf, frame_id, seq_num);

        INFO("[{:d}] Filling buffer...", frame_index);
        juliaCall([&]() {
            jl_module_t* const datagen_module =
                (jl_module_t*)jl_get_global(jl_main_module, jl_symbol("DataGen"));
            assert(datagen_module);
            jl_function_t* const fill_buffer = jl_get_function(datagen_module, "fill_buffer");
            assert(fill_buffer);
            jl_value_t* arg_ptr = nullptr;
            jl_value_t* arg_sz = nullptr;
            JL_GC_PUSH2(&arg_ptr, &arg_sz);
            arg_ptr = jl_box_uint8pointer(frame);
            arg_sz = jl_box_int64(num_elements);
            jl_value_t* const res = jl_call2(fill_buffer, arg_ptr, arg_sz);
            assert(res);
            JL_GC_POP();
        });
        INFO("[{:d}] Done filling buffer.", frame_index);

        mark_frame_full(buf, unique_name.c_str(), frame_id);
    }

    INFO("Exiting.");
    // `exit_kotekan` sents `SIGINT`, and Julia swallows this signal. Thus raise `SIGHUP`.
    exit_kotekan(CLEAN_EXIT);
    raise(SIGHUP);
}
