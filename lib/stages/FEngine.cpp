#include <Config.hpp>
#include <Stage.hpp>
#include <StageFactory.hpp>
#include <cassert>
#include <chimeMetadata.hpp>
#include <chordMetadata.hpp>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <julia.h>
#include <juliaManager.hpp>
#include <string>

class FEngine : public kotekan::Stage {
    const std::string unique_name;

    const int num_frames;
    const int samples_per_data_set;

    const int E_frame_size;
    const int A_frame_size;
    const int J_frame_size;

    Buffer* const E_buffer;
    Buffer* const A_buffer;
    Buffer* const J_buffer;

public:
    FEngine(kotekan::Config& config, const std::string& unique_name,
            kotekan::bufferContainer& buffer_conainer);
    virtual ~FEngine();
    void main_thread() override;
};

REGISTER_KOTEKAN_STAGE(FEngine);

FEngine::FEngine(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          [](const kotekan::Stage& stage) {
              return const_cast<kotekan::Stage&>(stage).main_thread();
          }),
    unique_name(unique_name), num_frames(config.get_default<int>(unique_name, "num_frames", 1)),
    samples_per_data_set(config.get<int>(unique_name, "samples_per_data_set")),
    E_frame_size(config.get<int>(unique_name, "E_frame_size")),
    A_frame_size(config.get<int>(unique_name, "A_frame_size")),
    J_frame_size(config.get<int>(unique_name, "J_frame_size")), E_buffer(get_buffer("E_buffer")),
    A_buffer(get_buffer("A_buffer")), J_buffer(get_buffer("J_buffer")) {
    assert(E_buffer);
    assert(A_buffer);
    assert(J_buffer);
    register_producer(E_buffer, unique_name.c_str());
    register_producer(A_buffer, unique_name.c_str());
    register_producer(J_buffer, unique_name.c_str());

    INFO("Starting Julia...");
    kotekan::juliaStartup();

    INFO("Defining Julia code...");
    {
        std::ifstream file("lib/stages/FEngine.jl");
        file.seekg(0, std::ios_base::end);
        const auto julia_source_length = file.tellg();
        file.seekg(0);
        std::vector<char> julia_source(julia_source_length);
        file.read(julia_source.data(), julia_source_length);
        file.close();
        kotekan::juliaCall([&]() {
            jl_value_t* const res = jl_eval_string(julia_source.data());
            assert(res);
        });
    }
}

FEngine::~FEngine() {
    INFO("Shutting down Julia...");
    kotekan::juliaShutdown();
    INFO("Done.");
}

void FEngine::main_thread() {
    static bool stale = false;
    assert(!stale);
    stale = true;

    INFO("Initializing F-Engine...");
    kotekan::juliaCall([&]() {
        jl_module_t* const f_engine_module =
            (jl_module_t*)jl_get_global(jl_main_module, jl_symbol("FEngine"));
        assert(f_engine_module);
        jl_function_t* const setup = jl_get_function(f_engine_module, "setup");
        assert(setup);
        const int nargs = 12;
        jl_value_t** args;
        JL_GC_PUSHARGS(args, nargs);
        args[0] /* source_amplitude */ = jl_box_float32(1.0);
        args[1] /* source_frequency */ = jl_box_float32(0.3e+9);
        args[2] /* source_position_sinx */ = jl_box_float32(0.02);
        args[3] /* source_position_siny */ = jl_box_float32(0.03);
        args[4] /* dish_separation_x */ = jl_box_float32(6.3);
        args[5] /* dish_separation_y */ = jl_box_float32(8.5);
        args[6] /* ndishes_i */ = jl_box_int64(32);
        args[7] /* ndishes_j */ = jl_box_int64(16);
        args[8] /* adc_frequency */ = jl_box_float32(3.0e+9);
        args[9] /* ntaps */ = jl_box_int64(4);
        args[10] /* nfreq */ = jl_box_int64(16);
        args[11] /* ntimes */ = jl_box_int64(32768);
        jl_value_t* const res = jl_call(setup, args, nargs);
        assert(res);
        JL_GC_POP();
    });
    INFO("Done initializing world.");

    for (int frame_index = 0; frame_index < num_frames; ++frame_index) {
        if (stop_thread)
            break;

        INFO("[{:d}] Getting buffers...", frame_index);
        const std::uint64_t seq_num = std::uint64_t(1) * samples_per_data_set * frame_index;

        const int A_frame_id = frame_index % A_buffer->num_frames;
        std::uint8_t* const A_frame =
            wait_for_empty_frame(A_buffer, unique_name.c_str(), A_frame_id);
        if (!A_frame)
            break;
        assert(std::ptrdiff_t(A_buffer->frame_size) == A_frame_size);
        allocate_new_metadata_object(A_buffer, A_frame_id);
        set_fpga_seq_num(A_buffer, A_frame_id, seq_num);

        const int E_frame_id = frame_index % E_buffer->num_frames;
        std::uint8_t* const E_frame =
            wait_for_empty_frame(E_buffer, unique_name.c_str(), E_frame_id);
        if (!E_frame)
            break;
        assert(std::ptrdiff_t(E_buffer->frame_size) == E_frame_size);
        allocate_new_metadata_object(E_buffer, E_frame_id);
        set_fpga_seq_num(E_buffer, E_frame_id, seq_num);

        const int J_frame_id = frame_index % J_buffer->num_frames;
        std::uint8_t* const J_frame =
            wait_for_empty_frame(J_buffer, unique_name.c_str(), J_frame_id);
        if (!J_frame)
            break;
        assert(std::ptrdiff_t(J_buffer->frame_size) == J_frame_size);
        allocate_new_metadata_object(J_buffer, J_frame_id);
        set_fpga_seq_num(J_buffer, J_frame_id, seq_num);

        INFO("[{:d}] Filling E buffer...", frame_index);
        kotekan::juliaCall([&]() {
            jl_module_t* const f_engine_module =
                (jl_module_t*)jl_get_global(jl_main_module, jl_symbol("FEngine"));
            assert(f_engine_module);
            jl_function_t* const set_E = jl_get_function(f_engine_module, "set_E");
            assert(set_E);
            jl_value_t* arg_ptr = nullptr;
            jl_value_t* arg_sz = nullptr;
            jl_value_t* arg_ndishes = nullptr;
            jl_value_t* arg_nfreqs = nullptr;
            jl_value_t* arg_npolrs = nullptr;
            jl_value_t* arg_ntimes = nullptr;
            jl_value_t* arg_index = nullptr;
            JL_GC_PUSH7(&arg_ptr, &arg_sz, &arg_ndishes, &arg_nfreqs, &arg_npolrs, &arg_ntimes,
                        &arg_index);
            arg_ptr = jl_box_uint8pointer(E_frame);
            arg_sz = jl_box_int64(E_frame_size);
            // TODO: Take these from the E metadata
            arg_ndishes = jl_box_int64(512);
            arg_nfreqs = jl_box_int64(16);
            arg_npolrs = jl_box_int64(2);
            arg_ntimes = jl_box_int64(32768);
            arg_index = jl_box_int64(frame_index + 1);
            const int nargs = 7;
            jl_value_t* args[nargs] = {
                arg_ptr, arg_sz, arg_ndishes, arg_nfreqs, arg_npolrs, arg_ntimes, arg_index,
            };
            jl_value_t* const res = jl_call(set_E, args, nargs);
            assert(res);
            JL_GC_POP();
        });
        INFO("[{:d}] Done filling E buffer.", frame_index);

        INFO("[{:d}] Filling A buffer...", frame_index);
        kotekan::juliaCall([&]() {
            jl_module_t* const f_engine_module =
                (jl_module_t*)jl_get_global(jl_main_module, jl_symbol("FEngine"));
            assert(f_engine_module);
            jl_function_t* const set_A = jl_get_function(f_engine_module, "set_A");
            assert(set_A);
            jl_value_t* arg_ptr = nullptr;
            jl_value_t* arg_sz = nullptr;
            jl_value_t* arg_ndishes = nullptr;
            jl_value_t* arg_nbeams = nullptr;
            jl_value_t* arg_npolrs = nullptr;
            jl_value_t* arg_nfreqs = nullptr;
            jl_value_t* arg_index = nullptr;
            JL_GC_PUSH7(&arg_ptr, &arg_sz, &arg_ndishes, &arg_nbeams, &arg_npolrs, &arg_nfreqs,
                        &arg_index);
            arg_ptr = jl_box_uint8pointer(A_frame);
            arg_sz = jl_box_int64(A_frame_size);
            // TODO: Take these from the A metadata
            arg_ndishes = jl_box_int64(512);
            arg_nbeams = jl_box_int64(96);
            arg_npolrs = jl_box_int64(2);
            arg_nfreqs = jl_box_int64(16);
            arg_index = jl_box_int64(frame_index + 1);
            const int nargs = 7;
            jl_value_t* args[nargs] = {
                arg_ptr, arg_sz, arg_ndishes, arg_nbeams, arg_npolrs, arg_nfreqs, arg_index,
            };
            jl_value_t* const res = jl_call(set_A, args, nargs);
            assert(res);
            JL_GC_POP();
        });
        INFO("[{:d}] Done filling A buffer.", frame_index);

        INFO("[{:d}] Filling J buffer...", frame_index);
        kotekan::juliaCall([&]() {
            jl_module_t* const f_engine_module =
                (jl_module_t*)jl_get_global(jl_main_module, jl_symbol("FEngine"));
            assert(f_engine_module);
            jl_function_t* const set_J = jl_get_function(f_engine_module, "set_J");
            assert(set_J);
            jl_value_t* arg_ptr = nullptr;
            jl_value_t* arg_sz = nullptr;
            jl_value_t* arg_ntimes = nullptr;
            jl_value_t* arg_npolrs = nullptr;
            jl_value_t* arg_nfreqs = nullptr;
            jl_value_t* arg_nbeams = nullptr;
            jl_value_t* arg_index = nullptr;
            JL_GC_PUSH7(&arg_ptr, &arg_sz, &arg_ntimes, &arg_npolrs, &arg_nfreqs, &arg_nbeams,
                        &arg_index);
            arg_ptr = jl_box_uint8pointer(J_frame);
            arg_sz = jl_box_int64(J_frame_size);
            // TODO: Take these from the J metadata
            arg_ntimes = jl_box_int64(32768);
            arg_npolrs = jl_box_int64(2);
            arg_nfreqs = jl_box_int64(16);
            arg_nbeams = jl_box_int64(96);
            arg_index = jl_box_int64(frame_index + 1);
            const int nargs = 7;
            jl_value_t* args[nargs] = {
                arg_ptr, arg_sz, arg_ntimes, arg_npolrs, arg_nfreqs, arg_nbeams, arg_index,
            };
            jl_value_t* const res = jl_call(set_J, args, nargs);
            assert(res);
            JL_GC_POP();
        });
        INFO("[{:d}] Done filling J buffer.", frame_index);

        // Set metadata
        chordMetadata* const E_metadata = get_chord_metadata(E_buffer, E_frame_id);
        chord_metadata_init(E_metadata);
        E_metadata->chime.fpga_seq_num = frame_index;
        E_metadata->frame_counter = frame_index;
        E_metadata->type = int4p4;
        E_metadata->dims = 4;
        std::strncpy(E_metadata->dim_name[0], "T", sizeof E_metadata->dim_name[0]);
        std::strncpy(E_metadata->dim_name[1], "P", sizeof E_metadata->dim_name[1]);
        std::strncpy(E_metadata->dim_name[2], "F", sizeof E_metadata->dim_name[2]);
        std::strncpy(E_metadata->dim_name[3], "D", sizeof E_metadata->dim_name[3]);
        E_metadata->dim[0] = 32768;
        E_metadata->dim[1] = 2;
        E_metadata->dim[2] = 16;
        E_metadata->dim[3] = 512;
        E_metadata->n_one_hot = -1;
        E_metadata->nfreq = 16;

        chordMetadata* const A_metadata = get_chord_metadata(A_buffer, A_frame_id);
        chord_metadata_init(A_metadata);
        A_metadata->chime.fpga_seq_num = frame_index;
        A_metadata->frame_counter = frame_index;
        A_metadata->type = int8;
        A_metadata->dims = 5;
        std::strncpy(A_metadata->dim_name[0], "F", sizeof A_metadata->dim_name[0]);
        std::strncpy(A_metadata->dim_name[1], "P", sizeof A_metadata->dim_name[1]);
        std::strncpy(A_metadata->dim_name[2], "B", sizeof A_metadata->dim_name[2]);
        std::strncpy(A_metadata->dim_name[3], "D", sizeof A_metadata->dim_name[3]);
        std::strncpy(A_metadata->dim_name[4], "C", sizeof A_metadata->dim_name[4]);
        A_metadata->dim[0] = 16;
        A_metadata->dim[1] = 2;
        A_metadata->dim[2] = 96;
        A_metadata->dim[3] = 512;
        A_metadata->dim[4] = 2;
        A_metadata->n_one_hot = -1;
        A_metadata->nfreq = 16;

        chordMetadata* const J_metadata = get_chord_metadata(J_buffer, J_frame_id);
        chord_metadata_init(J_metadata);
        J_metadata->chime.fpga_seq_num = frame_index;
        J_metadata->frame_counter = frame_index;
        J_metadata->type = int4p4;
        J_metadata->dims = 4;
        std::strncpy(J_metadata->dim_name[0], "B", sizeof J_metadata->dim_name[0]);
        std::strncpy(J_metadata->dim_name[1], "F", sizeof J_metadata->dim_name[1]);
        std::strncpy(J_metadata->dim_name[2], "P", sizeof J_metadata->dim_name[2]);
        std::strncpy(J_metadata->dim_name[3], "T", sizeof J_metadata->dim_name[3]);
        J_metadata->dim[0] = 96;
        J_metadata->dim[1] = 16;
        J_metadata->dim[2] = 2;
        J_metadata->dim[3] = 32768;
        J_metadata->n_one_hot = -1;
        J_metadata->nfreq = 16;

        mark_frame_full(E_buffer, unique_name.c_str(), E_frame_id);
        mark_frame_full(A_buffer, unique_name.c_str(), A_frame_id);
        mark_frame_full(J_buffer, unique_name.c_str(), J_frame_id);
    }

    INFO("Done.");
}
