#include <Config.hpp>
#include <Stage.hpp>
#include <StageFactory.hpp>
#include <cassert>
#include <chimeMetadata.hpp>
#include <chordMetadata.hpp>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <julia.h>
#include <juliaManager.hpp>
#include <string>
#include <vector>

class FEngine : public kotekan::Stage {
    const std::string unique_name;

    // Basic constants
    const int num_components;
    const int num_polarizations;

    // Sky
    const float source_amplitude;
    const float source_frequency;
    const float source_position_x;
    const float source_position_y;

    // Dishes
    const int num_dish_locations_M;
    const int num_dish_locations_N;
    const int num_dish_locations;
    const float dish_separation_x;
    const float dish_separation_y;
    const int num_dishes;
    const std::vector<int> dish_locations;

    // ADC
    const float adc_frequency;
    const int num_taps;
    const int num_frequencies;
    const int num_times;

    // Baseband beamformer setup
    const int bb_num_beams_P;
    const int bb_num_beams_Q;
    const float bb_beam_separation_x;
    const float bb_beam_separation_y;
    const int bb_num_beams;

    // Pipeline
    const int num_frames;

    // Kotekan
    const std::int64_t E_frame_size;
    const std::int64_t A_frame_size;
    const std::int64_t J_frame_size;
    const std::int64_t S_frame_size;
    const std::int64_t W_frame_size;

    Buffer* const E_buffer;
    Buffer* const A_buffer;
    Buffer* const J_buffer;
    Buffer* const S_buffer;
    Buffer* const W_buffer;

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
    unique_name(unique_name),
    // Basic constants
    num_components(config.get<int>(unique_name, "num_components")),
    num_polarizations(config.get<int>(unique_name, "num_polarizations")),
    // Sky
    source_amplitude(config.get<float>(unique_name, "source_amplitude")),
    source_frequency(config.get<float>(unique_name, "source_frequency")),
    source_position_x(config.get<float>(unique_name, "source_position_x")),
    source_position_y(config.get<float>(unique_name, "source_position_y")),
    // Dishes
    num_dish_locations_M(config.get<int>(unique_name, "num_dish_locations_M")),
    num_dish_locations_N(config.get<int>(unique_name, "num_dish_locations_N")),
    num_dish_locations(num_dish_locations_M * num_dish_locations_N),
    dish_separation_x(config.get<float>(unique_name, "dish_separation_x")),
    dish_separation_y(config.get<float>(unique_name, "dish_separation_y")),
    num_dishes(config.get<int>(unique_name, "num_dishes")),
    dish_locations(config.get<std::vector<int>>(unique_name, "dish_locations")),
    // ADC
    adc_frequency(config.get<float>(unique_name, "adc_frequency")),
    num_taps(config.get<int>(unique_name, "num_taps")),
    num_frequencies(config.get<int>(unique_name, "num_frequencies")),
    num_times(config.get<int>(unique_name, "num_times")),
    // Baseband beamformer setup
    bb_num_beams_P(config.get<int>(unique_name, "bb_num_beams_P")),
    bb_num_beams_Q(config.get<int>(unique_name, "bb_num_beams_Q")),
    bb_beam_separation_x(config.get<float>(unique_name, "bb_beam_separation_x")),
    bb_beam_separation_y(config.get<float>(unique_name, "bb_beam_separation_y")),
    bb_num_beams(bb_num_beams_P * bb_num_beams_Q),
    // Pipeline
    num_frames(config.get<int>(unique_name, "num_frames")),
    // Frame sizes
    E_frame_size(std::int64_t(1) * num_dishes * num_frequencies * num_polarizations * num_times),
    A_frame_size(std::int64_t(1) * num_components * num_dishes * bb_num_beams * num_polarizations
                 * num_frequencies),
    J_frame_size(std::int64_t(1) * num_times * num_polarizations * num_frequencies * bb_num_beams),
    S_frame_size(std::int64_t(1) * sizeof(short) * 2 * num_dish_locations),
    W_frame_size(std::int64_t(1) * sizeof(unsigned short) * num_components * num_dish_locations_M
                 * num_dish_locations_N * num_frequencies * num_polarizations),
    // Buffers
    E_buffer(get_buffer("E_buffer")), A_buffer(get_buffer("A_buffer")),
    J_buffer(get_buffer("J_buffer")), S_buffer(get_buffer("S_buffer")),
    W_buffer(get_buffer("W_buffer")) {
    assert(num_dishes <= num_dish_locations);
    assert(std::ptrdiff_t(dish_locations.size()) == 2 * num_dish_locations);

    assert(E_buffer);
    assert(A_buffer);
    assert(J_buffer);
    assert(S_buffer);
    assert(W_buffer);
    register_producer(E_buffer, unique_name.c_str());
    register_producer(A_buffer, unique_name.c_str());
    register_producer(J_buffer, unique_name.c_str());
    register_producer(S_buffer, unique_name.c_str());
    register_producer(W_buffer, unique_name.c_str());

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
        args[0] = jl_box_float32(source_amplitude);
        args[1] = jl_box_float32(source_frequency);
        args[2] = jl_box_float32(source_position_x);
        args[3] = jl_box_float32(source_position_y);
        args[4] = jl_box_float32(dish_separation_x);
        args[5] = jl_box_float32(dish_separation_y);
#warning "TODO: need to pass dish locations"
        args[6] = jl_box_int64(16);
        args[7] = jl_box_int64(32);
        args[8] = jl_box_float32(adc_frequency);
        args[9] = jl_box_int64(num_taps);
        args[10] = jl_box_int64(num_frequencies);
        args[11] = jl_box_int64(num_times);
        jl_value_t* const res = jl_call(setup, args, nargs);
        assert(res);
        JL_GC_POP();
    });
    INFO("Done initializing world.");

    for (int frame_index = 0; frame_index < num_frames; ++frame_index) {
        if (stop_thread)
            break;

        INFO("[{:d}] Getting buffers...", frame_index);
        const std::uint64_t seq_num = std::uint64_t(1) * num_times * frame_index;

        const int A_frame_id = frame_index % A_buffer->num_frames;
        std::uint8_t* const A_frame =
            wait_for_empty_frame(A_buffer, unique_name.c_str(), A_frame_id);
        if (!A_frame)
            break;
        if (!(std::ptrdiff_t(A_buffer->frame_size) == A_frame_size))
            ERROR("A_buffer->frame_size={:d} A_frame_size={:d}", A_buffer->frame_size,
                  A_frame_size);
        allocate_new_metadata_object(A_buffer, A_frame_id);
        set_fpga_seq_num(A_buffer, A_frame_id, seq_num);

        const int E_frame_id = frame_index % E_buffer->num_frames;
        std::uint8_t* const E_frame =
            wait_for_empty_frame(E_buffer, unique_name.c_str(), E_frame_id);
        if (!E_frame)
            break;
        if (!(std::ptrdiff_t(E_buffer->frame_size) == E_frame_size))
            ERROR("E_buffer->frame_size={:d} E_frame_size={:d}", E_buffer->frame_size,
                  E_frame_size);
        allocate_new_metadata_object(E_buffer, E_frame_id);
        set_fpga_seq_num(E_buffer, E_frame_id, seq_num);

        const int J_frame_id = frame_index % J_buffer->num_frames;
        std::uint8_t* const J_frame =
            wait_for_empty_frame(J_buffer, unique_name.c_str(), J_frame_id);
        if (!J_frame)
            break;
        if (!(std::ptrdiff_t(J_buffer->frame_size) == J_frame_size))
            ERROR("J_buffer->frame_size={:d} J_frame_size={:d}", J_buffer->frame_size,
                  J_frame_size);
        allocate_new_metadata_object(J_buffer, J_frame_id);
        set_fpga_seq_num(J_buffer, J_frame_id, seq_num);

        const int S_frame_id = frame_index % S_buffer->num_frames;
        std::uint8_t* const S_frame =
            wait_for_empty_frame(S_buffer, unique_name.c_str(), S_frame_id);
        if (!S_frame)
            break;
        if (!(std::ptrdiff_t(S_buffer->frame_size) == S_frame_size))
            ERROR("S_buffer->frame_size={:d} S_frame_size={:d}", S_buffer->frame_size,
                  S_frame_size);
        allocate_new_metadata_object(S_buffer, S_frame_id);
        set_fpga_seq_num(S_buffer, S_frame_id, seq_num);

        const int W_frame_id = frame_index % W_buffer->num_frames;
        std::uint8_t* const W_frame =
            wait_for_empty_frame(W_buffer, unique_name.c_str(), W_frame_id);
        if (!W_frame)
            break;
        if (!(std::ptrdiff_t(W_buffer->frame_size) == W_frame_size))
            ERROR("W_buffer->frame_size={:d} W_frame_size={:d}", W_buffer->frame_size,
                  W_frame_size);
        allocate_new_metadata_object(W_buffer, W_frame_id);
        set_fpga_seq_num(W_buffer, W_frame_id, seq_num);

        INFO("[{:d}] Filling E buffer...", frame_index);
        kotekan::juliaCall([&]() {
            jl_module_t* const f_engine_module =
                (jl_module_t*)jl_get_global(jl_main_module, jl_symbol("FEngine"));
            assert(f_engine_module);
            jl_function_t* const set_E = jl_get_function(f_engine_module, "set_E");
            assert(set_E);
            const int nargs = 7;
            jl_value_t** args;
            JL_GC_PUSHARGS(args, nargs);
            args[0] = jl_box_uint8pointer(E_frame);
            args[1] = jl_box_int64(E_frame_size);
            args[2] = jl_box_int64(num_dishes);
            args[3] = jl_box_int64(num_frequencies);
            args[4] = jl_box_int64(num_polarizations);
            args[5] = jl_box_int64(num_times);
            args[6] = jl_box_int64(frame_index + 1);
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
            const int nargs = 7;
            jl_value_t** args;
            JL_GC_PUSHARGS(args, nargs);
            args[0] = jl_box_uint8pointer(A_frame);
            args[1] = jl_box_int64(A_frame_size);
            args[2] = jl_box_int64(num_dishes);
            args[3] = jl_box_int64(bb_num_beams);
            args[4] = jl_box_int64(num_polarizations);
            args[5] = jl_box_int64(num_frequencies);
            args[6] = jl_box_int64(frame_index + 1);
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
            const int nargs = 7;
            jl_value_t** args;
            JL_GC_PUSHARGS(args, nargs);
            args[0] = jl_box_uint8pointer(J_frame);
            args[1] = jl_box_int64(J_frame_size);
            args[2] = jl_box_int64(num_times);
            args[3] = jl_box_int64(num_polarizations);
            args[4] = jl_box_int64(num_frequencies);
            args[5] = jl_box_int64(bb_num_beams);
            args[6] = jl_box_int64(frame_index + 1);
            jl_value_t* const res = jl_call(set_J, args, nargs);
            assert(res);
            JL_GC_POP();
        });
        INFO("[{:d}] Done filling J buffer.", frame_index);

        INFO("[{:d}] Filling S buffer...", frame_index);
        {
            std::int16_t* __restrict__ const S = (std::int16_t*)S_frame;
            for (int loc = 0; loc < num_dish_locations; ++loc) {
                S[2 * loc + 0] = dish_locations[2 * loc + 0];
                S[2 * loc + 1] = dish_locations[2 * loc + 1];
            }
        }
        INFO("[{:d}] Done filling S buffer.", frame_index);

        INFO("[{:d}] Filling W buffer...", frame_index);
        {
            std::complex<short>* __restrict__ const W = (std::complex<short>*)W_frame;
            for (int polr = 0; polr < num_polarizations; ++polr) {
                for (int freq = 0; freq < num_frequencies; ++freq) {
                    for (int dishN = 0; dishN < num_dish_locations_N; ++dishN) {
                        for (int dishM = 0; dishM < num_dish_locations_M; ++dishM) {
                            const std::size_t ind =
                                dishM
                                + num_dish_locations_M
                                      * (dishN
                                         + num_dish_locations_N
                                               * (freq
                                                  + num_frequencies * (polr + std::size_t(0))));
#warning "TODO: calculate W in Julia"
                            W[ind] = 0;
                        }
                    }
                }
            }
        }
        INFO("[{:d}] Done filling W buffer.", frame_index);

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
        E_metadata->dim[0] = num_times;
        E_metadata->dim[1] = num_polarizations;
        E_metadata->dim[2] = num_frequencies;
        E_metadata->dim[3] = num_dishes;
        E_metadata->n_one_hot = -1;
        E_metadata->nfreq = num_frequencies;

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
        A_metadata->dim[0] = num_frequencies;
        A_metadata->dim[1] = num_polarizations;
        A_metadata->dim[2] = bb_num_beams;
        A_metadata->dim[3] = num_dishes;
        A_metadata->dim[4] = num_polarizations;
        A_metadata->n_one_hot = -1;
        A_metadata->nfreq = num_frequencies;

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
        J_metadata->dim[0] = bb_num_beams;
        J_metadata->dim[1] = num_frequencies;
        J_metadata->dim[2] = num_polarizations;
        J_metadata->dim[3] = num_times;
        J_metadata->n_one_hot = -1;
        J_metadata->nfreq = num_frequencies;

        chordMetadata* const S_metadata = get_chord_metadata(S_buffer, S_frame_id);
        chord_metadata_init(S_metadata);
        S_metadata->chime.fpga_seq_num = frame_index;
        S_metadata->frame_counter = frame_index;
        S_metadata->type = int16;
        S_metadata->dims = 2;
        std::strncpy(S_metadata->dim_name[0], "D", sizeof S_metadata->dim_name[0]);
        std::strncpy(S_metadata->dim_name[1], "MN", sizeof S_metadata->dim_name[1]);
        S_metadata->dim[0] = num_dish_locations;
        S_metadata->dim[1] = 2;
        S_metadata->n_one_hot = -1;
        S_metadata->nfreq = -1;

        chordMetadata* const W_metadata = get_chord_metadata(W_buffer, W_frame_id);
        chord_metadata_init(W_metadata);
        W_metadata->chime.fpga_seq_num = frame_index;
        W_metadata->frame_counter = frame_index;
        W_metadata->type = float16;
        W_metadata->dims = 5;
        std::strncpy(W_metadata->dim_name[0], "P", sizeof W_metadata->dim_name[0]);
        std::strncpy(W_metadata->dim_name[1], "F", sizeof W_metadata->dim_name[1]);
        std::strncpy(W_metadata->dim_name[2], "dishN", sizeof W_metadata->dim_name[2]);
        std::strncpy(W_metadata->dim_name[3], "dishM", sizeof W_metadata->dim_name[3]);
        std::strncpy(W_metadata->dim_name[4], "C", sizeof W_metadata->dim_name[4]);
        W_metadata->dim[0] = num_polarizations;
        W_metadata->dim[1] = num_frequencies;
        W_metadata->dim[2] = num_dish_locations_N;
        W_metadata->dim[3] = num_dish_locations_M;
        W_metadata->dim[4] = num_components;
        W_metadata->n_one_hot = -1;
        W_metadata->nfreq = num_frequencies;

        mark_frame_full(E_buffer, unique_name.c_str(), E_frame_id);
        mark_frame_full(A_buffer, unique_name.c_str(), A_frame_id);
        mark_frame_full(J_buffer, unique_name.c_str(), J_frame_id);
        mark_frame_full(S_buffer, unique_name.c_str(), S_frame_id);
        mark_frame_full(W_buffer, unique_name.c_str(), W_frame_id);
    }

    INFO("Done.");
}
