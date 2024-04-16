#include <Config.hpp>
#include <FEngine.hpp>
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
#include <visUtil.hpp>

#ifdef WITH_CUDA
#include <nvToolsExt.h>
#endif

#if !KOTEKAN_FLOAT16
#warning "The F-Engine simulator requires float16 support"
#else

REGISTER_KOTEKAN_STAGE(FEngine);

FEngine::FEngine(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          [](const kotekan::Stage& stage) {
              return const_cast<kotekan::Stage&>(stage).main_thread();
          }),
    unique_name(unique_name),
    skip_julia(config.get_default<bool>(unique_name, "skip_julia", false)),
    // Basic constants
    num_components(config.get<int>(unique_name, "num_components")),
    num_polarizations(config.get<int>(unique_name, "num_polarizations")),
    // Sky
    source_amplitude(config.get<float>(unique_name, "source_amplitude")),
    source_frequency(config.get<float>(unique_name, "source_frequency")),
    source_position_ew(config.get<float>(unique_name, "source_position_ew")),
    source_position_ns(config.get<float>(unique_name, "source_position_ns")),
    // Dishes
    num_dish_locations_ew(config.get<int>(unique_name, "num_dish_locations_ew")),
    num_dish_locations_ns(config.get<int>(unique_name, "num_dish_locations_ns")),
    num_dish_locations(num_dish_locations_ew * num_dish_locations_ns),
    dish_separation_ew(config.get<float>(unique_name, "dish_separation_ew")),
    dish_separation_ns(config.get<float>(unique_name, "dish_separation_ns")),
    num_dishes(config.get<int>(unique_name, "num_dishes")),
    dish_indices(config.get<std::vector<int>>(unique_name, "dish_indices")),
    dish_locations(2 * num_dishes, -1),
    dish_indices_ptr(new int[num_dish_locations_ew * num_dish_locations_ns]),
    // ADC
    adc_frequency(config.get<float>(unique_name, "adc_frequency")),
    num_samples_per_frame(config.get<int>(unique_name, "num_samples_per_frame")),
    num_taps(config.get<int>(unique_name, "num_taps")),
    num_frequencies(config.get<int>(unique_name, "num_frequencies")),
    frequency_channels(config.get<std::vector<int>>(unique_name, "frequency_channels")),
    num_times(config.get<int>(unique_name, "num_times")),
    // Baseband beamformer setup
    bb_num_dishes_M(config.get<int>(unique_name, "bb_num_dishes_M")),
    bb_num_dishes_N(config.get<int>(unique_name, "bb_num_dishes_N")),
    bb_num_beams_P(config.get<int>(unique_name, "bb_num_beams_P")),
    bb_num_beams_Q(config.get<int>(unique_name, "bb_num_beams_Q")),
    bb_beam_separation_x(config.get<float>(unique_name, "bb_beam_separation_x")),
    bb_beam_separation_y(config.get<float>(unique_name, "bb_beam_separation_y")),
    bb_num_beams(bb_num_beams_P * bb_num_beams_Q),
    // Upchannelizer setup
    upchannelization_factor(config.get<int>(unique_name, "upchannelization_factor")),
    // FRB beamformer setup
    frb1_num_beams_P(2 * num_dish_locations_ns), frb1_num_beams_Q(2 * num_dish_locations_ew),
    frb2_num_beams_ns(config.get<int>(unique_name, "frb_num_beams_ns")),
    frb2_num_beams_ew(config.get<int>(unique_name, "frb_num_beams_ew")),
    frb2_bore_z(config.get<float>(unique_name, "frb_bore_z")),
    frb2_opening_angle_ns(config.get<float>(unique_name, "frb_opening_angle_ns")),
    frb2_opening_angle_ew(config.get<float>(unique_name, "frb_opening_angle_ew")),
    frb_num_times(num_times / upchannelization_factor / Tds),

    // Pipeline
    num_frames(config.get<int>(unique_name, "num_frames")),
    repeat_count(config.get_default<int>(unique_name, "repeat_count", 1)),
    // Frame sizes
    E_frame_size(sizeof(uint8_t) * num_dishes * num_polarizations * num_frequencies * num_times),
    A_frame_size(sizeof(int8_t) * num_components * num_dishes * bb_num_beams * num_polarizations
                 * num_frequencies),
    s_frame_size(sizeof(int32_t) * bb_num_beams * num_polarizations * num_frequencies),
    J_frame_size(num_times * num_polarizations * num_frequencies * bb_num_beams),
    G2_frame_size(sizeof(float16_t) * num_frequencies * 2),
    G4_frame_size(sizeof(float16_t) * num_frequencies * 4),
    G8_frame_size(sizeof(float16_t) * num_frequencies * 8),
    W11_frame_size(sizeof(float16_t) * num_components * num_dish_locations_ns
                   * num_dish_locations_ew * num_polarizations * (num_frequencies * 1)),
    W12_frame_size(sizeof(float16_t) * num_components * num_dish_locations_ns
                   * num_dish_locations_ew * num_polarizations * (num_frequencies * 2)),
    W14_frame_size(sizeof(float16_t) * num_components * num_dish_locations_ns
                   * num_dish_locations_ew * num_polarizations * (num_frequencies * 4)),
    W18_frame_size(sizeof(float16_t) * num_components * num_dish_locations_ns
                   * num_dish_locations_ew * num_polarizations * (num_frequencies * 8)),
    W2_frame_size(sizeof(float16_t) * (frb1_num_beams_P * frb1_num_beams_Q)
                  * (frb2_num_beams_ns * frb2_num_beams_ew)
                  * (num_frequencies * upchannelization_factor)),
    I11_frame_size(sizeof(float16_t) * frb1_num_beams_P * frb1_num_beams_Q * (num_frequencies * 1)
                   * frb_num_times),
    I12_frame_size(sizeof(float16_t) * frb1_num_beams_P * frb1_num_beams_Q * (num_frequencies * 2)
                   * frb_num_times),
    I14_frame_size(sizeof(float16_t) * frb1_num_beams_P * frb1_num_beams_Q * (num_frequencies * 4)
                   * frb_num_times),
    I18_frame_size(sizeof(float16_t) * frb1_num_beams_P * frb1_num_beams_Q * (num_frequencies * 8)
                   * frb_num_times),
    // Buffers
    E_buffer(get_buffer("E_buffer")), A_buffer(get_buffer("A_buffer")),
    s_buffer(get_buffer("s_buffer")), J_buffer(get_buffer("J_buffer")),
    G2_buffer(get_buffer("G2_buffer")), G4_buffer(get_buffer("G4_buffer")),
    G8_buffer(get_buffer("G8_buffer")), W11_buffer(get_buffer("W11_buffer")),
    W12_buffer(get_buffer("W12_buffer")), W14_buffer(get_buffer("W14_buffer")),
    W18_buffer(get_buffer("W18_buffer")), W2_buffer(get_buffer("W2_buffer")),
    I11_buffer(get_buffer("I11_buffer")), I12_buffer(get_buffer("I12_buffer")),
    I14_buffer(get_buffer("I14_buffer")), I18_buffer(get_buffer("I18_buffer")) {
    assert(num_dishes >= 0 && num_dishes <= num_dish_locations);
    assert(std::ptrdiff_t(dish_indices.size()) == num_dish_locations_ew * num_dish_locations_ns);
    int num_dishes_seen = 0;
    for (int loc_ns = 0; loc_ns < num_dish_locations_ns; ++loc_ns) {
        for (int loc_ew = 0; loc_ew < num_dish_locations_ew; ++loc_ew) {
            int loc = loc_ew + num_dish_locations_ew * loc_ns;
            int dish = dish_indices.at(loc);
            assert(dish == -1 || (dish >= 0 && dish < num_dishes));
            if (dish >= 0) {
                ++num_dishes_seen;
                // check for duplicate dish indices
                assert(dish_locations.at(2 * dish + 0) == -1);
                dish_locations.at(2 * dish + 0) = loc_ns;
                dish_locations.at(2 * dish + 1) = loc_ew;
            }
            dish_indices_ptr[loc] = dish;
        }
    }
    assert(num_dishes_seen == num_dishes);

    // TODO: Remove `num_frequencies`
    assert(int(frequency_channels.size()) == num_frequencies);

    assert(E_buffer);
    assert(A_buffer);
    assert(s_buffer);
    assert(J_buffer);
    assert(G2_buffer);
    assert(G4_buffer);
    assert(G8_buffer);
    assert(W11_buffer);
    assert(W12_buffer);
    assert(W14_buffer);
    assert(W18_buffer);
    assert(W2_buffer);
    assert(I11_buffer);
    assert(I12_buffer);
    assert(I14_buffer);
    assert(I18_buffer);
    E_buffer->register_producer(unique_name);
    A_buffer->register_producer(unique_name);
    s_buffer->register_producer(unique_name);
    J_buffer->register_producer(unique_name);
    G2_buffer->register_producer(unique_name);
    G4_buffer->register_producer(unique_name);
    G8_buffer->register_producer(unique_name);
    W11_buffer->register_producer(unique_name);
    W12_buffer->register_producer(unique_name);
    W14_buffer->register_producer(unique_name);
    W18_buffer->register_producer(unique_name);
    W2_buffer->register_producer(unique_name);
    I11_buffer->register_producer(unique_name);
    I12_buffer->register_producer(unique_name);
    I14_buffer->register_producer(unique_name);
    I18_buffer->register_producer(unique_name);

    INFO("Starting Julia...");
    kotekan::juliaStartup();

    if (!skip_julia) {
        INFO("Defining Julia code...");
        {
            const auto julia_source_filename = "lib/stages/FEngine.jl";
            std::ifstream file(julia_source_filename);
            if (!file.is_open())
                FATAL_ERROR(
                    "Could not open the file \"{:s}\" with the Julia source code for the F-Engine "
                    "simulator",
                    julia_source_filename);
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
    } // if !skip_julia
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

    if (!skip_julia) {
        INFO("Initializing F-Engine...");
        kotekan::juliaCall([&]() {
            jl_module_t* const f_engine_module =
                (jl_module_t*)jl_get_global(jl_main_module, jl_symbol("FEngine"));
            assert(f_engine_module);
            jl_function_t* const setup = jl_get_function(f_engine_module, "setup");
            assert(setup);
            const int nargs = 19;
            jl_value_t** args;
            JL_GC_PUSHARGS(args, nargs);
            int iargc = 0;
            args[iargc++] = jl_box_float32(source_amplitude);
            args[iargc++] = jl_box_float32(source_frequency);
            args[iargc++] = jl_box_float32(source_position_ew);
            args[iargc++] = jl_box_float32(source_position_ns);
            args[iargc++] = jl_box_int64(num_dish_locations_ew);
            args[iargc++] = jl_box_int64(num_dish_locations_ns);
            args[iargc++] = jl_box_voidpointer(dish_indices_ptr);
            args[iargc++] = jl_box_float32(dish_separation_ew);
            args[iargc++] = jl_box_float32(dish_separation_ns);
            args[iargc++] = jl_box_int64(num_dishes);
            args[iargc++] = jl_box_float32(adc_frequency);
            args[iargc++] = jl_box_int64(num_taps);
            args[iargc++] = jl_box_int64(num_frequencies);
            args[iargc++] = jl_box_int64(num_times);
            args[iargc++] = jl_box_int64(bb_num_dishes_M);
            args[iargc++] = jl_box_int64(bb_num_dishes_N);
            args[iargc++] = jl_box_int64(bb_num_beams_P);
            args[iargc++] = jl_box_int64(bb_num_beams_Q);
            args[iargc++] = jl_box_int64(num_frames);
            assert(iargc == nargs);
            jl_value_t* const res = jl_call(setup, args, nargs);
            assert(res);
            JL_GC_POP();
        });
        INFO("Done initializing world.");
    } // if !skip_julia

    // Produce baseband phase
    for (int A_frame_index = 0; A_frame_index < A_buffer->num_frames; ++A_frame_index) {
        const int A_frame_id = A_frame_index % A_buffer->num_frames;

        // Wait for buffer
        std::uint8_t* const A_frame = A_buffer->wait_for_empty_frame(unique_name, A_frame_id);
        if (!A_frame)
            return;
        if (!(std::ptrdiff_t(A_buffer->frame_size) == A_frame_size))
            FATAL_ERROR("A_buffer->frame_size={:d} A_frame_size={:d}", A_buffer->frame_size,
                        A_frame_size);
        assert(std::ptrdiff_t(A_buffer->frame_size) == A_frame_size);
        A_buffer->allocate_new_metadata_object(A_frame_id);
        set_fpga_seq_num(A_buffer, A_frame_id, -1);

        // Fill buffer
        DEBUG("[{:d}] Filling A buffer...", A_frame_index);
        if (!skip_julia) {
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
                args[6] = jl_box_int64(A_frame_index + 1);
                jl_value_t* const res = jl_call(set_A, args, nargs);
                assert(res);
                JL_GC_POP();
            });
        } else {
            for (int n = 0; n < num_components * num_dishes * bb_num_beams * num_polarizations
                                    * num_frequencies;
                 ++n)
                ((int8_t*)A_frame)[n] = 4;
        }
        DEBUG("[{:d}] Done filling A buffer.", A_frame_index);

        // Set metadata
        std::shared_ptr<chordMetadata> const A_metadata = get_chord_metadata(A_buffer, A_frame_id);
        A_metadata->frame_counter = 0; /*A_frame_index;*/
        std::strncpy(A_metadata->name, "A", sizeof A_metadata->name);
        A_metadata->type = int8;
        A_metadata->dims = 5;
        assert(A_metadata->dims <= CHORD_META_MAX_DIM);
        std::strncpy(A_metadata->dim_name[0], "F", sizeof A_metadata->dim_name[0]);
        std::strncpy(A_metadata->dim_name[1], "P", sizeof A_metadata->dim_name[1]);
        std::strncpy(A_metadata->dim_name[2], "B", sizeof A_metadata->dim_name[2]);
        std::strncpy(A_metadata->dim_name[3], "D", sizeof A_metadata->dim_name[3]);
        std::strncpy(A_metadata->dim_name[4], "C", sizeof A_metadata->dim_name[4]);
        A_metadata->dim[0] = num_frequencies;
        A_metadata->dim[1] = num_polarizations;
        A_metadata->dim[2] = bb_num_beams;
        A_metadata->dim[3] = num_dishes;
        A_metadata->dim[4] = num_components;
        A_metadata->sample0_offset = -1; // undefined
        A_metadata->nfreq = num_frequencies;
        assert(A_metadata->nfreq <= CHORD_META_MAX_FREQ);
        for (int freq = 0; freq < num_frequencies; ++freq) {
            A_metadata->coarse_freq[freq] = freq + 1; // See `FEngine.f_engine`
            A_metadata->freq_upchan_factor[freq] = 1;
            A_metadata->half_fpga_sample0[freq] = -1;      // undefined
            A_metadata->time_downsampling_fpga[freq] = -1; // undefined
        }
        A_metadata->ndishes = num_dishes;
        A_metadata->n_dish_locations_ew = num_dish_locations_ew;
        A_metadata->n_dish_locations_ns = num_dish_locations_ns;
        A_metadata->dish_index = dish_indices_ptr;

        // Mark buffer as full
        A_buffer->mark_frame_full(unique_name, A_frame_id);
    }

    // Produce baseband shift
    for (int s_frame_index = 0; s_frame_index < s_buffer->num_frames; ++s_frame_index) {
        const int s_frame_id = s_frame_index % s_buffer->num_frames;

        // Wait for buffer
        std::uint8_t* const s_frame = s_buffer->wait_for_empty_frame(unique_name, s_frame_id);
        if (!s_frame)
            return;
        if (!(std::ptrdiff_t(s_buffer->frame_size) == s_frame_size))
            FATAL_ERROR("s_buffer->frame_size={:d} s_frame_size={:d}", s_buffer->frame_size,
                        s_frame_size);
        assert(std::ptrdiff_t(s_buffer->frame_size) == s_frame_size);
        s_buffer->allocate_new_metadata_object(s_frame_id);
        set_fpga_seq_num(s_buffer, s_frame_id, -1);

        // Fill buffer
        DEBUG("[{:d}] Filling s buffer...", s_frame_index);
        for (int n = 0; n < bb_num_beams * num_polarizations * num_frequencies; ++n)
            ((int32_t*)s_frame)[n] = 13;
        DEBUG("[{:d}] Done filling s buffer.", s_frame_index);

        // Set metadata
        std::shared_ptr<chordMetadata> const s_metadata = get_chord_metadata(s_buffer, s_frame_id);
        s_metadata->frame_counter = s_frame_index;
        std::strncpy(s_metadata->name, "s", sizeof s_metadata->name);
        s_metadata->type = int32;
        s_metadata->dims = 3;
        assert(s_metadata->dims <= CHORD_META_MAX_DIM);
        std::strncpy(s_metadata->dim_name[0], "F", sizeof s_metadata->dim_name[0]);
        std::strncpy(s_metadata->dim_name[1], "P", sizeof s_metadata->dim_name[1]);
        std::strncpy(s_metadata->dim_name[2], "B", sizeof s_metadata->dim_name[2]);
        s_metadata->dim[0] = num_frequencies;
        s_metadata->dim[1] = num_polarizations;
        s_metadata->dim[2] = bb_num_beams;
        s_metadata->sample0_offset = -1; // undefined
        s_metadata->nfreq = num_frequencies;
        assert(s_metadata->nfreq <= CHORD_META_MAX_FREQ);
        for (int freq = 0; freq < num_frequencies; ++freq) {
            s_metadata->coarse_freq[freq] = freq + 1; // See `FEngine.f_engine`
            s_metadata->freq_upchan_factor[freq] = 1;
            s_metadata->half_fpga_sample0[freq] = -1;      // undefined
            s_metadata->time_downsampling_fpga[freq] = -1; // undefined
        }
        s_metadata->ndishes = num_dishes;
        s_metadata->n_dish_locations_ew = num_dish_locations_ew;
        s_metadata->n_dish_locations_ns = num_dish_locations_ns;
        s_metadata->dish_index = dish_indices_ptr;

        // Mark buffer as full
        s_buffer->mark_frame_full(unique_name, s_frame_id);
    }

    // Produce upchannelization-2 gains
    for (int G2_frame_index = 0; G2_frame_index < G2_buffer->num_frames; ++G2_frame_index) {
        const int U = 2;
        const int G2_frame_id = G2_frame_index % G2_buffer->num_frames;

        // Wait for buffer
        std::uint8_t* const G2_frame = G2_buffer->wait_for_empty_frame(unique_name, G2_frame_id);
        if (!G2_frame)
            return;
        if (!(std::ptrdiff_t(G2_buffer->frame_size) == G2_frame_size))
            FATAL_ERROR("G2_buffer->frame_size={:d} G2_frame_size={:d}", G2_buffer->frame_size,
                        G2_frame_size);
        assert(std::ptrdiff_t(G2_buffer->frame_size) == G2_frame_size);
        G2_buffer->allocate_new_metadata_object(G2_frame_id);
        set_fpga_seq_num(G2_buffer, G2_frame_id, -1);

        DEBUG("[{:d}] Filling G buffer...", G2_frame_index);
        for (int n = 0; n < num_frequencies * U; ++n)
            ((float16_t*)G2_frame)[n] = 1;
        DEBUG("[{:d}] Done filling G buffer.", G2_frame_index);

        // Set metadata
        std::shared_ptr<chordMetadata> const G2_metadata =
            get_chord_metadata(G2_buffer, G2_frame_id);
        G2_metadata->frame_counter = G2_frame_index;
        std::strncpy(G2_metadata->name, "G2", sizeof G2_metadata->name);
        G2_metadata->type = float16;
        G2_metadata->dims = 1;
        assert(G2_metadata->dims <= CHORD_META_MAX_DIM);
        std::strncpy(G2_metadata->dim_name[0], "Fbar", sizeof G2_metadata->dim_name[0]);
        G2_metadata->dim[0] = num_frequencies * U;
        G2_metadata->sample0_offset = -1; // undefined
        G2_metadata->nfreq = num_frequencies;
        assert(G2_metadata->nfreq <= CHORD_META_MAX_FREQ);
        for (int freq = 0; freq < num_frequencies; ++freq) {
            G2_metadata->coarse_freq[freq] = freq + 1;      // See `FEngine.f_engine`
            G2_metadata->freq_upchan_factor[freq] = 1;      // upchannelization_factor;
            G2_metadata->half_fpga_sample0[freq] = -1;      // undefined
            G2_metadata->time_downsampling_fpga[freq] = -1; // undefined
        }
        G2_metadata->ndishes = num_dishes;
        G2_metadata->n_dish_locations_ew = num_dish_locations_ew;
        G2_metadata->n_dish_locations_ns = num_dish_locations_ns;
        G2_metadata->dish_index = dish_indices_ptr;

        // Mark buffer as full
        G2_buffer->mark_frame_full(unique_name, G2_frame_id);
    }

    // Produce upchannelization-4 gains
    for (int G4_frame_index = 0; G4_frame_index < G4_buffer->num_frames; ++G4_frame_index) {
        const int U = 4;
        const int G4_frame_id = G4_frame_index % G4_buffer->num_frames;

        // Wait for buffer
        std::uint8_t* const G4_frame = G4_buffer->wait_for_empty_frame(unique_name, G4_frame_id);
        if (!G4_frame)
            return;
        if (!(std::ptrdiff_t(G4_buffer->frame_size) == G4_frame_size))
            FATAL_ERROR("G4_buffer->frame_size={:d} G4_frame_size={:d}", G4_buffer->frame_size,
                        G4_frame_size);
        assert(std::ptrdiff_t(G4_buffer->frame_size) == G4_frame_size);
        G4_buffer->allocate_new_metadata_object(G4_frame_id);
        set_fpga_seq_num(G4_buffer, G4_frame_id, -1);

        DEBUG("[{:d}] Filling G buffer...", G4_frame_index);
        for (int n = 0; n < num_frequencies * U; ++n)
            ((float16_t*)G4_frame)[n] = 1;
        DEBUG("[{:d}] Done filling G buffer.", G4_frame_index);

        // Set metadata
        std::shared_ptr<chordMetadata> const G4_metadata =
            get_chord_metadata(G4_buffer, G4_frame_id);
        G4_metadata->frame_counter = G4_frame_index;
        std::strncpy(G4_metadata->name, "G4", sizeof G4_metadata->name);
        G4_metadata->type = float16;
        G4_metadata->dims = 1;
        assert(G4_metadata->dims <= CHORD_META_MAX_DIM);
        std::strncpy(G4_metadata->dim_name[0], "Fbar", sizeof G4_metadata->dim_name[0]);
        G4_metadata->dim[0] = num_frequencies * U;
        G4_metadata->sample0_offset = -1; // undefined
        G4_metadata->nfreq = num_frequencies;
        assert(G4_metadata->nfreq <= CHORD_META_MAX_FREQ);
        for (int freq = 0; freq < num_frequencies; ++freq) {
            G4_metadata->coarse_freq[freq] = freq + 1;      // See `FEngine.f_engine`
            G4_metadata->freq_upchan_factor[freq] = 1;      // upchannelization_factor;
            G4_metadata->half_fpga_sample0[freq] = -1;      // undefined
            G4_metadata->time_downsampling_fpga[freq] = -1; // undefined
        }
        G4_metadata->ndishes = num_dishes;
        G4_metadata->n_dish_locations_ew = num_dish_locations_ew;
        G4_metadata->n_dish_locations_ns = num_dish_locations_ns;
        G4_metadata->dish_index = dish_indices_ptr;

        // Mark buffer as full
        G4_buffer->mark_frame_full(unique_name, G4_frame_id);
    }

    // Produce upchannelization-8 gains
    for (int G8_frame_index = 0; G8_frame_index < G8_buffer->num_frames; ++G8_frame_index) {
        const int U = 8;
        const int G8_frame_id = G8_frame_index % G8_buffer->num_frames;

        // Wait for buffer
        std::uint8_t* const G8_frame = G8_buffer->wait_for_empty_frame(unique_name, G8_frame_id);
        if (!G8_frame)
            return;
        if (!(std::ptrdiff_t(G8_buffer->frame_size) == G8_frame_size))
            FATAL_ERROR("G8_buffer->frame_size={:d} G8_frame_size={:d}", G8_buffer->frame_size,
                        G8_frame_size);
        assert(std::ptrdiff_t(G8_buffer->frame_size) == G8_frame_size);
        G8_buffer->allocate_new_metadata_object(G8_frame_id);
        set_fpga_seq_num(G8_buffer, G8_frame_id, -1);

        DEBUG("[{:d}] Filling G buffer...", G8_frame_index);
        for (int n = 0; n < num_frequencies * U; ++n)
            ((float16_t*)G8_frame)[n] = 1;
        DEBUG("[{:d}] Done filling G buffer.", G8_frame_index);

        // Set metadata
        std::shared_ptr<chordMetadata> const G8_metadata =
            get_chord_metadata(G8_buffer, G8_frame_id);
        G8_metadata->frame_counter = G8_frame_index;
        std::strncpy(G8_metadata->name, "G8", sizeof G8_metadata->name);
        G8_metadata->type = float16;
        G8_metadata->dims = 1;
        assert(G8_metadata->dims <= CHORD_META_MAX_DIM);
        std::strncpy(G8_metadata->dim_name[0], "Fbar", sizeof G8_metadata->dim_name[0]);
        G8_metadata->dim[0] = num_frequencies * U;
        G8_metadata->sample0_offset = -1; // undefined
        G8_metadata->nfreq = num_frequencies;
        assert(G8_metadata->nfreq <= CHORD_META_MAX_FREQ);
        for (int freq = 0; freq < num_frequencies; ++freq) {
            G8_metadata->coarse_freq[freq] = freq + 1;      // See `FEngine.f_engine`
            G8_metadata->freq_upchan_factor[freq] = 1;      // upchannelization_factor;
            G8_metadata->half_fpga_sample0[freq] = -1;      // undefined
            G8_metadata->time_downsampling_fpga[freq] = -1; // undefined
        }
        G8_metadata->ndishes = num_dishes;
        G8_metadata->n_dish_locations_ew = num_dish_locations_ew;
        G8_metadata->n_dish_locations_ns = num_dish_locations_ns;
        G8_metadata->dish_index = dish_indices_ptr;

        // Mark buffer as full
        G8_buffer->mark_frame_full(unique_name, G8_frame_id);
    }

    // Produce FRB1 upchannelization-1 phases
    for (int W11_frame_index = 0; W11_frame_index < W11_buffer->num_frames; ++W11_frame_index) {
        const int U = 1;
        const int W11_frame_id = W11_frame_index % W11_buffer->num_frames;

        // Wait for buffer
        std::uint8_t* const W11_frame = W11_buffer->wait_for_empty_frame(unique_name, W11_frame_id);
        if (!W11_frame)
            return;
        if (!(std::ptrdiff_t(W11_buffer->frame_size) == W11_frame_size))
            FATAL_ERROR("W11_buffer->frame_size={:d} W11_frame_size={:d}", W11_buffer->frame_size,
                        W11_frame_size);
        assert(std::ptrdiff_t(W11_buffer->frame_size) == W11_frame_size);
        W11_buffer->allocate_new_metadata_object(W11_frame_id);
        set_fpga_seq_num(W11_buffer, W11_frame_id, -1);

        DEBUG("[{:d}] Filling W11 buffer...", W11_frame_index);
        // Disable this because the F-Engine simulator doesn't upchannelize yet
        if (false && !skip_julia) {
            kotekan::juliaCall([&]() {
                jl_module_t* const f_engine_module =
                    (jl_module_t*)jl_get_global(jl_main_module, jl_symbol("FEngine"));
                assert(f_engine_module);
                jl_function_t* const set_W11 = jl_get_function(f_engine_module, "set_W");
                assert(set_W11);
                const int nargs = 7;
                jl_value_t** args;
                JL_GC_PUSHARGS(args, nargs);
                args[0] = jl_box_uint8pointer(W11_frame);
                args[1] = jl_box_int64(W11_frame_size);
                args[2] = jl_box_int64(num_dish_locations_ns);
                args[3] = jl_box_int64(num_dish_locations_ew);
                args[4] = jl_box_int64(num_polarizations);
                args[5] = jl_box_int64(num_frequencies * U);
                args[6] = jl_box_int64(W11_frame_index + 1);
                jl_value_t* const res = jl_call(set_W11, args, nargs);
                assert(res);
                JL_GC_POP();
            });
        } else {
            for (int n = 0; n < num_components * num_dish_locations_ns * num_dish_locations_ew
                                    * num_polarizations * num_frequencies * U;
                 ++n)
                ((float16_t*)W11_frame)[n] = 1;
        }
        DEBUG("[{:d}] Done filling W11 buffer.", W11_frame_index);

        // Set metadata
        std::shared_ptr<chordMetadata> const W11_metadata =
            get_chord_metadata(W11_buffer, W11_frame_id);
        W11_metadata->frame_counter = W11_frame_index;
        std::strncpy(W11_metadata->name, "W1", sizeof W11_metadata->name);
        W11_metadata->type = float16;
        W11_metadata->dims = 5;
        assert(W11_metadata->dims <= CHORD_META_MAX_DIM);
        std::strncpy(W11_metadata->dim_name[0], "Fbar", sizeof W11_metadata->dim_name[0]);
        std::strncpy(W11_metadata->dim_name[1], "P", sizeof W11_metadata->dim_name[1]);
        std::strncpy(W11_metadata->dim_name[2], "dishN", sizeof W11_metadata->dim_name[2]);
        std::strncpy(W11_metadata->dim_name[3], "dishM", sizeof W11_metadata->dim_name[3]);
        std::strncpy(W11_metadata->dim_name[4], "C", sizeof W11_metadata->dim_name[4]);
        W11_metadata->dim[0] = num_frequencies * U;
        W11_metadata->dim[1] = num_polarizations;
        W11_metadata->dim[2] = num_dish_locations_ew;
        W11_metadata->dim[3] = num_dish_locations_ns;
        W11_metadata->dim[4] = num_components;
        W11_metadata->sample0_offset = -1; // undefined
        W11_metadata->nfreq = num_frequencies;
        assert(W11_metadata->nfreq <= CHORD_META_MAX_FREQ);
        for (int freq = 0; freq < num_frequencies; ++freq) {
            W11_metadata->coarse_freq[freq] = freq + 1; // See `FEngine.f_engine`
            W11_metadata->freq_upchan_factor[freq] = U;
            W11_metadata->half_fpga_sample0[freq] = -1;      // undefined
            W11_metadata->time_downsampling_fpga[freq] = -1; // undefined
        }
        W11_metadata->ndishes = num_dishes;
        W11_metadata->n_dish_locations_ew = num_dish_locations_ew;
        W11_metadata->n_dish_locations_ns = num_dish_locations_ns;
        W11_metadata->dish_index = dish_indices_ptr;

        // Mark buffer as full
        W11_buffer->mark_frame_full(unique_name, W11_frame_id);
    }

    // Produce FRB1 upchannelization-2 phases
    for (int W12_frame_index = 0; W12_frame_index < W12_buffer->num_frames; ++W12_frame_index) {
        const int U = 2;
        const int W12_frame_id = W12_frame_index % W12_buffer->num_frames;

        // Wait for buffer
        std::uint8_t* const W12_frame = W12_buffer->wait_for_empty_frame(unique_name, W12_frame_id);
        if (!W12_frame)
            return;
        if (!(std::ptrdiff_t(W12_buffer->frame_size) == W12_frame_size))
            FATAL_ERROR("W12_buffer->frame_size={:d} W12_frame_size={:d}", W12_buffer->frame_size,
                        W12_frame_size);
        assert(std::ptrdiff_t(W12_buffer->frame_size) == W12_frame_size);
        W12_buffer->allocate_new_metadata_object(W12_frame_id);
        set_fpga_seq_num(W12_buffer, W12_frame_id, -1);

        DEBUG("[{:d}] Filling W12 buffer...", W12_frame_index);
        // Disable this because the F-Engine simulator doesn't upchannelize yet
        if (false && !skip_julia) {
            kotekan::juliaCall([&]() {
                jl_module_t* const f_engine_module =
                    (jl_module_t*)jl_get_global(jl_main_module, jl_symbol("FEngine"));
                assert(f_engine_module);
                jl_function_t* const set_W12 = jl_get_function(f_engine_module, "set_W");
                assert(set_W12);
                const int nargs = 7;
                jl_value_t** args;
                JL_GC_PUSHARGS(args, nargs);
                args[0] = jl_box_uint8pointer(W12_frame);
                args[1] = jl_box_int64(W12_frame_size);
                args[2] = jl_box_int64(num_dish_locations_ns);
                args[3] = jl_box_int64(num_dish_locations_ew);
                args[4] = jl_box_int64(num_polarizations);
                args[5] = jl_box_int64(num_frequencies * U);
                args[6] = jl_box_int64(W12_frame_index + 1);
                jl_value_t* const res = jl_call(set_W12, args, nargs);
                assert(res);
                JL_GC_POP();
            });
        } else {
            for (int n = 0; n < num_components * num_dish_locations_ns * num_dish_locations_ew
                                    * num_polarizations * num_frequencies * U;
                 ++n)
                ((float16_t*)W12_frame)[n] = 1;
        }
        DEBUG("[{:d}] Done filling W12 buffer.", W12_frame_index);

        // Set metadata
        std::shared_ptr<chordMetadata> const W12_metadata =
            get_chord_metadata(W12_buffer, W12_frame_id);
        W12_metadata->frame_counter = W12_frame_index;
        std::strncpy(W12_metadata->name, "W2", sizeof W12_metadata->name);
        W12_metadata->type = float16;
        W12_metadata->dims = 5;
        assert(W12_metadata->dims <= CHORD_META_MAX_DIM);
        std::strncpy(W12_metadata->dim_name[0], "Fbar", sizeof W12_metadata->dim_name[0]);
        std::strncpy(W12_metadata->dim_name[1], "P", sizeof W12_metadata->dim_name[1]);
        std::strncpy(W12_metadata->dim_name[2], "dishN", sizeof W12_metadata->dim_name[2]);
        std::strncpy(W12_metadata->dim_name[3], "dishM", sizeof W12_metadata->dim_name[3]);
        std::strncpy(W12_metadata->dim_name[4], "C", sizeof W12_metadata->dim_name[4]);
        W12_metadata->dim[0] = num_frequencies * U;
        W12_metadata->dim[1] = num_polarizations;
        W12_metadata->dim[2] = num_dish_locations_ew;
        W12_metadata->dim[3] = num_dish_locations_ns;
        W12_metadata->dim[4] = num_components;
        W12_metadata->sample0_offset = -1; // undefined
        W12_metadata->nfreq = num_frequencies;
        assert(W12_metadata->nfreq <= CHORD_META_MAX_FREQ);
        for (int freq = 0; freq < num_frequencies; ++freq) {
            W12_metadata->coarse_freq[freq] = freq + 1; // See `FEngine.f_engine`
            W12_metadata->freq_upchan_factor[freq] = U;
            W12_metadata->half_fpga_sample0[freq] = -1;      // undefined
            W12_metadata->time_downsampling_fpga[freq] = -1; // undefined
        }
        W12_metadata->ndishes = num_dishes;
        W12_metadata->n_dish_locations_ew = num_dish_locations_ew;
        W12_metadata->n_dish_locations_ns = num_dish_locations_ns;
        W12_metadata->dish_index = dish_indices_ptr;

        // Mark buffer as full
        W12_buffer->mark_frame_full(unique_name, W12_frame_id);
    }

    // Produce FRB1 upchannelization-4 phases
    for (int W14_frame_index = 0; W14_frame_index < W14_buffer->num_frames; ++W14_frame_index) {
        const int U = 4;
        const int W14_frame_id = W14_frame_index % W14_buffer->num_frames;

        // Wait for buffer
        std::uint8_t* const W14_frame = W14_buffer->wait_for_empty_frame(unique_name, W14_frame_id);
        if (!W14_frame)
            return;
        if (!(std::ptrdiff_t(W14_buffer->frame_size) == W14_frame_size))
            FATAL_ERROR("W14_buffer->frame_size={:d} W14_frame_size={:d}", W14_buffer->frame_size,
                        W14_frame_size);
        assert(std::ptrdiff_t(W14_buffer->frame_size) == W14_frame_size);
        W14_buffer->allocate_new_metadata_object(W14_frame_id);
        set_fpga_seq_num(W14_buffer, W14_frame_id, -1);

        DEBUG("[{:d}] Filling W14 buffer...", W14_frame_index);
        // Disable this because the F-Engine simulator doesn't upchannelize yet
        if (false && !skip_julia) {
            kotekan::juliaCall([&]() {
                jl_module_t* const f_engine_module =
                    (jl_module_t*)jl_get_global(jl_main_module, jl_symbol("FEngine"));
                assert(f_engine_module);
                jl_function_t* const set_W14 = jl_get_function(f_engine_module, "set_W");
                assert(set_W14);
                const int nargs = 7;
                jl_value_t** args;
                JL_GC_PUSHARGS(args, nargs);
                args[0] = jl_box_uint8pointer(W14_frame);
                args[1] = jl_box_int64(W14_frame_size);
                args[2] = jl_box_int64(num_dish_locations_ns);
                args[3] = jl_box_int64(num_dish_locations_ew);
                args[4] = jl_box_int64(num_polarizations);
                args[5] = jl_box_int64(num_frequencies * U);
                args[6] = jl_box_int64(W14_frame_index + 1);
                jl_value_t* const res = jl_call(set_W14, args, nargs);
                assert(res);
                JL_GC_POP();
            });
        } else {
            for (int n = 0; n < num_components * num_dish_locations_ns * num_dish_locations_ew
                                    * num_polarizations * num_frequencies * U;
                 ++n)
                ((float16_t*)W14_frame)[n] = 1;
        }
        DEBUG("[{:d}] Done filling W14 buffer.", W14_frame_index);

        // Set metadata
        std::shared_ptr<chordMetadata> const W14_metadata =
            get_chord_metadata(W14_buffer, W14_frame_id);
        W14_metadata->frame_counter = W14_frame_index;
        std::strncpy(W14_metadata->name, "W4", sizeof W14_metadata->name);
        W14_metadata->type = float16;
        W14_metadata->dims = 5;
        assert(W14_metadata->dims <= CHORD_META_MAX_DIM);
        std::strncpy(W14_metadata->dim_name[0], "Fbar", sizeof W14_metadata->dim_name[0]);
        std::strncpy(W14_metadata->dim_name[1], "P", sizeof W14_metadata->dim_name[1]);
        std::strncpy(W14_metadata->dim_name[2], "dishN", sizeof W14_metadata->dim_name[2]);
        std::strncpy(W14_metadata->dim_name[3], "dishM", sizeof W14_metadata->dim_name[3]);
        std::strncpy(W14_metadata->dim_name[4], "C", sizeof W14_metadata->dim_name[4]);
        W14_metadata->dim[0] = num_frequencies * U;
        W14_metadata->dim[1] = num_polarizations;
        W14_metadata->dim[2] = num_dish_locations_ew;
        W14_metadata->dim[3] = num_dish_locations_ns;
        W14_metadata->dim[4] = num_components;
        W14_metadata->sample0_offset = -1; // undefined
        W14_metadata->nfreq = num_frequencies;
        assert(W14_metadata->nfreq <= CHORD_META_MAX_FREQ);
        for (int freq = 0; freq < num_frequencies; ++freq) {
            W14_metadata->coarse_freq[freq] = freq + 1; // See `FEngine.f_engine`
            W14_metadata->freq_upchan_factor[freq] = U;
            W14_metadata->half_fpga_sample0[freq] = -1;      // undefined
            W14_metadata->time_downsampling_fpga[freq] = -1; // undefined
        }
        W14_metadata->ndishes = num_dishes;
        W14_metadata->n_dish_locations_ew = num_dish_locations_ew;
        W14_metadata->n_dish_locations_ns = num_dish_locations_ns;
        W14_metadata->dish_index = dish_indices_ptr;

        // Mark buffer as full
        W14_buffer->mark_frame_full(unique_name, W14_frame_id);
    }

    // Produce FRB1 upchannelization-8 phases
    for (int W18_frame_index = 0; W18_frame_index < W18_buffer->num_frames; ++W18_frame_index) {
        const int U = 8;
        const int W18_frame_id = W18_frame_index % W18_buffer->num_frames;

        // Wait for buffer
        std::uint8_t* const W18_frame = W18_buffer->wait_for_empty_frame(unique_name, W18_frame_id);
        if (!W18_frame)
            return;
        if (!(std::ptrdiff_t(W18_buffer->frame_size) == W18_frame_size))
            FATAL_ERROR("W18_buffer->frame_size={:d} W18_frame_size={:d}", W18_buffer->frame_size,
                        W18_frame_size);
        assert(std::ptrdiff_t(W18_buffer->frame_size) == W18_frame_size);
        W18_buffer->allocate_new_metadata_object(W18_frame_id);
        set_fpga_seq_num(W18_buffer, W18_frame_id, -1);

        DEBUG("[{:d}] Filling W18 buffer...", W18_frame_index);
        // Disable this because the F-Engine simulator doesn't upchannelize yet
        if (false && !skip_julia) {
            kotekan::juliaCall([&]() {
                jl_module_t* const f_engine_module =
                    (jl_module_t*)jl_get_global(jl_main_module, jl_symbol("FEngine"));
                assert(f_engine_module);
                jl_function_t* const set_W18 = jl_get_function(f_engine_module, "set_W");
                assert(set_W18);
                const int nargs = 7;
                jl_value_t** args;
                JL_GC_PUSHARGS(args, nargs);
                args[0] = jl_box_uint8pointer(W18_frame);
                args[1] = jl_box_int64(W18_frame_size);
                args[2] = jl_box_int64(num_dish_locations_ns);
                args[3] = jl_box_int64(num_dish_locations_ew);
                args[4] = jl_box_int64(num_polarizations);
                args[5] = jl_box_int64(num_frequencies * U);
                args[6] = jl_box_int64(W18_frame_index + 1);
                jl_value_t* const res = jl_call(set_W18, args, nargs);
                assert(res);
                JL_GC_POP();
            });
        } else {
            for (int n = 0; n < num_components * num_dish_locations_ns * num_dish_locations_ew
                                    * num_polarizations * num_frequencies * U;
                 ++n)
                ((float16_t*)W18_frame)[n] = 1;
        }
        DEBUG("[{:d}] Done filling W18 buffer.", W18_frame_index);

        // Set metadata
        std::shared_ptr<chordMetadata> const W18_metadata =
            get_chord_metadata(W18_buffer, W18_frame_id);
        W18_metadata->frame_counter = W18_frame_index;
        std::strncpy(W18_metadata->name, "W8", sizeof W18_metadata->name);
        W18_metadata->type = float16;
        W18_metadata->dims = 5;
        assert(W18_metadata->dims <= CHORD_META_MAX_DIM);
        std::strncpy(W18_metadata->dim_name[0], "Fbar", sizeof W18_metadata->dim_name[0]);
        std::strncpy(W18_metadata->dim_name[1], "P", sizeof W18_metadata->dim_name[1]);
        std::strncpy(W18_metadata->dim_name[2], "dishN", sizeof W18_metadata->dim_name[2]);
        std::strncpy(W18_metadata->dim_name[3], "dishM", sizeof W18_metadata->dim_name[3]);
        std::strncpy(W18_metadata->dim_name[4], "C", sizeof W18_metadata->dim_name[4]);
        W18_metadata->dim[0] = num_frequencies * U;
        W18_metadata->dim[1] = num_polarizations;
        W18_metadata->dim[2] = num_dish_locations_ew;
        W18_metadata->dim[3] = num_dish_locations_ns;
        W18_metadata->dim[4] = num_components;
        W18_metadata->sample0_offset = -1; // undefined
        W18_metadata->nfreq = num_frequencies;
        assert(W18_metadata->nfreq <= CHORD_META_MAX_FREQ);
        for (int freq = 0; freq < num_frequencies; ++freq) {
            W18_metadata->coarse_freq[freq] = freq + 1; // See `FEngine.f_engine`
            W18_metadata->freq_upchan_factor[freq] = U;
            W18_metadata->half_fpga_sample0[freq] = -1;      // undefined
            W18_metadata->time_downsampling_fpga[freq] = -1; // undefined
        }
        W18_metadata->ndishes = num_dishes;
        W18_metadata->n_dish_locations_ew = num_dish_locations_ew;
        W18_metadata->n_dish_locations_ns = num_dish_locations_ns;
        W18_metadata->dish_index = dish_indices_ptr;

        // Mark buffer as full
        W18_buffer->mark_frame_full(unique_name, W18_frame_id);
    }

    // Produce FRB2 phases
    for (int W2_frame_index = 0; W2_frame_index < W2_buffer->num_frames; ++W2_frame_index) {
        const int W2_frame_id = W2_frame_index % W2_buffer->num_frames;

        // Wait for buffer
        std::uint8_t* const W2_frame = W2_buffer->wait_for_empty_frame(unique_name, W2_frame_id);
        if (!W2_frame)
            return;
        if (!(std::ptrdiff_t(W2_buffer->frame_size) == W2_frame_size))
            FATAL_ERROR("W2_buffer->frame_size={:d} W2_frame_size={:d}", W2_buffer->frame_size,
                        W2_frame_size);
        assert(std::ptrdiff_t(W2_buffer->frame_size) == W2_frame_size);
        W2_buffer->allocate_new_metadata_object(W2_frame_id);
        set_fpga_seq_num(W2_buffer, W2_frame_id, -1);

        DEBUG("[{:d}] Filling W2 buffer...", W2_frame_index);
        float16_t* __restrict__ const W2 = (float16_t*)W2_frame;
        constexpr std::ptrdiff_t beamIn_ns_stride = 1;
        const std::ptrdiff_t beamIn_ew_stride = beamIn_ns_stride * 2 * num_dish_locations_ns;
        const std::ptrdiff_t beamOut_ns_stride = beamIn_ew_stride * 2 * num_dish_locations_ew;
        const std::ptrdiff_t beamOut_ew_stride = beamOut_ns_stride * frb2_num_beams_ns;
        const std::ptrdiff_t freq_stride = beamOut_ew_stride * frb2_num_beams_ew;
        const std::ptrdiff_t npoints = freq_stride * num_frequencies * upchannelization_factor;
        assert(std::ptrdiff_t(sizeof(float16_t)) * npoints == W2_frame_size);

        {
            using std::cos, std::sin;

            // Kendrick's FRB beamforming note, equation 7:
            //   theta = M (nhat ⋅ sigma) / lambda
            // where nhat is the unit vector in the direction of the sky location
            // sigma is the dish displacement in meters East-West.
            //
            // We'll assume that the dishes are pointed along the meridian, so
            // the boresight lies in the y,z plane (x=0)
            //   nhat_0_x = 0  (x is the direction of RA = EW, y of Dec = NS)
            //   nhat_0_y = cos(zd)
            //   nhat_0_z = sin(zd)
            // And nhat for each beam will be
            //   nhat_z ~ sin(zd - ddec)
            //   nhat_x ~ cos(zd - ddec) * sin(dra)
            //   nhat_y ~ cos(zd - ddec) * cos(dra)
            // We could probably get away with small-angle
            // approximations of beam_dra,beam_ddec,
            //   nhat_z ~ sin(zd)
            //   nhat_y ~ cos(zd) - ddec * sin(zd)    (cos(a-b) ~ cos(a) + b sin(a) when b->0);
            //   cos(dra)~1 nhat_x ~ cos(zd) * dra
            // (but here we don't use the small-angle approx)

            // This matches a function defined in Kendrick's beamforming note (eqn
            // 7/8).
            const auto Ufunc = [](int p, int M, float theta) {
                float acc = 0.0f;
                for (int s = 0; s <= M; s++) {
                    float A = s == 0 || s == M ? 0.5f : 1.0f;
                    acc += A * cos(float(M_PI) * (2 * theta - p) * s / M);
                }
                return acc;
            };
            std::vector<float> Up(2 * num_dish_locations_ew);
            std::vector<float> Uq(2 * num_dish_locations_ns);

            for (int freq0 = 0; freq0 < num_frequencies; ++freq0) {
                for (int freq1 = 0; freq1 < upchannelization_factor; ++freq1) {
                    const int freq = freq1 + upchannelization_factor * freq0;

                    // Calculate physical frequency from channel index
                    const float dfreq = adc_frequency / num_samples_per_frame;
                    const float afreq =
                        dfreq
                        * (frequency_channels.at(freq0)
                           + ((freq1 + 0.5f) / float(upchannelization_factor) - 0.5f));
                    const float c = 299792458; // speed of light
                    const float wavelength = c / afreq;

                    for (int beamOut_ew = 0; beamOut_ew < frb2_num_beams_ew; ++beamOut_ew) {
                        for (int beamOut_ns = 0; beamOut_ns < frb2_num_beams_ns; ++beamOut_ns) {

                            const float beam_dec = frb2_opening_angle_ns
                                                   * (beamOut_ns / (frb2_num_beams_ns - 1) - 0.5f);
                            const float beam_ra = frb2_opening_angle_ew
                                                  * (beamOut_ew / (frb2_num_beams_ew - 1) - 0.5f);

                            const float theta_ns = cos(frb2_bore_z - beam_dec) * cos(beam_ra)
                                                   * num_dish_locations_ns * dish_separation_ns
                                                   / wavelength;
                            const float theta_ew = cos(frb2_bore_z - beam_dec) * sin(beam_ra)
                                                   * num_dish_locations_ew * dish_separation_ew
                                                   / wavelength;

                            for (int i = 0; i < 2 * num_dish_locations_ns; i++)
                                Uq[i] = Ufunc(i, num_dish_locations_ns, theta_ns);
                            for (int i = 0; i < 2 * num_dish_locations_ew; i++)
                                Up[i] = Ufunc(i, num_dish_locations_ew, theta_ew);

                            for (int beamIn_ew = 0; beamIn_ew < 2 * num_dish_locations_ew;
                                 ++beamIn_ew) {
                                for (int beamIn_ns = 0; beamIn_ns < 2 * num_dish_locations_ns;
                                     ++beamIn_ns) {
                                    const std::ptrdiff_t n =
                                        beamIn_ns * beamIn_ns_stride + beamIn_ew * beamIn_ew_stride
                                        + beamOut_ns * beamOut_ns_stride
                                        + beamOut_ew * beamOut_ew_stride + freq * freq_stride;

                                    W2[n] = Up[beamIn_ew] * Uq[beamIn_ns];
                                }
                            }
                        }
                    }
                }
            }
        }
        DEBUG("[{:d}] Done filling W2 buffer.", W2_frame_index);

        // Set metadata
        std::shared_ptr<chordMetadata> const W2_metadata =
            get_chord_metadata(W2_buffer, W2_frame_id);
        W2_metadata->frame_counter = W2_frame_index;
        std::strncpy(W2_metadata->name, "W2", sizeof W2_metadata->name);
        W2_metadata->type = float16;
        W2_metadata->dims = 4;
        assert(W2_metadata->dims <= CHORD_META_MAX_DIM);
        std::strncpy(W2_metadata->dim_name[0], "Fbar", sizeof W2_metadata->dim_name[0]);
        std::strncpy(W2_metadata->dim_name[2], "R", sizeof W2_metadata->dim_name[1]);
        std::strncpy(W2_metadata->dim_name[3], "beamQ", sizeof W2_metadata->dim_name[2]);
        std::strncpy(W2_metadata->dim_name[4], "beamP", sizeof W2_metadata->dim_name[3]);
        W2_metadata->dim[0] = num_frequencies * upchannelization_factor;
        W2_metadata->dim[1] = frb2_num_beams_ns * frb2_num_beams_ew;
        W2_metadata->dim[2] = 2 * num_dish_locations_ew;
        W2_metadata->dim[3] = 2 * num_dish_locations_ns;
        W2_metadata->sample0_offset = -1; // undefined
        W2_metadata->nfreq = num_frequencies;
        assert(W2_metadata->nfreq <= CHORD_META_MAX_FREQ);
        for (int freq = 0; freq < num_frequencies; ++freq) {
            W2_metadata->coarse_freq[freq] = freq + 1; // See `FEngine.f_engine`
            W2_metadata->freq_upchan_factor[freq] = upchannelization_factor;
            W2_metadata->half_fpga_sample0[freq] = -1;      // undefined
            W2_metadata->time_downsampling_fpga[freq] = -1; // undefined
        }
        W2_metadata->ndishes = num_dishes;
        W2_metadata->n_dish_locations_ew = num_dish_locations_ew;
        W2_metadata->n_dish_locations_ns = num_dish_locations_ns;
        W2_metadata->dish_index = dish_indices_ptr;

        // Mark buffer as full
        W2_buffer->mark_frame_full(unique_name, W2_frame_id);
    }

    for (int E_frame_index = 0; E_frame_index < num_frames * repeat_count; ++E_frame_index) {
        const std::uint64_t seq_num = std::uint64_t(1) * num_times * E_frame_index;
        if (stop_thread)
            break;

        {
            // Produce E-field
            const int E_frame_id = E_frame_index % E_buffer->num_frames;

            // Wait for buffer
            profile_range_push("E_frame::wait_for_empty_frame");
            std::uint8_t* const E_frame = E_buffer->wait_for_empty_frame(unique_name, E_frame_id);
            profile_range_pop();
            if (!E_frame)
                break;
            if (!(std::ptrdiff_t(E_buffer->frame_size) == E_frame_size))
                FATAL_ERROR("E_buffer->frame_size={:d} E_frame_size={:d}", E_buffer->frame_size,
                            E_frame_size);
            assert(std::ptrdiff_t(E_buffer->frame_size) == E_frame_size);
            E_buffer->allocate_new_metadata_object(E_frame_id);
            set_fpga_seq_num(E_buffer, E_frame_id, seq_num);

            DEBUG("[{:d}] Filling E buffer...", E_frame_index);
            profile_range_push("E_frame::fill");
            if (!skip_julia) {
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
                    args[3] = jl_box_int64(num_polarizations);
                    args[4] = jl_box_int64(num_frequencies);
                    args[5] = jl_box_int64(num_times);
                    args[6] = jl_box_int64(E_frame_index % num_frames + 1);
                    jl_value_t* const res = jl_call(set_E, args, nargs);
                    assert(res);
                    JL_GC_POP();
                });
            } else {
                // for (int n = 0; n < num_dishes * num_polarizations * num_frequencies * num_times;
                //      ++n)
                //     ((uint8_t*)E_frame)[n] = 0x44;
                // std::memset(E_frame, 0x44,
                //        num_dishes * num_polarizations * num_frequencies * num_times);
                if (E_frame_index < E_buffer->num_frames)
                    std::memset(E_frame, 0x44,
                                num_dishes * num_polarizations * num_frequencies * num_times);
            }
            profile_range_pop();
            DEBUG("[{:d}] Done filling E buffer.", E_frame_index);

            // Set metadata
            std::shared_ptr<chordMetadata> const E_metadata =
                get_chord_metadata(E_buffer, E_frame_id);
            E_metadata->frame_counter = E_frame_index;
            std::strncpy(E_metadata->name, "E", sizeof E_metadata->name);
            E_metadata->type = int4p4;
            E_metadata->dims = 4;
            assert(E_metadata->dims <= CHORD_META_MAX_DIM);
            std::strncpy(E_metadata->dim_name[0], "T", sizeof E_metadata->dim_name[0]);
            std::strncpy(E_metadata->dim_name[1], "F", sizeof E_metadata->dim_name[1]);
            std::strncpy(E_metadata->dim_name[2], "P", sizeof E_metadata->dim_name[2]);
            std::strncpy(E_metadata->dim_name[3], "D", sizeof E_metadata->dim_name[3]);
            E_metadata->dim[0] = num_times;
            E_metadata->dim[1] = num_frequencies;
            E_metadata->dim[2] = num_polarizations;
            E_metadata->dim[3] = num_dishes;
            E_metadata->sample0_offset = seq_num;
            E_metadata->nfreq = num_frequencies;
            assert(E_metadata->nfreq <= CHORD_META_MAX_FREQ);
            for (int freq = 0; freq < num_frequencies; ++freq) {
                E_metadata->coarse_freq[freq] = frequency_channels.at(freq);
                E_metadata->freq_upchan_factor[freq] = 1;
                E_metadata->half_fpga_sample0[freq] = 0;
                E_metadata->time_downsampling_fpga[freq] = 1;
            }
            E_metadata->ndishes = num_dishes;
            E_metadata->n_dish_locations_ew = num_dish_locations_ew;
            E_metadata->n_dish_locations_ns = num_dish_locations_ns;
            E_metadata->dish_index = dish_indices_ptr;

            // Mark buffer as full
            profile_mark("E_frame::mark_frame_full");
            E_buffer->mark_frame_full(unique_name, E_frame_id);
        }

#if 0
	// We don't know how fast the expected output should be produced
        {
            // Produce expected baseband beams
            const int J_frame_index = 0;
            const int J_frame_id = s_frame_index % J_buffer->num_frames;

            // Wait for buffer
            std::uint8_t* const J_frame = J_buffer->wait_for_empty_frame(unique_name, J_frame_id);
            if (!J_frame)
                break;
            if (!(std::ptrdiff_t(J_buffer->frame_size) == J_frame_size))
                FATAL_ERROR("J_buffer->frame_size={:d} J_frame_size={:d}", J_buffer->frame_size,
                            J_frame_size);
            assert(std::ptrdiff_t(J_buffer->frame_size) == J_frame_size);
            J_buffer->allocate_new_metadata_object(J_frame_id);
            set_fpga_seq_num(J_buffer, J_frame_id, seq_num);

            // Fill buffer
            if (!skip_julia) {
                DEBUG("[{:d}] Filling J buffer...", J_frame_index);
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
                    args[6] = jl_box_int64(J_frame_index + 1);
                    jl_value_t* const res = jl_call(set_J, args, nargs);
                    assert(res);
                    JL_GC_POP();
                });
                DEBUG("[{:d}] Done filling J buffer.", J_frame_index);
            }

            // Set metadata
            std::shared_ptr<chordMetadata> const J_metadata =
                get_chord_metadata(J_buffer, J_frame_id);
            J_metadata->frame_counter = J_frame_index;
	    std::strncpy(J_metadata->name, "J", sizeof J_metadata->name);
            J_metadata->type = int4p4;
            J_metadata->dims = 4;
            assert(J_metadata->dims <= CHORD_META_MAX_DIM);
            std::strncpy(J_metadata->dim_name[0], "B", sizeof J_metadata->dim_name[0]);
            std::strncpy(J_metadata->dim_name[1], "F", sizeof J_metadata->dim_name[1]);
            std::strncpy(J_metadata->dim_name[2], "P", sizeof J_metadata->dim_name[2]);
            std::strncpy(J_metadata->dim_name[3], "T", sizeof J_metadata->dim_name[3]);
            J_metadata->dim[0] = bb_num_beams;
            J_metadata->dim[1] = num_frequencies;
            J_metadata->dim[2] = num_polarizations;
            J_metadata->dim[3] = num_times;
            J_metadata->sample0_offset = seq_num;
            J_metadata->nfreq = num_frequencies;
            assert(J_metadata->nfreq <= CHORD_META_MAX_FREQ);
            for (int freq = 0; freq < num_frequencies; ++freq) {
                J_metadata->coarse_freq[freq] = freq + 1; // See `FEngine.f_engine`
                J_metadata->freq_upchan_factor[freq] = 1;
                J_metadata->half_fpga_sample0[freq] = 0;
                J_metadata->time_downsampling_fpga[freq] = 1;
            }
            J_metadata->ndishes = num_dishes;
            J_metadata->n_dish_locations_ew = num_dish_locations_ew;
            J_metadata->n_dish_locations_ns = num_dish_locations_ns;
            J_metadata->dish_index = dish_indices_ptr;

            // Mark buffer as full
            J_buffer->mark_frame_full(unique_name, J_frame_id);
        }

        {
            // Produce expected FRB upchannelization-1 beams
            const int U = 1;
            const int I11_frame_index = 0;
            const int I11_frame_id = I11_frame_index % I11_buffer->num_frames;

            // Wait for buffer
            std::uint8_t* const I11_frame = I11_buffer->wait_for_empty_frame(unique_name, I11_frame_id);
            if (!I11_frame)
                break;
            if (!(std::ptrdiff_t(I11_buffer->frame_size) == I11_frame_size))
                FATAL_ERROR("I11_buffer->frame_size={:d} I11_frame_size={:d}", I11_buffer->frame_size,
                            I11_frame_size);
            assert(std::ptrdiff_t(I11_buffer->frame_size) == I11_frame_size);
            I11_buffer->allocate_new_metadata_object(I11_frame_id);
            set_fpga_seq_num(I11_buffer, I11_frame_id, seq_num);

            if (!skip_julia) {
                DEBUG("[{:d}] Filling I11 buffer...", I11_frame_index);
#if 0
                    // Disable this because the F-Engine simulator doesn't upchannelize yet
                    kotekan::juliaCall([&]() {
                        jl_module_t* const f_engine_module =
                            (jl_module_t*)jl_get_global(jl_main_module, jl_symbol("FEngine"));
                        assert(f_engine_module);
                        jl_function_t* const set_I11 = jl_get_function(f_engine_module, "set_I");
                        assert(set_I11);
                        const int nargs = 7;
                        jl_value_t** args;
                        JL_GC_PUSHARGS(args, nargs);
                        args[0] = jl_box_uint8pointer(I11_frame);
                        args[1] = jl_box_int64(I11_frame_size);
                        args[2] = jl_box_int64(frb_num_beams_P);
                        args[3] = jl_box_int64(frb_num_beams_Q);
                        args[4] = jl_box_int64(frb_num_times);
                        args[5] = jl_box_int64(num_frequencies * U);
                        args[6] = jl_box_int64(I11_frame_index + 1);
                        jl_value_t* const res = jl_call(set_I, args, nargs);
                        assert(res);
                        JL_GC_POP();
                    });
#else
                std::memset(I11_frame, 0, I11_frame_size);
#endif
                DEBUG("[{:d}] Done filling I11 buffer.", I11_frame_index);
            }

            std::shared_ptr<chordMetadata> const I11_metadata =
                get_chord_metadata(I11_buffer, I11_frame_id);
            I11_metadata->frame_counter = I11_frame_index;
	    std::strncpy(I11_metadata->name, "I11", sizeof I11_metadata->name);
            I11_metadata->type = float16;
            I11_metadata->dims = 4;
            assert(I11_metadata->dims <= CHORD_META_MAX_DIM);
            std::strncpy(I11_metadata->dim_name[0], "Ttilde", sizeof I11_metadata->dim_name[0]);
            std::strncpy(I11_metadata->dim_name[1], "Fbar", sizeof I11_metadata->dim_name[1]);
            std::strncpy(I11_metadata->dim_name[2], "beamQ", sizeof I11_metadata->dim_name[2]);
            std::strncpy(I11_metadata->dim_name[3], "beamP", sizeof I11_metadata->dim_name[3]);
            I11_metadata->dim[0] = frb_num_times;
            I11_metadata->dim[1] = num_frequencies * U;
            I11_metadata->dim[2] = frb_num_beams_Q;
            I11_metadata->dim[3] = frb_num_beams_P;
            I11_metadata->sample0_offset = seq_num;
            I11_metadata->nfreq = num_frequencies;
            assert(I11_metadata->nfreq <= CHORD_META_MAX_FREQ);
            for (int freq = 0; freq < num_frequencies; ++freq) {
                I11_metadata->coarse_freq[freq] = freq + 1; // See `FEngine.f_engine`
                I11_metadata->freq_upchan_factor[freq] = U;
                I11_metadata->half_fpga_sample0[freq] = 2 * Tds - 1;
                I11_metadata->time_downsampling_fpga[freq] = U * Tds;
            }
            I11_metadata->ndishes = num_dishes;
            I11_metadata->n_dish_locations_ew = num_dish_locations_ew;
            I11_metadata->n_dish_locations_ns = num_dish_locations_ns;
            I11_metadata->dish_index = dish_indices_ptr;

            // Mark buffer as full
            I11_buffer->mark_frame_full(unique_name, I11_frame_id);
        }

        {
            // Produce expected FRB upchannelization-2 beams
            const int U = 2;
            const int I12_frame_index = 0;
            const int I12_frame_id = I12_frame_index % I12_buffer->num_frames;

            // Wait for buffer
            std::uint8_t* const I12_frame = I12_buffer->wait_for_empty_frame(unique_name, I12_frame_id);
            if (!I12_frame)
                break;
            if (!(std::ptrdiff_t(I12_buffer->frame_size) == I12_frame_size))
                FATAL_ERROR("I12_buffer->frame_size={:d} I12_frame_size={:d}", I12_buffer->frame_size,
                            I12_frame_size);
            assert(std::ptrdiff_t(I12_buffer->frame_size) == I12_frame_size);
            I12_buffer->allocate_new_metadata_object(I12_frame_id);
            set_fpga_seq_num(I12_buffer, I12_frame_id, seq_num);

            if (!skip_julia) {
                DEBUG("[{:d}] Filling I12 buffer...", I12_frame_index);
#if 0
                    // Disable this because the F-Engine simulator doesn't upchannelize yet
                    kotekan::juliaCall([&]() {
                        jl_module_t* const f_engine_module =
                            (jl_module_t*)jl_get_global(jl_main_module, jl_symbol("FEngine"));
                        assert(f_engine_module);
                        jl_function_t* const set_I12 = jl_get_function(f_engine_module, "set_I");
                        assert(set_I12);
                        const int nargs = 7;
                        jl_value_t** args;
                        JL_GC_PUSHARGS(args, nargs);
                        args[0] = jl_box_uint8pointer(I12_frame);
                        args[1] = jl_box_int64(I12_frame_size);
                        args[2] = jl_box_int64(frb_num_beams_P);
                        args[3] = jl_box_int64(frb_num_beams_Q);
                        args[4] = jl_box_int64(frb_num_times);
                        args[5] = jl_box_int64(num_frequencies * U);
                        args[6] = jl_box_int64(I12_frame_index + 1);
                        jl_value_t* const res = jl_call(set_I, args, nargs);
                        assert(res);
                        JL_GC_POP();
                    });
#else
                std::memset(I12_frame, 0, I12_frame_size);
#endif
                DEBUG("[{:d}] Done filling I12 buffer.", I12_frame_index);
            }

            std::shared_ptr<chordMetadata> const I12_metadata =
                get_chord_metadata(I12_buffer, I12_frame_id);
            I12_metadata->frame_counter = I12_frame_index;
	    std::strncpy(I12_metadata->name, "I12", sizeof I12_metadata->name);
            I12_metadata->type = float16;
            I12_metadata->dims = 4;
            assert(I12_metadata->dims <= CHORD_META_MAX_DIM);
            std::strncpy(I12_metadata->dim_name[0], "Ttilde", sizeof I12_metadata->dim_name[0]);
            std::strncpy(I12_metadata->dim_name[1], "Fbar", sizeof I12_metadata->dim_name[1]);
            std::strncpy(I12_metadata->dim_name[2], "beamQ", sizeof I12_metadata->dim_name[2]);
            std::strncpy(I12_metadata->dim_name[3], "beamP", sizeof I12_metadata->dim_name[3]);
            I12_metadata->dim[0] = frb_num_times;
            I12_metadata->dim[1] = num_frequencies * U;
            I12_metadata->dim[2] = frb_num_beams_Q;
            I12_metadata->dim[3] = frb_num_beams_P;
            I12_metadata->sample0_offset = seq_num;
            I12_metadata->nfreq = num_frequencies;
            assert(I12_metadata->nfreq <= CHORD_META_MAX_FREQ);
            for (int freq = 0; freq < num_frequencies; ++freq) {
                I12_metadata->coarse_freq[freq] = freq + 1; // See `FEngine.f_engine`
                I12_metadata->freq_upchan_factor[freq] = U;
                I12_metadata->half_fpga_sample0[freq] = 2 * Tds - 1;
                I12_metadata->time_downsampling_fpga[freq] = U * Tds;
            }
            I12_metadata->ndishes = num_dishes;
            I12_metadata->n_dish_locations_ew = num_dish_locations_ew;
            I12_metadata->n_dish_locations_ns = num_dish_locations_ns;
            I12_metadata->dish_index = dish_indices_ptr;

            // Mark buffer as full
            I12_buffer->mark_frame_full(unique_name, I12_frame_id);
        }

        {
            // Produce expected FRB upchannelization-4 beams
            const int U = 4;
            const int I14_frame_index = 0;
            const int I14_frame_id = I14_frame_index % I14_buffer->num_frames;

            // Wait for buffer
            std::uint8_t* const I14_frame = I14_buffer->wait_for_empty_frame(unique_name, I14_frame_id);
            if (!I14_frame)
                break;
            if (!(std::ptrdiff_t(I14_buffer->frame_size) == I14_frame_size))
                FATAL_ERROR("I14_buffer->frame_size={:d} I14_frame_size={:d}", I14_buffer->frame_size,
                            I14_frame_size);
            assert(std::ptrdiff_t(I14_buffer->frame_size) == I14_frame_size);
            I14_buffer->allocate_new_metadata_object(I14_frame_id);
            set_fpga_seq_num(I14_buffer, I14_frame_id, seq_num);

            if (!skip_julia) {
                DEBUG("[{:d}] Filling I14 buffer...", I14_frame_index);
#if 0
                    // Disable this because the F-Engine simulator doesn't upchannelize yet
                    kotekan::juliaCall([&]() {
                        jl_module_t* const f_engine_module =
                            (jl_module_t*)jl_get_global(jl_main_module, jl_symbol("FEngine"));
                        assert(f_engine_module);
                        jl_function_t* const set_I14 = jl_get_function(f_engine_module, "set_I");
                        assert(set_I14);
                        const int nargs = 7;
                        jl_value_t** args;
                        JL_GC_PUSHARGS(args, nargs);
                        args[0] = jl_box_uint8pointer(I14_frame);
                        args[1] = jl_box_int64(I14_frame_size);
                        args[2] = jl_box_int64(frb_num_beams_P);
                        args[3] = jl_box_int64(frb_num_beams_Q);
                        args[4] = jl_box_int64(frb_num_times);
                        args[5] = jl_box_int64(num_frequencies * U);
                        args[6] = jl_box_int64(I14_frame_index + 1);
                        jl_value_t* const res = jl_call(set_I, args, nargs);
                        assert(res);
                        JL_GC_POP();
                    });
#else
                std::memset(I14_frame, 0, I14_frame_size);
#endif
                DEBUG("[{:d}] Done filling I14 buffer.", I14_frame_index);
            }

            std::shared_ptr<chordMetadata> const I14_metadata =
                get_chord_metadata(I14_buffer, I14_frame_id);
            I14_metadata->frame_counter = I14_frame_index;
	    std::strncpy(I14_metadata->name, "I14", sizeof I14_metadata->name);
            I14_metadata->type = float16;
            I14_metadata->dims = 4;
            assert(I14_metadata->dims <= CHORD_META_MAX_DIM);
            std::strncpy(I14_metadata->dim_name[0], "Ttilde", sizeof I14_metadata->dim_name[0]);
            std::strncpy(I14_metadata->dim_name[1], "Fbar", sizeof I14_metadata->dim_name[1]);
            std::strncpy(I14_metadata->dim_name[2], "beamQ", sizeof I14_metadata->dim_name[2]);
            std::strncpy(I14_metadata->dim_name[3], "beamP", sizeof I14_metadata->dim_name[3]);
            I14_metadata->dim[0] = frb_num_times;
            I14_metadata->dim[1] = num_frequencies * U;
            I14_metadata->dim[2] = frb_num_beams_Q;
            I14_metadata->dim[3] = frb_num_beams_P;
            I14_metadata->sample0_offset = seq_num;
            I14_metadata->nfreq = num_frequencies;
            assert(I14_metadata->nfreq <= CHORD_META_MAX_FREQ);
            for (int freq = 0; freq < num_frequencies; ++freq) {
                I14_metadata->coarse_freq[freq] = freq + 1; // See `FEngine.f_engine`
                I14_metadata->freq_upchan_factor[freq] = U;
                I14_metadata->half_fpga_sample0[freq] = 2 * Tds - 1;
                I14_metadata->time_downsampling_fpga[freq] = U * Tds;
            }
            I14_metadata->ndishes = num_dishes;
            I14_metadata->n_dish_locations_ew = num_dish_locations_ew;
            I14_metadata->n_dish_locations_ns = num_dish_locations_ns;
            I14_metadata->dish_index = dish_indices_ptr;

            // Mark buffer as full
            I14_buffer->mark_frame_full(unique_name, I14_frame_id);
        }

        {
            // Produce expected FRB upchannelization-8 beams
            const int U = 8;
            const int I18_frame_index = 0;
            const int I18_frame_id = I18_frame_index % I18_buffer->num_frames;

            // Wait for buffer
            std::uint8_t* const I18_frame = I18_buffer->wait_for_empty_frame(unique_name, I18_frame_id);
            if (!I18_frame)
                break;
            if (!(std::ptrdiff_t(I18_buffer->frame_size) == I18_frame_size))
                FATAL_ERROR("I18_buffer->frame_size={:d} I18_frame_size={:d}", I18_buffer->frame_size,
                            I18_frame_size);
            assert(std::ptrdiff_t(I18_buffer->frame_size) == I18_frame_size);
            I18_buffer->allocate_new_metadata_object(I18_frame_id);
            set_fpga_seq_num(I18_buffer, I18_frame_id, seq_num);

            if (!skip_julia) {
                DEBUG("[{:d}] Filling I18 buffer...", I18_frame_index);
#if 0
                    // Disable this because the F-Engine simulator doesn't upchannelize yet
                    kotekan::juliaCall([&]() {
                        jl_module_t* const f_engine_module =
                            (jl_module_t*)jl_get_global(jl_main_module, jl_symbol("FEngine"));
                        assert(f_engine_module);
                        jl_function_t* const set_I18 = jl_get_function(f_engine_module, "set_I");
                        assert(set_I18);
                        const int nargs = 7;
                        jl_value_t** args;
                        JL_GC_PUSHARGS(args, nargs);
                        args[0] = jl_box_uint8pointer(I18_frame);
                        args[1] = jl_box_int64(I18_frame_size);
                        args[2] = jl_box_int64(frb_num_beams_P);
                        args[3] = jl_box_int64(frb_num_beams_Q);
                        args[4] = jl_box_int64(frb_num_times);
                        args[5] = jl_box_int64(num_frequencies * U);
                        args[6] = jl_box_int64(I18_frame_index + 1);
                        jl_value_t* const res = jl_call(set_I, args, nargs);
                        assert(res);
                        JL_GC_POP();
                    });
#else
                std::memset(I18_frame, 0, I18_frame_size);
#endif
                DEBUG("[{:d}] Done filling I18 buffer.", I18_frame_index);
            }

            std::shared_ptr<chordMetadata> const I18_metadata =
                get_chord_metadata(I18_buffer, I18_frame_id);
            I18_metadata->frame_counter = I18_frame_index;
	    std::strncpy(I18_metadata->name, "I18", sizeof I18_metadata->name);
            I18_metadata->type = float16;
            I18_metadata->dims = 4;
            assert(I18_metadata->dims <= CHORD_META_MAX_DIM);
            std::strncpy(I18_metadata->dim_name[0], "Ttilde", sizeof I18_metadata->dim_name[0]);
            std::strncpy(I18_metadata->dim_name[1], "Fbar", sizeof I18_metadata->dim_name[1]);
            std::strncpy(I18_metadata->dim_name[2], "beamQ", sizeof I18_metadata->dim_name[2]);
            std::strncpy(I18_metadata->dim_name[3], "beamP", sizeof I18_metadata->dim_name[3]);
            I18_metadata->dim[0] = frb_num_times;
            I18_metadata->dim[1] = num_frequencies * U;
            I18_metadata->dim[2] = frb_num_beams_Q;
            I18_metadata->dim[3] = frb_num_beams_P;
            I18_metadata->sample0_offset = seq_num;
            I18_metadata->nfreq = num_frequencies;
            assert(I18_metadata->nfreq <= CHORD_META_MAX_FREQ);
            for (int freq = 0; freq < num_frequencies; ++freq) {
                I18_metadata->coarse_freq[freq] = freq + 1; // See `FEngine.f_engine`
                I18_metadata->freq_upchan_factor[freq] = U;
                I18_metadata->half_fpga_sample0[freq] = 2 * Tds - 1;
                I18_metadata->time_downsampling_fpga[freq] = U * Tds;
            }
            I18_metadata->ndishes = num_dishes;
            I18_metadata->n_dish_locations_ew = num_dish_locations_ew;
            I18_metadata->n_dish_locations_ns = num_dish_locations_ns;
            I18_metadata->dish_index = dish_indices_ptr;

            // Mark buffer as full
            I18_buffer->mark_frame_full(unique_name, I18_frame_id);
        }
#endif

    } // for E_frame_index

    INFO("Done.");
}

#endif
