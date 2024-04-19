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
    upchan_max_num_channelss{
        config.get<int>(unique_name, "upchan_U1_max_num_channels"),
        config.get<int>(unique_name, "upchan_U2_max_num_channels"),
        config.get<int>(unique_name, "upchan_U4_max_num_channels"),
        config.get<int>(unique_name, "upchan_U8_max_num_channels"),
        config.get<int>(unique_name, "upchan_U16_max_num_channels"),
        config.get<int>(unique_name, "upchan_U32_max_num_channels"),
        config.get<int>(unique_name, "upchan_U64_max_num_channels"),
    },
    upchan_min_channels{
        config.get<int>(unique_name, "upchan_U1_min_channel"),
        config.get<int>(unique_name, "upchan_U2_min_channel"),
        config.get<int>(unique_name, "upchan_U4_min_channel"),
        config.get<int>(unique_name, "upchan_U8_min_channel"),
        config.get<int>(unique_name, "upchan_U16_min_channel"),
        config.get<int>(unique_name, "upchan_U32_min_channel"),
        config.get<int>(unique_name, "upchan_U64_min_channel"),
    },
    upchan_max_channels{
        config.get<int>(unique_name, "upchan_U1_max_channel"),
        config.get<int>(unique_name, "upchan_U2_max_channel"),
        config.get<int>(unique_name, "upchan_U4_max_channel"),
        config.get<int>(unique_name, "upchan_U8_max_channel"),
        config.get<int>(unique_name, "upchan_U16_max_channel"),
        config.get<int>(unique_name, "upchan_U32_max_channel"),
        config.get<int>(unique_name, "upchan_U64_max_channel"),
    },
    upchan_all_max_num_output_channels(
        config.get<int>(unique_name, "upchan_all_max_num_output_channels")),
    upchan_all_min_output_channel(config.get<int>(unique_name, "upchan_all_min_output_channel")),
    upchan_all_max_output_channel(config.get<int>(unique_name, "upchan_all_max_output_channel")),

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
    G_frame_sizes{
        0,
        std::int64_t(sizeof(float16_t)) * upchan_max_num_channelss[U2] * upchan_factor(U2),
        std::int64_t(sizeof(float16_t)) * upchan_max_num_channelss[U4] * upchan_factor(U4),
        std::int64_t(sizeof(float16_t)) * upchan_max_num_channelss[U8] * upchan_factor(U8),
        std::int64_t(sizeof(float16_t)) * upchan_max_num_channelss[U16] * upchan_factor(U16),
        std::int64_t(sizeof(float16_t)) * upchan_max_num_channelss[U32] * upchan_factor(U32),
        std::int64_t(sizeof(float16_t)) * upchan_max_num_channelss[U64] * upchan_factor(U64),
    },
    W1_frame_sizes{
        std::int64_t(sizeof(float16_t)) * num_components * num_dish_locations_ns
            * num_dish_locations_ew * num_polarizations * upchan_max_num_channelss[U1]
            * upchan_factor(U1),
        std::int64_t(sizeof(float16_t)) * num_components * num_dish_locations_ns
            * num_dish_locations_ew * num_polarizations * upchan_max_num_channelss[U2]
            * upchan_factor(U2),
        std::int64_t(sizeof(float16_t)) * num_components * num_dish_locations_ns
            * num_dish_locations_ew * num_polarizations * upchan_max_num_channelss[U4]
            * upchan_factor(U4),
        std::int64_t(sizeof(float16_t)) * num_components * num_dish_locations_ns
            * num_dish_locations_ew * num_polarizations * upchan_max_num_channelss[U8]
            * upchan_factor(U8),
        std::int64_t(sizeof(float16_t)) * num_components * num_dish_locations_ns
            * num_dish_locations_ew * num_polarizations * upchan_max_num_channelss[U16]
            * upchan_factor(U16),
        std::int64_t(sizeof(float16_t)) * num_components * num_dish_locations_ns
            * num_dish_locations_ew * num_polarizations * upchan_max_num_channelss[U32]
            * upchan_factor(U32),
        std::int64_t(sizeof(float16_t)) * num_components * num_dish_locations_ns
            * num_dish_locations_ew * num_polarizations * upchan_max_num_channelss[U64]
            * upchan_factor(U64),
    },
    W2_frame_size(sizeof(float16_t) * (frb1_num_beams_P * frb1_num_beams_Q)
                  * (frb2_num_beams_ns * frb2_num_beams_ew)
                  // TODO: Fix this once the frequencies are recombined
                  * (upchan_max_num_channelss[U4] * 4)),
    I1_frame_sizes{
        std::int64_t(sizeof(float16_t)) * frb1_num_beams_P * frb1_num_beams_Q
            * (upchan_max_num_channelss[U1] * upchan_factor(U1)) * frb_num_times,
        std::int64_t(sizeof(float16_t)) * frb1_num_beams_P * frb1_num_beams_Q
            * (upchan_max_num_channelss[U2] * upchan_factor(U2)) * frb_num_times,
        std::int64_t(sizeof(float16_t)) * frb1_num_beams_P * frb1_num_beams_Q
            * (upchan_max_num_channelss[U4] * upchan_factor(U4)) * frb_num_times,
        std::int64_t(sizeof(float16_t)) * frb1_num_beams_P * frb1_num_beams_Q
            * (upchan_max_num_channelss[U8] * upchan_factor(U8)) * frb_num_times,
        std::int64_t(sizeof(float16_t)) * frb1_num_beams_P * frb1_num_beams_Q
            * (upchan_max_num_channelss[U16] * upchan_factor(U32)) * frb_num_times,
        std::int64_t(sizeof(float16_t)) * frb1_num_beams_P * frb1_num_beams_Q
            * (upchan_max_num_channelss[U32] * upchan_factor(U64)) * frb_num_times,
    },

    // Buffers
    E_buffer(get_buffer("E_buffer")), A_buffer(get_buffer("A_buffer")),
    s_buffer(get_buffer("s_buffer")), J_buffer(get_buffer("J_buffer")),
    G_buffers{
        nullptr,
        get_buffer("G_U2_buffer"),
        get_buffer("G_U4_buffer"),
        get_buffer("G_U8_buffer"),
        get_buffer("G_U16_buffer"),
        get_buffer("G_U32_buffer"),
        get_buffer("G_U64_buffer"),
    },
    W1_buffers{
        get_buffer("W1_U1_buffer"),  get_buffer("W1_U2_buffer"),  get_buffer("W1_U4_buffer"),
        get_buffer("W1_U8_buffer"),  get_buffer("W1_U16_buffer"), get_buffer("W1_U32_buffer"),
        get_buffer("W1_U64_buffer"),
    },
    W2_buffer(get_buffer("W2_buffer")),
    I1_buffers{
        get_buffer("I1_U1_buffer"),  get_buffer("I1_U2_buffer"),  get_buffer("I1_U4_buffer"),
        get_buffer("I1_U8_buffer"),  get_buffer("I1_U16_buffer"), get_buffer("I1_U32_buffer"),
        get_buffer("I1_U64_buffer"),
    }

{
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

    assert(upchan_all_min_output_channel >= 0);
    assert(upchan_all_min_output_channel <= upchan_all_max_output_channel);
    assert(upchan_all_max_output_channel <= upchan_all_max_num_output_channels);
    for (int Uindex = 0; Uindex < Usize; ++Uindex) {
        assert(upchan_min_channels[Uindex] >= upchan_all_min_output_channel);
        assert(upchan_min_channels[Uindex] <= upchan_max_channels[Uindex]);
        assert(upchan_max_channels[Uindex] <= upchan_all_max_output_channel);
        assert(upchan_max_channels[Uindex] <= num_frequencies);
        assert(upchan_max_num_channelss[Uindex] >= 0);
        assert(upchan_max_channels[Uindex] - upchan_min_channels[Uindex]
               <= upchan_max_num_channelss[Uindex]);
        // Check that the local channels are contiguous
        if (Uindex == Usize - 1)
            assert(upchan_min_channels[Uindex] == 0);
        else
            assert(upchan_min_channels[Uindex] == upchan_max_channels[Uindex + 1]);
        if (Uindex == 0)
            assert(upchan_max_channels[Uindex] == num_frequencies);
    }

    assert(E_buffer);
    assert(A_buffer);
    assert(s_buffer);
    assert(J_buffer);
    for (int Uindex = 0; Uindex < Usize; ++Uindex)
        if (Uindex == 0)
            assert(!G_buffers[Uindex]);
        else
            assert(G_buffers[Uindex]);
    for (auto W1_buffer : W1_buffers)
        assert(W1_buffer);
    assert(W2_buffer);
    for (auto I1_buffer : I1_buffers)
        assert(I1_buffer);
    E_buffer->register_producer(unique_name);
    A_buffer->register_producer(unique_name);
    s_buffer->register_producer(unique_name);
    J_buffer->register_producer(unique_name);
    for (auto G_buffer : G_buffers)
        if (G_buffer)
            G_buffer->register_producer(unique_name);
    for (auto W1_buffer : W1_buffers)
        W1_buffer->register_producer(unique_name);
    W2_buffer->register_producer(unique_name);
    for (auto I1_buffer : I1_buffers)
        I1_buffer->register_producer(unique_name);

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
        for (int d = A_metadata->dims - 1; d >= 0; --d)
            if (d == A_metadata->dims - 1)
                A_metadata->stride[d] = 1;
            else
                A_metadata->stride[d] = A_metadata->stride[d + 1] * A_metadata->dim[d + 1];
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
        for (int d = s_metadata->dims - 1; d >= 0; --d)
            if (d == s_metadata->dims - 1)
                s_metadata->stride[d] = 1;
            else
                s_metadata->stride[d] = s_metadata->stride[d + 1] * s_metadata->dim[d + 1];
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

    // Produce upchannelization gains
    for (int Uindex = 0; Uindex < Usize; ++Uindex) {
        const upchan_factor_t Ufactor = upchan_factor_t(Uindex);
        const int U = upchan_factor(Ufactor);
        // Skip U = 1
        if (U == 1)
            continue;
        for (int G_frame_index = 0; G_frame_index < G_buffers[Ufactor]->num_frames;
             ++G_frame_index) {
            const int G_frame_id = G_frame_index % G_buffers[Ufactor]->num_frames;

            // Wait for buffer
            std::uint8_t* const G_frame =
                G_buffers[Ufactor]->wait_for_empty_frame(unique_name, G_frame_id);
            if (!G_frame)
                return;
            if (!(std::ptrdiff_t(G_buffers[Ufactor]->frame_size) == G_frame_sizes[Ufactor]))
                FATAL_ERROR("G_buffers[U{:d}]->frame_size={:d} G_frame_sizes[U{:d}]={:d}", U,
                            G_buffers[Ufactor]->frame_size, U, G_frame_sizes[Ufactor]);
            assert(std::ptrdiff_t(G_buffers[Ufactor]->frame_size) == G_frame_sizes[Ufactor]);
            G_buffers[Ufactor]->allocate_new_metadata_object(G_frame_id);
            set_fpga_seq_num(G_buffers[Ufactor], G_frame_id, -1);

            DEBUG("[{:d}] Filling G_U{:d} buffer...", G_frame_index, U);
            const int num_local_channels =
                upchan_max_channels[Ufactor] - upchan_min_channels[Ufactor];
            for (int n = 0; n < num_local_channels * U; ++n)
                ((float16_t*)G_frame)[n] = 1;
            DEBUG("[{:d}] Done filling G_U{:d} buffer.", G_frame_index, U);

            // Set metadata
            std::shared_ptr<chordMetadata> const G_metadata =
                get_chord_metadata(G_buffers[Ufactor], G_frame_id);
            G_metadata->frame_counter = G_frame_index;
            std::snprintf(G_metadata->name, sizeof G_metadata->name, "G_U%d", U);
            G_metadata->type = float16;
            G_metadata->dims = 1;
            assert(G_metadata->dims <= CHORD_META_MAX_DIM);
            std::snprintf(G_metadata->dim_name[0], sizeof G_metadata->dim_name[0], "Fbar_U%d", U);
            G_metadata->dim[0] = upchan_max_num_channelss[Ufactor] * U;
            for (int d = G_metadata->dims - 1; d >= 0; --d)
                if (d == G_metadata->dims - 1)
                    G_metadata->stride[d] = 1;
                else
                    G_metadata->stride[d] = G_metadata->stride[d + 1] * G_metadata->dim[d + 1];
            G_metadata->sample0_offset = -1; // undefined
            G_metadata->nfreq = num_frequencies;
            assert(G_metadata->nfreq <= CHORD_META_MAX_FREQ);
            for (int freq = 0; freq < num_frequencies; ++freq) {
                G_metadata->coarse_freq[freq] = freq + 1;      // See `FEngine.f_engine`
                G_metadata->freq_upchan_factor[freq] = 1;      // upchannelization_factor;
                G_metadata->half_fpga_sample0[freq] = -1;      // undefined
                G_metadata->time_downsampling_fpga[freq] = -1; // undefined
            }
            G_metadata->ndishes = num_dishes;
            G_metadata->n_dish_locations_ew = num_dish_locations_ew;
            G_metadata->n_dish_locations_ns = num_dish_locations_ns;
            G_metadata->dish_index = dish_indices_ptr;

            // Mark buffer as full
            G_buffers[Ufactor]->mark_frame_full(unique_name, G_frame_id);
        }
    }

    // Produce FRB1 phases
    for (int Uindex = 0; Uindex < Usize; ++Uindex) {
        const upchan_factor_t Ufactor = upchan_factor_t(Uindex);
        const int U = upchan_factor(Ufactor);
        for (int W1_frame_index = 0; W1_frame_index < W1_buffers[Ufactor]->num_frames;
             ++W1_frame_index) {
            const int W1_frame_id = W1_frame_index % W1_buffers[Ufactor]->num_frames;

            // Wait for buffer
            std::uint8_t* const W1_frame =
                W1_buffers[Ufactor]->wait_for_empty_frame(unique_name, W1_frame_id);
            if (!W1_frame)
                return;
            if (!(std::ptrdiff_t(W1_buffers[Ufactor]->frame_size) == W1_frame_sizes[Ufactor]))
                FATAL_ERROR("W1_buffers[U{:d}]->frame_size={:d} W1_frame_sizes[U{:d}]={:d}", U,
                            W1_buffers[Ufactor]->frame_size, U, W1_frame_sizes[Ufactor]);
            assert(std::ptrdiff_t(W1_buffers[Ufactor]->frame_size) == W1_frame_sizes[Ufactor]);
            W1_buffers[Ufactor]->allocate_new_metadata_object(W1_frame_id);
            set_fpga_seq_num(W1_buffers[Ufactor], W1_frame_id, -1);

            DEBUG("[{:d}] Filling W1_U{:d} buffer...", W1_frame_index, U);
            const int num_local_channels =
                upchan_max_channels[Ufactor] - upchan_min_channels[Ufactor];
            // Disable this because the F-Engine simulator doesn't upchannelize yet
            if (false && !skip_julia) {
                kotekan::juliaCall([&]() {
                    jl_module_t* const f_engine_module =
                        (jl_module_t*)jl_get_global(jl_main_module, jl_symbol("FEngine"));
                    assert(f_engine_module);
                    jl_function_t* const set_W1 = jl_get_function(f_engine_module, "set_W");
                    assert(set_W1);
                    const int nargs = 7;
                    jl_value_t** args;
                    JL_GC_PUSHARGS(args, nargs);
                    args[0] = jl_box_uint8pointer(W1_frame);
                    args[1] = jl_box_int64(W1_frame_sizes[Ufactor]);
                    args[2] = jl_box_int64(num_dish_locations_ns);
                    args[3] = jl_box_int64(num_dish_locations_ew);
                    args[4] = jl_box_int64(num_polarizations);
                    args[5] = jl_box_int64(num_local_channels * U);
                    args[6] = jl_box_int64(W1_frame_index + 1);
                    jl_value_t* const res = jl_call(set_W1, args, nargs);
                    assert(res);
                    JL_GC_POP();
                });
            } else {
                for (int n = 0; n < num_components * num_dish_locations_ns * num_dish_locations_ew
                                        * num_polarizations * num_local_channels * U;
                     ++n)
                    ((float16_t*)W1_frame)[n] = 1;
            }
            DEBUG("[{:d}] Done filling W1_U{:d} buffer.", W1_frame_index, U);

            // Set metadata
            std::shared_ptr<chordMetadata> const W1_metadata =
                get_chord_metadata(W1_buffers[Ufactor], W1_frame_id);
            W1_metadata->frame_counter = W1_frame_index;
            std::snprintf(W1_metadata->name, sizeof W1_metadata->name, "W_U%d", U);
            W1_metadata->type = float16;
            W1_metadata->dims = 5;
            assert(W1_metadata->dims <= CHORD_META_MAX_DIM);
            std::snprintf(W1_metadata->dim_name[0], sizeof W1_metadata->dim_name[0], "Fbar_U%d", U);
            std::strncpy(W1_metadata->dim_name[1], "P", sizeof W1_metadata->dim_name[1]);
            std::strncpy(W1_metadata->dim_name[2], "dishN", sizeof W1_metadata->dim_name[2]);
            std::strncpy(W1_metadata->dim_name[3], "dishM", sizeof W1_metadata->dim_name[3]);
            std::strncpy(W1_metadata->dim_name[4], "C", sizeof W1_metadata->dim_name[4]);
            W1_metadata->dim[0] = upchan_max_num_channelss[Ufactor] * U;
            W1_metadata->dim[1] = num_polarizations;
            W1_metadata->dim[2] = num_dish_locations_ew;
            W1_metadata->dim[3] = num_dish_locations_ns;
            W1_metadata->dim[4] = num_components;
            for (int d = W1_metadata->dims - 1; d >= 0; --d)
                if (d == W1_metadata->dims - 1)
                    W1_metadata->stride[d] = 1;
                else
                    W1_metadata->stride[d] = W1_metadata->stride[d + 1] * W1_metadata->dim[d + 1];
            W1_metadata->sample0_offset = -1; // undefined
            W1_metadata->nfreq = num_local_channels;
            assert(W1_metadata->nfreq <= CHORD_META_MAX_FREQ);
            for (int freq = 0; freq < num_frequencies; ++freq) {
                W1_metadata->coarse_freq[freq] = freq + 1; // See `FEngine.f_engine`
                W1_metadata->freq_upchan_factor[freq] = U;
                W1_metadata->half_fpga_sample0[freq] = -1;      // undefined
                W1_metadata->time_downsampling_fpga[freq] = -1; // undefined
            }
            W1_metadata->ndishes = num_dishes;
            W1_metadata->n_dish_locations_ew = num_dish_locations_ew;
            W1_metadata->n_dish_locations_ns = num_dish_locations_ns;
            W1_metadata->dish_index = dish_indices_ptr;

            // Mark buffer as full
            W1_buffers[Ufactor]->mark_frame_full(unique_name, W1_frame_id);
        }
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
        // TODO: Fix this once the frequencies are recombined
        // const std::ptrdiff_t tmp_num_frequencies = num_frequencies * upchannelization_factor;
        const std::ptrdiff_t tmp_num_frequencies = upchan_max_num_channelss[U4] * 4;
        const std::ptrdiff_t npoints = freq_stride * tmp_num_frequencies;
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

            for (int freq0 = 0; freq0 < tmp_num_frequencies / 4; ++freq0) {
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
        W2_metadata->dim[0] = tmp_num_frequencies;
        W2_metadata->dim[1] = frb2_num_beams_ns * frb2_num_beams_ew;
        W2_metadata->dim[2] = 2 * num_dish_locations_ew;
        W2_metadata->dim[3] = 2 * num_dish_locations_ns;
        for (int d = W2_metadata->dims - 1; d >= 0; --d)
            if (d == W2_metadata->dims - 1)
                W2_metadata->stride[d] = 1;
            else
                W2_metadata->stride[d] = W2_metadata->stride[d + 1] * W2_metadata->dim[d + 1];
        W2_metadata->sample0_offset = -1; // undefined
        W2_metadata->nfreq = tmp_num_frequencies / 4;
        assert(W2_metadata->nfreq <= CHORD_META_MAX_FREQ);
        for (int freq = 0; freq < tmp_num_frequencies / 4; ++freq) {
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
            for (int d = E_metadata->dims - 1; d >= 0; --d)
                if (d == E_metadata->dims - 1)
                    E_metadata->stride[d] = 1;
                else
                    E_metadata->stride[d] = E_metadata->stride[d + 1] * E_metadata->dim[d + 1];
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
            for (int d = J_metadata->dims - 1; d >= 0; --d)
                if (d == J_metadata->dims - 1)
                    J_metadata->stride[d] = 1;
                else
                    J_metadata->stride[d] = J_metadata->stride[d + 1] * J_metadata->dim[d + 1];
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

        for (int Uindex = 0; Uindex < Usize; ++Uindex) {
            const upchan_factor_t Ufactor = upchan_factor_t(Uindex);
            const int U = upchan_factor(Ufactor);

            const int I1_frame_index = 0;
            const int I1_frame_id = I1_frame_index % I1_buffer->num_frames;

            // Wait for buffer
            std::uint8_t* const I1_frame = I1_buffer->wait_for_empty_frame(unique_name, I1_frame_id);
            if (!I1_frame)
                break;
            if (!(std::ptrdiff_t(I1_buffer->frame_size) == I1_frame_size))
                FATAL_ERROR("I1_buffer->frame_size={:d} I1_frame_size={:d}", I1_buffer->frame_size,
                            I1_frame_size);
            assert(std::ptrdiff_t(I1_buffer->frame_size) == I1_frame_size);
            I1_buffer->allocate_new_metadata_object(I1_frame_id);
            set_fpga_seq_num(I1_buffer, I1_frame_id, seq_num);

            if (!skip_julia) {
                DEBUG("[{:d}] Filling I1 buffer...", I1_frame_index);
#if 0
                    // Disable this because the F-Engine simulator doesn't upchannelize yet
                    kotekan::juliaCall([&]() {
                        jl_module_t* const f_engine_module =
                            (jl_module_t*)jl_get_global(jl_main_module, jl_symbol("FEngine"));
                        assert(f_engine_module);
                        jl_function_t* const set_I1 = jl_get_function(f_engine_module, "set_I");
                        assert(set_I1);
                        const int nargs = 7;
                        jl_value_t** args;
                        JL_GC_PUSHARGS(args, nargs);
                        args[0] = jl_box_uint8pointer(I1_frame);
                        args[1] = jl_box_int64(I1_frame_size);
                        args[2] = jl_box_int64(frb_num_beams_P);
                        args[3] = jl_box_int64(frb_num_beams_Q);
                        args[4] = jl_box_int64(frb_num_times);
                        args[5] = jl_box_int64(num_frequencies * U);
                        args[6] = jl_box_int64(I1_frame_index + 1);
                        jl_value_t* const res = jl_call(set_I, args, nargs);
                        assert(res);
                        JL_GC_POP();
                    });
#else
                std::memset(I1_frame, 0, I1_frame_size);
#endif
                DEBUG("[{:d}] Done filling I1 buffer.", I1_frame_index);
            }

            std::shared_ptr<chordMetadata> const I1_metadata =
                get_chord_metadata(I1_buffer, I1_frame_id);
            I1_metadata->frame_counter = I1_frame_index;
	    std::strncpy(I1_metadata->name, "I1", sizeof I1_metadata->name);
            I1_metadata->type = float16;
            I1_metadata->dims = 4;
            assert(I1_metadata->dims <= CHORD_META_MAX_DIM);
            std::strncpy(I1_metadata->dim_name[0], "Ttilde", sizeof I1_metadata->dim_name[0]);
            std::strncpy(I1_metadata->dim_name[1], "Fbar", sizeof I1_metadata->dim_name[1]);
            std::strncpy(I1_metadata->dim_name[2], "beamQ", sizeof I1_metadata->dim_name[2]);
            std::strncpy(I1_metadata->dim_name[3], "beamP", sizeof I1_metadata->dim_name[3]);
            I1_metadata->dim[0] = frb_num_times;
            I1_metadata->dim[1] = num_frequencies * U;
            I1_metadata->dim[2] = frb_num_beams_Q;
            I1_metadata->dim[3] = frb_num_beams_P;
            for (int d = I1_metadata->dims - 1; d >= 0; --d)
                if (d == I1_metadata->dims - 1)
                    I1_metadata->stride[d] = 1;
                else
                    I1_metadata->stride[d] = I1_metadata->stride[d + 1] * I1_metadata->dim[d + 1];
            I1_metadata->sample0_offset = seq_num;
            I1_metadata->nfreq = num_frequencies;
            assert(I1_metadata->nfreq <= CHORD_META_MAX_FREQ);
            for (int freq = 0; freq < num_frequencies; ++freq) {
                I1_metadata->coarse_freq[freq] = freq + 1; // See `FEngine.f_engine`
                I1_metadata->freq_upchan_factor[freq] = U;
                I1_metadata->half_fpga_sample0[freq] = 2 * Tds - 1;
                I1_metadata->time_downsampling_fpga[freq] = U * Tds;
            }
            I1_metadata->ndishes = num_dishes;
            I1_metadata->n_dish_locations_ew = num_dish_locations_ew;
            I1_metadata->n_dish_locations_ns = num_dish_locations_ns;
            I1_metadata->dish_index = dish_indices_ptr;

            // Mark buffer as full
            I1_buffer->mark_frame_full(unique_name, I1_frame_id);
        }

#endif

    } // for E_frame_index

    INFO("Done.");
}

#endif
