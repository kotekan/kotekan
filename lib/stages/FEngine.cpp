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
#include <visUtil.hpp>

#if !KOTEKAN_FLOAT16
#warning "The F-Engine simulator requires float16 support"
#else

class FEngine : public kotekan::Stage {
    const std::string unique_name;

    const bool skip_julia;

    // Basic constants
    const int num_components;
    const int num_polarizations;

    // Sky
    const float source_amplitude;
    const float source_frequency;
    const float source_position_ew;
    const float source_position_ns;

    // Dishes
    const int num_dish_locations_ew;
    const int num_dish_locations_ns;
    const int num_dish_locations;
    const float dish_separation_ew;
    const float dish_separation_ns;
    const int num_dishes;
    const std::vector<int> dish_indices;
    std::vector<int> dish_locations;
    int* dish_indices_ptr;

    // ADC
    const float adc_frequency;
    const int num_taps;
    const int num_frequencies;
    const int num_times;

    // Baseband beamformer setup
    const int bb_num_dishes_M;
    const int bb_num_dishes_N;
    const int bb_num_beams_P;
    const int bb_num_beams_Q;
    const float bb_beam_separation_x;
    const float bb_beam_separation_y;
    const int bb_num_beams;

    // Upchannelizer setup
    const int upchannelization_factor;

    // FRB beamformer setup
    const int frb_num_beams_P;
    const int frb_num_beams_Q;
    const int Tds = 40;
    const int frb_num_times;

    // Pipeline
    const int num_frames;

    // Kotekan
    const std::int64_t E_frame_size;
    const std::int64_t A_frame_size;
    const std::int64_t s_frame_size;
    const std::int64_t J_frame_size;
    const std::int64_t G_frame_size;
    const std::int64_t W_frame_size;
    const std::int64_t I_frame_size;

    Buffer* const E_buffer;
    Buffer* const A_buffer;
    Buffer* const s_buffer;
    Buffer* const J_buffer;
    Buffer* const G_buffer;
    Buffer* const W_buffer;
    Buffer* const I_buffer;

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
    num_taps(config.get<int>(unique_name, "num_taps")),
    num_frequencies(config.get<int>(unique_name, "num_frequencies")),
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
    frb_num_beams_P(2 * num_dish_locations_ns), frb_num_beams_Q(2 * num_dish_locations_ew),
    frb_num_times(num_times / upchannelization_factor / Tds),
    // Pipeline
    num_frames(config.get<int>(unique_name, "num_frames")),
    // Frame sizes
    E_frame_size(std::int64_t(1) * num_dishes * num_polarizations * num_frequencies * num_times),
    A_frame_size(std::int64_t(1) * num_components * num_dishes * bb_num_beams * num_polarizations
                 * num_frequencies),
    s_frame_size(std::int64_t(1) * sizeof(int32_t) * bb_num_beams * num_polarizations
                 * num_frequencies),
    J_frame_size(std::int64_t(1) * num_times * num_polarizations * num_frequencies * bb_num_beams),
    G_frame_size(std::int64_t(1) * sizeof(float16_t) * num_frequencies * upchannelization_factor),
    W_frame_size(std::int64_t(1) * sizeof(float16_t) * num_components * num_dish_locations_ns
                 * num_dish_locations_ew * num_polarizations
                 * (num_frequencies * upchannelization_factor)),
    I_frame_size(std::int64_t(1) * sizeof(float16_t) * frb_num_beams_P * frb_num_beams_Q
                 * (num_frequencies * upchannelization_factor) * frb_num_times),
    // Buffers
    E_buffer(get_buffer("E_buffer")), A_buffer(get_buffer("A_buffer")),
    s_buffer(get_buffer("s_buffer")), J_buffer(get_buffer("J_buffer")),
    G_buffer(get_buffer("G_buffer")), W_buffer(get_buffer("W_buffer")),
    I_buffer(get_buffer("I_buffer")) {
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

    assert(E_buffer);
    assert(A_buffer);
    assert(s_buffer);
    assert(J_buffer);
    assert(G_buffer);
    assert(W_buffer);
    assert(I_buffer);
    E_buffer->register_producer(unique_name);
    A_buffer->register_producer(unique_name);
    s_buffer->register_producer(unique_name);
    J_buffer->register_producer(unique_name);
    G_buffer->register_producer(unique_name);
    W_buffer->register_producer(unique_name);
    I_buffer->register_producer(unique_name);

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
        if (!skip_julia) {
            INFO("[{:d}] Filling A buffer...", A_frame_index);
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
            INFO("[{:d}] Done filling A buffer.", A_frame_index);
        }

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
        if (!skip_julia) {
            INFO("[{:d}] Filling s buffer...", s_frame_index);
            for (int n = 0; n < bb_num_beams * num_polarizations * num_frequencies; ++n)
                ((int32_t*)s_frame)[n] = 13;
            INFO("[{:d}] Done filling s buffer.", s_frame_index);
        }

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

    // Produce upchannelization gains
    for (int G_frame_index = 0; G_frame_index < G_buffer->num_frames; ++G_frame_index) {
        const int G_frame_id = G_frame_index % G_buffer->num_frames;

        // Wait for buffer
        std::uint8_t* const G_frame = G_buffer->wait_for_empty_frame(unique_name, G_frame_id);
        if (!G_frame)
            return;
        if (!(std::ptrdiff_t(G_buffer->frame_size) == G_frame_size))
            FATAL_ERROR("G_buffer->frame_size={:d} G_frame_size={:d}", G_buffer->frame_size,
                        G_frame_size);
        assert(std::ptrdiff_t(G_buffer->frame_size) == G_frame_size);
        G_buffer->allocate_new_metadata_object(G_frame_id);
        set_fpga_seq_num(G_buffer, G_frame_id, -1);

        // Set metadata
        std::shared_ptr<chordMetadata> const G_metadata = get_chord_metadata(G_buffer, G_frame_id);
        G_metadata->frame_counter = G_frame_index;
        std::strncpy(G_metadata->name, "G", sizeof G_metadata->name);
        G_metadata->type = float16;
        G_metadata->dims = 1;
        assert(G_metadata->dims <= CHORD_META_MAX_DIM);
        std::strncpy(G_metadata->dim_name[0], "Fbar", sizeof G_metadata->dim_name[0]);
        G_metadata->dim[0] = num_frequencies * upchannelization_factor;
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

        if (!skip_julia) {
            INFO("[{:d}] Filling G buffer...", G_frame_index);
            _Float16* __restrict__ const G = (_Float16*)G_frame;
            for (int freqbar = 0; freqbar < num_frequencies * upchannelization_factor; ++freqbar) {
                const std::size_t ind = freqbar + std::size_t(0);
                G[ind] = 1;
            }
            INFO("[{:d}] Done filling G buffer.", G_frame_index);
        }

        // Mark buffer as full
        G_buffer->mark_frame_full(unique_name, G_frame_id);
    }

    // Produce FRB phases
    for (int W_frame_index = 0; W_frame_index < W_buffer->num_frames; ++W_frame_index) {
        const int W_frame_id = W_frame_index % W_buffer->num_frames;

        // Wait for buffer
        std::uint8_t* const W_frame = W_buffer->wait_for_empty_frame(unique_name, W_frame_id);
        if (!W_frame)
            return;
        if (!(std::ptrdiff_t(W_buffer->frame_size) == W_frame_size))
            FATAL_ERROR("W_buffer->frame_size={:d} W_frame_size={:d}", W_buffer->frame_size,
                        W_frame_size);
        assert(std::ptrdiff_t(W_buffer->frame_size) == W_frame_size);
        W_buffer->allocate_new_metadata_object(W_frame_id);
        set_fpga_seq_num(W_buffer, W_frame_id, -1);

        if (!skip_julia) {
            INFO("[{:d}] Filling W buffer...", W_frame_index);
#if 0
                    // Disable this because the F-Engine simulator doesn't upchannelize yet
                    kotekan::juliaCall([&]() {
                        jl_module_t* const f_engine_module =
                            (jl_module_t*)jl_get_global(jl_main_module, jl_symbol("FEngine"));
                        assert(f_engine_module);
                        jl_function_t* const set_W = jl_get_function(f_engine_module, "set_W");
                        assert(set_W);
                        const int nargs = 7;
                        jl_value_t** args;
                        JL_GC_PUSHARGS(args, nargs);
                        args[0] = jl_box_uint8pointer(W_frame);
                        args[1] = jl_box_int64(W_frame_size);
                        args[2] = jl_box_int64(num_dish_locations_ns);
                        args[3] = jl_box_int64(num_dish_locations_ew);
                        args[4] = jl_box_int64(num_polarizations);
                        args[5] = jl_box_int64(num_frequencies * upchannelization_factor);
                        args[6] = jl_box_int64(W_frame_index + 1);
                        jl_value_t* const res = jl_call(set_W, args, nargs);
                        assert(res);
                        JL_GC_POP();
                    });
#else
            std::memset(W_frame, 0, W_frame_size);
#endif
            INFO("[{:d}] Done filling W buffer.", W_frame_index);
        }

        // Set metadata
        std::shared_ptr<chordMetadata> const W_metadata = get_chord_metadata(W_buffer, W_frame_id);
        W_metadata->frame_counter = W_frame_index;
        std::strncpy(W_metadata->name, "W", sizeof W_metadata->name);
        W_metadata->type = float16;
        W_metadata->dims = 5;
        assert(W_metadata->dims <= CHORD_META_MAX_DIM);
        std::strncpy(W_metadata->dim_name[0], "Fbar", sizeof W_metadata->dim_name[0]);
        std::strncpy(W_metadata->dim_name[1], "P", sizeof W_metadata->dim_name[1]);
        std::strncpy(W_metadata->dim_name[2], "dishN", sizeof W_metadata->dim_name[2]);
        std::strncpy(W_metadata->dim_name[3], "dishM", sizeof W_metadata->dim_name[3]);
        std::strncpy(W_metadata->dim_name[4], "C", sizeof W_metadata->dim_name[4]);
        W_metadata->dim[0] = num_frequencies * upchannelization_factor;
        W_metadata->dim[1] = num_polarizations;
        W_metadata->dim[2] = num_dish_locations_ew;
        W_metadata->dim[3] = num_dish_locations_ns;
        W_metadata->dim[4] = num_components;
        W_metadata->sample0_offset = -1; // undefined
        W_metadata->nfreq = num_frequencies;
        assert(W_metadata->nfreq <= CHORD_META_MAX_FREQ);
        for (int freq = 0; freq < num_frequencies; ++freq) {
            W_metadata->coarse_freq[freq] = freq + 1; // See `FEngine.f_engine`
            W_metadata->freq_upchan_factor[freq] = upchannelization_factor;
            W_metadata->half_fpga_sample0[freq] = -1;      // undefined
            W_metadata->time_downsampling_fpga[freq] = -1; // undefined
        }
        W_metadata->ndishes = num_dishes;
        W_metadata->n_dish_locations_ew = num_dish_locations_ew;
        W_metadata->n_dish_locations_ns = num_dish_locations_ns;
        W_metadata->dish_index = dish_indices_ptr;

        // Mark buffer as full
        W_buffer->mark_frame_full(unique_name, W_frame_id);
    }


    for (int E_frame_index = 0; E_frame_index < num_frames; ++E_frame_index) {
        const std::uint64_t seq_num = std::uint64_t(1) * num_times * E_frame_index;
        if (stop_thread)
            break;

        {
            // Produce E-field
            const int E_frame_id = E_frame_index % E_buffer->num_frames;

            // Wait for buffer
            std::uint8_t* const E_frame = E_buffer->wait_for_empty_frame(unique_name, E_frame_id);
            if (!E_frame)
                break;
            if (!(std::ptrdiff_t(E_buffer->frame_size) == E_frame_size))
                FATAL_ERROR("E_buffer->frame_size={:d} E_frame_size={:d}", E_buffer->frame_size,
                            E_frame_size);
            assert(std::ptrdiff_t(E_buffer->frame_size) == E_frame_size);
            E_buffer->allocate_new_metadata_object(E_frame_id);
            set_fpga_seq_num(E_buffer, E_frame_id, seq_num);

            if (!skip_julia) {
                INFO("[{:d}] Filling E buffer...", E_frame_index);
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
                    args[6] = jl_box_int64(E_frame_index + 1);
                    jl_value_t* const res = jl_call(set_E, args, nargs);
                    assert(res);
                    JL_GC_POP();
                });
                INFO("[{:d}] Done filling E buffer.", E_frame_index);
            }

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
                E_metadata->coarse_freq[freq] = freq + 1; // See `FEngine.f_engine`
                E_metadata->freq_upchan_factor[freq] = 1;
                E_metadata->half_fpga_sample0[freq] = 0;
                E_metadata->time_downsampling_fpga[freq] = 1;
            }
            E_metadata->ndishes = num_dishes;
            E_metadata->n_dish_locations_ew = num_dish_locations_ew;
            E_metadata->n_dish_locations_ns = num_dish_locations_ns;
            E_metadata->dish_index = dish_indices_ptr;

            // Mark buffer as full
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
                INFO("[{:d}] Filling J buffer...", J_frame_index);
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
                INFO("[{:d}] Done filling J buffer.", J_frame_index);
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
            // Produce expected FRB beams
            const int I_frame_index = 0;
            const int I_frame_id = I_frame_index % I_buffer->num_frames;

            // Wait for buffer
            std::uint8_t* const I_frame = I_buffer->wait_for_empty_frame(unique_name, I_frame_id);
            if (!I_frame)
                break;
            if (!(std::ptrdiff_t(I_buffer->frame_size) == I_frame_size))
                FATAL_ERROR("I_buffer->frame_size={:d} I_frame_size={:d}", I_buffer->frame_size,
                            I_frame_size);
            assert(std::ptrdiff_t(I_buffer->frame_size) == I_frame_size);
            I_buffer->allocate_new_metadata_object(I_frame_id);
            set_fpga_seq_num(I_buffer, I_frame_id, seq_num);

            if (!skip_julia) {
                INFO("[{:d}] Filling I buffer...", I_frame_index);
#if 0
                    // Disable this because the F-Engine simulator doesn't upchannelize yet
                    kotekan::juliaCall([&]() {
                        jl_module_t* const f_engine_module =
                            (jl_module_t*)jl_get_global(jl_main_module, jl_symbol("FEngine"));
                        assert(f_engine_module);
                        jl_function_t* const set_I = jl_get_function(f_engine_module, "set_I");
                        assert(set_I);
                        const int nargs = 7;
                        jl_value_t** args;
                        JL_GC_PUSHARGS(args, nargs);
                        args[0] = jl_box_uint8pointer(I_frame);
                        args[1] = jl_box_int64(I_frame_size);
                        args[2] = jl_box_int64(frb_num_beams_P);
                        args[3] = jl_box_int64(frb_num_beams_Q);
                        args[4] = jl_box_int64(frb_num_times);
                        args[5] = jl_box_int64(num_frequencies * upchannelization_factor);
                        args[6] = jl_box_int64(I_frame_index + 1);
                        jl_value_t* const res = jl_call(set_I, args, nargs);
                        assert(res);
                        JL_GC_POP();
                    });
#else
                std::memset(I_frame, 0, I_frame_size);
#endif
                INFO("[{:d}] Done filling I buffer.", I_frame_index);
            }

            std::shared_ptr<chordMetadata> const I_metadata =
                get_chord_metadata(I_buffer, I_frame_id);
            I_metadata->frame_counter = I_frame_index;
	    std::strncpy(I_metadata->name, "I", sizeof I_metadata->name);
            I_metadata->type = float16;
            I_metadata->dims = 4;
            assert(I_metadata->dims <= CHORD_META_MAX_DIM);
            std::strncpy(I_metadata->dim_name[0], "Ttilde", sizeof I_metadata->dim_name[0]);
            std::strncpy(I_metadata->dim_name[1], "Fbar", sizeof I_metadata->dim_name[1]);
            std::strncpy(I_metadata->dim_name[2], "beamQ", sizeof I_metadata->dim_name[2]);
            std::strncpy(I_metadata->dim_name[3], "beamP", sizeof I_metadata->dim_name[3]);
            I_metadata->dim[0] = frb_num_times;
            I_metadata->dim[1] = num_frequencies * upchannelization_factor;
            I_metadata->dim[2] = frb_num_beams_Q;
            I_metadata->dim[3] = frb_num_beams_P;
            I_metadata->sample0_offset = seq_num;
            I_metadata->nfreq = num_frequencies;
            assert(I_metadata->nfreq <= CHORD_META_MAX_FREQ);
            for (int freq = 0; freq < num_frequencies; ++freq) {
                I_metadata->coarse_freq[freq] = freq + 1; // See `FEngine.f_engine`
                I_metadata->freq_upchan_factor[freq] = upchannelization_factor;
                I_metadata->half_fpga_sample0[freq] = 2 * Tds - 1;
                I_metadata->time_downsampling_fpga[freq] = upchannelization_factor * Tds;
            }
            I_metadata->ndishes = num_dishes;
            I_metadata->n_dish_locations_ew = num_dish_locations_ew;
            I_metadata->n_dish_locations_ns = num_dish_locations_ns;
            I_metadata->dish_index = dish_indices_ptr;

            // Mark buffer as full
            I_buffer->mark_frame_full(unique_name, I_frame_id);
        }
#endif

    } // for E_frame_index

    INFO("Done.");
}

#endif
