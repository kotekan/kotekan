#include "FakeVisPattern.hpp"

#include "Config.hpp"         // for Config
#include "Hash.hpp"           // for Hash
#include "CHORDTelescope.hpp" // for CHORDTelescope
#include "datasetManager.hpp" // for datasetManager, state_id_t, dset_id_t
#include "datasetState.hpp"   // for flagState, inputState
#include "timeUtil.hpp"       // for get_ERA_from_GPS
#include "visBuffer.hpp"      // for VisFrameView
#include "visUtil.hpp"        // for cfloat, input_ctype, ts_to_double, cmap

#include "fmt.hpp"      // for format
#include "gsl-lite.hpp" // for span
#include "json.hpp"     // for json, basic_json, basic_json<>::object_t

#include <algorithm> // for copy, max, copy_backward
#include <complex>   // for complex, operator*
#include <cstdint>   // for uint32_t, uint16_t
#include <exception> // for exception
#include <lapacke.h> // for LAPACKE_cheevr, LAPACK_ROW_MAJOR
#include <map>       // for map, map<>::mapped_type
#include <math.h>    // for cosf, sinf
#include <regex>     // for match_results<>::_Base_type
#include <stdexcept> // for invalid_argument, runtime_error
#include <tuple>     // for get
#include <vector>    // for vector, __alloc_traits<>::value_type

static constexpr double C = 299792458.0;

// Register test patterns
REGISTER_FAKE_VIS_PATTERN(DefaultVisPattern, "default");
REGISTER_FAKE_VIS_PATTERN(FillIJVisPattern, "fill_ij");
REGISTER_FAKE_VIS_PATTERN(FillIJMissingVisPattern, "fill_ij_missing");
REGISTER_FAKE_VIS_PATTERN(PhaseIJVisPattern, "phase_ij");
REGISTER_FAKE_VIS_PATTERN(ChimeVisPattern, "chime");
REGISTER_FAKE_VIS_PATTERN(TestPatternSimpleVisPattern, "test_pattern_simple");
REGISTER_FAKE_VIS_PATTERN(TestPatternFreqVisPattern, "test_pattern_freq");
REGISTER_FAKE_VIS_PATTERN(TestPatternInputVisPattern, "test_pattern_inputs");
REGISTER_FAKE_VIS_PATTERN(ChangeStatePattern, "change_state");
REGISTER_FAKE_VIS_PATTERN(PointSourceVisPattern, "point_source");


FakeVisPattern::FakeVisPattern(kotekan::Config& config, const std::string& path) {
    set_log_level(config.get<std::string>(path, "log_level"));
    set_log_prefix(path);
}


DefaultVisPattern::DefaultVisPattern(kotekan::Config& config, const std::string& path) :
    FakeVisPattern(config, path) {}


void DefaultVisPattern::fill(VisFrameView& frame) {
    auto out_vis = frame.vis;
    // Set diagonal elements to (0, row)
    for (uint32_t i = 0; i < frame.num_elements; i++) {
        uint32_t pi = cmap(i, i, frame.num_elements);
        out_vis[pi] = {0., (float)i};
    }
    // Save metadata in first few cells
    if (out_vis.size() < 3) {
        FATAL_ERROR("Number of elements ({:d}) is too small to encode the 3 debugging values of "
                    "fill-mode 'default' in fake visibilities.\nExiting...",
                    frame.num_elements);
    } else {
        // For simplicity overwrite diagonal if needed
        out_vis[0] = {(float)std::get<0>(frame.time), 0.0};
        out_vis[1] = {(float)ts_to_double(std::get<1>(frame.time)), 0.0};
        out_vis[2] = {(float)frame.freq_id, 0.};
        // out_vis[3] = {(float) output_frame_id, 0.};
    }
}

void DefaultVisPattern::fill(N2FrameView& frame) {
    auto out_vis = frame.vis;
    // Set diagonal elements to (0, row)
    for (uint32_t i = 0; i < frame.num_elements; i++) {
        uint32_t pi = cmap(i, i, frame.num_elements);
        out_vis[pi] = {0., (float)i};
    }
    // Save metadata in first few cells
    if (out_vis.size() < 3) {
        FATAL_ERROR("Number of elements ({:d}) is too small to encode the 3 debugging values of "
                    "fill-mode 'default' in fake visibilities.\nExiting...",
                    frame.num_elements);
    } else {
        out_vis[0] = {(float)frame._metadata->frame_start_time_ns, 0.0};
        out_vis[1] = {(float)frame._metadata->frame_start_time_ns, 0.0};
        out_vis[2] = {(float)frame.freq_id, 0.};
    }
}

FillIJVisPattern::FillIJVisPattern(kotekan::Config& config, const std::string& path) :
    FakeVisPattern(config, path) {}


void FillIJVisPattern::fill(VisFrameView& frame) {
    int ind = 0;
    for (uint32_t i = 0; i < frame.num_elements; i++) {
        for (uint32_t j = i; j < frame.num_elements; j++) {
            frame.vis[ind] = {(float)i, (float)j};
            ind++;
        }
    }
}

void FillIJVisPattern::fill(N2FrameView& frame) {
    int ind = 0;
    for (uint32_t i = 0; i < frame.num_elements; i++) {
        for (uint32_t j = i; j < frame.num_elements; j++) {
            frame.vis[ind] = {(float)i, (float)j};
            ind++;
        }
    }
}


FillIJMissingVisPattern::FillIJMissingVisPattern(kotekan::Config& config, const std::string& path) :
    FillIJVisPattern(config, path) {}


void FillIJMissingVisPattern::fill(VisFrameView& frame) {
    FillIJVisPattern::fill(frame);

    frame.fpga_seq_total = frame.fpga_seq_length - 2;
    frame.rfi_total = 1;
}

void FillIJMissingVisPattern::fill(N2FrameView& frame) {
    FillIJVisPattern::fill(frame);

    frame._metadata->n_valid_fpga_ticks_in_frame = frame._metadata->frame_length_fpga_ticks - 2;
    frame._metadata->n_rfi_fpga_ticks = 1;
}

PhaseIJVisPattern::PhaseIJVisPattern(kotekan::Config& config, const std::string& path) :
    FakeVisPattern(config, path) {}


void PhaseIJVisPattern::fill(VisFrameView& frame) {
    int ind = 0;
    for (uint32_t i = 0; i < frame.num_elements; i++) {
        for (uint32_t j = i; j < frame.num_elements; j++) {
            float phase = (float)i - (float)j;
            frame.vis[ind] = {cosf(phase), sinf(phase)};
            ind++;
        }
    }
}

void PhaseIJVisPattern::fill(N2FrameView& frame) {
    int ind = 0;
    for (uint32_t i = 0; i < frame.num_elements; i++) {
        for (uint32_t j = i; j < frame.num_elements; j++) {
            float phase = (float)i - (float)j;
            frame.vis[ind] = {cosf(phase), sinf(phase)};
            ind++;
        }
    }
}

ChimeVisPattern::ChimeVisPattern(kotekan::Config& config, const std::string& path) :
    FakeVisPattern(config, path) {}

void ChimeVisPattern::fill(VisFrameView& frame) {
    int ind = 0;
    for (uint32_t i = 0; i < frame.num_elements; i++) {
        for (uint32_t j = i; j < frame.num_elements; j++) {
            int cyl_i = i / 512;
            int cyl_j = j / 512;

            int pos_i = i % 256;
            int pos_j = j % 256;

            frame.vis[ind] = {(float)(cyl_j - cyl_i), (float)(pos_j - pos_i)};
            ind++;
        }
    }
}

void ChimeVisPattern::fill(N2FrameView& frame) {
    std::shared_ptr<metadataPool> pool = frame._metadata->parent_pool.lock();
    ERROR("Invalid fill pattern (ChimeVisPattern) for N2 frame {:d}.", pool->type_name);
}


TestPatternSimpleVisPattern::TestPatternSimpleVisPattern(kotekan::Config& config,
                                                         const std::string& path) :
    FakeVisPattern(config, path) {
    test_pattern_value = config.get_default<cfloat>(path, "default_val", {1., 0.});
}


void TestPatternSimpleVisPattern::fill(VisFrameView& frame) {
    // Fill vis
    int ind = 0;
    for (uint32_t i = 0; i < frame.num_elements; i++) {
        for (uint32_t j = i; j < frame.num_elements; j++) {
            frame.vis[ind] = test_pattern_value;
            ind++;
        }
    }

    // Fill ev (slightly different to the vals in fill_non_vis)
    for (uint32_t i = 0; i < frame.num_ev; i++) {
        for (uint32_t j = 0; j < frame.num_elements; j++) {
            int k = i * frame.num_elements + j;
            frame.evec[k] = {(float)i, 1};
        }
        frame.eval[i] = i;
    }

    // Fill weights (slightly different to the vals in fill_non_vis)
    ind = 0;
    for (uint32_t i = 0; i < frame.num_elements; i++) {
        for (uint32_t j = i; j < frame.num_elements; j++) {
            frame.weight[ind] = 1.;
            ind++;
        }
    }
}

void TestPatternSimpleVisPattern::fill(N2FrameView& frame) {
    // Fill vis
    int ind = 0;
    for (uint32_t i = 0; i < frame.num_elements; i++) {
        for (uint32_t j = i; j < frame.num_elements; j++) {
            frame.vis[ind] = test_pattern_value;
            ind++;
        }
    }

    // Fill ev (slightly different to the vals in fill_non_vis)
    for (uint32_t i = 0; i < frame.num_ev; i++) {
        for (uint32_t j = 0; j < frame.num_elements; j++) {
            int k = i * frame.num_elements + j;
            frame.evec[k] = {(float)i, 1};
        }
        frame.eval[i] = i;
    }

    // Fill weights (slightly different to the vals in fill_non_vis)
    ind = 0;
    for (uint32_t i = 0; i < frame.num_elements; i++) {
        for (uint32_t j = i; j < frame.num_elements; j++) {
            frame.weight[ind] = 1.;
            ind++;
        }
    }
}


TestPatternFreqVisPattern::TestPatternFreqVisPattern(kotekan::Config& config,
                                                     const std::string& path) :
    FakeVisPattern(config, path) {
    auto default_val = config.get_default<cfloat>(path, "default_val", {128., 0.});
    auto bins = config.get<std::vector<uint32_t>>(path, "frequencies");
    auto bin_values = config.get<std::vector<cfloat>>(path, "freq_values");
    auto freq = config.get<std::vector<uint32_t>>(path, "freq_ids");

    if (bins.size() != bin_values.size()) {
        throw std::invalid_argument(
            "fakeVis: lengths of frequencies (" + std::to_string(bins.size()) + ") and freq_value ("
            + std::to_string(bin_values.size()) + ") arrays have to be equal.");
    }
    if (bins.size() > freq.size()) {
        throw std::invalid_argument("fakeVis: length of frequencies array ("
                                    + std::to_string(bins.size())
                                    + ") can not be larger "
                                      "than size of freq_ids array ("
                                    + std::to_string(freq.size()) + ").");
    }

    test_pattern_value = std::vector<cfloat>(freq.size());
    for (size_t i = 0; i < freq.size(); i++) {
        size_t j;
        for (j = 0; j < bins.size(); j++) {
            if (bins.at(j) == i)
                break;
        }
        if (j == bins.size())
            test_pattern_value[i] = default_val * std::conj(default_val);
        else
            test_pattern_value[i] = bin_values.at(j) * std::conj(bin_values.at(j));
    }
    DEBUG("Using test pattern `test_pattern_freq` with default value {:f}+{:f}j and {:d} frequency "
          "values",
          default_val.real(), default_val.imag(), bins.size());
}

void TestPatternFreqVisPattern::fill(VisFrameView& frame) {
    cfloat fill_value = test_pattern_value.at(frame.freq_id);

    // Fill vis
    int ind = 0;
    for (uint32_t i = 0; i < frame.num_elements; i++) {
        for (uint32_t j = i; j < frame.num_elements; j++) {
            frame.vis[ind] = fill_value;
            ind++;
        }
    }

    // Fill ev
    for (uint32_t i = 0; i < frame.num_ev; i++) {
        for (uint32_t j = 0; j < frame.num_elements; j++) {
            int k = i * frame.num_elements + j;
            frame.evec[k] = {(float)i, fill_value.real()};
        }
        frame.eval[i] = i;
    }
    frame.erms = fill_value.real();

    // Fill weights
    ind = 0;
    for (uint32_t i = 0; i < frame.num_elements; i++) {
        for (uint32_t j = i; j < frame.num_elements; j++) {
            frame.weight[ind] = fill_value.real();
            ind++;
        }
    }
}

void TestPatternFreqVisPattern::fill(N2FrameView& frame) {
    std::shared_ptr<metadataPool> pool = frame._metadata->parent_pool.lock();
    ERROR("Invalid fill pattern (ChimeVisPattern) for N2 frame {:d}.", pool->type_name);
}


TestPatternInputVisPattern::TestPatternInputVisPattern(kotekan::Config& config,
                                                       const std::string& path) :
    FakeVisPattern(config, path) {
    auto input_values = config.get<std::vector<cfloat>>(path, "input_values");
    auto num_elements = config.get<size_t>(path, "num_elements");

    if (input_values.size() != num_elements) {
        throw std::invalid_argument(
            "fakeVis: lengths of input values (" + std::to_string(input_values.size())
            + ") and number of elements (" + std::to_string(num_elements) + ") have to be equal.");
    }

    size_t num_prods = num_elements * (num_elements + 1) / 2;
    test_pattern_value = std::vector<cfloat>(num_prods);
    size_t ind = 0;
    for (size_t i = 0; i < num_elements; i++) {
        for (size_t j = 0; j <= i; j++) {
            test_pattern_value[ind] = input_values.at(j) * std::conj(input_values.at(i));
            ind++;
        }
    }
    DEBUG("Using test pattern mode `test_pattern_inputs` with {:d} input values",
          input_values.size());
}

void TestPatternInputVisPattern::fill(VisFrameView& frame) {
    // Fill vis
    int ind = 0;
    for (uint32_t i = 0; i < frame.num_elements; i++) {
        for (uint32_t j = i; j < frame.num_elements; j++) {
            frame.vis[ind] = test_pattern_value[ind];
            ind++;
        }
    }
    // Fill ev
    for (uint32_t i = 0; i < frame.num_ev; i++) {
        for (uint32_t j = 0; j < frame.num_elements; j++) {
            int k = i * frame.num_elements + j;
            frame.evec[k] = {(float)i, 1};
        }
        frame.eval[i] = i;
    }

    // Fill weights
    ind = 0;
    for (uint32_t i = 0; i < frame.num_elements; i++) {
        for (uint32_t j = i; j < frame.num_elements; j++) {
            frame.weight[ind] = 1;
            ind++;
        }
    }
}

void TestPatternInputVisPattern::fill(N2FrameView& frame) {
    std::shared_ptr<metadataPool> pool = frame._metadata->parent_pool.lock();
    ERROR("Invalid fill pattern (ChimeVisPattern) for N2 frame {:d}.", pool->type_name);
}


ChangeStatePattern::ChangeStatePattern(kotekan::Config& config, const std::string& path) :
    DefaultVisPattern(config, path) {

    // Map for state generators
    std::map<std::string, gen_state> gen_state_map;
    gen_state_map["inputs"] = std::bind(&ChangeStatePattern::gen_state_inputs, this);
    gen_state_map["flags"] = std::bind(&ChangeStatePattern::gen_state_flags, this);
    gen_state_map["gains"] = std::bind(&ChangeStatePattern::gen_state_gains, this);

    auto state_changes = config.get_default<nlohmann::json>(path, "state_changes", {});

    for (const auto& s : state_changes) {
        double ts = s["timestamp"].get<double>();
        std::string type = s["type"].get<std::string>();

        if (gen_state_map.count(type) == 0) {
            FATAL_ERROR("State type '{}' not understood.", type);
        }

        _dataset_changes.push_back({ts, gen_state_map.at(type)});
    }

    num_elements = config.get<size_t>(path, "num_elements");
}

void ChangeStatePattern::fill(VisFrameView& frame) {

    auto& dm = datasetManager::instance();

    DefaultVisPattern::fill(frame);
    double frame_time = ts_to_double(std::get<1>(frame.time));

    if (!current_dset_id) {
        current_dset_id = frame.dataset_id;
    }

    // If there are still changes to apply, check that we've exceeded the start
    // time and then update the state
    while (_dataset_changes.size() > 0) {
        auto& [ts, func] = _dataset_changes[0];

        if (frame_time < ts)
            break;

        state_id_t id = func();
        current_dset_id = dm.add_dataset(id, current_dset_id.value());
        _dataset_changes.pop_front();
    }

    frame.dataset_id = current_dset_id.value();
}


state_id_t ChangeStatePattern::gen_state_inputs() {

    std::vector<input_ctype> inputs;

    for (uint16_t i = 0; i < num_elements; i++) {
        inputs.emplace_back(i, fmt::format("input_{}_{}", _input_update_ind++, i));
    }

    auto& dm = datasetManager::instance();
    return dm.create_state<inputState>(inputs).first;
}


state_id_t ChangeStatePattern::gen_state_flags() {

    std::string update_id = fmt::format("flag_update_{}", _flag_update_ind++);
    auto& dm = datasetManager::instance();
    return dm.create_state<flagState>(update_id).first;
}

state_id_t ChangeStatePattern::gen_state_gains() {

    std::string update_id = fmt::format("gain_update_{}", _gain_update_ind);
    auto& dm = datasetManager::instance();
    return dm.create_state<gainState>(update_id, _gain_update_ind++).first;
}

PointSourceVisPattern::PointSourceVisPattern(kotekan::Config& config,
                                             const std::string& path) : 
        FakeVisPattern(config, path) {
    ra = config.get_default<double>(path, "ra", 0.0);
    dec = config.get_default<double>(path, "dec", 45.0);
    stokes_I = config.get_default<double>(path, "stokes_I", 1.0);
    stokes_Q = config.get_default<double>(path, "stokes_Q", 0.0);
    stokes_U = config.get_default<double>(path, "stokes_U", 0.0);
    stokes_V = config.get_default<double>(path, "stokes_V", 0.0);
    noise_var = config.get_default<double>(path, "noise_var", 0.01);
}

void PointSourceVisPattern::fill(VisFrameView& frame) {

}

void PointSourceVisPattern::fill(N2FrameView& frame) {


    const CHORDTelescope& tel = Telescope::instance().cast<CHORDTelescope>();

    uint32_t num_dishes = tel.get_num_dishes();
    uint32_t num_elements = frame.num_elements;

    double dut1 = tel.get_dut1();
    double dtai = tel.get_dtai();

    timespec gps_time = tel.to_time(frame.fpga_start_tick
                                    + frame.frame_length_fpga_ticks/2);

    double era = get_ERA_from_GPS(gps_time, dtai, dut1);

    std::array<double, 3> n = tel.get_sky_vec_in_dish_coords(ra, dec, era);

    double f = tel.to_freq(frame.freq_id);
    double lambda = C / f;

    INFO("Making fake vis at t: {:d} s + {:d} ns",
            gps_time.tv_sec, gps_time.tv_nsec);
    INFO("                   f: {:d} = {:e} Hz", frame.freq_id, f);
    INFO("                   ERA: {:f} deg", era);

    int ind = 0;
    for (uint32_t el_i = 0; el_i < num_elements; el_i++) {
        for (uint32_t el_j = el_i; el_j < num_elements; el_j++) {

            uint32_t dish_i = el_i % (num_dishes);
            uint32_t dish_j = el_j % (num_dishes);
            uint32_t pol_i = el_i / num_dishes;
            uint32_t pol_j = el_j / num_dishes;

            double phase = 2*M_PI * (
                      (tel.get_dish_coord(dish_i, 0)
                        - tel.get_dish_coord(dish_j, 0)) * n[0]
                    + (tel.get_dish_coord(dish_i, 1)
                        - tel.get_dish_coord(dish_j, 1)) * n[1]
                    + (tel.get_dish_coord(dish_i, 2)
                        - tel.get_dish_coord(dish_j, 2)) * n[2])
                    / lambda;

            double cp = cos(phase);
            double sp = sin(phase);

            double power_r = 0;
            double power_i = 0;
            if (pol_i == 0 && pol_j == 0)
                power_r = stokes_I + stokes_Q;
            else if (pol_i == 1 && pol_j == 1)
                power_r = stokes_I - stokes_Q;
            else if (pol_i == 0 && pol_j == 1) {
                power_r = stokes_U;
                power_i = stokes_V;
            }
            else if (pol_i == 1 && pol_j == 0) {
                power_r = stokes_U;
                power_i = -stokes_V;
            }

            frame.vis[ind] = {(float)(power_r * cp - power_i * sp),
                              (float)(power_r * sp + power_i * cp)};
            ind++;
        }
    }

    //Memory for LAPACK
    std::vector<N2::cfloat> vis_square;
    std::vector<N2::cfloat> evecs;
    std::vector<float> evals;
    
    vis_square.resize(num_elements * num_elements, 0);
    evecs.resize(num_elements * frame.num_ev);
    evals.resize(num_elements);

    uint32_t vis_idx = 0;

    for(uint32_t i = 0; i < num_elements; i++) {
        for(uint32_t j = i; j < num_elements; j++) {
            vis_square[i * num_elements + j] = frame.vis[vis_idx];
            vis_idx++;
        }
    }


    int info;
    int64_t nside = num_elements;
    int ev_found;

    LAPACKE_cheevr(LAPACK_ROW_MAJOR, 'V', 'I', 'U', nside,
                   (lapack_complex_float *)vis_square.data(), nside,
                   0.0, 0.0, nside-frame.num_ev, nside, 
                   0.0, &ev_found, evals.data(),
                   (lapack_complex_float *)evecs.data(), nside,
                   nullptr);

    for(uint32_t i = 0; i < frame.num_ev; i++)
    {
        frame.eval[i] = evals[nside-i-1];
        for(uint32_t j = 0; j < num_elements; j++)
        {
            frame.evec[i*num_elements + j]
                = evecs[(nside-i-1)*num_elements + j];
        }
    }
}
