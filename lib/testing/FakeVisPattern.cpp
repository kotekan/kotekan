#include "FakeVisPattern.hpp"

#include "visUtil.hpp"

#include <inttypes.h>
#include <vector>


// Register test patterns
REGISTER_FAKE_VIS_PATTERN(DefaultVisPattern, "default");
REGISTER_FAKE_VIS_PATTERN(FillIJVisPattern, "fill_ij");
REGISTER_FAKE_VIS_PATTERN(FillIJMissingVisPattern, "fill_ij_missing");
REGISTER_FAKE_VIS_PATTERN(PhaseIJVisPattern, "phase_ij");
REGISTER_FAKE_VIS_PATTERN(ChimeVisPattern, "chime");
REGISTER_FAKE_VIS_PATTERN(TestPatternSimpleVisPattern, "test_pattern_simple");
REGISTER_FAKE_VIS_PATTERN(TestPatternFreqVisPattern, "test_pattern_freq");
REGISTER_FAKE_VIS_PATTERN(TestPatternInputVisPattern, "test_pattern_inputs");


FakeVisPattern::FakeVisPattern(kotekan::Config& config, const std::string& path) {
    set_log_level(config.get<std::string>(path, "log_level"));
    set_log_prefix(path);
}


DefaultVisPattern::DefaultVisPattern(kotekan::Config& config, const std::string& path) :
    FakeVisPattern(config, path) {}


void DefaultVisPattern::fill(visFrameView& frame) {
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

FillIJVisPattern::FillIJVisPattern(kotekan::Config& config, const std::string& path) :
    FakeVisPattern(config, path) {}


void FillIJVisPattern::fill(visFrameView& frame) {
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


void FillIJMissingVisPattern::fill(visFrameView& frame) {
    FillIJVisPattern::fill(frame);

    frame.fpga_seq_total = frame.fpga_seq_length - 2;
    frame.rfi_total = 1;
}


PhaseIJVisPattern::PhaseIJVisPattern(kotekan::Config& config, const std::string& path) :
    FakeVisPattern(config, path) {}


void PhaseIJVisPattern::fill(visFrameView& frame) {
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

void ChimeVisPattern::fill(visFrameView& frame) {
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


TestPatternSimpleVisPattern::TestPatternSimpleVisPattern(kotekan::Config& config,
                                                         const std::string& path) :
    FakeVisPattern(config, path) {
    test_pattern_value = config.get_default<cfloat>(path, "default_val", {1., 0.});
}


void TestPatternSimpleVisPattern::fill(visFrameView& frame) {
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

void TestPatternFreqVisPattern::fill(visFrameView& frame) {
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

void TestPatternInputVisPattern::fill(visFrameView& frame) {
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