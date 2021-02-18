#include "fakeGpuPattern.hpp"

#include "Config.hpp"        // for Config
#include "chimeMetadata.hpp" // for chimeMetadata
#include "visUtil.hpp"       // for prod_index

#include "gsl-lite.hpp" // for span, span<>::iterator

#include <algorithm> // for fill
#include <cmath>     // for lroundf, pow
#include <exception> // for exception
#include <regex>     // for match_results<>::_Base_type
#include <stdexcept> // for runtime_error
#include <time.h>    // for timespec  // IWYU pragma: keep
#include <vector>    // for vector

// Register test patterns
REGISTER_FAKE_GPU_PATTERN(BlockGpuPattern, "block");
REGISTER_FAKE_GPU_PATTERN(LostSamplesGpuPattern, "lostsamples");
REGISTER_FAKE_GPU_PATTERN(LostWeightsGpuPattern, "lostweights");
REGISTER_FAKE_GPU_PATTERN(AccumulateGpuPattern, "accumulate");
REGISTER_FAKE_GPU_PATTERN(GaussianGpuPattern, "gaussian");
REGISTER_FAKE_GPU_PATTERN(PulsarGpuPattern, "pulsar");
REGISTER_FAKE_GPU_PATTERN(MultiFreqGpuPattern, "multifreq");


FakeGpuPattern::FakeGpuPattern(kotekan::Config& config, const std::string& path) {
    _num_elements = config.get<int>(path, "num_elements");
    _block_size = config.get<int>(path, "block_size");
    _samples_per_data_set = config.get<int>(path, "samples_per_data_set");
    _num_freq_in_frame = config.get_default<int>(path, "num_freq_in_frame", 1);
    set_log_level(config.get<std::string>(path, "log_level"));
    set_log_prefix(path);
}


BlockGpuPattern::BlockGpuPattern(kotekan::Config& config, const std::string& path) :
    FakeGpuPattern(config, path) {}


void BlockGpuPattern::fill(gsl::span<int32_t>& data, chimeMetadata* metadata, int frame_number,
                           freq_id_t freq_id) {

    (void)metadata;
    (void)frame_number;
    (void)freq_id;

    unsigned int nb1 = _num_elements / _block_size;
    unsigned int num_blocks = nb1 * (nb1 + 1) / 2;

    DEBUG2("Block size %i, num blocks %i", _block_size, num_blocks);

    for (unsigned int b = 0; b < num_blocks; ++b) {
        for (unsigned int y = 0; y < _block_size; ++y) {
            for (unsigned int x = 0; x < _block_size; ++x) {
                unsigned int ind = b * _block_size * _block_size + x + y * _block_size;
                data[2 * ind + 0] = y;
                data[2 * ind + 1] = x;
            }
        }
    }
}


LostSamplesGpuPattern::LostSamplesGpuPattern(kotekan::Config& config, const std::string& path) :
    FakeGpuPattern(config, path) {}

void LostSamplesGpuPattern::fill(gsl::span<int32_t>& data, chimeMetadata* metadata,
                                 int frame_number, freq_id_t freq_id) {
    (void)freq_id;

    uint32_t norm = _samples_per_data_set - frame_number;

    // Every frame has one more lost packet than the last
    for (size_t i = 0; i < _num_elements; i++) {
        for (size_t j = i; j < _num_elements; j++) {
            uint32_t bi = prod_index(i, j, _block_size, _num_elements);

            // The visibilities are row + 1j * col, scaled by the total number
            // of frames.
            data[2 * bi] = j * norm;     // Imag
            data[2 * bi + 1] = i * norm; // Real
        }
    }

    metadata->lost_timesamples = frame_number;
}


LostWeightsGpuPattern::LostWeightsGpuPattern(kotekan::Config& config, const std::string& path) :
    FakeGpuPattern(config, path), _b(config.get_default<uint32_t>(path, "b", 1)) {}

void LostWeightsGpuPattern::fill(gsl::span<int32_t>& data, chimeMetadata* metadata,
                                 int frame_number, freq_id_t freq_id) {
    (void)freq_id;

    int32_t lost = ((frame_number + 1) % 4 < 2) ? _b : 0;
    uint32_t norm = (uint32_t)(_samples_per_data_set - lost);
    int32_t noise = (frame_number % 2) ? 1 : -1;

    // Every frame has one more lost packet than the last
    for (size_t i = 0; i < _num_elements; i++) {
        for (size_t j = i; j < _num_elements; j++) {
            uint32_t bi = prod_index(i, j, _block_size, _num_elements);

            // The visibilities are row + 1j * col, scaled by the total number
            // of frames.
            data[2 * bi] = j * norm + noise;     // Imag
            data[2 * bi + 1] = i * norm + noise; // Real
        }
    }

    metadata->lost_timesamples = lost;
    metadata->rfi_flagged_samples = lost;
}

AccumulateGpuPattern::AccumulateGpuPattern(kotekan::Config& config, const std::string& path) :
    FakeGpuPattern(config, path) {}


void AccumulateGpuPattern::fill(gsl::span<int32_t>& data, chimeMetadata* metadata, int frame_number,
                                freq_id_t freq_id) {

    (void)metadata;
    (void)freq_id;

    for (size_t i = 0; i < _num_elements; i++) {
        for (size_t j = i; j < _num_elements; j++) {
            uint32_t bi = prod_index(i, j, _block_size, _num_elements);

            // Every 4th sample the imaginary part is boosted by 4 * samples,
            // but we subtract off a constant to make it average the to be the
            // column index.
            data[2 * bi] = // Imag
                (j + 4 * (frame_number % 4 == 0) - 1) * _samples_per_data_set;

            // ... similar for the real part, except we subtract every 4th
            // frame, and boost by a constant to ensure the average value is the
            // row.
            data[2 * bi + 1] = // Real
                (i - 4 * ((frame_number + 1) % 4 == 0) + 1) * _samples_per_data_set;
        }
    }
}


GaussianGpuPattern::GaussianGpuPattern(kotekan::Config& config, const std::string& path) :
    FakeGpuPattern(config, path), rd(), gen(rd()), gaussian(0, 1) {}


void GaussianGpuPattern::fill(gsl::span<int32_t>& data, chimeMetadata* metadata, int frame_number,
                              freq_id_t freq_id) {

    (void)metadata;
    (void)frame_number;
    (void)freq_id;

    float f_auto = pow(_samples_per_data_set, 0.5);
    float f_cross = pow(_samples_per_data_set / 2, 0.5);

    for (size_t i = 0; i < _num_elements; i++) {
        for (size_t j = i; j < _num_elements; j++) {
            uint32_t bi = prod_index(i, j, _block_size, _num_elements);

            if (i == j) {
                data[2 * bi + 1] = (int32_t)lroundf(_samples_per_data_set + f_auto * gaussian(gen));
                data[2 * bi] = 0;
            } else {
                data[2 * bi + 1] = (int32_t)lroundf(f_cross * gaussian(gen));
                data[2 * bi] = (int32_t)lroundf(f_cross * gaussian(gen));
            }
        }
    }
}


PulsarGpuPattern::PulsarGpuPattern(kotekan::Config& config, const std::string& path) :
    FakeGpuPattern(config, path) {
    // set up pulsar polyco
    auto coeff = config.get<std::vector<float>>(path, "coeff");
    auto dm = config.get<float>(path, "dm");
    auto t_ref = config.get<double>(path, "t_ref");         // in days since MJD
    auto phase_ref = config.get<double>(path, "phase_ref"); // in number of rotations
    _rot_freq = config.get<double>(path, "rot_freq");       // in Hz
    _pulse_width = config.get<float>(path, "pulse_width");
    _polyco = Polyco(t_ref, dm, phase_ref, _rot_freq, coeff);
}


void PulsarGpuPattern::fill(gsl::span<int32_t>& data, chimeMetadata* metadata, int frame_number,
                            freq_id_t freq_id) {
    (void)frame_number;

    auto& tel = Telescope::instance();

    // Fill frame with zeros
    std::fill(data.begin(), data.end(), 0);

    DEBUG2("GPS time %ds%dns", metadata->gps_time.tv_sec, metadata->gps_time.tv_nsec);

    // Figure out if we are in a pulse
    double toa = _polyco.next_toa(metadata->gps_time, tel.to_freq(freq_id));
    double last_toa = toa - 1. / _rot_freq;
    DEBUG2("TOA: %f, last TOA: %f", toa, last_toa);

    // If so, add 10 to real part
    if (toa < _samples_per_data_set * tel.seq_length_nsec() * 1e-9 || last_toa + _pulse_width > 0) {
        // DEBUG("Found pulse!");
        for (size_t i = 0; i < _num_elements; i++) {
            for (size_t j = i; j < _num_elements; j++) {
                uint32_t bi = prod_index(i, j, _block_size, _num_elements);
                data[2 * bi + 1] += 10 * _samples_per_data_set;
            }
        }
    }
}


MultiFreqGpuPattern::MultiFreqGpuPattern(kotekan::Config& config, const std::string& path) :
    FakeGpuPattern(config, path) {}

void MultiFreqGpuPattern::fill(gsl::span<int32_t>& data, chimeMetadata* metadata, int frame_number,
                               freq_id_t freq_id) {
    (void)frame_number;
    (void)metadata;

    // Label the real with the freq_id and the imag with the product id.
    uint32_t prod_id = 0;
    for (size_t i = 0; i < _num_elements; i++) {
        for (size_t j = i; j < _num_elements; j++) {
            uint32_t bi = prod_index(i, j, _block_size, _num_elements);
            data[2 * bi] = prod_id * _samples_per_data_set;     // Imag
            data[2 * bi + 1] = freq_id * _samples_per_data_set; // Real
            prod_id++;
        }
    }
}
