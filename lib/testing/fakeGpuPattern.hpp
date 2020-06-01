/*****************************************
@file
@brief Patterns for fake GPU data
- fakeGpuPattern
- blockGpuPattern
- lostSamplesGpuPattern
- accumulateGpuPattern
- gaussianGpuPattern
- pulsarGpuPulsar
*****************************************/
#ifndef FAKE_GPU_PATTERN_HPP
#define FAKE_GPU_PATTERN_HPP

#include "Config.hpp" // for Config
#include "Telescope.hpp"
#include "chimeMetadata.h"    // for chimeMetadata
#include "factory.hpp"        // for REGISTER_NAMED_TYPE_WITH_FACTORY, CREATE_FACTORY, Factory
#include "kotekanLogging.hpp" // for kotekanLogging
#include "pulsarTiming.hpp"   // for Polyco

#include "gsl-lite.hpp" // for span

#include <random>   // for mt19937, normal_distribution, random_device
#include <stddef.h> // for size_t
#include <stdint.h> // for int32_t, uint32_t
#include <string>   // for string

// Create the abstract factory for generating patterns
class FakeGpuPattern;

CREATE_FACTORY(FakeGpuPattern, kotekan::Config&, const std::string&);
#define REGISTER_FAKE_GPU_PATTERN(patternType, name)                                               \
    REGISTER_NAMED_TYPE_WITH_FACTORY(FakeGpuPattern, patternType, name)


/**
 * @class fakeGpuPattern
 * @brief Base class for fakeGpu test patterns.
 *
 * @conf  num_elements          Int. The number of elements (i.e. inputs) in
 *                              the correlator data
 * @conf  block_size            Int. The block size of the packed data
 * @conf  samples_per_data_set  Int. FPGA seq ticks per frame. Only use for
 *                              `pre_accumulate`.
 * @conf  num_freq_in_frame     Int. Number of frequencies packed into each GPU
 *                              frame.
 **/
class FakeGpuPattern : public kotekan::kotekanLogging {
public:
    /**
     * @brief Create a class that makes a fakeGpu test mode.
     *
     * @param  config  Kotekan configuration.
     * @param  path    Path to start search from. Will generally be the path for
     *                 the process creating the pattern.
     **/
    FakeGpuPattern(kotekan::Config& config, const std::string& path);

    virtual ~FakeGpuPattern() = default;

    /**
     * @brief Fill the data with a test pattern.
     *
     * @note You can also modify the metadata (for example to set the number of
     *       lost frames), but be careful doing this if there are multiple
     *       frequencies in the buffer.
     *
     * @param  data       Data for the specific frequency in the buffer.
     * @param  metadata   Metadata for sample. Potentially shared across
     *                    frequencies.
     * @param  frame_num  Index of frame since start.
     * @param  freq_id    Global frequency ID.
     **/
    virtual void fill(gsl::span<int32_t>& data, chimeMetadata* metadata, const int frame_num,
                      freq_id_t freq_id) = 0;

protected:
    // Configuration info
    size_t _num_elements;
    size_t _block_size;
    size_t _samples_per_data_set;
    size_t _num_freq_in_frame;
};


/**
 * @brief Fill with a pattern useful for debugging the packing.
 *
 * Fill each element with its block row (real value) and block column
 * (imaginary).
 **/
class BlockGpuPattern : public FakeGpuPattern {
public:
    /// @sa fakeGpuPattern::fakeGpuPattern
    BlockGpuPattern(kotekan::Config& config, const std::string& path);

    /// @sa fakeGpuPattern::fill
    void fill(gsl::span<int32_t>& data, chimeMetadata* metadata, const int frame_num,
              freq_id_t freq_id) override;
};


/**
 * @brief Fill with a pattern for testing lost packet renormalisation.
 *
 * Fill each element with its block row (real value) and block column
 * (imaginary). Each frame has a number of lost packets equivalent to the
 * frame number.
 **/
class LostSamplesGpuPattern : public FakeGpuPattern {
public:
    /// @sa fakeGpuPattern::fakeGpuPattern
    LostSamplesGpuPattern(kotekan::Config& config, const std::string& path);

    /// @sa fakeGpuPattern::fill
    void fill(gsl::span<int32_t>& data, chimeMetadata* metadata, const int frame_num,
              freq_id_t freq_id) override;
};


/**
 * @brief A pattern for testing the weight calculation with lost samples
 *
 * The model is that each frame is produced with a value
 *     x_i = alpha_i x + n_i
 * where i is the frame index. If we have a four iteration cycle where
 *     alpha_i = (a, a - b, a - b, a) and,
 *     n_i = (1, -1, 1, -1)
 * we can exactly calculate the expected the visibilities and the weights,
 * which should be
 *     V = x
 *     W = N (2a - b)^2 / 16.
 *
 * This also sets the RFI flagged count to the number of sample lost to allow
 * the `VisFrameView.rfi_total` accumulation to be checked.
 *
 * @conf  b  Int. Number of samples to "drop" on the 2nd and 3rd frames above.
 *                Default is 1.
 **/
class LostWeightsGpuPattern : public FakeGpuPattern {
public:
    /// @sa fakeGpuPattern::fakeGpuPattern
    LostWeightsGpuPattern(kotekan::Config& config, const std::string& path);

    /// @sa fakeGpuPattern::fill
    void fill(gsl::span<int32_t>& data, chimeMetadata* metadata, const int frame_num,
              freq_id_t freq_id) override;

private:
    uint32_t _b;
};


/**
 * @brief Fill with a pattern for debugging the accumulation.
 *
 * Fill each element with its full correlation index (real = row; column =
 * imag), with real and imaginary parts being shifted every 4th frame.
 *
 * Overall this mode should average to (row + column * J) and the
 * inverse variance should be num_gpu_frames / 8.
 **/
class AccumulateGpuPattern : public FakeGpuPattern {
public:
    /// @sa fakeGpuPattern::fakeGpuPattern
    AccumulateGpuPattern(kotekan::Config& config, const std::string& path);

    /// @sa fakeGpuPattern::fill
    void fill(gsl::span<int32_t>& data, chimeMetadata* metadata, const int frame_num,
              freq_id_t freq_id) override;
};


/**
 * @brief Fill with a pattern with Gaussian noise with radiometer variance.
 *
 * The underlying inputs are uncorrelated with variance of 1.
 **/
class GaussianGpuPattern : public FakeGpuPattern {
public:
    /// @sa fakeGpuPattern::fakeGpuPattern
    GaussianGpuPattern(kotekan::Config& config, const std::string& path);

    /// @sa fakeGpuPattern::fill
    void fill(gsl::span<int32_t>& data, chimeMetadata* metadata, const int frame_num,
              freq_id_t freq_id) override;

private:
    std::random_device rd;
    std::mt19937 gen;
    std::normal_distribution<float> gaussian;
};


/**
 * @brief Fill with pulsar pulses.
 *
 * The phase of the pulses will be calculated from the polyco
 * provided in the config. Gaussian noise can be added as a background.
 *
 * @conf  dm                    Float. Dispersion measure of pulsar (cm^-3 pc).
 * @conf  t_ref                 Float. Reference time for polyco in MJD (days).
 * @conf  phase_ref             Float. Reference phase for polyco (number of
 *                              rotations).
 * @conf  rot_freq              Float. Rotation frequency of the pulsar (Hz).
 * @conf  pulse_width           Float. Width of the pulse (s).
 * @conf  coeff                 List of floats. Polynomial coeffecients for
 *                              pulsar mode. Use Tempo convention.
 * @conf  gaussian_bgnd         Bool. Fill background with gaussian noise.
 **/
class PulsarGpuPattern : public FakeGpuPattern {
public:
    /// @sa fakeGpuPattern::fakeGpuPattern
    PulsarGpuPattern(kotekan::Config& config, const std::string& path);

    /// @sa fakeGpuPattern::fill
    void fill(gsl::span<int32_t>& data, chimeMetadata* metadata, const int frame_num,
              freq_id_t freq_id) override;

private:
    float _pulse_width; // in s
    float _rot_freq;    // in Hz
    Polyco _polyco;
};


/**
 * @brief Fill with a pattern useful for debugging the packing.
 *
 * Fill each element with its freq_id (real value) and product index
 * (imaginary). Each of these is multiplied by samples_per_data_set such that
 * they can be tested *after* accumulation.
 **/
class MultiFreqGpuPattern : public FakeGpuPattern {
public:
    /// @sa fakeGpuPattern::fakeGpuPattern
    MultiFreqGpuPattern(kotekan::Config& config, const std::string& path);

    /// @sa fakeGpuPattern::fill
    void fill(gsl::span<int32_t>& data, chimeMetadata* metadata, const int frame_num,
              freq_id_t freq_id) override;
};
#endif // FAKE_GPU_PATTERN
