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

#include <string>
#include <stdint.h>

#include "gsl-lite.hpp"

#include "factory.hpp"
#include "Config.hpp"
#include "chimeMetadata.h"
#include "pulsarTiming.hpp"

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
class fakeGpuPattern {
public:

    /**
     * @brief Create a class that makes a fakeGpu test mode.
     *
     * @param  config  Kotekan configuration.
     * @param  path    Path to start search from. Will generally be the path for
     *                 the process creating the pattern.
     **/
    fakeGpuPattern(Config& config, const std::string& path);

    virtual ~fakeGpuPattern() = default;

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
    virtual void fill(gsl::span<int32_t>& data, chimeMetadata* metadata,
                      const int frame_num, const int freq_id) {}

protected:

    // Configuration info
    size_t _num_elements;
    size_t _block_size;
    size_t _samples_per_data_set;
    size_t _num_freq_in_frame;
};

// Create the abstract factory for generating patterns
CREATE_FACTORY(fakeGpuPattern, Config&, const std::string&);
#define REGISTER_FAKE_GPU_PATTERN(patternType, name) \
    REGISTER_NAMED_TYPE_WITH_FACTORY(fakeGpuPattern, patternType, name)


/**
 * @brief Fill with a pattern useful for debugging the packing.
 *
 * Fill each element with its block row (real value) and block column
 * (imaginary).
 **/
class blockGpuPattern : public fakeGpuPattern {
public:

    /// @sa fakeGpuPattern::fakeGpuPattern
    blockGpuPattern(Config& config, const std::string& path);

    /// @sa fakeGpuPattern::fill
    void fill(gsl::span<int32_t>& data, chimeMetadata* metadata,
              const int frame_num, const int freq_id) override;
};


/**
 * @brief Fill with a pattern for testing lost packet renormalisation.
 *
 * Fill each element with its block row (real value) and block column
 * (imaginary). Each frame has a number of lost packets equivalent to the
 * frame number.
 **/
class lostSamplesGpuPattern : public fakeGpuPattern {
public:
    /// @sa fakeGpuPattern::fakeGpuPattern
    lostSamplesGpuPattern(Config& config, const std::string& path);

    /// @sa fakeGpuPattern::fill
    void fill(gsl::span<int32_t>& data, chimeMetadata* metadata,
              const int frame_num, const int freq_id) override;
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
class accumulateGpuPattern : public fakeGpuPattern {
public:
    /// @sa fakeGpuPattern::fakeGpuPattern
    accumulateGpuPattern(Config& config, const std::string& path);

    /// @sa fakeGpuPattern::fill
    void fill(gsl::span<int32_t>& data, chimeMetadata* metadata,
              const int frame_num, const int freq_id) override;
};



/**
 * @brief Fill with a pattern with Gaussian noise with radiometer variance.
 *
 * The underlying inputs are uncorrelated with variance of 1.
 **/
class gaussianGpuPattern : public fakeGpuPattern {
public:
    /// @sa fakeGpuPattern::fakeGpuPattern
    gaussianGpuPattern(Config& config, const std::string& path);

    /// @sa fakeGpuPattern::fill
    void fill(gsl::span<int32_t>& data, chimeMetadata* metadata,
              const int frame_num, const int freq_id) override;
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
class pulsarGpuPattern : public fakeGpuPattern {
public:
    /// @sa fakeGpuPattern::fakeGpuPattern
    pulsarGpuPattern(Config& config, const std::string& path);

    /// @sa fakeGpuPattern::fill
    void fill(gsl::span<int32_t>& data, chimeMetadata* metadata,
              const int frame_num, const int freq_id) override;

private:
    float _pulse_width;  // in s
    float _rot_freq; // in Hz
    Polyco _polyco;
};


/**
 * @brief Fill with a pattern useful for debugging the packing.
 *
 * Fill each element with its freq_id (real value) and product index
 * (imaginary). Each of these is multiplied by samples_per_data_set such that
 * they can be tested *after* accumulation.
 **/
class multiFreqGpuPattern : public fakeGpuPattern {
public:

    /// @sa fakeGpuPattern::fakeGpuPattern
    multiFreqGpuPattern(Config& config, const std::string& path);

    /// @sa fakeGpuPattern::fill
    void fill(gsl::span<int32_t>& data, chimeMetadata* metadata,
              const int frame_num, const int freq_id) override;
};
#endif  // FAKE_GPU_PATTERN