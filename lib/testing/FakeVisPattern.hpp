/*****************************************
@file
@brief Patterns for generating fake visibility data
- FakeVisPattern
- DefaultVisPattern
- FillIJVisPattern
- FillIJMissingVisPattern
- PhaseIJVisPattern
- ChimeVisPattern
- TestPatternSimpleVisPattern
- TestPatternFreqVisPattern
- TestPatternInputVisPattern
*****************************************/
#ifndef FAKE_VIS_PATTERN_HPP
#define FAKE_VIS_PATTERN_HPP

#include "Config.hpp"         // for Config
#include "datasetManager.hpp" // for state_id_t, dset_id_t
#include "factory.hpp"        // for REGISTER_NAMED_TYPE_WITH_FACTORY, CREATE_FACTORY, Factory
#include "kotekanLogging.hpp" // for kotekanLogging
#include "visBuffer.hpp"      // for VisFrameView
#include "visUtil.hpp"        // for cfloat

#include <random>   // for mt19937, normal_distribution, random_device
#include <deque>      // for deque
#include <functional> // for function
#include <optional>   // for optional
#include <stddef.h>   // for size_t
#include <string>     // for string
#include <utility>    // for pair
#include <vector>     // for vector

/**
 * @class FakeVisPattern
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
class FakeVisPattern : public kotekan::kotekanLogging {
public:
    /**
     * @brief Create a class that makes a fakeGpu test mode.
     *
     * @param  config  Kotekan configuration.
     * @param  path    Path to start search from. Will generally be the path for
     *                 the process creating the pattern.
     **/
    FakeVisPattern(kotekan::Config& config, const std::string& path);

    virtual ~FakeVisPattern() = default;

    /**
     * @brief Fill the data with a test pattern.
     *
     * @note The weights, eigenvalues, eigenvectors, erms, gains and flags will
     *       have been pre-filled with some reasonable values. However, they can be
     *       overwritten if desired.
     *
     * @param  frame  The vis buffer frame to fill with data.
     **/
    virtual void fill(VisFrameView& frame) = 0;
};

// Create the abstract factory for generating patterns
CREATE_FACTORY(FakeVisPattern, kotekan::Config&, const std::string&);
#define REGISTER_FAKE_VIS_PATTERN(PatternType, name)                                               \
    REGISTER_NAMED_TYPE_WITH_FACTORY(FakeVisPattern, PatternType, name)


/**
 * @brief Default fill pattern.
 *
 * The visibility array is populated with integers increasing from zero on
 * the diagonal (imaginary part) and FPGA sequence number, timestamp,
 * frequency, and frame ID in the first four elements (real part). The
 * remaining elements are zero.
 **/
class DefaultVisPattern : public FakeVisPattern {
public:
    /// @sa FakeVisPattern::FakeVisPattern
    DefaultVisPattern(kotekan::Config& config, const std::string& path);

    /// @sa FakeVisPattern::fill
    void fill(VisFrameView& frame) override;
};


/**
 * @brief Default fill pattern.
 *
 * Fill the real part with the index of feed i and the imaginary part with
 * the index of j.
 **/
class FillIJVisPattern : public FakeVisPattern {
public:
    /// @sa FakeVisPattern::FakeVisPattern
    FillIJVisPattern(kotekan::Config& config, const std::string& path);

    /// @sa FakeVisPattern::fill
    void fill(VisFrameView& frame) override;
};


/**
 * @brief Default fill pattern.
 *
 * Fill the real part with the index of feed i and the imaginary part with
 * the index of j. Each frame is marked as missing two samples of data, one
 * of which is RFI.
 *
 **/
// TODO: merge with FillIJPattern
class FillIJMissingVisPattern : public FillIJVisPattern {
public:
    /// @sa FakeVisPattern::FakeVisPattern
    FillIJMissingVisPattern(kotekan::Config& config, const std::string& path);

    /// @sa FakeVisPattern::fill
    void fill(VisFrameView& frame) override;
};


/**
 * @brief Fill with a factorisable pattern.
 *
 * Fill with unit amplitude numbers with phase ``i - j``
 * radians.
 **/
class PhaseIJVisPattern : public FakeVisPattern {
public:
    /// @sa FakeVisPattern::FakeVisPattern
    PhaseIJVisPattern(kotekan::Config& config, const std::string& path);

    /// @sa FakeVisPattern::fill
    void fill(VisFrameView& frame) override;
};


/**
 * @brief Fill with a pattern to test CHIME redundant stacking.
 *
 * Fill real and imaginary parts with normally distributed random numbers.
 * Specify mean and standard deviation with additional parameters. Will use
 * the same distribution to set the weights. Note that the seed for the
 * generator is not random.
 **/
class ChimeVisPattern : public FakeVisPattern {
public:
    /// @sa FakeVisPattern::FakeVisPattern
    ChimeVisPattern(kotekan::Config& config, const std::string& path);

    /// @sa FakeVisPattern::fill
    void fill(VisFrameView& frame) override;
};

/**
 * @brief Fill with a simple test pattern, where all visibilities have
 * the value of 'default_val'.
 *
 * @conf  default_val  Cfloat. The default test pattern value.
 **/
class TestPatternSimpleVisPattern : public FakeVisPattern {
public:
    /// @sa FakeVisPattern::FakeVisPattern
    TestPatternSimpleVisPattern(kotekan::Config& config, const std::string& path);

    /// @sa FakeVisPattern::fill
    void fill(VisFrameView& frame) override;

private:
    cfloat test_pattern_value;
};


/**
 * @brief Fill with a frequency dependent test pattern.
 *
 * Here the frequencies defined in the config value 'frequencies' have the
 * values defined in 'freq_values'. All other visibility values have the value
 * defined in 'default_val'.
 *
 * @conf  default_val  Cfloat. The default test pattern value.
 * @conf  freq_values  Array of CFloat. Values for the frequency IDs
 *
 **/
class TestPatternFreqVisPattern : public FakeVisPattern {
public:
    /// @sa FakeVisPattern::FakeVisPattern
    TestPatternFreqVisPattern(kotekan::Config& config, const std::string& path);

    /// @sa FakeVisPattern::fill
    void fill(VisFrameView& frame) override;

private:
    std::vector<cfloat> test_pattern_value;
};


/**
 * @brief Fill with a input dependent test pattern.
 *
 * Here the input values are defined in the config value 'input_values'.
 *
 * @conf  input_values  Array of CFloat. Values for the frequency IDs
 *
 **/
class TestPatternInputVisPattern : public FakeVisPattern {
public:
    /// @sa FakeVisPattern::FakeVisPattern
    TestPatternInputVisPattern(kotekan::Config& config, const std::string& path);

    /// @sa FakeVisPattern::fill
    void fill(VisFrameView& frame) override;

private:
    std::vector<cfloat> test_pattern_value;
};


/**
 * @brief Fill with a pattern with Gaussian noise.
 *
 * The underlying inputs are uncorrelated with variance of 1.
 **/
class NoiseVisPattern : public FakeVisPattern {
public:
    /// @sa FakeGpuPattern::FakeVisPattern
    NoiseVisPattern(kotekan::Config& config, const std::string& path);

    /// @sa FakeVisPattern::fill
    void fill(VisFrameView& frame) override;

private:
    std::random_device rd;
    std::mt19937 gen;
    std::normal_distribution<float> gaussian;
};


/**
 * @brief Send out some data but change the dataset IDs
 *
 * @conf  state_changes  A series of timestamp-state type pairs. Supported state types are `inputs`,
 * `gains` and `flags`.
 *
 **/
class ChangeStatePattern : public DefaultVisPattern {
public:
    /// @sa FakeVisPattern::FakeVisPattern
    ChangeStatePattern(kotekan::Config& config, const std::string& path);

    /// @sa FakeVisPattern::fill
    void fill(VisFrameView& frame) override;

private:
    // Alias for the type of a function that will generate the state
    using gen_state = std::function<state_id_t()>;

    std::deque<std::pair<double, gen_state>> _dataset_changes;

    // Methods to generate state changes. These should not change any structural
    // parameters.
    state_id_t gen_state_inputs();
    state_id_t gen_state_flags();
    state_id_t gen_state_gains();

    std::optional<dset_id_t> current_dset_id;

    size_t num_elements;

    size_t _input_update_ind = 0;
    size_t _flag_update_ind = 0;
    size_t _gain_update_ind = 0;
};
#endif // FAKE_VIS_PATTERN
