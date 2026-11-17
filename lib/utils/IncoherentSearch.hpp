/**
 * @file
 * @brief Search of feed-summed power for dispersed pulses.
 *  - IncoherentSearch
 */
#ifndef INCOHERENT_SEARCH_HPP
#define INCOHERENT_SEARCH_HPP

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

/// Parameters of IncoherentSearch.
struct IncoherentSearchParams {
    /// Number of frequency channels, in ascending frequency
    int num_freq = 0;
    /// Centre frequency of channel 0 and the channel spacing, MHz
    double freq0_mhz = 0;
    double dfreq_mhz = 0;
    /// Length of one time sample, seconds
    double sample_s = 0;
    /// Trial dispersion measures, pc/cm^3
    std::vector<double> dms;
    /// Time constant of every running mean and variance, seconds
    double tau_s = 5.0;
    /// Thermal sigma/mean of one power sample: 1 / sqrt(voltage samples summed)
    double thermal = 0;
    /// Frequency ranges that are always masked, MHz
    std::vector<std::pair<double, double>> mask_bands_mhz;
    /// Candidates are reported above this S/N
    double threshold = 8.0;
    /// No candidates before this time after the first sample, seconds
    double holdoff_s = 120.0;
    /// OpenMP threads for the dedispersion
    int num_threads = 4;
};

/// One pulse candidate.
struct IncoherentCandidate {
    /// Blocks of channels in the band occupancy
    static constexpr int num_blocks = 16;
    /// S/N of the strongest sample, with its DM, pulse width in samples and arrival at the top
    /// channel in samples since the first one
    double snr = 0;
    double dm = 0;
    int width = 0;
    int64_t sample = 0;
    /// Samples above threshold that were merged into the candidate
    int num_hits = 0;
    /// S/N of each block of channels along the dispersion curve. A pulse fills most blocks;
    /// narrow-band interference fills one or two.
    std::array<float, num_blocks> block_snr{};
};

/**
 * @class IncoherentSearch
 * @brief Searches power summed over feeds for dispersed pulses, one time sample at a time.
 *
 * Each channel keeps an exponential running mean and variance and is normalized by them. Channels
 * that are missing, in a fixed band, noisier than their neighbours or drifting (the means of
 * 64-sample blocks vary more than white noise allows) are masked. The channel average is
 * subtracted at each time (zero DM). The normalized samples are summed along
 * each trial DM's dispersion curve with delays rounded to the nearest sample, relative to the top
 * channel. Each sum is tried at several widths against a baseline from the samples on both sides,
 * and normalized by its own running variance.
 *
 * Samples above threshold are merged into candidates: with the strongest sample of an open
 * candidate when they lie within the dispersion sweep between the two DMs and are either close in
 * DM or much weaker (the residue of a bright pulse at another DM).
 *
 * Widths of 1 to 16 samples are tried, in passes of 256 samples; these and the mask thresholds
 * are constants in the source.
 *
 * The object has no dependencies in kotekan and is not thread safe.
 */
class IncoherentSearch {
public:
    explicit IncoherentSearch(const IncoherentSearchParams& params);

    /**
     * @brief Adds one time sample.
     * @param power  Power per channel; a value <= 0 marks a missing channel.
     * @return True when this sample completed a search pass: there may be new candidates.
     */
    bool add_sample(const float* power);

    /// Returns the candidates completed since the last call.
    std::vector<IncoherentCandidate> take_candidates();

    /// Highest S/N of any DM and width since the last call, after the hold-off.
    double take_max_snr();

    /// Samples added so far
    int64_t num_samples() const {
        return ns;
    }

    /// Channels used in the most recent sample
    int channels_used() const {
        return num_used;
    }

private:
    struct Hit {
        float snr;
        int32_t dm_index;
        int32_t width_index;
        int64_t sample;
    };
    struct Event {
        Hit best;
        int num_hits;
        int64_t created; // value of ns when the event was opened
        bool reported;
    };

    void update_reference();
    void dedisperse_chunk();
    void search_dm(int d, const float* sums, int64_t n, std::vector<double>& prefix,
                   std::vector<Hit>& hits, double& max_snr);
    void merge_hits(const std::vector<Hit>& hits);
    void close_events();
    void band_occupancy(IncoherentCandidate& cand, const Hit& hit) const;
    const int8_t* row(int channel, int64_t sample) const {
        return &w[(size_t)channel * buffer_len + (size_t)(sample - base)];
    }

    // A channel in the sample being added: no data or no statistics yet, normalized but masked,
    // or in the search.
    enum class ChannelState : uint8_t { missing, masked, used };

    const IncoherentSearchParams p;
    const int F;    // channels
    const int D;    // trial DMs
    const int W;    // pulse widths
    const double a; // weight of one sample in a running statistic
    // In samples:
    int64_t holdoff = 0;
    int64_t min_count = 0;    // updates a statistic needs before it is used
    int64_t close_after = 0;  // a candidate is reported this long after its first sample
    int64_t ref_interval = 0; // between updates of the neighbours' reference
    int64_t lookahead = 0;    // sums needed after a window's start

    // Per channel
    std::vector<double> mean, var;   // running mean and variance of the power
    std::vector<int64_t> count;      // updates of mean and var
    std::vector<float> ref;          // sigma/mean of the neighbours, in units of thermal
    std::vector<uint8_t> band_ok;    // 0 in a fixed masked band
    std::vector<ChannelState> state; // in the sample being added
    std::vector<float> scratch;      // normalized value in the sample being added
    // Drift test: the sum of the current block of normalized samples, and the running variance
    // of the block means in units of that of white noise.
    std::vector<double> block_sum, drift;

    // Normalized samples [channel][time] in steps of 1/32 sigma, a linear buffer that is shifted
    // down when full. The samples are limited to a few sigma, so 8 bits lose nothing, and the
    // dedispersion is bound by the memory it reads.
    std::vector<int8_t> w;
    int64_t buffer_len = 0; // samples per channel in w
    int64_t keep = 0;       // samples that a shift keeps
    int64_t base = 0;       // sample index of column 0
    int64_t ns = 0;         // samples added
    int num_used = 0;

    // Per DM
    std::vector<int32_t> delay;     // [D][F], in samples, relative to the top channel
    std::vector<int64_t> max_delay; // the bottom channel's delay
    int64_t max_delay_all = 0;
    double sweep_per_dm = 0;        // samples of sweep across the band per unit DM
    std::vector<int64_t> next_sum;  // first top-channel time not yet dedispersed
    std::vector<int64_t> next_eval; // first window start not yet evaluated
    int64_t history_len = 0;
    std::vector<float> history;      // [D][history_len + chunk]: the most recent sums
    std::vector<int64_t> history_n;  // sums held in history
    std::vector<double> sum_var;     // [D][W]: running variance of each width's statistic
    std::vector<int64_t> sum_count;  // [D][W]: updates of sum_var
    std::vector<double> sigma_floor; // [W]: lower limit of the sigma that divides a statistic

    std::vector<Event> events; // open candidates, and reported ones that still absorb residue
    std::vector<IncoherentCandidate> candidates;
    double max_snr_seen = 0;
};

#endif
