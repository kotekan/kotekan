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
    /// A channel that becomes masked is also masked for this long before, seconds
    double mask_margin_s = 0.17;
    /// Candidates are reported above this S/N
    double threshold = 8.0;
    /// No candidates before this time after the first sample, seconds
    double holdoff_s = 120.0;
    /// OpenMP threads for the dedispersion
    int num_threads = 4;
    /// Keep each pass's dedispersed sums for IncoherentSearch::last_sums()
    bool keep_sums = false;
};

/// One pulse candidate.
struct IncoherentCandidate {
    /// Blocks of channels in the band occupancy
    static constexpr int num_blocks = 16;
    /// S/N of the strongest sample, with its DM, pulse width in samples and arrival at the top
    /// channel in samples since the first one
    double snr = 0;
    double dm = 0;
    /// Index of dm in IncoherentSearchParams::dms
    int dm_index = 0;
    int width = 0;
    int64_t sample = 0;
    /// Samples above threshold that were merged into the candidate
    int num_hits = 0;
    /// S/N of each block of channels along the dispersion curve. A pulse fills most blocks;
    /// narrow-band interference fills one or two.
    std::array<float, num_blocks> block_snr{};
    /// Channels with data along the dispersion curve
    int num_channels = 0;
    /// Other candidates more than 20 DM units away that fit the same arrival without dispersion,
    /// with their highest S/N in or beside a block where this one has half its own highest
    int num_partners = 0;
    /// The candidate has two or more partners, and with them covers more than 50 DM units:
    /// interference, not a pulse.
    bool undispersed = false;
    /// 8 samples or wider, with more than 0.8 times its S/N in one block. Interference a few MHz
    /// wide looks like this; so may a wide burst in a narrow band.
    bool one_band = false;
    /// The highest S/N of the candidate in a contiguous part of the band, of 2 to 16 sixteenths
    /// of the channels with data, after the sum of every 8 adjacent channels over the window is
    /// limited to 3 sigma, or to what a pulse in two sixteenths would give; and its ratio to the
    /// S/N of the whole band from the same samples. The ratio is about 1 for noise, below 1 for
    /// interference a few MHz wide, and above 1 for a pulse that fills part of the band.
    float part_snr = 0;
    float part_ratio = 0;
};

/**
 * @class IncoherentSearch
 * @brief Searches power summed over feeds for dispersed pulses, one time sample at a time.
 *
 * Each channel keeps an exponential running mean and variance and is normalized by them. Channels
 * that are missing, in a fixed band, noisier than their neighbours or drifting (the means of
 * 64-sample blocks vary more than white noise allows) are masked, with two channels on each side
 * of a noisy or drifting one and with the mask_margin_s before the mask. So is, for one time
 * sample, a sixteenth of the band whose channels add up to far more than their noise: interference
 * in part of the band, which the limit on single samples does not stop. The channel average is
 * subtracted at each time (zero DM). The normalized samples are summed along
 * each trial DM's dispersion curve with delays rounded to the nearest sample, relative to the top
 * channel. Each sum is tried at several widths against a baseline from the samples on both sides,
 * and normalized by its own running variance.
 *
 * Samples above threshold are merged into candidates: with the strongest sample of an open
 * candidate when they lie within the dispersion sweep between the two DMs and are either close in
 * DM or much weaker (the residue of a bright pulse at another DM, which falls as one over the
 * difference of DM).
 *
 * Interference in part of the band has no dispersion, and is found at every DM: at a higher DM
 * earlier at the top channel, by the sweep from its own frequency to the top. A pulse is found
 * at its own DM. A partner of a candidate is another one of at least half its S/N and of at
 * least 6 sigma, more than 20 DM units away at such a time, with its highest S/N in or beside a
 * block of the band where the candidate has at least half its own highest: a pulse that arrives
 * during interference has little S/N where the interference is. A candidate is marked
 * undispersed when it has two or more partners and they cover, with it, more than 50 DM units.
 * For that a candidate is reported only when the same arrival has been searched at the highest
 * DM: the sweep of the highest DM after it reached the top channel.
 *
 * Widths of 1 to 16 samples are tried, in passes of `chunk` samples; the widths and the mask
 * thresholds are constants in the source.
 *
 * The object has no dependencies in kotekan and is not thread safe.
 */
class IncoherentSearch {
public:
    /// Samples per search pass
    static constexpr int chunk = 256;
    /// Time samples in a cutout
    static constexpr int cutout_samples = 256;
    /// A stored sample is in units of 1 / sample_scale sigma of its channel: int8 reaches 4 sigma
    static constexpr double sample_scale = 32.0;
    /// A waterfall value is in units of 1 / waterfall_scale sigma of its bin: up to 7.9 sigma
    static constexpr double waterfall_scale = 16.0;
    /// The waterfall value of a bin without a sample in use
    static constexpr int8_t waterfall_no_data = -128;

    explicit IncoherentSearch(const IncoherentSearchParams& params);

    /**
     * @brief Adds one time sample.
     * @param power  Power per channel; a value <= 0 marks a missing channel.
     * @return True when this sample completed a search pass: there may be new candidates.
     */
    bool add_sample(const float* power);

    /// Returns the candidates completed since the last call.
    std::vector<IncoherentCandidate> take_candidates();

    /**
     * @brief Copies the samples around a candidate as the search saw them, dedispersed at its DM.
     *
     * A sample is the channel's power less its running mean, in units of 1 / sample_scale of
     * its sigma, limited to 3 sigma, less the average of the channels at that time; 0 where
     * the channel was masked or the sample is not held.
     *
     * @param cand  A candidate of the last take_candidates(), before the next add_sample().
     * @param out   [channel][cutout_samples]. Column k is the top channel's sample
     *              cand.sample - cutout_samples / 2 + k, and in each channel the sample that
     *              the candidate's DM delays to it.
     */
    void cutout(const IncoherentCandidate& cand, int8_t* out) const;

    /**
     * @brief Bins the stored samples in time and frequency, for a display.
     *
     * A value is the sum of the bin's samples (see cutout()) over the square root of the number
     * that are not 0, in units of 1 / waterfall_scale sigma: thermal noise gives about 1 sigma
     * in every bin, whatever part of it is masked. waterfall_no_data where every sample of the
     * bin is 0 (masked or missing) or the bin is not held.
     *
     * @param first     First sample of the first row.
     * @param rows      Number of rows.
     * @param time_bin  Samples per row.
     * @param freq_bin  Channels per column; the last column takes what is left.
     * @param out       [rows][columns], with (channels + freq_bin - 1) / freq_bin columns.
     */
    void waterfall(int64_t first, int rows, int time_bin, int freq_bin, int8_t* out) const;

    /// Highest S/N of any DM and width since the last call, after the hold-off.
    double take_max_snr();

    /**
     * @brief The dedispersed sums of the last search pass, [DM][chunk], when params.keep_sums.
     *
     * A value is the sum over the channels of their normalized samples along the dispersion
     * curve, before any width or baseline. Column k is the curve that reaches the bottom
     * channel at sample num_samples() - chunk + k, so that all DMs fill a pass together; it
     * left the top channel sweep_samples() earlier. 0 where there is no sum yet.
     */
    const std::vector<float>& last_sums() const {
        return pass_sums;
    }

    /// Highest S/N of each trial DM in the last search pass, over its times and widths; 0 where
    /// none was computed (the hold-off, no data)
    const std::vector<float>& last_max_snr() const {
        return pass_max;
    }

    /// Delay of the bottom channel against the top channel at trial DM d, in samples
    int64_t sweep_samples(int d) const {
        return max_delay[d];
    }

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
        int block; // the block of channels with the highest S/N of `best`
    };

    void update_reference();
    void dedisperse_chunk();
    void search_dm(int d, const float* sums, int64_t n, std::vector<double>& prefix,
                   std::vector<Hit>& hits, double& max_snr);
    void merge_hits(const std::vector<Hit>& hits);
    void close_events();
    int count_partners(const Event& e,
                       const std::array<float, IncoherentCandidate::num_blocks>& block_snr,
                       double& dm_range) const;
    int strongest_block(const Hit& hit) const;
    void band_occupancy(IncoherentCandidate& cand, const Hit& hit) const;
    void band_parts(IncoherentCandidate& cand) const;
    const int8_t* row(int channel, int64_t sample) const {
        return &w[(size_t)channel * buffer_len + (size_t)(sample - base)];
    }
    // First channel of block b of the IncoherentCandidate::num_blocks equal parts of the band
    int block_start(int b) const {
        return (int)((int64_t)b * F / IncoherentCandidate::num_blocks);
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
    int64_t mask_margin = 0;  // masked before a channel's mask
    int64_t ref_interval = 0; // between updates of the neighbours' reference
    int64_t lookahead = 0;    // sums needed after a window's start

    // Per channel
    std::vector<double> mean, var;   // running mean and variance of the power
    std::vector<int64_t> count;      // updates of mean and var
    std::vector<float> ref;          // sigma/mean of the neighbours, in units of thermal
    std::vector<uint8_t> band_ok;    // 0 in a fixed masked band
    std::vector<ChannelState> state; // in the sample being added
    std::vector<float> scratch;      // normalized value in the sample being added
    std::vector<uint8_t> noisy;      // fails its own test in the sample being added
    std::vector<uint8_t> was_used;   // used in the last sample, before the cut of blocks
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
    std::vector<float> pass_sums;    // [D][chunk] when p.keep_sums: see last_sums()
    std::vector<float> pass_max;     // [D]: see last_max_snr()

    // Open candidates, and reported ones that still absorb residue and mark others undispersed
    std::vector<Event> events;
    std::vector<IncoherentCandidate> candidates;
    double max_snr_seen = 0;
};

#endif
