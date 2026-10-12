#ifndef GNSS_CHAN_POWER_HPP
#define GNSS_CHAN_POWER_HPP

#include <cmath>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace gnss {

/**
 * @brief Process-wide per-channel, per-element mean power |x|^2 (4+4b lsb^2), keyed by F-engine
 *        freq_id.
 *
 * GnssChordVoltageTap publishes it on every band-power pass; GnssGpuRecordAssemble reads the
 * mean over ITS OWN elements to weight its cross-channel sums (@c chan_noise_norm, #163). Per
 * element, because the frame carries inputs the beam never sums -- the RFI antennas with their
 * own gain table, dead slots -- and a frame-wide mean lets them set the weights. One table per
 * process: every stage of an instance sees the same frames.
 */
class ChanPower {
public:
    static ChanPower& instance() {
        static ChanPower p;
        return p;
    }

    /// Fold a pass in: @c power is [freq_ids.size()][n_elem]; an EMA (@c alpha) per entry, the
    /// first value taken as is. A channel whose element count changes starts over.
    void publish(const std::vector<int>& freq_ids, const std::vector<double>& power, int n_elem,
                 double alpha = 0.2) {
        if (n_elem <= 0)
            return;
        std::lock_guard<std::mutex> lk(_m);
        for (size_t i = 0; i < freq_ids.size() && (i + 1) * (size_t)n_elem <= power.size(); ++i) {
            std::vector<double>& v = _pw[freq_ids[i]];
            const double* src = power.data() + i * (size_t)n_elem;
            if (v.size() != (size_t)n_elem) {
                v.assign(src, src + n_elem);
                continue;
            }
            for (int e = 0; e < n_elem; ++e)
                v[(size_t)e] += alpha * (src[e] - v[(size_t)e]);
        }
    }

    /// @c power[i] = the mean over the listed frame elements WITH POWER of channel
    /// @c freq_ids[i]; 0 where the channel is unknown or none of the elements has any.
    /// Returns how many channels were known.
    int lookup(const int* freq_ids, int n, const int* elems, int n_elems, double* power) const {
        std::lock_guard<std::mutex> lk(_m);
        int found = 0;
        for (int i = 0; i < n; ++i) {
            power[i] = 0.0;
            auto it = _pw.find(freq_ids[i]);
            if (it == _pw.end())
                continue;
            ++found;
            double s = 0.0;
            int k = 0;
            for (int j = 0; j < n_elems; ++j) {
                const int e = elems[j];
                if (e >= 0 && e < (int)it->second.size() && it->second[(size_t)e] > 0.0) {
                    s += it->second[(size_t)e];
                    ++k;
                }
            }
            power[i] = k > 0 ? s / k : 0.0;
        }
        return found;
    }

private:
    mutable std::mutex _m;
    std::unordered_map<int, std::vector<double>> _pw;
};

/// Per-channel weights 1/sqrt(power), scaled to mean 1 over the @c n channels, so a channel
/// enters the cross-channel sums by its signal-to-noise and not by the F-engine digital gain.
/// Returns false, @c w untouched, when any power is missing or not positive: the caller keeps
/// unit weights rather than mixing normalised and raw channels.
inline bool chan_norm_weights(const double* power, int n, double* w) {
    if (n <= 0)
        return false;
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        if (!(power[i] > 0.0))
            return false;
        s += 1.0 / std::sqrt(power[i]);
    }
    const double k = (double)n / s;
    for (int i = 0; i < n; ++i)
        w[i] = k / std::sqrt(power[i]);
    return true;
}

} // namespace gnss

#endif // GNSS_CHAN_POWER_HPP
