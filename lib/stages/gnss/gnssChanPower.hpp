#ifndef GNSS_CHAN_POWER_HPP
#define GNSS_CHAN_POWER_HPP

#include <cmath>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace gnss {

/**
 * @brief Process-wide per-channel mean power |x|^2 (4+4b lsb^2), keyed by F-engine freq_id.
 *
 * GnssChordVoltageTap publishes it on every band-power pass; GnssGpuRecordAssemble reads it to
 * weight its cross-channel sums (@c chan_noise_norm, #163). One table per process: every stage
 * of an instance sees the same frames, so the tap's channels are the assemblers' channels.
 */
class ChanPower {
public:
    static ChanPower& instance() {
        static ChanPower p;
        return p;
    }

    /// Fold a pass in: an EMA per channel (@c alpha), the first value taken as is.
    void publish(const std::vector<int>& freq_ids, const std::vector<double>& power,
                 double alpha = 0.2) {
        std::lock_guard<std::mutex> lk(_m);
        for (size_t i = 0; i < freq_ids.size() && i < power.size(); ++i) {
            auto it = _pw.find(freq_ids[i]);
            if (it == _pw.end())
                _pw[freq_ids[i]] = power[i];
            else
                it->second += alpha * (power[i] - it->second);
        }
    }

    /// @c power[i] for @c freq_ids[i], 0 where unknown; returns how many were known.
    int lookup(const int* freq_ids, int n, double* power) const {
        std::lock_guard<std::mutex> lk(_m);
        int found = 0;
        for (int i = 0; i < n; ++i) {
            auto it = _pw.find(freq_ids[i]);
            power[i] = (it == _pw.end()) ? 0.0 : it->second;
            found += (it != _pw.end()) ? 1 : 0;
        }
        return found;
    }

private:
    mutable std::mutex _m;
    std::unordered_map<int, double> _pw;
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
