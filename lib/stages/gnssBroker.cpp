#include "gnssBroker.hpp"

#include "gnssSignal.hpp" // for signal_by_name, SignalDescriptor

#include <algorithm> // for sort
#include <set>       // for set

namespace gnss {

namespace {

// True if every covering channel id lies within the worker's [chan_lo, chan_hi].
bool range_holds(const Worker& w, const std::vector<freq_id_t>& covering) {
    for (freq_id_t id : covering)
        if (id < w.chan_lo || id > w.chan_hi)
            return false;
    return true;
}

} // namespace

Assignment assign_targets(const std::vector<Worker>& workers, const std::vector<Target>& targets,
                          const std::vector<ChannelBand>& channels, double max_doppler_hz) {
    // Consider workers in name order for deterministic tie-breaks.
    std::vector<const Worker*> order;
    order.reserve(workers.size());
    for (const Worker& w : workers)
        order.push_back(&w);
    std::sort(order.begin(), order.end(),
              [](const Worker* a, const Worker* b) { return a->name < b->name; });

    Assignment out;
    std::map<std::string, int> load; // worker name -> #PRNs assigned so far
    for (const Worker* w : order)
        out.worker_prns[w->name]; // ensure every worker appears, even if empty

    for (const Target& t : targets) {
        if (!t.visible)
            continue;
        const SignalDescriptor* sig = signal_by_name(t.signal);
        if (sig == nullptr) {
            out.unplaced.push_back(t);
            continue;
        }
        const auto covering = covering_channels(channels, *sig, max_doppler_hz);

        const Worker* best = nullptr;
        if (!covering.empty()) {
            for (const Worker* w : order) {
                if (w->signal != t.signal || t.prn < w->prn_lo || t.prn > w->prn_hi)
                    continue;
                if (!range_holds(*w, covering))
                    continue; // would straddle this worker's channels
                if (best == nullptr || load[w->name] < load[best->name])
                    best = w;
            }
        }
        if (best == nullptr) {
            out.unplaced.push_back(t);
        } else {
            out.worker_prns[best->name].push_back(t.prn);
            load[best->name]++;
        }
    }

    for (auto& kv : out.worker_prns)
        std::sort(kv.second.begin(), kv.second.end());
    return out;
}

} // namespace gnss
