#include "GnssTrackState.hpp"

namespace gnss {

void TrackState::promote(const Seed& s) {
    std::lock_guard<std::mutex> lk(_mtx);
    for (auto& p : _pending) {
        if (p.prn == s.prn) {
            p = s; // keep only the newest seed per PRN
            return;
        }
    }
    _pending.push_back(s);
}

std::vector<Seed> TrackState::take_pending() {
    std::lock_guard<std::mutex> lk(_mtx);
    std::vector<Seed> out;
    out.swap(_pending);
    return out;
}

void TrackState::set_owned(const std::set<int>& owned) {
    std::lock_guard<std::mutex> lk(_mtx);
    _owned = owned;
}

std::set<int> TrackState::owned() const {
    std::lock_guard<std::mutex> lk(_mtx);
    return _owned;
}

TrackState& track_state(const std::string& group) {
    // Registry of named states; intentionally leaked (process-lifetime).
    static std::mutex reg_mtx;
    static std::map<std::string, TrackState*> reg;
    std::lock_guard<std::mutex> lk(reg_mtx);
    auto it = reg.find(group);
    if (it == reg.end())
        it = reg.emplace(group, new TrackState()).first;
    return *it->second;
}

} // namespace gnss
