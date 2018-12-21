#include "basebandReadoutManager.hpp"


void basebandReadoutManager::add(basebandRequest req) {
    std::unique_lock<std::mutex> lock(requests_mtx);

    basebandDumpStatus ev{req};
    requests.insert_after(tail, ev);
    tail++;

    has_request.notify_all();
}


std::unique_ptr<basebandReadoutManager::requestStatusMutex>
basebandReadoutManager::get_next_waiting_request() {
    std::unique_lock<std::mutex> lock(requests_mtx);

    // NB: the requests_mtx is released while the thread is waiting on
    // `has_request`, and reacquired once woken
    using namespace std::chrono_literals;
    has_request.wait_for(lock, 0.1s);

    if (waiting == tail) {
        return nullptr;
    }
    basebandDumpStatus& ev = *++waiting;
    return std::make_unique<basebandReadoutManager::requestStatusMutex>(ev, waiting_mtx);
}


basebandReadoutManager::requestStatusMutex basebandReadoutManager::get_next_ready_request() {
    std::unique_lock<std::mutex> lock(requests_mtx);

    while (current != waiting && current != tail) {
        basebandDumpStatus& ev = *++current;
        if (ev.state == basebandDumpStatus::State::INPROGRESS) {
            return {ev, current_mtx};
        }
    }

    throw std::runtime_error("No ready request");
}


std::vector<basebandDumpStatus> basebandReadoutManager::all() {
    std::vector<basebandDumpStatus> v;
    std::lock_guard<std::mutex> lock(requests_mtx);
    for (auto it = requests.begin(); it != requests.end(); it++) {
        std::unique_lock<std::mutex> waiting_lock(waiting_mtx, std::defer_lock);
        std::unique_lock<std::mutex> current_lock(current_mtx, std::defer_lock);
        if (it == waiting) {
            waiting_lock.lock();
        }
        if (it == current) {
            current_lock.lock();
        }
        v.push_back(*it);
    }

    return v;
}


std::unique_ptr<basebandDumpStatus> basebandReadoutManager::find(uint64_t event_id) {
    std::lock_guard<std::mutex> lock(requests_mtx);
    for (auto it = requests.begin(); it != requests.end(); it++) {
        std::unique_lock<std::mutex> waiting_lock(waiting_mtx, std::defer_lock);
        std::unique_lock<std::mutex> current_lock(current_mtx, std::defer_lock);
        if (it == waiting) {
            waiting_lock.lock();
        }
        if (it == current) {
            current_lock.lock();
        }
        if (it->request.event_id == event_id) {
            return std::make_unique<basebandDumpStatus>(*it);
        }
    }

    return nullptr;
}
