#include "basebandReadoutManager.hpp"


void basebandReadoutManager::add(basebandRequest req) {
    std::unique_lock<std::mutex> lock(requests_mtx);

    basebandDumpStatus ev{req};
    requests.insert_after(tail, ev);
    tail++;

    has_request.notify_all();
}


std::tuple<basebandDumpStatus*, std::mutex*> basebandReadoutManager::get_next_waiting_request() {
    std::unique_lock<std::mutex> lock(requests_mtx);

    // NB: the requests_mtx is released while the thread is waiting on
    // `has_request`, and reacquired once woken
    using namespace std::chrono_literals;
    has_request.wait_for(lock, 0.1s);

    if (waiting == tail) {
        return std::make_tuple(nullptr, nullptr);
    }
    basebandDumpStatus* ev = &(*++waiting);
    return std::make_tuple(ev, &waiting_mtx);
}


std::tuple<basebandDumpStatus*, std::mutex*> basebandReadoutManager::get_next_ready_request() {
    std::unique_lock<std::mutex> lock(requests_mtx);

    basebandDumpStatus* ev;
    do {
        if (current == waiting || current == tail) {
            throw std::runtime_error("No ready request");
        }
        ev = &(*++current);
    } while (ev->state != basebandDumpStatus::State::INPROGRESS);

    return std::make_tuple(ev, &current_mtx);
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
