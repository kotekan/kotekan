#include "basebandReadoutManager.hpp"

#include <optional> // for optional


namespace kotekan {


basebandDumpData::basebandDumpData(const uint64_t event_id_, const uint32_t freq_id_,
                                   const uint32_t stream_freq_idx_, int64_t trigger_start_fpga_,
                                   int64_t trigger_length_fpga_, int dump_start_frame_,
                                   int dump_end_frame_) :
    event_id(event_id_),
    freq_id(freq_id_),
    stream_freq_idx(stream_freq_idx_),
    trigger_start_fpga(trigger_start_fpga_),
    trigger_length_fpga(trigger_length_fpga_),
    dump_start_frame(dump_start_frame_),
    dump_end_frame(dump_end_frame_),
    status(basebandDumpData::Status::Ok) {}


basebandDumpData::basebandDumpData(basebandDumpData::Status status_) :
    event_id(0),
    freq_id(0),
    stream_freq_idx(0),
    trigger_start_fpga(0),
    trigger_length_fpga(0),
    dump_start_frame(0),
    dump_end_frame(0),
    status(status_) {}


void basebandReadoutManager::add(basebandRequest req) {
    std::unique_lock<std::mutex> lock(requests_mtx);

    basebandDumpStatus ev{req};
    requests.push_front(ev);

    waiting_queue.put(std::ref(requests.front()));
}


void basebandReadoutManager::ready(const basebandReadoutManager::ReadyRequest& req) {
    ready_queue.put(req);
}

void basebandReadoutManager::stop() {
    waiting_queue.cancel();
    ready_queue.cancel();
}

std::unique_ptr<basebandReadoutManager::requestStatusMutex>
basebandReadoutManager::get_next_waiting_request() {
    {
        std::unique_lock<std::mutex> lock(requests_mtx);
        readout_current = nullptr;
    }
    auto req = waiting_queue.get();
    if (req) {
        std::unique_lock<std::mutex> lock(requests_mtx);
        readout_current = &(req->get());
        return std::make_unique<requestStatusMutex>(*req, readout_mtx);
    }
    return nullptr;
}


std::unique_ptr<basebandReadoutManager::ReadyRequestMutex>
basebandReadoutManager::get_next_ready_request() {
    {
        std::unique_lock<std::mutex> lock(requests_mtx);
        writeout_current = nullptr;
    }
    auto req = ready_queue.get();
    if (req) {
        std::unique_lock<std::mutex> lock(requests_mtx);
        writeout_current = &(std::get<0>(*req));
        return std::make_unique<ReadyRequestMutex>(*req, writeout_mtx);
    }

    return nullptr;
}


std::vector<basebandDumpStatus> basebandReadoutManager::all() {
    std::vector<basebandDumpStatus> v;
    std::lock_guard<std::mutex> lock(requests_mtx);
    for (auto it = requests.begin(); it != requests.end(); it++) {
        std::unique_lock<std::mutex> readout_lock(readout_mtx, std::defer_lock);
        std::unique_lock<std::mutex> writeout_lock(writeout_mtx, std::defer_lock);
        if (&(*it) == readout_current) {
            readout_lock.lock();
        }
        if (&(*it) == writeout_current) {
            writeout_lock.lock();
        }
        v.push_back(*it);
    }

    return v;
}


std::unique_ptr<basebandDumpStatus> basebandReadoutManager::find(uint64_t event_id) {
    std::lock_guard<std::mutex> lock(requests_mtx);
    for (auto it = requests.begin(); it != requests.end(); it++) {
        if (it->request.event_id == event_id) {
            std::unique_lock<std::mutex> readout_lock(readout_mtx, std::defer_lock);
            std::unique_lock<std::mutex> writeout_lock(writeout_mtx, std::defer_lock);
            if (&(*it) == readout_current) {
                readout_lock.lock();
            }
            if (&(*it) == writeout_current) {
                writeout_lock.lock();
            }
            return std::make_unique<basebandDumpStatus>(*it);
        }
    }

    return nullptr;
}

} // namespace kotekan
