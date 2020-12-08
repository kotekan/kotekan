#include "basebandReadoutManager.hpp"

#include <optional> // for optional


namespace kotekan {


basebandDumpData::basebandDumpData(uint64_t event_id_, uint32_t freq_id_, uint32_t num_elements_,
                                   int64_t data_start_fpga_, uint64_t data_length_fpga_,
                                   timespec data_start_ctime_, const gsl::span<uint8_t>& data_) :
    event_id(event_id_),
    freq_id(freq_id_),
    num_elements(num_elements_),
    data_start_fpga(data_start_fpga_),
    data_length_fpga(data_length_fpga_),
    data_start_ctime(data_start_ctime_),
    data(span_from_length_aligned(data_)),
    reservation_length(data_.size()),
    status(basebandDumpData::Status::Ok) {}

basebandDumpData::basebandDumpData(basebandDumpData::Status status_) :
    event_id(0),
    freq_id(0),
    num_elements(0),
    data_start_fpga(0),
    data_length_fpga(0),
    data_start_ctime({0, 0}),
    data(),
    reservation_length(0),
    status(status_) {}

gsl::span<uint8_t> basebandDumpData::span_from_length_aligned(const gsl::span<uint8_t>& span_) {

    intptr_t span_start_int = (intptr_t)span_.data() + 15;
    span_start_int -= span_start_int % 16;
    uint8_t* span_start = (uint8_t*)span_start_int;
    size_t length = span_.size();
    if (span_start > span_.data()) {
        length -= (span_start - span_.data());
    }
    uint8_t* span_end = span_start + length;

    return gsl::span<uint8_t>(span_start, span_end);
}


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
