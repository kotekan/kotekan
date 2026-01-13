#ifndef FRAME_PREFETCH_SERVICE_HPP
#define FRAME_PREFETCH_SERVICE_HPP

#include "buffer.hpp"
#include "kotekanLogging.hpp"
#include <atomic>
#include <thread>
#include <vector>
#include <string>
#include <memory>
#include <cmath>

#ifndef MAC_OSX
#include <pthread.h>
#include <sched.h>
#endif

namespace kotekan {

struct FrameInfo {
    uint8_t* frame_ptr = nullptr;
    uint8_t* receipt_bitmap_ptr = nullptr;
    uint64_t start_seq = 0;
    int frame_id = -1;
};

class FramePrefetchService : public kotekanLogging {
public:
    FramePrefetchService(Buffer* buf, Buffer* receipt_bitmap_buf, std::string name, uint32_t port, uint64_t samples_per_frame, int depth, std::vector<int> cpu_affinity, uint64_t capture_n_frames = 0, bool allocate_metadata = true);
    ~FramePrefetchService();

    void start(uint64_t start_seq, const std::vector<uint32_t>& expected_stream_ids);
    void stop();
    
    // Called by consumer to indicate it has finished with the current frame and moved to the next.
    void advance();

    // Get the frame info for a relative index from the current consumed cursor.
    const FrameInfo* get_frame(int relative_offset);
    
    bool is_ready() const { return ready.load(std::memory_order_acquire); }
    bool has_error() const { return error_flag.load(std::memory_order_acquire); }
    bool is_complete() const { return complete_flag.load(std::memory_order_acquire); }
    uint64_t get_start_seq() const { return initial_start_seq.load(std::memory_order_acquire); }

private:
    void prefetcher_loop();
    void apply_affinity();

    // Adds expected stream IDs for this port
    // This function also checks for duplicates across ports and calls to this function.
    void add_expected_stream_ids(const std::vector<uint32_t>& ids);

    Buffer* buf;
    Buffer* receipt_bitmap_buf;
    std::string unique_name;
    uint32_t port;
    uint64_t samples_per_frame;
    int depth;
    std::vector<int> cpu_affinity;
    uint64_t capture_n_frames;
    bool allocate_metadata;

    std::vector<FrameInfo> frames;
    size_t mask;

    uint64_t marked_full_cursor = 0;

    std::vector<uint32_t> expected_stream_ids;

    // Global sorted list of expected stream IDs for all ports, format is [port][stream_id]
    inline static std::vector<std::vector<uint32_t>> global_expected_stream_ids;
    // Global mutex to protect access to the list of expected stream IDs.
    inline static std::mutex global_stream_id_mutex;

    // Written by prefetcher, read by consumer
    alignas(64) std::atomic<uint64_t> produced_cursor{0};
    std::atomic<bool> ready{false};
    std::atomic<bool> error_flag{false};
    std::atomic<bool> complete_flag{false};
    std::atomic<bool> running{false};

    // Written by consumer, read by prefetcher
    alignas(64) std::atomic<uint64_t> consumed_cursor{0};
    std::atomic<uint64_t> initial_start_seq{0};
    std::atomic<bool> start_requested{false};
    
    std::thread prefetcher_thread;
};

FramePrefetchService::FramePrefetchService(Buffer* buf, Buffer* receipt_bitmap_buf, std::string name, uint32_t port, uint64_t samples_per_frame, int depth, std::vector<int> cpu_affinity, uint64_t capture_n_frames, bool allocate_metadata)
    : buf(buf), receipt_bitmap_buf(receipt_bitmap_buf), unique_name(name), port(port), samples_per_frame(samples_per_frame), depth(depth), cpu_affinity(cpu_affinity), capture_n_frames(capture_n_frames), allocate_metadata(allocate_metadata) {
    
    size_t d = 1;
    while (d < (size_t)depth) d <<= 1;
    this->depth = d;
    this->mask = d - 1;
    frames.resize(d);

    if (buf->num_frames != receipt_bitmap_buf->num_frames) {
        throw std::runtime_error("FramePrefetchService: buf and receipt_bitmap_buf must have the same number of frames");
    }

    if (buf->num_frames < depth + 2) {
        throw std::runtime_error("FramePrefetchService: Buffer does not have enough frames for the requested prefetch depth");
    }
    
    running = true;
    prefetcher_thread = std::thread(&FramePrefetchService::prefetcher_loop, this);
}

FramePrefetchService::~FramePrefetchService() {
    stop();
}

void FramePrefetchService::stop() {
    running = false;
    if (prefetcher_thread.joinable()) {
        prefetcher_thread.join();
    }
}

void FramePrefetchService::start(uint64_t start_seq, const std::vector<uint32_t>& expected_stream_ids) {
    initial_start_seq = start_seq;
    add_expected_stream_ids(expected_stream_ids);
    start_requested = true;
}

inline void FramePrefetchService::advance() {
    INFO("FramePrefetchService {} advancing from frame {}", unique_name, consumed_cursor.load(std::memory_order_acquire));
    consumed_cursor.fetch_add(1, std::memory_order_release);
}

inline const FrameInfo* FramePrefetchService::get_frame(int relative_offset) {
    uint64_t current = consumed_cursor.load(std::memory_order_acquire);
    uint64_t target = current + relative_offset;
    
    if (target >= produced_cursor.load(std::memory_order_acquire)) {
        return nullptr;
    }
    
    return &frames[target & mask];
}

void FramePrefetchService::prefetcher_loop() {
    apply_affinity();
    
    while (running) {
        if (!start_requested.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::microseconds(10000));
            continue;
        }
        
        // Check if we need to mark frames full
        uint64_t consumed = consumed_cursor.load(std::memory_order_acquire);
        while (marked_full_cursor < consumed) {
            FrameInfo& info = frames[marked_full_cursor & mask];
            INFO("FramePrefetchService {} marking frame {} full (seq {})", unique_name, info.frame_id, info.start_seq);
            buf->mark_frame_full(unique_name, info.frame_id);
            receipt_bitmap_buf->mark_frame_full(unique_name, info.frame_id);
            marked_full_cursor++;
            
            if (capture_n_frames > 0 && marked_full_cursor >= capture_n_frames) {
                complete_flag = true;
                running = false;
                return;
            }
        }
        
        // Check if we can produce more frames
        uint64_t produced = produced_cursor.load(std::memory_order_acquire);
        if (produced - consumed < (size_t)depth) {
            // Produce next frame
            uint64_t seq;
            if (produced == 0) {
                seq = initial_start_seq.load(std::memory_order_acquire);
            } else {
                seq = initial_start_seq.load(std::memory_order_acquire) + produced * samples_per_frame;
            }
            
            int frame_id = produced % buf->num_frames;
            
            INFO("FramePrefetchService {} prefetching frame {} (seq {})", unique_name, frame_id, seq);
            uint8_t* ptr = buf->wait_for_empty_frame(unique_name, frame_id);
            if (ptr == nullptr) {
                error_flag = true;
                running = false;
                return;
            }
            uint8_t* receipt_bitmap_ptr = receipt_bitmap_buf->wait_for_empty_frame(unique_name, frame_id);
            if (receipt_bitmap_ptr == nullptr) {
                error_flag = true;
                running = false;
                return;
            }
            INFO("FramePrefetchService {} obtained frame {}", unique_name, frame_id);
            
            // Only the worker assigned to allocate metadata does so, as the data buffer 
            // is shared across all workers.
            if (allocate_metadata) {
                buf->allocate_new_metadata_object(frame_id);
                //auto metadata = buf->get_metadata(frame_id);
                //metadata->set_time_sample_start_seq(seq);

                // Print the list of expected stream IDs for this port
                std::string stream_id_list;
                for (size_t i = 0; i < global_expected_stream_ids.at(port).size(); ++i) {
                    stream_id_list += std::to_string(global_expected_stream_ids.at(port)[i]);  
                    if (i < global_expected_stream_ids.at(port).size() - 1) {
                        stream_id_list += ", ";
                    }
                }
                INFO("FramePrefetchService {}: Setting expected stream IDs for port {}: {}", unique_name, port, stream_id_list); 
            }

            // The receipt bitmap metadata is always set.  One bitmap per worker.
            receipt_bitmap_buf->allocate_new_metadata_object(frame_id);
            
            FrameInfo& info = frames[produced & mask];
            info.frame_ptr = ptr;
            info.receipt_bitmap_ptr = receipt_bitmap_ptr;
            info.start_seq = seq;
            info.frame_id = frame_id;
            
            produced_cursor.store(produced + 1, std::memory_order_release);
            
            if (produced + 1 >= 2) {
                ready.store(true, std::memory_order_release);
            }
        } else {
            // Wait a bit to avoid busy loop
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }
}

inline void FramePrefetchService::apply_affinity() {
#ifndef MAC_OSX
    if (cpu_affinity.empty())
        return;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int cpu : cpu_affinity)
        CPU_SET(cpu, &cpuset);

    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    // Set name for easier identification in htop/top
    std::string thread_name = "frame_prefetch";
    pthread_setname_np(pthread_self(), thread_name.c_str());
#endif
}

void FramePrefetchService::add_expected_stream_ids(const std::vector<uint32_t>& ids) {
    std::lock_guard<std::mutex> lock(global_stream_id_mutex);
    
    // Check for duplicates across ports
    for (const auto& existing_ids : global_expected_stream_ids) {
        for (uint32_t id : ids) {
            if (std::find(existing_ids.begin(), existing_ids.end(), id) != existing_ids.end()) {
                throw std::runtime_error("FramePrefetchService: Duplicate stream ID " + std::to_string(id) + " found across ports");
            }
        }
    }
    
    // Add the stream ids to the global list based on the local port number.
    if (global_expected_stream_ids.size() <= port) {
        global_expected_stream_ids.resize(port + 1);
    }
    for (uint32_t id : ids) {
        global_expected_stream_ids[port].push_back(id);
    }
    // Sort the stream IDs for this port.
    std::sort(global_expected_stream_ids[port].begin(), global_expected_stream_ids[port].end());
    
}

} // namespace kotekan

#endif
