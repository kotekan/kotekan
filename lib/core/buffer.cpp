#include "buffer.hpp"

#include "errors.h"     // for CHECK_ERROR_F, ERROR_F, CHECK_MEM_F, INFO_F, DEBUG_F, WARN_F
#include "metadata.hpp" // for metadataContainer, decrement_metadata_ref_count, increment_...
#include "nt_memset.h"  // for nt_memset
#include "util.h"       // for e_time
#ifdef WITH_HSA
#include "hsaBase.h" // for hsa_host_free, hsa_host_malloc
#endif
#include "kotekanLogging.hpp"

#include "fmt.hpp" // for fmt, basic_string_view, make_format_args, FMT_STRING

#include <chrono>
#include <stdexcept>

// IWYU pragma: no_include <asm/mman-common.h>
// IWYU pragma: no_include <asm/mman.h>
#include <assert.h>   // for assert
#include <errno.h>    // for errno, ETIMEDOUT
#include <sched.h>    // for cpu_set_t, CPU_SET, CPU_ZERO
#include <stdio.h>    // for snprintf
#include <stdlib.h>   // for free, malloc
#include <string.h>   // for memset, strerror, memcpy, strdup, strncmp, strncpy
#include <sys/mman.h> // for mlock, mmap, munmap, MAP_FAILED
#ifndef MAC_OSX
#include <linux/mman.h> // for MAP_HUGE_2MB
#endif
#include <time.h> // for NULL, size_t, timespec
#ifdef WITH_NUMA
#include <numa.h>   // for numa_allocate_nodemask, numa_bitmask_free, numa_bitmask_setbit
#include <numaif.h> // for set_mempolicy, mbind, MPOL_BIND, MPOL_DEFAULT, MPOL_MF_STRICT
#endif

// It is assumed this is a power of two in the code.
#define HUGE_PAGE_SIZE 2097152

struct zero_frames_thread_args {
    Buffer* buf;
    int ID;
};

typedef std::lock_guard<std::recursive_mutex> buffer_lock;

GenericBuffer::GenericBuffer(const std::string& _buffer_name, const std::string& _buffer_type,
                             std::shared_ptr<metadataPool> pool, int _num_frames) :
    num_frames(_num_frames),
    shutdown_signal(false), buffer_name(_buffer_name), buffer_type(_buffer_type),
    metadata_pool(pool), metadata(num_frames, nullptr) {
    set_log_prefix(_buffer_type + "Buffer \"" + _buffer_name + "\"");
}

GenericBuffer::~GenericBuffer() {}

bool GenericBuffer::set_metadata(int ID, std::shared_ptr<metadataObject> meta) {
    assert(ID >= 0);
    assert(ID < num_frames);
    buffer_lock lock(mutex);
    metadata[ID] = meta;
    return true;
}

void GenericBuffer::register_consumer(const std::string& name) {
    buffer_lock lock(mutex);

    DEBUG("Registering consumer {:s} for buffer {:s}", name, buffer_name);

    auto const& ins = consumers.try_emplace(name, name, num_frames);
    if (!ins.second) {
        ERROR("You cannot register two consumers with the same name (\"{:s}\")!", name);
        assert(0); // Optional
        return;
    }
}

void GenericBuffer::unregister_consumer(const std::string& name) {
    {
        buffer_lock lock(mutex);
        DEBUG("Unregistering consumer {:s} for buffer {:s}", name, buffer_name);
        size_t nrem = consumers.erase(name);
        if (nrem == 0) {
            ERROR("The consumer {:s} hasn't been registered, cannot unregister!", name);
            return;
        }
    }
    // Signal producers in case removing this consumer causes a buffer to become writable
    empty_cond.notify_all();
}

void GenericBuffer::register_producer(const std::string& name) {
    buffer_lock lock(mutex);
    DEBUG("Buffer: {:s} Registering producer: {:s}", buffer_name, name);

    auto const& ins = producers.try_emplace(name, name, num_frames);
    if (!ins.second) {
        ERROR("You cannot register two producers with the same name (\"{:s}\")!", name);
        assert(0); // Optional
        return;
    }
}

int GenericBuffer::get_num_consumers() {
    buffer_lock lock(mutex);
    return consumers.size();
}

int GenericBuffer::get_num_producers() {
    buffer_lock lock(mutex);
    return producers.size();
}

void GenericBuffer::pass_metadata(int from_ID, GenericBuffer* to_buf, int to_ID) {
    buffer_lock lock(mutex);
    if (metadata[from_ID] == nullptr) {
        WARN("No metadata in source buffer {:s}[{:d}], was this intended?", buffer_name, from_ID);
        return;
    }
    std::shared_ptr<metadataObject> metadata_container = metadata[from_ID];
    bool set = to_buf->set_metadata(to_ID, metadata_container);
    assert(set);
}

void GenericBuffer::copy_metadata(int from_ID, GenericBuffer* to_buf, int to_ID) {
    buffer_lock lock(mutex);
    if (!metadata[from_ID]) {
        WARN("No metadata in source buffer {:s}[{:d}], was this intended?", buffer_name, from_ID);
        return;
    }
    if (!to_buf->metadata[to_ID]) {
        WARN("No metadata in dest buffer {:s}[{:d}], was this intended?", to_buf->buffer_name,
             to_ID);
        return;
    }
    to_buf->private_copy_metadata(to_ID, this, from_ID);
}

void GenericBuffer::private_copy_metadata(int dest_frame_id, GenericBuffer* src, int src_frame_id) {
    std::shared_ptr<metadataObject> from_metadata_container = src->metadata[src_frame_id];
    std::shared_ptr<metadataObject> to_metadata_container = metadata[dest_frame_id];
    if (from_metadata_container->get_object_size() != to_metadata_container->get_object_size()) {
        WARN("Metadata sizes don't match, cannot copy metadata!!");
        return;
    }
    *to_metadata_container = *from_metadata_container;
}

void GenericBuffer::allocate_new_metadata_object(int ID) {
    assert(ID >= 0);
    assert(ID < num_frames);

    buffer_lock lock(mutex);
    if (metadata_pool == nullptr) {
        FATAL_ERROR_F("No metadata pool on %s but metadata was needed by a producer", buffer_name);
    }
    DEBUG2_F("Called allocate_new_metadata_object, buf %p, %d", this, ID);

    if (!metadata[ID])
        metadata[ID] = metadata_pool->request_metadata_object();
}

// Might return empty (null)
std::shared_ptr<metadataObject> GenericBuffer::get_metadata(int ID) {
    assert(ID >= 0);
    assert(ID < num_frames);
    return metadata[ID];
}

void GenericBuffer::send_shutdown_signal() {
    DEBUG("Setting shutdown signal");
    {
        buffer_lock lock(mutex);
        shutdown_signal = true;
    }
    empty_cond.notify_all();
    full_cond.notify_all();
}

void GenericBuffer::json_description(nlohmann::json& buf_json) {
    buf_json["consumers"];
    for (auto& cit : consumers) {
        auto& c = cit.second;
        std::string consumer_name = c.name;
        buf_json["consumers"][consumer_name] = {};
        buf_json["consumers"][consumer_name]["last_frame_acquired"] = c.last_frame_acquired;
        buf_json["consumers"][consumer_name]["last_frame_released"] = c.last_frame_released;
        for (int f = 0; f < num_frames; ++f)
            buf_json["consumers"][consumer_name]["marked_frame_empty"].push_back(c.is_done[f] ? 1
                                                                                              : 0);
    }
    buf_json["producers"];
    for (auto& pit : producers) {
        auto& p = pit.second;
        std::string producer_name = p.name;
        buf_json["producers"][producer_name] = {};
        buf_json["producers"][producer_name]["last_frame_acquired"] = p.last_frame_acquired;
        buf_json["producers"][producer_name]["last_frame_released"] = p.last_frame_released;
        for (int f = 0; f < num_frames; ++f)
            buf_json["producers"][producer_name]["marked_frame_empty"].push_back(p.is_done[f] ? 1
                                                                                              : 0);
    }
    buf_json["num_frames"] = num_frames;
    buf_json["type"] = buffer_type;
}

std::string GenericBuffer::get_dot_node_label() {
    return buffer_name;
}

Buffer::Buffer(int num_frames, size_t len, std::shared_ptr<metadataPool> pool,
               const std::string& _buffer_name, const std::string& _buffer_type, int _numa_node,
               bool _use_hugepages, bool _mlock_frames, bool zero_new_frames) :
    GenericBuffer(_buffer_name, _buffer_type, pool, num_frames),
    frame_size(len),
    // By default don't zero buffers at the end of their use.
    _zero_frames(false), frames(num_frames, nullptr), is_full(num_frames, false),
    last_arrival_time(0), use_hugepages(_use_hugepages), mlock_frames(_mlock_frames),
    numa_node(_numa_node) {
    assert(num_frames > 0);

#if defined(WITH_NUMA) && !defined(WITH_NO_MEMLOCK)
    // Allocate all memory for a buffer on the NUMA domain it's frames are located.
    struct bitmask* node_mask = numa_allocate_nodemask();
    numa_bitmask_setbit(node_mask, numa_node);
    if (set_mempolicy(MPOL_BIND, node_mask ? node_mask->maskp : NULL,
                      node_mask ? node_mask->size + 1 : 0)
        < 0) {
        throw std::runtime_error(
            fmt::format(fmt("Failed to set memory policy: {:s} {:d}"), strerror(errno), errno));
    }
    numa_bitmask_free(node_mask);
#endif

    if (use_hugepages) {
        // Round up to the nearest huge page size multiple.
        aligned_frame_size =
            (int)(((size_t)len + (size_t)HUGE_PAGE_SIZE - 1) & -(size_t)HUGE_PAGE_SIZE);
    } else {
        aligned_frame_size = len;
    }

    // Create the frames.
    for (int i = 0; i < num_frames; ++i) {
        if (len) {
            frames[i] = buffer_malloc(aligned_frame_size, numa_node, use_hugepages, mlock_frames,
                                      zero_new_frames);
            if (frames[i] == nullptr) {
                throw std::runtime_error(
                    fmt::format(fmt("Failed to allocate Buffer memory: %d bytes: %s (%d)"),
                                aligned_frame_size, strerror(errno), errno));
            }
        } else {
            // Put in a pattern != NULL, because NULL is used for signalling, eg in
            // wait_for_empty_frame.
            frames[i] = (uint8_t*)0xffffffff;
        }
    }

#if defined(WITH_NUMA) && !defined(WITH_NO_MEMLOCK)
    // Reset the memory policy so that we don't impact other parts of the
    if (set_mempolicy(MPOL_DEFAULT, nullptr, 0) < 0) {
        throw std::runtime_error(fmt::format(
            fmt("Failed to reset memory policy to default: %s (%d)"), strerror(errno), errno));
    }
#endif
}

Buffer::~Buffer() {
    for (int i = 0; i < num_frames; ++i)
        buffer_free(frames[i], aligned_frame_size, use_hugepages);
}

double Buffer::get_last_arrival_time() {
    return last_arrival_time;
}

void Buffer::private_reset_producers(const int ID) {
    for (auto& x : producers)
        x.second.is_done[ID] = false;
}

void Buffer::private_reset_consumers(const int ID) {
    for (auto& x : consumers)
        x.second.is_done[ID] = false;
}

/*
void Buffer::private_mark_consumer_done(const std::string& name, const int ID) {
    auto& con = consumers.at(name);
    assert(con.is_done[ID] == false);
    con.last_frame_released = ID;
    con.is_done[ID] = true;
}
*/

void Buffer::private_mark_producer_done(const std::string& name, const int ID) {
    auto& pro = producers.at(name);
    assert(pro.is_done[ID] == false);
    pro.last_frame_released = ID;
    pro.is_done[ID] = true;
}

bool Buffer::private_consumers_done(const int ID) {
    for (auto& c : consumers)
        if (!c.second.is_done[ID])
            return false;
    return true;
}

bool Buffer::private_producers_done(const int ID) {
    for (auto& x : producers)
        if (!x.second.is_done[ID])
            return false;
    return true;
}

bool Buffer::is_frame_empty(const int ID) {
    assert(ID >= 0);
    assert(ID < num_frames);
    bool empty = true;
    buffer_lock lock(mutex);
    if (is_full[ID])
        empty = false;
    return empty;
}

uint8_t* Buffer::wait_for_full_frame(const std::string& name, const int ID) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    auto& con = consumers.at(name);
    // This loop exists when is_full == 1 (i.e. a full buffer) AND
    // when this producer hasn't already marked this buffer as
    // while ((!is_full[ID] || con.is_done[ID])
    //       && !shutdown_signal)
    //    full_cond.wait(lock);

    while (true) {
        DEBUG("wait_for_full_frame: frame {:d}, is full? {:s}  is consumer done? {:s}  shutdown? "
              "{:s}",
              ID, is_full[ID] ? "T" : "F", con.is_done[ID] ? "T" : "F",
              shutdown_signal ? "T" : "F");
        if ((!is_full[ID] || con.is_done[ID]) && !shutdown_signal) {
            DEBUG("waiting on condition...");
            full_cond.wait(lock);
        } else
            break;
    }
    lock.unlock();

    if (shutdown_signal)
        return nullptr;
    con.last_frame_acquired = ID;
    return frames[ID];
}

int Buffer::wait_for_full_frame_timeout(const std::string& name, const int ID,
                                        const struct timespec timeout_time) {
    std::chrono::duration dur =
        std::chrono::seconds{timeout_time.tv_sec} + std::chrono::nanoseconds{timeout_time.tv_nsec};
    std::chrono::time_point<std::chrono::system_clock> deadline(dur);
    std::cv_status st = std::cv_status::no_timeout;
    std::unique_lock<std::recursive_mutex> lock(mutex);
    auto& con = consumers.at(name);

    while ((!is_full[ID] || con.is_done[ID]) && !shutdown_signal) {
        st = full_cond.wait_until(lock, deadline);
        if (st == std::cv_status::timeout)
            break;
    }
    lock.unlock();

    if (shutdown_signal)
        return -1;

    if (st == std::cv_status::timeout)
        return 1;

    con.last_frame_acquired = ID;
    return 0;
}

int Buffer::get_num_full_frames() {
    buffer_lock lock(mutex);
    int numFull = 0;
    for (int i = 0; i < num_frames; ++i)
        if (is_full[i])
            numFull++;
    return numFull;
}

void Buffer::json_description(nlohmann::json& buf_json) {
    GenericBuffer::json_description(buf_json);
    buf_json["frames"];
    for (int i = 0; i < num_frames; ++i)
        buf_json["frames"].push_back(is_full[i] ? 1 : 0);
    buf_json["num_full_frame"] = get_num_full_frames();
    buf_json["frame_size"] = frame_size;
    buf_json["last_frame_arrival_time"] = last_arrival_time;
}

std::string Buffer::get_dot_node_label() {
    return fmt::format(fmt("{:s}<BR/>{:d}/{:d} ({:.1f}%)"), buffer_name, get_num_full_frames(),
                       num_frames, (float)get_num_full_frames() / num_frames * 100);
}

void Buffer::print_buffer_status() {
    std::vector<bool> local_is_full;
    {
        buffer_lock lock(mutex);
        local_is_full = is_full;
    }
    char status_string[num_frames + 1];
    for (int i = 0; i < num_frames; ++i) {
        if (local_is_full[i])
            status_string[i] = 'X';
        else
            status_string[i] = '_';
    }
    status_string[num_frames] = '\0';
    INFO("Buffer {:s}, status: {:s}", buffer_name, std::string(status_string));
}

void Buffer::print_full_status() {

    buffer_lock lock(mutex);

    char status_string[num_frames + 1];
    status_string[num_frames] = '\0';

    INFO_F("--------------------- %s ---------------------", buffer_name);

    for (int i = 0; i < num_frames; ++i) {
        if (is_full[i])
            status_string[i] = 'X';
        else
            status_string[i] = '_';
    }

    INFO_F("Full Frames (X)                : %s", status_string);

    INFO_F("---- Producers ----");
    for (auto& xit : producers)
        for (int i = 0; i < num_frames; ++i) {
            auto& x = xit.second;
            if (x.is_done[i])
                status_string[i] = '+';
            else
                status_string[i] = '_';
            INFO_F("%-30s : %s (%d, %d)", x.name, status_string, x.last_frame_acquired,
                   x.last_frame_released);
        }

    INFO_F("---- Consumers ----");
    for (auto& xit : consumers)
        for (int i = 0; i < num_frames; ++i) {
            auto& x = xit.second;
            if (x.is_done[i])
                status_string[i] = '=';
            else
                status_string[i] = '_';
            INFO_F("%-30s : %s (%d, %d)", x.name, status_string, x.last_frame_acquired,
                   x.last_frame_released);
        }
}

void Buffer::mark_frame_full(const std::string& name, const int ID) {
    assert(ID >= 0);
    assert(ID < num_frames);

    // DEBUG_F("Frame %s[%d] being marked full by producer %s\n", buf->buffer_name, ID, name);

    bool set_full = false;
    bool set_empty = false;
    {
        buffer_lock lock(mutex);

        private_mark_producer_done(name, ID);
        if (private_producers_done(ID)) {
            private_reset_producers(ID);
            is_full[ID] = true;
            last_arrival_time = e_time();
            set_full = true;

            // If there are no consumers registered then we can just mark the buffer empty
            if (private_consumers_done(ID)) {
                DEBUG("No consumers are registered on {:s} dropping data in frame {:d}...",
                      buffer_name, ID);
                is_full[ID] = false;
                metadata[ID].reset();
                set_empty = true;
                private_reset_consumers(ID);
            }
        }
    }

    // Signal consumer
    if (set_full)
        full_cond.notify_all();

    // Signal producer
    if (set_empty) {
        // empty_cond.notify_all();
    }
}

// function passed to pthreads
void* private_zero_frames(void* args) {
    int ID = ((struct zero_frames_thread_args*)(args))->ID;
    Buffer* buf = ((struct zero_frames_thread_args*)(args))->buf;
    free(args);
    buf->_impl_zero_frame(ID);
    return nullptr;
}

void Buffer::_impl_zero_frame(const int ID) {
    assert(ID >= 0);
    assert(ID <= num_frames);

    // This zeros everything, but for VDIF we just need to header zeroed.
    int div_256 = 256 * (frame_size / 256);
    nt_memset((void*)frames[ID], 0x00, div_256);
    memset((void*)&frames[ID][div_256], 0x00, frame_size - div_256);

    // HACK: Just zero the first two words of the VDIF header
    // for (int i = 0; i < frame_size/1056; ++i) {
    //    *((uint64_t*)&frames[ID][i*1056]) = 0;
    //}
    {
        buffer_lock lock(mutex);

        is_full[ID] = false;
        private_reset_consumers(ID);
    }
    empty_cond.notify_all();

    int ret = 0;
    pthread_exit(&ret);
}

void Buffer::zero_frames() {
    buffer_lock lock(mutex);
    _zero_frames = true;
}

void Buffer::mark_frame_empty(const std::string& consumer_name, const int ID) {
    assert(ID >= 0);
    assert(ID < num_frames);
    bool broadcast = false;

    // If we've been asked to zero the buffer do it here.
    // This needs to happen out side of the critical section
    // so that we don't block for a long time here.
    {
        buffer_lock lock(mutex);
        consumers.at(consumer_name).is_done[ID] = true;
        if (private_consumers_done(ID)) {
            broadcast = private_mark_frame_empty(ID);
        }
    }
    // Signal producer
    if (broadcast)
        empty_cond.notify_all();
}

bool Buffer::private_mark_frame_empty(const int ID) {
    bool broadcast = false;
    if (_zero_frames) {
        pthread_t zero_t;
        struct zero_frames_thread_args* zero_args =
            (struct zero_frames_thread_args*)malloc(sizeof(struct zero_frames_thread_args));
        zero_args->ID = ID;
        zero_args->buf = this;

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        // TODO: Move this to the config file (when buffers.c updated to C++11)
        CPU_SET(5, &cpuset);

        CHECK_ERROR_F(pthread_create(&zero_t, nullptr, &private_zero_frames, (void*)zero_args));
        CHECK_ERROR_F(pthread_setaffinity_np(zero_t, sizeof(cpu_set_t), &cpuset));
        CHECK_ERROR_F(pthread_detach(zero_t));
    } else {
        is_full[ID] = 0;
        private_reset_consumers(ID);
        broadcast = true;
    }
    if (metadata[ID])
        metadata[ID].reset();
    return broadcast;
}

uint8_t* Buffer::wait_for_empty_frame(const std::string& producer_name, const int ID) {
    assert(ID >= 0);
    assert(ID < num_frames);

    StageInfo* pro;
    bool print_stat = 0;

    std::unique_lock<std::recursive_mutex> lock(mutex);

    pro = &producers.at(producer_name);
    // If the buffer isn't full, i.e. is_full[ID] == 0, then we never sleep on the cond var.
    // The second condition stops us from using a buffer we've already filled,
    // and forces a wait until that buffer has been marked as empty.
    while ((is_full[ID] || pro->is_done[ID]) && !shutdown_signal) {
        DEBUG("wait_for_empty_frame: {:s} waiting for empty frame ID = {:d} in buffer {:s}",
              producer_name, ID, buffer_name);
        print_stat = true;
        empty_cond.wait(lock);
    }
    lock.unlock();

    // TODO: remove this output until we have a solution which has better control over log levels
    // if (print_stat == 1)
    //     print_buffer_status(buf);
    (void)print_stat;

    if (shutdown_signal)
        return nullptr;
    pro->last_frame_acquired = ID;
    return frames[ID];
}

void Buffer::unregister_consumer(const std::string& name) {
    int broadcast = 0;
    {
        buffer_lock lock(mutex);
        DEBUG("Unregistering consumer {:s} for buffer {:s}", name, buffer_name);
        size_t nrem = consumers.erase(name);
        if (nrem == 0) {
            ERROR("The consumer {:s} hasn't been registered, cannot unregister!", name);
            return;
        }
        // Check if removing this consumer would cause any of the frames
        // which are currently full to become empty.
        for (int id = 0; id < num_frames; ++id)
            if (private_consumers_done(id))
                broadcast |= private_mark_frame_empty(id);
    }
    // Signal producers if we found something could be empty after
    // removal of this consumer.
    if (broadcast)
        empty_cond.notify_all();
}

uint8_t* Buffer::swap_external_frame(int frame_id, uint8_t* external_frame) {

    buffer_lock lock(mutex);

    // Check that we don't have more than one producer.
    assert(producers.size() == 1);

    uint8_t* temp_frame = frames[frame_id];
    frames[frame_id] = external_frame;

    return temp_frame;
}

void Buffer::swap_frames(int from_frame_id, Buffer* to_buf, int to_frame_id) {

    assert(this != to_buf);
    assert(to_buf != nullptr);
    assert(from_frame_id >= 0);
    assert(from_frame_id < num_frames);
    assert(to_frame_id >= 0);
    assert(to_frame_id < to_buf->num_frames);
    assert(aligned_frame_size == to_buf->aligned_frame_size);

    buffer_lock lock(mutex);

    int num_consumers = get_num_consumers();
    assert(num_consumers == 1);
    (void)num_consumers;
    int num_producers = to_buf->get_num_producers();
    assert(num_producers == 1);
    (void)num_producers;

    // Swap the frame in -- the "to_buf" will lock its own mutex
    uint8_t* temp_frame = to_buf->swap_external_frame(to_frame_id, frames[from_frame_id]);
    frames[from_frame_id] = temp_frame;
}

void Buffer::safe_swap_frame(int src_frame_id, Buffer* dest_buf, int dest_frame_id) {
    assert(this != dest_buf);
    assert(dest_buf != nullptr);
    assert(src_frame_id >= 0);
    assert(src_frame_id < num_frames);
    assert(dest_frame_id >= 0);
    assert(dest_frame_id < dest_buf->num_frames);

    // Buffer sizes must match exactly
    if (frame_size != dest_buf->frame_size) {
        FATAL_ERROR("Buffer sizes must match for direct copy ({:s}.frame_size != {:s}.frame_size)",
                    buffer_name, dest_buf->buffer_name);
    }
    if (dest_buf->get_num_producers() > 1) {
        FATAL_ERROR("Cannot swap/copy frames into dest buffer {:s} with more than one producer",
                    dest_buf->buffer_name);
    }

    buffer_lock lock(mutex);

    int num_consumers = get_num_consumers();

    // Copy or transfer the data part.
    if (num_consumers == 1) {
        // Swap the frames
        swap_frames(src_frame_id, dest_buf, dest_frame_id);
    } else if (num_consumers > 1) {
        // Copy the frame data over, leaving the source intact.
        // the dest will lock its mutex in this call
        dest_buf->private_copy_frame(dest_frame_id, this, src_frame_id);
    }
}

void Buffer::private_copy_frame(int dest_frame_id, Buffer* src, int src_frame_id) {
    buffer_lock lock(mutex);
    memcpy(frames[dest_frame_id], src->frames[src_frame_id], src->frame_size);
}

bool is_frame_buffer(GenericBuffer* buf) {
    // See also bufferFactor::new_buffer()
    return (buf->buffer_type == "standard") || (buf->buffer_type == "vis")
           || (buf->buffer_type == "hfb");
}

uint8_t* buffer_malloc(size_t len, int numa_node, bool use_hugepages, bool mlock_frames,
                       bool zero_new_frames) {

    uint8_t* frame = nullptr;

#ifdef WITH_HSA // Support for legacy HSA support used in CHIME
    (void)use_hugepages;
    frame = (uint8_t*)hsa_host_malloc(len, numa_node);
    if (frame == nullptr) {
        return nullptr;
    }
#else
    if (use_hugepages) {
#ifndef MAC_OSX
        void* mapped_frame = mmap(nullptr, len, PROT_READ | PROT_WRITE,
                                  MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB, -1, 0);
        if (mapped_frame == MAP_FAILED) {
            ERROR_F("Error mapping huge pages, check available huge pages: %s (%d)",
                    strerror(errno), errno);
            return nullptr;
        }
        // Strictly bind the memory to the required NUMA domain
#ifdef WITH_NUMA
        struct bitmask* node_mask = numa_allocate_nodemask();
        numa_bitmask_setbit(node_mask, numa_node);
        if (mbind(mapped_frame, len, MPOL_BIND, node_mask ? node_mask->maskp : NULL,
                  node_mask ? node_mask->size + 1 : 0, MPOL_MF_STRICT)
            < 0) {
            ERROR_F("Failed to bind huge page frames to requested NUMA node: %s (%d)",
                    strerror(errno), errno);
            return nullptr;
        }
        numa_bitmask_free(node_mask);
#endif
        frame = (uint8_t*)mapped_frame;
#else
        ERROR_F("Huge pages not supported in Mac OSX.");
        return nullptr;
#endif
    } else {
#ifdef WITH_NUMA
        frame = (uint8_t*)numa_alloc_onnode(len, numa_node);
        CHECK_MEM_F(frame);
#else
        (void)numa_node;
        // Create a page aligned block of memory for the buffer
        int err = posix_memalign((void**)&(frame), PAGESIZE_MEM, len);
        if (err != 0) {
            ERROR_F("Error creating aligned memory: %d", err);
            return nullptr;
        }
        CHECK_MEM_F(frame);
#endif
    }
#endif

#ifndef WITH_NO_MEMLOCK
    if (mlock_frames) {
        // Ask that all pages be kept in memory
        if (mlock((void*)frame, len) != 0) {
            ERROR_F("Error locking memory: %d - check ulimit -a to check memlock limits", errno);
            buffer_free(frame, len, use_hugepages);
            return nullptr;
        }
    }
#else
    (void)mlock_frames;
#endif
    // Zero the new frame
    if (zero_new_frames)
        memset(frame, 0x0, len);

    return frame;
}

void buffer_free(uint8_t* frame_pointer, size_t size, bool use_hugepages) {
#ifdef WITH_HSA
    (void)size;
    (void)use_hugepages;
    hsa_host_free(frame_pointer);
#else
    if (use_hugepages) {
        munmap(frame_pointer, size);
    } else {
#ifdef WITH_NUMA
        numa_free(frame_pointer, size);
#else
        (void)size;
        free(frame_pointer);
#endif
    }
#endif
}
