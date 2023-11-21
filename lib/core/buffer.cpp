#include "buffer.hpp"

#include "errors.h"    // for CHECK_ERROR_F, ERROR_F, CHECK_MEM_F, INFO_F, DEBUG_F, WARN_F
#include "metadata.h"  // for metadataContainer, decrement_metadata_ref_count, increment_...
#include "nt_memset.h" // for nt_memset
#include "util.h"      // for e_time
#ifdef WITH_HSA
#include "hsaBase.h" // for hsa_host_free, hsa_host_malloc
#endif
#include "kotekanLogging.hpp"
#include "fmt.hpp" // for fmt, basic_string_view, make_format_args, FMT_STRING

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

//#define BUFFER_LOCK std::lock_guard<std::recursive_mutex> _lock(mutex)
typedef std::lock_guard<std::recursive_mutex> buffer_lock;

/*
void* private_zero_frames(void* args);

// Returns -1 if there is no producer with that name
int private_get_producer_id(Buffer* buf, const char* name);

// Marks the consumer named by `name` as done for the given ID
void private_mark_consumer_done(Buffer* buf, const char* name, const int ID);

// Marks the producer named by `name` as done for the given ID
void private_mark_producer_done(Buffer* buf, const char* name, const int ID);
*/

GenericBuffer::GenericBuffer(const std::string& _buffer_name, const std::string& _buffer_type,
                             metadataPool* pool, int _num_frames) :
    num_frames(_num_frames),
    buffer_name(_buffer_name),
    buffer_type(_buffer_type),
    metadata_pool(pool),
    metadata(num_frames, NULL) {
    //CHECK_ERROR_F(pthread_mutex_init(&lock, nullptr));
    //CHECK_ERROR_F(pthread_cond_init(&full_cond, nullptr));
    //CHECK_ERROR_F(pthread_cond_init(&empty_cond, nullptr));
}

GenericBuffer::~GenericBuffer() {
    // Free locks and cond vars
    //CHECK_ERROR_F(pthread_mutex_destroy(&lock));
    //CHECK_ERROR_F(pthread_cond_destroy(&full_cond));
    //CHECK_ERROR_F(pthread_cond_destroy(&empty_cond));
}

RingBuffer::RingBuffer(metadataPool* pool, const std::string& _buffer_name, const std::string& _buffer_type) :
    GenericBuffer(_buffer_name, _buffer_type, pool, 1) {
}

Buffer::Buffer(int num_frames, size_t len, metadataPool* pool, const std::string& _buffer_name,
               const std::string& _buffer_type, int _numa_node, bool _use_hugepages, bool _mlock_frames,
               bool zero_new_frames) :
    GenericBuffer(_buffer_name, _buffer_type, pool, num_frames),
    frame_size(len),
    // By default don't zero buffers at the end of their use.
    zero_frames(false),
    frames(num_frames, nullptr),
    is_full(num_frames, false),
    last_arrival_time(0),
    use_hugepages(_use_hugepages),
    mlock_frames(_mlock_frames),
    numa_node(_numa_node)
{
    assert(num_frames > 0);

#if defined(WITH_NUMA) && !defined(WITH_NO_MEMLOCK)
    // Allocate all memory for a buffer on the NUMA domain it's frames are located.
    struct bitmask* node_mask = numa_allocate_nodemask();
    numa_bitmask_setbit(node_mask, numa_node);
    if (set_mempolicy(MPOL_BIND, node_mask ? node_mask->maskp : NULL,
                      node_mask ? node_mask->size + 1 : 0)
        < 0) {
        throw std::runtime_error(fmt::format(fmt("Failed to set memory policy: %s (%d)"), strerror(errno), errno));
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
            frames[i] = buffer_malloc(aligned_frame_size, numa_node, use_hugepages,
                                           mlock_frames, zero_new_frames);
            if (frames[i] == nullptr) {
                throw std::runtime_error(fmt::format(fmt("Failed to allocate Buffer memory: %d bytes: %s (%d)"), aligned_frame_size, strerror(errno), errno));
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
        throw std::runtime_error(fmt::format(fmt("Failed to reset memory policy to default: %s (%d)"), strerror(errno), errno));
    }
#endif
}

Buffer::~Buffer() {
    for (int i = 0; i < num_frames; ++i)
        buffer_free(frames[i], aligned_frame_size, use_hugepages);
}

void Buffer::mark_frame_full(const std::string& name, const int ID) {
    assert(ID >= 0);
    assert(ID < buf->num_frames);

    // DEBUG_F("Frame %s[%d] being marked full by producer %s\n", buf->buffer_name, ID, name);

    {
        buffer_lock lock(mutex);

        bool set_full = false;
        bool set_empty = false;

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
                if (metadata[ID] != NULL) {
                    decrement_metadata_ref_count(metadata[ID]);
                    metadata[ID] = NULL;
                }
                set_empty = true;
                private_reset_consumers(ID);
            }
        }
    }

    // Signal consumer
    if (set_full)
        //CHECK_ERROR_F(pthread_cond_broadcast(&buf->full_cond));
        full_cond.notify_all();

    // Signal producer
    if (set_empty) {
        // CHECK_ERROR_F( pthread_cond_broadcast(&buf->empty_cond) );
    }
}

void* private_zero_frames(void* args) {

    int ID = ((struct zero_frames_thread_args*)(args))->ID;
    Buffer* buf = ((struct zero_frames_thread_args*)(args))->buf;

    assert(ID >= 0);
    assert(ID <= buf->num_frames);

    // This zeros everything, but for VDIF we just need to header zeroed.
    int div_256 = 256 * (buf->frame_size / 256);
    nt_memset((void*)buf->frames[ID], 0x00, div_256);
    memset((void*)&buf->frames[ID][div_256], 0x00, buf->frame_size - div_256);

    // HACK: Just zero the first two words of the VDIF header
    // for (int i = 0; i < buf->frame_size/1056; ++i) {
    //    *((uint64_t*)&buf->frames[ID][i*1056]) = 0;
    //}

    CHECK_ERROR_F(pthread_mutex_lock(&buf->lock));

    buf->is_full[ID] = 0;
    private_reset_consumers(ID);

    CHECK_ERROR_F(pthread_mutex_unlock(&buf->lock));

    CHECK_ERROR_F(pthread_cond_broadcast(&buf->empty_cond));

    free(args);

    int ret = 0;
    pthread_exit(&ret);
}

void zero_frames(Buffer* buf) {
    CHECK_ERROR_F(pthread_mutex_lock(&buf->lock));
    buf->zero_frames = 1;
    CHECK_ERROR_F(pthread_mutex_unlock(&buf->lock));
}

void mark_frame_empty(Buffer* buf, const std::string& consumer_name, const int ID) {
    assert(ID >= 0);
    assert(ID < buf->num_frames);
    int broadcast = 0;

    // If we've been asked to zero the buffer do it here.
    // This needs to happen out side of the critical section
    // so that we don't block for a long time here.
    CHECK_ERROR_F(pthread_mutex_lock(&buf->lock));

    if (buf->ring_buffer_size) {

        // Reader has consumed ring_buffer_read_size bytes!
        INFO_NON_OO("Ring buffer: mark_frame_empty.  Available: {:d} - Claimed: {:d},  Read size: {:d}",
                    buf->ring_buffer_elements, buf->ring_buffer_elements_claimed, buf->ring_buffer_read_size);
        assert(buf->ring_buffer_elements >= buf->ring_buffer_read_size);
        assert(buf->ring_buffer_elements_claimed >= buf->ring_buffer_read_size);
        buf->ring_buffer_elements -= buf->ring_buffer_read_size;
        buf->ring_buffer_elements_claimed -= buf->ring_buffer_read_size;
        buf->ring_buffer_read_cursor = (buf->ring_buffer_read_cursor + buf->ring_buffer_read_size) % buf->ring_buffer_size;
        INFO_NON_OO("Ring buffer: mark_frame_empty (read {:d}).  Now available: {:d} - Claimed: {:d}",
                    buf->ring_buffer_read_size, buf->ring_buffer_elements, buf->ring_buffer_elements_claimed);
        broadcast = 1;

    } else {

        buf->consumers.at(consumer_name).is_done[ID] = true;

        if (private_consumers_done(ID)) {
            broadcast = private_mark_frame_empty(ID);
        }

    }

    CHECK_ERROR_F(pthread_mutex_unlock(&buf->lock));

    // Signal producer
    if (broadcast == 1) {
        CHECK_ERROR_F(pthread_cond_broadcast(&buf->empty_cond));
    }
}

bool Buffer::private_mark_frame_empty(const int ID) {
    bool broadcast = false;
    if (zero_frames == 1) {
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
    if (metadata[ID] != NULL) {
        decrement_metadata_ref_count(metadata[ID]);
        metadata[ID] = NULL;
    }
    return broadcast;
}

uint8_t* wait_for_empty_frame(Buffer* buf, const std::string& producer_name, const int ID) {
    assert(ID >= 0);
    assert(ID < buf->num_frames);

    int print_stat = 0;

    CHECK_ERROR_F(pthread_mutex_lock(&buf->lock));

    auto& pro = buf->producers[producer_name];

    // If the buffer isn't full, i.e. is_full[ID] == 0, then we never sleep on the cond var.
    // The second condition stops us from using a buffer we've already filled,
    // and forces a wait until that buffer has been marked as empty.
    while ((buf->is_full[ID] == 1 || pro.is_done[ID])
           && buf->shutdown_signal == 0) {
        DEBUG_F("wait_for_empty_frame: %s waiting for empty frame ID = %d in buffer %s",
                producer_name, ID, buf->buffer_name);
        print_stat = 1;
        pthread_cond_wait(&buf->empty_cond, &buf->lock);
    }

    CHECK_ERROR_F(pthread_mutex_unlock(&buf->lock));

    // TODO: remove this output until we have a solution which has better control over log levels
    // if (print_stat == 1)
    //     print_buffer_status(buf);
    (void)print_stat;

    if (buf->shutdown_signal == 1)
        return NULL;

    pro.last_frame_acquired = ID;
    return buf->frames[ID];
}

void buffer_set_ring_buffer_size(Buffer* buf, size_t sz) {
    CHECK_ERROR_F(pthread_mutex_lock(&buf->lock));
    buf->ring_buffer_size = sz;
    CHECK_ERROR_F(pthread_mutex_unlock(&buf->lock));
}

void buffer_set_ring_buffer_read_size(Buffer* buf, size_t sz) {
    CHECK_ERROR_F(pthread_mutex_lock(&buf->lock));
    buf->ring_buffer_read_size = sz;
    CHECK_ERROR_F(pthread_mutex_unlock(&buf->lock));
}

int buffer_wait_for_ring_buffer_writable(Buffer* buf, size_t sz) {
    CHECK_ERROR_F(pthread_mutex_lock(&buf->lock));

    while (1) {
        INFO_NON_OO("Ring buffer: waiting to write {:d} elements.  Currently available: {:d}, size {:d}",
               sz, buf->ring_buffer_elements, buf->ring_buffer_size);
        if (buf->ring_buffer_elements + sz <= buf->ring_buffer_size)
            break;
        if (buf->shutdown_signal == 1)
            break;
        INFO_NON_OO("Ring buffer: waiting for space to write...");
        pthread_cond_wait(&buf->empty_cond, &buf->lock);
    }

    CHECK_ERROR_F(pthread_mutex_unlock(&buf->lock));
    return 0;
}

void buffer_wrote_to_ring_buffer(Buffer* buf, /*const char* name,*/ size_t sz) {
    CHECK_ERROR_F(pthread_mutex_lock(&buf->lock));

    buf->ring_buffer_elements += sz;
    INFO_NON_OO("Ring buffer: wrote {:d}.  Now available: {:d}", sz, buf->ring_buffer_elements);
    buf->ring_buffer_write_cursor = (buf->ring_buffer_write_cursor + sz) % buf->ring_buffer_size;

    // If we filled a read-size block, mark the next frame as full.
    //if (buf->ring_buffer_elements >= buf->ring_buffer_read_size)
    //private_mark_frame_full(buf, name, ring_buffer_full_frame);
    //ring_buffer_full_frame = (ring_buffer_full_frame + 1) % buf->num_frames;

    // If there are no consumers registered then we can just mark the buffer empty
    /*
    if (private_consumers_done(buf, ID) == 1) {
        DEBUG_F("No consumers are registered on %s dropping data in frame %d...",
                buf->buffer_name, ID);
        buf->is_full[ID] = 0;
        if (buf->metadata[ID] != NULL) {
            decrement_metadata_ref_count(buf->metadata[ID]);
            buf->metadata[ID] = NULL;
        }
        set_empty = 1;
        private_reset_consumers(buf, ID);
    }
    */
    
    CHECK_ERROR_F(pthread_mutex_unlock(&buf->lock));

    // Signal consumer
    CHECK_ERROR_F(pthread_cond_broadcast(&buf->full_cond));
}

uint8_t* buffer_claim_next_full_frame(Buffer* buf, const std::string&, const int frame_id) {
    CHECK_ERROR_F(pthread_mutex_lock(&buf->lock));

    //int consumer_id = private_get_consumer_id(buf, name);
    //assert(consumer_id != -1);

    assert(buf->ring_buffer_size);

    // HACK -- only works for single consumer
    //assert(consumer_id == 0);

    // DEBUG - prints
    while (1) {
        INFO_NON_OO("Ring buffer: waiting for input data frame {:d}.  Need: {:d}.  Available: {:d} - Claimed: {:d}",
                    frame_id, buf->ring_buffer_read_size, buf->ring_buffer_elements, buf->ring_buffer_elements_claimed);
        if (buf->ring_buffer_elements - buf->ring_buffer_elements_claimed >= buf->ring_buffer_read_size)
                break;
        if (buf->shutdown_signal)
            break;
        // FIXME???
        //if (buf->consumers_done[ID][consumer_id] == 1)
        //break;
        INFO_NON_OO("Ring buffer: waiting on condition variable for input data");
        pthread_cond_wait(&buf->full_cond, &buf->lock);
    }
    // Claim!
    buf->ring_buffer_elements_claimed += buf->ring_buffer_read_size;
    // (read cursor is moved when elements are released??? no, that's not right!)

    CHECK_ERROR_F(pthread_mutex_unlock(&buf->lock));
    if (buf->shutdown_signal == 1)
        return NULL;
    //buf->consumers[consumer_id].last_frame_acquired = ID;
    return buf->frames[frame_id];
}

void GenericBuffer::register_consumer(const std::string& name) {
    CHECK_ERROR_F(pthread_mutex_lock(&lock));

    DEBUG("Registering consumer {:s} for buffer {:s}", name, buffer_name);

    auto const& ins = consumers.try_emplace(name, name, num_frames);
    if (!ins.second) {
        ERROR_F("You cannot register two consumers with the same name (\"{:s}\")!", name);
        assert(0); // Optional
        CHECK_ERROR_F(pthread_mutex_unlock(&lock));
        return;
    }
    //registered_consumer(ins.second);

    CHECK_ERROR_F(pthread_mutex_unlock(&lock));
}

void GenericBuffer::unregister_consumer(const std::string& name) {
    //int broadcast = 0;
    CHECK_ERROR_F(pthread_mutex_lock(&lock));
    DEBUG("Unregistering consumer {:s} for buffer {:s}", name, buffer_name);

    size_t nrem = consumers.erase(name);
    if (nrem == 0) {
        ERROR("The consumer {:s} hasn't been registered, cannot unregister!", name);
    }

    /*
    // Check if removing this consumer would cause any of the frames
    // which are currently full to become empty.
    for (int id = 0; id < buf->num_frames; ++id) {
        if (private_consumers_done(buf, id) == 1) {
            broadcast |= private_mark_frame_empty(buf, id);
        }
    }
    */

    CHECK_ERROR_F(pthread_mutex_unlock(&lock));

    // Signal producers if we found something could be empty after
    // removal of this consumer.
    //if (broadcast == 1) {
    CHECK_ERROR_F(pthread_cond_broadcast(&empty_cond));
    //}
}


void GenericBuffer::register_producer(const std::string& name) {
    CHECK_ERROR_F(pthread_mutex_lock(&lock));
    DEBUG("Buffer: {:s} Registering producer: {:s}", buffer_name, name);

    auto const& ins = producers.try_emplace(name, name, num_frames);
    if (!ins.second) {
        ERROR("You cannot register two producers with the same name (\"{:s}\")!", name);
        assert(0); // Optional
        CHECK_ERROR_F(pthread_mutex_unlock(&lock));
        return;
    }
    //registered_producer(ins.first.second);

    CHECK_ERROR_F(pthread_mutex_unlock(&lock));
}

/*
  void Buffer::registered_producer(StageInfo&) {
  }
*/

void Buffer::private_reset_producers(const int ID) {
    for (auto& x : producers)
        x.second.is_done[ID] = false;
}

void Buffer::private_reset_consumers(const int ID) {
    for (auto& x : consumers)
        x.second.is_done[ID] = false;
}

void private_mark_consumer_done(Buffer* buf, const std::string& name, const int ID) {
    auto& con = buf->consumers.at(name);
    assert(con.is_done[ID] == false);
    con.last_frame_released = ID;
    con.is_done[ID] = true;
}

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

int is_frame_empty(Buffer* buf, const int ID) {
    assert(ID >= 0);
    assert(buf != NULL);
    assert(ID < buf->num_frames);

    int empty = 1;

    CHECK_ERROR_F(pthread_mutex_lock(&buf->lock));

    if (buf->is_full[ID] == 1) {
        empty = 0;
    }

    CHECK_ERROR_F(pthread_mutex_unlock(&buf->lock));

    return empty;
}

uint8_t* wait_for_full_frame(Buffer* buf, const std::string& name, const int ID) {
    CHECK_ERROR_F(pthread_mutex_lock(&buf->lock));

    auto& cons = buf->consumers.at(name);

    if (buf->ring_buffer_size) {
        // HACK -- only works for single consumer
        // DEBUG - prints
        while (1) {
            INFO_NON_OO("Ring buffer: waiting for input data frame {:d}.  Need: {:d}.  Available: {:d}",
                        ID, buf->ring_buffer_read_size, buf->ring_buffer_elements);
            if (buf->ring_buffer_elements >= buf->ring_buffer_read_size)
                break;
            if (buf->shutdown_signal)
                break;
            // FIXME???
            //if (buf->consumers_done[ID][consumer_id] == 1)
            //break;
            INFO_NON_OO("Ring buffer: waiting on condition variable for input data");
            pthread_cond_wait(&buf->full_cond, &buf->lock);
        }
        /*
        INFO_F("Ring buffer: waiting for input data.  Need: {:d}.  Available: {:d}",
               ring_buffer_read_size, ring_buffer_elements);
        while ((ring_buffer_elements <= ring_buffer_read_size ||
                buf->consumers_done[ID][consumer_id] == 1)
               && buf->shutdown_signal == 0) {
            INFO_F("Ring buffer: waiting on condition variable for input data");
            pthread_cond_wait(&buf->full_cond, &buf->lock);
         }
        */

    } else {

        // This loop exists when is_full == 1 (i.e. a full buffer) AND
        // when this producer hasn't already marked this buffer as
        while ((buf->is_full[ID] == 0 || cons.is_done[ID])
               && buf->shutdown_signal == 0) {
            pthread_cond_wait(&buf->full_cond, &buf->lock);
        }

    }

    CHECK_ERROR_F(pthread_mutex_unlock(&buf->lock));

    if (buf->shutdown_signal == 1)
        return NULL;

    cons.last_frame_acquired = ID;
    return buf->frames[ID];
}

int wait_for_full_frame_timeout(Buffer* buf, const std::string& name, const int ID,
                                const struct timespec timeout) {
    CHECK_ERROR_F(pthread_mutex_lock(&buf->lock));

    auto& cons = buf->consumers.at(name);
    int err = 0;

    // This loop exists when is_full == 1 (i.e. a full buffer) AND
    // when this producer hasn't already marked this buffer as
    while ((buf->is_full[ID] == 0 || cons.is_done[ID])
           && buf->shutdown_signal == 0 && err == 0) {
        err = pthread_cond_timedwait(&buf->full_cond, &buf->lock, &timeout);
    }

    CHECK_ERROR_F(pthread_mutex_unlock(&buf->lock));

    if (buf->shutdown_signal == 1)
        return -1;

    if (err == ETIMEDOUT)
        return 1;

    cons.last_frame_acquired = ID;
    return 0;
}

int get_num_full_frames(Buffer* buf) {
    int numFull = 0;

    CHECK_ERROR_F(pthread_mutex_lock(&buf->lock));

    for (int i = 0; i < buf->num_frames; ++i) {
        if (buf->is_full[i] == 1) {
            numFull++;
        }
    }

    CHECK_ERROR_F(pthread_mutex_unlock(&buf->lock));

    return numFull;
}

int GenericBuffer::get_num_consumers() {
    CHECK_ERROR_F(pthread_mutex_lock(&lock));
    int n = consumers.size();
    CHECK_ERROR_F(pthread_mutex_unlock(&lock));
    return n;
}

int GenericBuffer::get_num_producers() {
    CHECK_ERROR_F(pthread_mutex_lock(&lock));
    int n = producers.size();
    CHECK_ERROR_F(pthread_mutex_unlock(&lock));
    return n;
}

void print_buffer_status(Buffer* buf) {
    int is_full[buf->num_frames];

    CHECK_ERROR_F(pthread_mutex_lock(&buf->lock));

    memcpy(is_full, buf->is_full, buf->num_frames * sizeof(int));

    CHECK_ERROR_F(pthread_mutex_unlock(&buf->lock));

    char status_string[buf->num_frames + 1];

    for (int i = 0; i < buf->num_frames; ++i) {
        if (is_full[i] == 1) {
            status_string[i] = 'X';
        } else {
            status_string[i] = '_';
        }
    }
    status_string[buf->num_frames] = '\0';

    INFO_F("Buffer %s, status: %s", buf->buffer_name, status_string);
}

void Buffer::print_full_status() {

    CHECK_ERROR_F(pthread_mutex_lock(&lock));

    char status_string[num_frames + 1];
    status_string[num_frames] = '\0';

    INFO_F("--------------------- %s ---------------------", buffer_name);

    for (int i = 0; i < num_frames; ++i) {
        if (is_full[i] == 1) {
            status_string[i] = 'X';
        } else {
            status_string[i] = '_';
        }
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
            INFO_F("%-30s : %s (%d, %d)", x.name, status_string,
                   x.last_frame_acquired,
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
            INFO_F("%-30s : %s (%d, %d)", x.name, status_string,
                   x.last_frame_acquired,
                   x.last_frame_released);
        }

    CHECK_ERROR_F(pthread_mutex_unlock(&lock));
}


void pass_metadata(Buffer* from_buf, int from_ID, Buffer* to_buf, int to_ID) {

    if (from_buf->metadata[from_ID] == NULL) {
        WARN_F("No metadata in source buffer %s[%d], was this intended?", from_buf->buffer_name,
               from_ID);
        return;
    }

    metadataContainer* metadata_container = from_buf->metadata[from_ID];

    CHECK_ERROR_F(pthread_mutex_lock(&to_buf->lock));

    // In case we've already moved the metadata we don't want to increment the ref count.
    if (to_buf->metadata[to_ID] == NULL) {
        to_buf->metadata[to_ID] = metadata_container;
        increment_metadata_ref_count(metadata_container);
    }

    // If this is true then the to_buf already has a metadata container for this ID and its
    // different!
    assert(to_buf->metadata[to_ID] == metadata_container);
    CHECK_ERROR_F(pthread_mutex_unlock(&to_buf->lock));
}

void copy_metadata(Buffer* from_buf, int from_ID, Buffer* to_buf, int to_ID) {
    metadataContainer* from_metadata_container;
    metadataContainer* to_metadata_container;
    CHECK_ERROR_F(pthread_mutex_lock(&from_buf->lock));
    CHECK_ERROR_F(pthread_mutex_lock(&to_buf->lock));

    if (from_buf->metadata[from_ID] == NULL) {
        WARN_F("No metadata in source buffer %s[%d], was this intended?", from_buf->buffer_name,
               from_ID);
        // Cannot wait to update this to C++14 locks...
        goto unlock_exit;
    }

    if (to_buf->metadata[to_ID] == NULL) {
        WARN_F("No metadata in dest buffer %s[%d], was this intended?", from_buf->buffer_name,
               from_ID);
        goto unlock_exit;
    }

    from_metadata_container = from_buf->metadata[from_ID];
    to_metadata_container = to_buf->metadata[to_ID];

    if (from_metadata_container->metadata_size != to_metadata_container->metadata_size) {
        WARN_F("Metadata sizes don't match, cannot copy metadata!!");
        goto unlock_exit;
    }

    memcpy(to_metadata_container->metadata, from_metadata_container->metadata,
           from_metadata_container->metadata_size);

unlock_exit:
    CHECK_ERROR_F(pthread_mutex_unlock(&to_buf->lock));
    CHECK_ERROR_F(pthread_mutex_unlock(&from_buf->lock));
}

void allocate_new_metadata_object(Buffer* buf, int ID) {
    assert(ID >= 0);
    assert(ID < buf->num_frames);

    CHECK_ERROR_F(pthread_mutex_lock(&buf->lock));

    if (buf->metadata_pool == NULL) {
        FATAL_ERROR_F("No metadata pool on %s but metadata was needed by a producer",
                      buf->buffer_name);
    }

    DEBUG2_F("Called allocate_new_metadata_object, buf %p, %d", buf, ID);

    if (buf->metadata[ID] == NULL) {
        buf->metadata[ID] = request_metadata_object(buf->metadata_pool);
    }

    // Make sure we got a metadata object.
    CHECK_MEM_F(buf->metadata[ID]);

    CHECK_ERROR_F(pthread_mutex_unlock(&buf->lock));
}

uint8_t* swap_external_frame(Buffer* buf, int frame_id, uint8_t* external_frame) {

    CHECK_ERROR_F(pthread_mutex_lock(&buf->lock));

    // Check that we don't have more than one producer.
    assert(buf->producers.size() == 1);

    uint8_t* temp_frame = buf->frames[frame_id];
    buf->frames[frame_id] = external_frame;

    CHECK_ERROR_F(pthread_mutex_unlock(&buf->lock));

    return temp_frame;
}

void swap_frames(Buffer* from_buf, int from_frame_id, Buffer* to_buf, int to_frame_id) {

    assert(from_buf != to_buf);
    assert(from_buf != NULL);
    assert(to_buf != NULL);
    assert(from_frame_id >= 0);
    assert(from_frame_id < from_buf->num_frames);
    assert(to_frame_id >= 0);
    assert(to_frame_id < to_buf->num_frames);
    assert(from_buf->aligned_frame_size == to_buf->aligned_frame_size);

    int num_consumers = from_buf->get_num_consumers();
    assert(num_consumers == 1);
    (void)num_consumers;
    int num_producers = to_buf->get_num_producers();
    assert(num_producers == 1);
    (void)num_producers;

    // Swap the frames
    uint8_t* temp_frame = from_buf->frames[from_frame_id];
    from_buf->frames[from_frame_id] = to_buf->frames[to_frame_id];
    to_buf->frames[to_frame_id] = temp_frame;
}

void safe_swap_frame(Buffer* src_buf, int src_frame_id, Buffer* dest_buf, int dest_frame_id) {
    assert(src_buf != dest_buf);
    assert(src_buf != NULL);
    assert(dest_buf != NULL);
    assert(src_frame_id >= 0);
    assert(src_frame_id < src_buf->num_frames);
    assert(dest_frame_id >= 0);
    assert(dest_frame_id < dest_buf->num_frames);

    // Buffer sizes must match exactly
    if (src_buf->frame_size != dest_buf->frame_size) {
        FATAL_ERROR_F("Buffer sizes must match for direct copy (%s.frame_size != %s.frame_size)",
                      src_buf->buffer_name, dest_buf->buffer_name);
    }

    if (dest_buf->get_num_producers() > 1) {
        FATAL_ERROR_F("Cannot swap/copy frames into dest buffer %s with more than one producer",
                      dest_buf->buffer_name);
    }

    int num_consumers = src_buf->get_num_consumers();

    // Copy or transfer the data part.
    if (num_consumers == 1) {
        // Swap the frames
        uint8_t* temp_frame = src_buf->frames[src_frame_id];
        src_buf->frames[src_frame_id] = dest_buf->frames[dest_frame_id];
        dest_buf->frames[dest_frame_id] = temp_frame;
    } else if (num_consumers > 1) {
        // Copy the frame data over, leaving the source intact
        memcpy(dest_buf->frames[dest_frame_id], src_buf->frames[src_frame_id], src_buf->frame_size);
    }
}

uint8_t* buffer_malloc(size_t len, int numa_node, bool use_hugepages, bool mlock_frames,
                       bool zero_new_frames) {

    uint8_t* frame = NULL;

#ifdef WITH_HSA // Support for legacy HSA support used in CHIME
    (void)use_hugepages;
    frame = (uint8_t*)hsa_host_malloc(len, numa_node);
    if (frame == NULL) {
        return NULL;
    }
#else
    if (use_hugepages) {
#ifndef MAC_OSX
        void* mapped_frame = mmap(nullptr, len, PROT_READ | PROT_WRITE,
                                  MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB, -1, 0);
        if (mapped_frame == MAP_FAILED) {
            ERROR_F("Error mapping huge pages, check available huge pages: %s (%d)",
                    strerror(errno), errno);
            return NULL;
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
            return NULL;
        }
        numa_bitmask_free(node_mask);
#endif
        frame = (uint8_t*)mapped_frame;
#else
        ERROR_F("Huge pages not supported in Mac OSX.");
        return NULL;
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
            return NULL;
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
            return NULL;
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

// Do not call if there is no metadata
void* get_metadata(Buffer* buf, int ID) {
    assert(ID >= 0);
    assert(ID < buf->num_frames);
    assert(buf->metadata[ID] != NULL);

    return buf->metadata[ID]->metadata;
}

// Might return NULLL
metadataContainer* get_metadata_container(Buffer* buf, int ID) {
    assert(ID >= 0);
    assert(ID < buf->num_frames);

    return buf->metadata[ID];
}

double get_last_arrival_time(Buffer* buf) {
    return buf->last_arrival_time;
}

void GenericBuffer::send_shutdown_signal() {
    CHECK_ERROR_F(pthread_mutex_lock(&lock));
    shutdown_signal = 1;
    CHECK_ERROR_F(pthread_mutex_unlock(&lock));
    CHECK_ERROR_F(pthread_cond_broadcast(&empty_cond));
    CHECK_ERROR_F(pthread_cond_broadcast(&full_cond));
}
