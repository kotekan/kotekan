#ifndef KOTEKAN_STAGE_H
#define KOTEKAN_STAGE_H

#include "Config.hpp"          // for Config
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for kotekanLogging

#include <atomic>      // for atomic_bool
#include <functional>  // for function
#include <mutex>       // for mutex
#include <stdint.h>    // for uint32_t
#include <string>      // for string
#include <sys/types.h> // for pid_t
#include <thread>      // for thread
#include <utility>     // for std::pair
#include <vector>      // for vector

#ifdef MAC_OSX
#include "osxBindCPU.hpp"

#include <immintrin.h>
#endif

namespace kotekan {

class Stage : public kotekanLogging {
public:
    Stage(Config& config, const std::string& unique_name, bufferContainer& buffer_container,
          std::function<void(const Stage&)> main_thread_ref);
    virtual ~Stage();
    virtual void start();
    virtual void main_thread();
    /**
     * @brief Called when the stage is about to exit its main thread.
     *
     * Default: no-op. Override in derived stages to unregister from buffers,
     * release external resources, etc. The base class also calls any tracked
     * unregisters recorded via mark_* helpers (see below).
     */
    virtual void unregister();

    std::string get_unique_name() const;

    /**
     * @brief Attempts to join the stage's @c main_thread with a timeout
     *
     * Should only be called after a call to @c stop()
     *
     * If the thread doesn't exit within the timeout given by @c join_timeout
     * then this function will output an error and perform a hard exit of kotekan.
     */
    void join();
    void stop();

    /**
     * @brief Generates a graphviz "dot" string for this stage.
     *
     * By default this is just the stage name plus some default formatting.
     *
     * @return "dot" style graph description for this stage.
     */
    virtual std::string dot_string(const std::string& prefix) const;

    /**
     * @brief Add newly created stage tid to thread_list for cpu usage tracking.
     */
    void register_tid(pid_t ptr);

    /**
     * @brief Remove a tid from thread_list.
     */
    void unregister_tid(pid_t ptr);

    /**
     * @brief Get tids from the current stage.
     */
    const std::vector<pid_t>& get_tids();

protected:
    std::atomic_bool stop_thread;
    Config& config;

    std::string unique_name;

    std::thread this_thread;

    // --- Optional helpers for derived classes --------------------------------
    //
    // If your stage calls into Buffer APIs like register_consumer/register_producer,
    // call these helpers *after* a successful register to let the base class track
    // what to undo. During Stage::unregister() we’ll invoke the corresponding
    // unregister callbacks you record here (in reverse order).
    //
    // Use whichever overload matches how you prefer to store callsites:
    //  - direct function pointers/lambdas, or
    //  - just mark a Buffer* and we’ll call provided functors later.
    //
    using UnregFn = std::function<void()>;

    // Record an explicit unregistration action (e.g., a lambda that calls
    // buf->unregister_consumer(unique_name) or similar).
    void track_consumer_unreg(UnregFn fn);
    void track_producer_unreg(UnregFn fn);

    // Convenience: track a Buffer* plus an unregistration functor you supply.
    void track_consumer_unreg_for(Buffer* buf, std::function<void(Buffer*)> fn);
    void track_producer_unreg_for(Buffer* buf, std::function<void(Buffer*)> fn);

    // Set the cores the main thread is allowed to run on to the
    // cores given in cpu_affinity_
    // Also applies the list to the main thread if it is running.
    void set_cpu_affinity(const std::vector<int>& cpu_affinity_);

    // Applies the cpu_list to the thread affinity of the main thread.
    void apply_cpu_affinity();

    /**
     * @brief Get a buffer pointer by config tag.
     *
     * @param name The config tag with the buffer name
     * @return A pointer to the buffer
     */
    Buffer* get_buffer(const std::string& name);

    /**
     * @brief Gets an array of buffer pointers linked to the @c name in the config.
     *
     * @param name The name of the array in the config.
     * @return A vector of pointers to the buffers requested
     */
    std::vector<Buffer*> get_buffer_array(const std::string& name);

    /**
     * @brief Convenience: fetch and register as consumer on buffer referenced by config tag.
     *
     * Calls `get_buffer(tag)`, then `buf->register_consumer(unique_name)`, and records an
     * unregister action so the base class will unregister on thread exit.
     */
    Buffer* get_buffer_as_consumer(const std::string& tag);

    /**
     * @brief Convenience: fetch and register as producer on buffer referenced by config tag.
     *
     * Calls `get_buffer(tag)`, then `buf->register_producer(unique_name)`, and records an
     * unregister action so the base class will unregister on thread exit.
     */
    Buffer* get_buffer_as_producer(const std::string& tag);

    /**
     * @brief Convenience: array variant for consumers.
     */
    std::vector<Buffer*> get_buffer_array_as_consumer(const std::string& tag);

    /**
     * @brief Convenience: array variant for producers.
     */
    std::vector<Buffer*> get_buffer_array_as_producer(const std::string& tag);

    bufferContainer& buffer_container;

private:
    std::function<void(const Stage&)> main_thread_fn;

    // List of CPU cores that the main thread is allowed to run on.
    // CPU core numbers are zero based.
    std::vector<int> cpu_affinity;

    // Lock for changing or using the cpu_affinity variable.
    std::mutex cpu_affinity_lock;

    /// The number of seconds to wait for a kotekan stage thread to be
    /// joined after the exit signal has been given before exiting ungracefully.
    uint32_t join_timeout;

    // List of stage tids used for CPU usage tracking
    std::vector<pid_t> thread_list;

    // Tracked unregistration actions (executed on thread exit).
    std::vector<UnregFn> consumer_unregs_;
    std::vector<UnregFn> producer_unregs_;
};

} // namespace kotekan

/// Helper defined to reduce the boiler plate needed to crate the
/// standarized constructor in sub classes
#define STAGE_CONSTRUCTOR(T)                                                                       \
    T::T(kotekan::Config& config, const std::string& unique_name,                                  \
         kotekan::bufferContainer& buffer_container) :                                             \
        Stage(config, unique_name, buffer_container, std::bind(&T::main_thread, this))

#endif /* KOTEKAN_STAGE_H */
