#ifndef KOTEKAN_STAGE_H
#define KOTEKAN_STAGE_H

#include "Config.hpp"          // for Config
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for kotekanLogging

#include <atomic>     // for atomic_bool
#include <functional> // for function
#include <mutex>      // for mutex
#include <stdint.h>   // for uint32_t
#include <string>     // for string
#include <thread>     // for thread
#include <vector>     // for vector

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

    std::string get_unique_name() const;

    /**
     * @brief Attempts to join the stage's @c main_thread with a tineout
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
    void register_tid(pthread_t ptr);

    /**
     * @brief Remove the current stage tid from thread_list.
     */
    void unregister_tid();

    /**
     * @brief Get the list of all registered tid.
     * 
     * @return the copy of the thread list.
     */
    static std::map<std::string, pid_t> get_thread_list();

protected:
    std::atomic_bool stop_thread;
    Config& config;

    std::string unique_name;

    std::thread this_thread;

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
    struct Buffer* get_buffer(const std::string& name);

    /**
     * @brief Gets an array of buffer pointers linked to the @c name in the config.
     *
     * @param name The name of the array in the config.
     * @return A vector of pointers to the buffers requested
     */
    std::vector<struct Buffer*> get_buffer_array(const std::string& name);

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

    static std::map<std::string, pid_t> thread_list;
};

} // namespace kotekan

/// Helper defined to reduce the boiler plate needed to crate the
/// standarized constructor in sub classes
#define STAGE_CONSTRUCTOR(T)                                                                       \
    T::T(kotekan::Config& config, const std::string& unique_name,                                  \
         kotekan::bufferContainer& buffer_container) :                                             \
        Stage(config, unique_name, buffer_container, std::bind(&T::main_thread, this))

#endif /* KOTEKAN_STAGE_H */
