#ifndef KOTEKAN_TRACKERS_HPP
#define KOTEKAN_TRACKERS_HPP

#include "restServer.hpp" // for connectionInstance, restServer
#include "visUtil.hpp"    // for StatTracker

#include <map>      // for map
#include <memory>   // for shared_ptr
#include <mutex>    // for mutex
#include <stddef.h> // for size_t
#include <string>   // for string

namespace kotekan {

typedef std::map<std::string, std::shared_ptr<StatTracker>> stage_trackers_t;

class KotekanTrackers {

public:
    /**
     * @brief Set and apply the static config to KotekanTrackers
     * @param config         The config.
     *
     * @returns A reference to the global KotekanTrackers instance.
     */
    static KotekanTrackers& instance(const kotekan::Config& config);

    /**
     * @brief Get the global KotekanTrackers.
     *
     * @returns A reference to the global KotekanTrackers instance.
     **/
    static KotekanTrackers& instance();

    /**
     * @brief Registers this class with the REST server, creating the
     *        /trackers end point
     * @param rest_server The server to register with.
     */
    void register_with_server(restServer* rest_server);

    /**
     * @brief The call back function for the REST server to use.
     * This returns all contents from every tracker and grouped by stage.
     *
     * This function is never called directly.
     *
     * @param conn The connection instance to send results to.
     */
    void trackers_callback(connectionInstance& conn);

    /**
     * @brief The call back function for the REST server to use.
     * This returns min/max/avg/std of each tracker and grouped by stage.
     *
     * This function is never called directly.
     *
     * @param conn The connection instance to send results to.
     */
    void trackers_current_callback(connectionInstance& conn);

    /**
     * @brief Adds a new tracker
     *
     * @param stage_name The name of the stage.
     * @param tracker_name The name of the tracker.
     * @param unit The unit of the tracker.
     * @param size The size of the tracker with default of 100.
     * @param is_optimized The switch of min/max methods of the tracker(see statTracker for
     * details).
     * @return a shared pointer to the newly created tracker
     * @throw std::runtime_error if the tracker with that name is already registered.
     */
    std::shared_ptr<StatTracker> add_tracker(std::string stage_name, std::string tracker_name,
                                             std::string unit, size_t size = 100,
                                             bool is_optimized = true);

    /**
     * @brief Remove all trackers in the given stage
     *
     * @param stage_name The given stage name.
     */
    void remove_tracker(std::string stage_name);

    /**
     * @brief Remove a single tracker.
     *
     * @param stage_name The given stage name.
     * @param tracker_name The given tracker name.
     */
    void remove_tracker(std::string stage_name, std::string tracker_name);

    /**
     * @brief Dump all trackers.
     */
    void dump_trackers();

private:
    KotekanTrackers();
    ~KotekanTrackers();

    // Generate a private static instance.
    static KotekanTrackers& private_instance();

    // A map to store all trackers <stage_name, <tracker_name, tracker_ptr>>
    std::map<std::string, stage_trackers_t> trackers;

    std::string dump_path;

    std::mutex trackers_lock;
};

} // namespace kotekan

#endif /* KOTEKAN_TRACKERS_HPP */
