#ifndef KOTEKAN_TRACKERS_HPP
#define KOTEKAN_TRACKERS_HPP

#include "kotekanLogging.hpp"
#include "restServer.hpp"
#include "visUtil.hpp"

namespace kotekan {
namespace trackers {

class KotekanTrackers {

public:
    /**
     * @brief Returns an instance of the kotekan trackers.
     *
     * @return Returns the kotekan trackers instance.
     */
    static KotekanTrackers& instance();

    /**
     * @brief Registers this class with the REST server, creating the
     *        /trackers end point
     * @param rest_server The server to register with.
     */
    void register_with_server(restServer* rest_server);

    /**
     * @brief The call back function for the REST server to use.
     *
     * This function is never called directly.
     *
     * @param conn The connection instance to send results to.
     */
    void trackers_callback(connectionInstance& conn);

    /**
     * @brief Adds a new tracker
     *
     * @param name The name of the tracker.
     * @param unit The unit of the tracker.
     * @param size The size of the tracker with default of 100.
     * @param is_optimized The switch of min/max methods of the tracker.
     * @return a shared pointer to the newly created tracker
     * @throw std::runtime_error if the tracker with that name is already registered.
     */
    std::shared_ptr<StatTracker> add_tracker(std::string name, std::string unit, size_t size = 100,
                                             bool is_optimized = true);

private:
    KotekanTrackers();
    ~KotekanTrackers();

    // A map to store all trackers <tracker_name, tracker_ptr>
    std::map<std::string, std::shared_ptr<StatTracker>> trackers;

    std::mutex trackers_lock;
};

} // namespace trackers
} // namespace kotekan

#endif /* KOTEKAN_TRACKERS_HPP */
