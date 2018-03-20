#ifndef BASEBAND_MANAGER_HPP
#define BASEBAND_MANAGER_HPP

#include "json.hpp"
#include "restServer.hpp"

class BasebandManager {
public:
    /**
     * @brief Returns the singleton instance of the BasebandManager object.
     * @return A pointer to the BasebandManager object
     */
    static BasebandManager& instance();

    /**
     * @brief Registers this class with the REST server, creating the
     *        /baseband end point
     * @param rest_server The server to register with.
     */
    void register_with_server(restServer * rest_server);

    /**
     * @brief The call back function for the REST server to use.
     *
     * This function is never called directly.
     *
     * @param conn The connection instance to send results too.
     */
    void status_callback(connectionInstance& conn);

private:
    /// Constructor, not used directly
    BasebandManager();

};

#endif /* BASEBAND_MANAGER_HPP */
