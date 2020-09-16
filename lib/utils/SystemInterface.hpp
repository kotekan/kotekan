#ifndef SYSTEM_INTERFACE_HPP
#define SYSTEM_INTERFACE_HPP

#include <string> // for string, operator+


/**
 * @brief Contains wrappers for system calls.
 *
 **/

/**
 * @brief Get the user name from the system.
 *
 * @returns Username as a string.
 **/
const std::string get_user_name();

/**
 * @brief Get the host name of the system.
 *
 * @returns Host name as a string.
 **/
const std::string get_host_name();

#endif // SYSTEM_INTERFACE_HPP
