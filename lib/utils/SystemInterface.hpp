#ifndef SYSTEM_INTERFACE_HPP
#define SYSTEM_INTERFACE_HPP

#include <string> // for string, operator+


/**
 * @brief Contains wrappers for system calls.
 *
 **/

/**
 * @brief Get the username from the system.
 *
 * @returns Username as a string.
 **/
const std::string get_username();

/**
 * @brief Set a pretend hostname for the system, used for testing.
 *
 * @param hostname Host name as a string.
 **/
void set_fake_hostname(const std::string hostname);

/**
 * @brief Get the hostname of the system.
 *
 * @returns Host name as a string.
 **/
const std::string get_hostname();

#endif // SYSTEM_INTERFACE_HPP
