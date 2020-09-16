#include "SystemInterface.hpp"

#include <string>   // for string, operator+
#include <unistd.h> // for access, F_OK

const std::string get_user_name() {

    std::string user(256, '\0');
    user = (getlogin_r(&user[0], 256) == 0) ? user.c_str() : "unknown";

    return user;
}

const std::string get_host_name() {

    std::string hostname(256, '\0');
    gethostname(&hostname[0], 256);

    return hostname.c_str();
}
