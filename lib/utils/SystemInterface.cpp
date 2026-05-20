#include "SystemInterface.hpp"

#include <limits.h> // for HOST_NAME_MAX
#include <string>   // for string, operator+
#include <unistd.h> // for access, F_OK

// used for testing
static std::string fake_hostname;

const std::string get_username() {

    std::string user(256, '\0');
    user = (getlogin_r(&user[0], 256) == 0) ? user.c_str() : "unknown";

    return user;
}

void set_fake_hostname(const std::string hostname) {
    fake_hostname = hostname;
}

const std::string get_hostname() {

    std::string hostname(HOST_NAME_MAX, '\0');
    if (fake_hostname.empty()) {
        gethostname(&hostname[0], hostname.size());
        hostname.back() = '\0'; // ensure NUL termination even if gethostname failed
        hostname.resize(hostname.find('\0'));
    } else {
        hostname = fake_hostname;
    }

    return hostname;
}
