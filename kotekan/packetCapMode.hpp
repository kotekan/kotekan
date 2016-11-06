#ifndef PACKET_CAP_HPP
#define PACKET_CAP_HPP

#include "Config.hpp"
#include "kotekanMode.hpp"

class packetCapMode : public kotekanMode {

public:
    packetCapMode(Config &config);
    virtual ~packetCapMode();

    void initalize_processes();
};

#endif /* PACKET_CAP_HPP */
