#ifndef CHIME_SHUFFLE_HPP
#define CHIME_SHUFFLE_HPP

#include "kotekanMode.hpp"

class chimeShuffleMode : public kotekanMode {

public:
    chimeShuffleMode(Config &config);
    virtual ~chimeShuffleMode();

    void initalize_processes();
};

#endif /* CHIME_HPP */
