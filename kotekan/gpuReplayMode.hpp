#ifndef GPU_REPLAY_MODE_HPP
#define GPU_REPLAY_MODE_HPP

#include "kotekanMode.hpp"
#include "bufferContainer.hpp"

class gpuReplayMode : public kotekanMode {

public:
    gpuReplayMode(Config &config);
    virtual ~gpuReplayMode();

    void initalize_processes();

};

#endif /* SINGLE_DISH_MODE_HPP */
