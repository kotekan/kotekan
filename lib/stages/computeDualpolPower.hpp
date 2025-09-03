#ifndef COMPUTE_DUALPOL_POWER
#define COMPUTE_DUALPOL_POWER

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "bufferContainer.hpp"

#include <string>      // for string
#include <sys/types.h> // for uint

class computeDualpolPower : public kotekan::Stage {
public:
    computeDualpolPower(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& buffer_container);
    ~computeDualpolPower();
    void main_thread() override;

private:
    inline void fastSqSumVdif(unsigned char* data, uint* temp_buf, uint* output);
    void parallelSqSumVdif(int loop_idx, int loop_length);
    Buffer* buf_in;
    Buffer* buf_out;

    int num_freq;
    int num_elem;
    int timesteps_in;
    int timesteps_out;
    int integration_length;
    unsigned int* integration_count;
    unsigned char* in_local;
    unsigned char* out_local;
};

#endif
