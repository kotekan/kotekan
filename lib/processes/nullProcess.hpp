#ifndef NULL_THREAD
#define NULL_THREAD

#include "KotekanProcess.hpp"

class nullProcess : public KotekanProcess {
public:
    nullProcess(Config& config, const string& unique_name,
                         bufferContainer &buffer_container);
    virtual ~nullProcess();
    void main_thread();
    virtual void apply_config(uint64_t fpga_seq);
private:
    struct Buffer *buf;
};

#endif