#ifndef AIRSPY_INPUT_HPP
#define AIRSPY_INPUT_HPP

#include "KotekanProcess.hpp"
#include "airspy_control.h"

#include <string>
using std::string;

//int my_callback(airspy_transfer_t* transfer);

class airspyInput : public KotekanProcess {
public:
    airspyInput(Config& config, const string& unique_name,
                         bufferContainer &buffer_container);
    virtual ~airspyInput();
    void main_thread();
    virtual void apply_config(uint64_t fpga_seq);
    int buf_id;
    int airspy_producer(void *in, int bt, int ID);
private:
    struct command_queue cmd;
    struct Buffer *buf;
    struct airspy_device* a_device;
    
};


#endif 
