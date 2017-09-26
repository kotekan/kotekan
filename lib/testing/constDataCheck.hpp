#ifndef CONST_DATA_CHECK_H
#define CONST_DATA_CHECK_H

#include "buffer.h"
#include "KotekanProcess.hpp"
#include "errors.h"
#include "util.h"
#include <unistd.h>

/*
 * Checks that the contents of "buf" match the complex number given by "real" and "imag"
 * Configuration options
 * "buf": String with the name of the buffer to check
 * "real": Expected real value (int)
 * "imag": Expected imaginary value (int)
 */

class constDataCheck : public KotekanProcess {
public:
    constDataCheck(Config &config,
                  const string& unique_name,
                  bufferContainer &buffer_container);
    ~constDataCheck();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread();
private:
    struct Buffer *buf;
    int32_t ref_real;
    int32_t ref_imag;
};

#endif