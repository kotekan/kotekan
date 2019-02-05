#ifndef HEX_DUMP_H
#define HEX_DUMP_H

#include "Stage.hpp"
#include "buffer.h"
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

class hexDump : public kotekan::Stage {
public:
    hexDump(kotekan::Config& config, const string& unique_name,
            kotekan::bufferContainer& buffer_container);
    ~hexDump();
    void main_thread() override;

private:
    struct Buffer* buf;
    int32_t len;
    int32_t offset;
};

#endif
