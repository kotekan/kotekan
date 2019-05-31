/**
 * @file
 * @brief Contains airspy producer for kotekan.
 *  - SynthesisDeWalsh : public KotekanProcess
 */

#ifndef SYNTHESIS_DEWALSH_HPP
#define SYNTHESIS_DEWALSH_HPP

#include "Stage.hpp"
#include "buffer.h"
#include "errors.h"
#include "util.h"
#include <termios.h>
#include <fcntl.h>
#include <string>
#include <string>
#include <stdio.h>
#include <unistd.h>
#include <signal.h>

using std::string;

/**
 * @class SynthesisDeWalsh
 * @brief Producer ``KotekanProcess`` which streams simple-formatted data
 * from a serial port ``Buffer``.
 *
 * This is a simple producer which initializes a UART.
 *
 * @par Buffers
 * @buffer out_buf The kotekan buffer which will be fed, can be any size.
 *     @buffer_format Array of @c shorts
 *     @buffer_metadata none
 *
 * @conf   device       String, the tty or usb device to read from.
 *
 * @author Keith Vanderlinde
 *
 */
class SynthesisDeWalsh : public kotekan::Stage {
public:
    /// Constructor, also initializes internal variables from config.
    SynthesisDeWalsh(kotekan::Config& config, const string& unique_name,
                     kotekan::bufferContainer &buffer_container);

    /// Destructor, cleans up local allocs.
    virtual ~SynthesisDeWalsh();

    /// Primary loop to wait for buffers, stuff in data, mark full, lather, rinse and repeat.
    void main_thread() override;

private:
    int read_data(void *dest, int src_fd, int length);
    int set_interface_attribs (int fd, int speed, int parity);

    /// kotekan buffer object which will be fed
    struct Buffer *buf;

    string dev_name;
};


#endif
