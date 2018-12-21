/**
 * @file
 * @brief A process to read VDIF files from multiple drives.
 *  - pyPlotN2 : public KotekanProcess
 */

#ifndef PY_PLOT_N2_H
#define PY_PLOT_N2_H

#include "KotekanProcess.hpp"
#include "accumulate.hpp"
#include "buffer.h"
#include "chimeMetadata.h"
#include "errors.h"
#include "fpga_header_functions.h"
#include "restServer.hpp"

#include <cstdio>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

/**
 * @class pyPlotN2
 * @brief Consumer ``KotekanProcess`` to produce PDF plots of correlation matrices.
 *
 * This process does nothing until it receives a REST request from an outside user.
 * Upon receipt, it spawns a companion python script (``pyPlotN2.py``),
 * and pipes a short configuration header to it, followed by the contents of the next available
 * buffer. The python script generates a pdf plot of the visibilitiy matrix, saving it to a
 * timestamped file.
 *
 * @par REST Endpoints
 * @endpoint /plot_corr_matrix/``gpu_id`` Any contact here triggers a plot dump.
 *
 * @par Buffers
 * @buffer in_buf Buffer containing the data to be plotted. Should be a blocked upper triangle
 * correlation matrix
 *  @buffer_format Array of complex uint32_t values.
 *  @buffer_metadata none
 *
 * @conf gpu_id         Int, used to generate the REST endpoint,
 *                      needed in case of multiple streams in a single kotekan process.
 *
 * @todo    Make the location of the python plotting script more robust / permanent.
 * @todo    Move config parsing to the constructor.
 * @todo    Spin the triggered dump/executre out into a new thread.
 *
 * @author Keith Vanderlinde
 */
class pyPlotN2 : public KotekanProcess {
public:
    /// Constructor
    pyPlotN2(Config& config, const string& unique_name, bufferContainer& buffer_container);

    /// Destructor, currently does nothing
    virtual ~pyPlotN2();

    /// Creates n safe instances of the file_read_thread thread
    void main_thread() override;

    /**
     * Function to receive and receive the request for a new plot.
     * Sets a flag informing the main loop to produce a plot of the next available buffer frame.
     * @param conn         Connection object requesting the plot. Only used to reply
     * `HTTP_RESPONSE::OK`.
     * @param json_request The contents of the REST json transmission. Ignored.
     *
     * @warning        Nobody should ever call this directly, it's only meant to service the
     *                 RESTful callback.
     */
    void request_plot_callback(connectionInstance& conn);

private:
    void make_plot(void);

    /// The kotekan buffer object the processes is producing for
    struct Buffer* buf;
    unsigned char* in_local;
    std::string endpoint;

    int gpu_id = -1;
    bool dump_plot = false;
    bool busy = false;
    stream_id_t stream_id;
};

#endif