/**
 * @file
 * @brief A stage to read VDIF files from multiple drives.
 *  - pyPlotN2 : public kotekan::Stage
 */

#ifndef PY_PLOT_N2_H
#define PY_PLOT_N2_H

#include "Config.hpp"          // for Config
#include "ICETelescope.hpp"    // for ice_stream_id_t
#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // for bufferContainer
#include "restServer.hpp"      // for connectionInstance

#include <string> // for string

/**
 * @class pyPlotN2
 * @brief Consumer ``kotekan::Stage`` to produce PDF plots of correlation matrices.
 *
 * This stage does nothing until it receives a REST request from an outside user.
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
 *                      needed in case of multiple streams in a single kotekan stage.
 *
 * @todo    Make the location of the python plotting script more robust / permanent.
 * @todo    Move config parsing to the constructor.
 * @todo    Spin the triggered dump/executre out into a new thread.
 *
 * @author Keith Vanderlinde
 */
class pyPlotN2 : public kotekan::Stage {
public:
    /// Constructor
    pyPlotN2(kotekan::Config& config, const std::string& unique_name,
             kotekan::bufferContainer& buffer_container);

    /// Destructor, currently does nothing
    virtual ~pyPlotN2();

    /// Creates n safe instances of the file_read_thread thread
    void main_thread() override;

    /**
     * Function to receive and receive the request for a new plot.
     * Sets a flag informing the main loop to produce a plot of the next available buffer frame.
     * @param conn         Connection object requesting the plot. Only used to reply
     * `HTTP_RESPONSE::OK`.
     *
     * @warning        Nobody should ever call this directly, it's only meant to service the
     *                 RESTful callback.
     */
    void request_plot_callback(kotekan::connectionInstance& conn);

private:
    void make_plot(void);

    /// The kotekan buffer object the stage is producing for
    struct Buffer* buf;
    unsigned char* in_local;
    std::string endpoint;

    int gpu_id = -1;
    bool dump_plot = false;
    bool busy = false;
    ice_stream_id_t stream_id;
};

#endif
