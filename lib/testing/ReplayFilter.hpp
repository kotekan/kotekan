/*****************************************
@file
@brief Stages for replaying old data
- ReplayFilter : public Stage
*****************************************/

#ifndef REPLAY_FILTER
#define REPLAY_FILTER

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.h"            // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "dataset.hpp"         // for dset_id_t
#include "visBuffer.hpp"       // for visFrameView
#include "visUtil.hpp"         // for cfloat


/**
 * @brief Transform incoming data to look like current acquisition data.
 *
 *
 * @par Buffers
 * @buffer in_buf The buffer which will be transformed.
 *     @buffer_format visBuffer structured
 *     @buffer_metadata visMetadata
 * @buffer out_buf The replayed data.
 *     @buffer_format visBuffer structured
 *     @buffer_metadata visMetadata
 *
 * @conf  modify_times  Bool. Rewrite the timestamps to start at a given time.
 * @conf  start_time    Double. If modifying the timestamps, this is the
 *                      initial time. If not set defaults to the current time.
 * @conf  wait          Bool. Wait until the expected amount of time has elapsed
 *                      between time samples. Default is true.
 * @conf  fpga_length   Float. If modifying time stamps this is the number of
 *                      seconds per FPGA tick and is used to construct the time
 *                      of all samples after `start_time`.
 * @conf  drop_empty    Bool. Drop empty frames rather than passing them onwards.
 *                      Default is true.
 *
 * @author  Richard Shaw
 *
 */
class ReplayFilter : public kotekan::Stage {

public:
    /// Constructor. Loads config options.
    ReplayFilter(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);

    /// Primary loop to wait for buffers, stuff in data, mark full, lather, rinse and repeat.
    void main_thread() override;

private:

    Buffer* in_buf;
    Buffer* out_buf;

    const double _start_time;
    const uint64_t _fpga_length;
    const double _ctime_length;
    const bool _wait;
    const bool _modify_times;

    const bool _drop_empty;
};

#endif // REPLAY_FILTER