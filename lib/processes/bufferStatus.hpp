#ifndef BUFFER_STATUS_H
#define BUFFER_STATUS_H

#include "KotekanProcess.hpp"
#include "bufferContainer.hpp"

/**
 * @class bufferStatus
 *
 * @brief Exports buffer metrics and prints out buffer status
 *
 * Exports buffer size and current load (number of full buffers)
 * for all buffers in the system, and prints this information
 * to the logs and/or stdout depending on the system settings.
 *
 * Note at the moment this class requires the global log_level to be
 * set to INFO or highter for the buffer metrics to be output to the logs.
 *
 * @config time_delay    Int. Microseconds, default 1000000 (1 second)
 *                       The number of micro seconds between buffer print outs.
 *                       Will not output more frequently than 100ms.
 * @config print_status  Bool. Default true.
 *                       If true buffer stats are send to the logs/stderr
 *
 * @par Metrics
 * @metric kotekan_bufferstatus_full_frames_total
 *         The number of full frames for a given buffer
 * @metric kotekan_bufferstatus_frames_total
 *         The total number of frames in a given buffer (buffer depth)
 *
 * @author Jacob Taylor, Andre Renard
 */
class bufferStatus : public KotekanProcess {
public:
    bufferStatus(Config& config, const string& unique_name,
                         bufferContainer &buffer_container);
    virtual ~bufferStatus();
    void main_thread() override;
private:
    map<string, Buffer*> buffers;

    /// The time in microseconds between print updates,
    /// only used if print_status == true
    int time_delay;

    /// Set to true to output the status to the logs
    bool print_status;
};

#endif
