/*****************************************
@file
@brief Remove the cross-correlations from a buffer.
- selectAutos : public KotekanProcess
*****************************************/
#ifndef SELECT_AUTOS_HPP
#define SELECT_AUTOS_HPP

#include <unistd.h>
#include "buffer.h"
#include "KotekanProcess.hpp"

/**
 * @class selectAutos
 * @brief ``KotekanProcess`` that consumes a full set of visibilities from a ``visBuffer``
 *        and passes on only auto-correlations to an output ``visBuffer``.
 *
 * @par Buffers
 * @buffer in_buf The kotekan buffer from which the visibilities are read, can be any size.
 *     @buffer_format visBuffer structured
+*     @buffer_metadata visMetadata
 * @buffer out_buf The kotekan buffer which will be fed the auto-correlations-only visibilities.
 *     @buffer_format visBuffer structured
+*     @buffer_metadata visMetadata
 *
 * @conf  out_buf           string. Name of buffer to output auto-correlations to.
 * @conf  in_buf            string. Name of buffer to read from.
 *
 * @author Mateus Fandino
 */

class selectAutos : public KotekanProcess {

public:

    /// Default constructor
    selectAutos(Config &config,
               const string& unique_name,
               bufferContainer &buffer_container);

    void apply_config(uint64_t fpga_seq);

    /// Main loop for the process
    void main_thread();

private:

    /// Output buffer with subset of frequencies
    Buffer * out_buf;
    /// Input buffer with all frequencies
    Buffer * in_buf;
};



#endif

