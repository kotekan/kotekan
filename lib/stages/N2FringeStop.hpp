/*****************************************
@file
@brief Reduce cadence of a single-frequency.
- N2FringeStop : public kotekan::Stage
*****************************************/
#ifndef N2_FRINGE_STOP_HPP
#define N2_FRINGE_STOP_HPP

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <stddef.h> // for size_t
#include <string>   // for string

/**
 * @class N2FringeStop
 * @brief Average a set number of frames on a single-frequency stream to
 *        effectively reduce the cadence of the acquisition.
 *
 * This stage accumulates and averages a specified number of incoming frames on
 * a single-frequency stream to reduce the cadence of the acquisition.
 * Visibilities, eigenvectors, eigenvalues, eigen-rms are averaged. Inverse
 * weights are averaged and divided by number of combined frames to track
 * reduction in variance. Metadata from the first frame is passed on and that
 * of the others discarded.
 * Will throw an exception if more than one frequency is found in the stream.
 *
 * @par Buffers
 * @buffer in_buf The kotekan buffer of the incoming single-frequency stream.
 *     @buffer_format VisBuffer structured
 *     @buffer_metadata VisMetadata
 * @buffer out_buf The kotekan buffer into which low cadence stream is fed.
 *     @buffer_format VisBuffer structured
 *     @buffer_metadata VisMetadata
 *
 * @conf  in_buf           String. Input buffer.
 * @conf  out_buf          String. Output buffer.
 * @conf  fringestop_mode  Int (default 1). 0=no fringestop, 1=multiply vis, 2=overwrite with
 *                         pure phase term.
 * @conf  num_rot_target   Int (default 9000). Target ERA rotation count for fringestopping.
 * @conf  era_target_deg   Double (default 0.0). Target ERA in degrees.
 * @conf  xp_target_as     Double (default 0.0). Target x polar motion (arcsec).
 * @conf  yp_target_as     Double (default 0.0). Target y polar motion (arcsec).
 *
 * @par Example
 * @code
 * N2FringeStop:
 *   in_buf: vis_in
 *   out_buf: vis_fs
 *   fringestop_mode: 1
 *   num_rot_target: 9000
 *   era_target_deg: 0.0
 *   xp_target_as: 0.0
 *   yp_target_as: 0.0
 * @endcode
 *
 * @author  Tristan Pinsonneault-Marotte
 *
 */
class N2FringeStop : public kotekan::Stage {

public:
    /// Default constructor, loads config params.
    N2FringeStop(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);

    /// Main loop for the stage
    void main_thread() override;

private:
    // Frame parameters
    size_t num_elements;
    [[maybe_unused]] size_t num_eigenvectors = 0;
    size_t nprod;

    // Whether to apply fringestopping phases.
    int fringestop_mode;

    int num_rot_target;
    double era_target_deg;
    double xp_target_as;
    double yp_target_as;

    // Buffers
    Buffer* in_buf;
    Buffer* out_buf;
};

#endif
