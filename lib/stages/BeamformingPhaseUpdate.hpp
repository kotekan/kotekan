#ifndef BEAMFORMING_PHASE_UPDATE_HPP
#define BEAMFORMING_PHASE_UPDATE_HPP

#include "Config.hpp" // for Config
#include "Stage.hpp"  // for Stage
#include "Telescope.hpp"
#include "bufferContainer.hpp" // for bufferContainer
#include "chimeMetadata.hpp"

#include <stdint.h> // for int32_t

/**
 * @brief Base class stage which generates phases for a tracking beamformer
 *
 * @note This stage is not used directly, but instead via one of the subclasses
 *
 * @par Buffers
 * @buffer in_buf The incoming voltage data, along with metadata
 *        @buffer_format baseband voltage data
 *        @buffer_metadata CHIME Metadata
 *
 * @buffer out_buf Phase buffer sent to the GPU in the format the kernel requires
 *        @buffer_format Format determined by subclass/kernel requirements
 *        @buffer_metadata none
 *
 * @buffer gain_buf Complex gains to multiply into the phases
 *                   Only used for kernels which apply gains and phases in one step.
 *        @buffer_format Format determined by subclass/kernel requirements
 *        @buffer_metadata none
 *
 * @conf   num_elements             Int. Number of elements (2 * num_feeds)
 * @conf   num_beams                Int. Number of beams
 * @conf   inst_lat                 Double. The Instrument latitude
 * @conf   inst_long                Double. The Instrument longitude
 * @conf   num_local_freq           Int. Number of local frequencies
 * @conf   beam_pointing/i/ra       Float. initial RA (in deg) to form beams on for beam_id=i
 * @conf   beam_pointing/i/dec      Float. initial Dec (in deg) to form beams on for beam_id=i
 * @conf   beam_pointing/i/scaling  Int. scaling for beam_id=i, not used by all subclasses
 *
 * @author Andre Renard
 */
class BeamformingPhaseUpdate : public kotekan::Stage {
public:
    /// Constructor
    BeamformingPhaseUpdate(kotekan::Config& config, const std::string& unique_name,
                           kotekan::bufferContainer& buffer_container);
    ~BeamformingPhaseUpdate();

    /// Main thread/loop
    void main_thread() override;

    /// Endpoint for providing new UT1-UTC offset
    bool update_UT1_UTC_offset(nlohmann::json& json);

    /// Endpoint for providing new tracking target (RA, Dec, beam_id)
    bool tracking_update_callback(nlohmann::json& json, const uint8_t beamID);

protected:
    /// Save the beam scaling values to one of the scaling arrays
    void copy_scaling(const beamCoord& beam_coord, float* scaling);

    /// The baseband buffer which will provide the time and frequency data
    struct Buffer* in_buf = nullptr;
    /// The phase buffer used by the GPU kernel to generate the beams
    struct Buffer* out_buf = nullptr;
    /// (Optional) Buffer with gains for the cases we combine phase + gains
    struct Buffer* gains_buf = nullptr;

    /// Number of elements
    uint32_t _num_elements;
    /// Number of beams
    int16_t _num_beams;

    // Coordinates of the phase center.
    /// The instrument latitude
    double _inst_lat;
    /// The instrument longitude
    double _inst_long;

    /// Number of frequencies in a frame
    uint32_t _num_local_freq;

    /// List of frequencies in the current frame
    std::vector<freq_id_t> frequencies_in_frame;

    /// The feed locations [feed][x,y] in meters from phase center
    std::vector<std::pair<double, double>> feed_locations;

    /// Beam pointing coordinates
    struct beamCoord _beam_coord;

    /// mutex lock prevent beam_coord to be read while it is being updated.
    std::mutex beam_lock;

    double _DUT1;

    /**
     * @brief Pure virtual function for computing the phases of the beamformer.
     *
     * This fucntion implements the beamformer phase generation code.
     * The fastest changing parameters are passed to this function, but all the
     * static configuration values can be accessed via the protected members of
     * the base class (e.g. feed positions)
     *
     * @param out_frame Location to put the gains
     *                  (do not index past out_buf->frame_size bytes)
     * @param gps_time The GPS timestamp of the first sample in the incoming data
     *                  frame which will be assoicated with these gains
     * @param frequencies_in_frame An array of all the frequency ids needed for the
     *                             given out_frame
     * @param gains_frame (Optional) The gains frame, used if we have combined gains
     *                               phases.   For arrays with seperated gains/phases
     *                               this will be set to nullptr
     */
    virtual void compute_phases(uint8_t* out_frame, const timespec& gps_time,
                                const std::vector<freq_id_t>& frequencies_in_frame,
                                uint8_t* gains_frame) = 0;
};


#endif // BEAMFORMING_PHASE_UPDATE_HPP
