#ifndef JSON_METADATA
#define JSON_METADATA


#include "json.hpp"

#include <cassert>
#include <cstdint>
#include <ctime>
#include <string>
#include <sys/time.h>
#include <vector>

#pragma pack()

// json based metadata structures to hold "physics" metadata, ie. data that
// varies at most once per frame

// These must match chimeMetadata and chordMetadata for now. They are used
// unqualified all over the code base, so they live in the global namespace
// (they used to be macros).
constexpr int MAX_NUM_BEAMS = 20;
constexpr int MAX_NUM_RFI_THRESHOLDS = 8;

namespace jsonMetadata {

typedef nlohmann::json metadata;

// known types of metadata, use name variables instead of bare strings to have
// the compiler catch (some) typos

const std::string BEAM_COORD("BEAM_COORD");           // a struct beamCoord
const std::string RIGHT_ASCENSION("RIGHT_ASCENSION"); // an array of float of size MAX_NUM_BEAMS
const std::string DECLINATION("DECLINATION");         // an array of float of size MAX_NUM_BEAMS
const std::string SCALING("SCALING");                 // an array of uint32 of size MAX_NUM_BEAMS
const std::string FPGA_SEQ_NUM("FPGA_SEQ_NUM");       // an uint64
const std::string TIME_DOWNSAMPLING_FPGA("TIME_DOWNSAMPLING_FPGA"); // an int
// frequencies -- integer (0-8192) identifier for FPGA coarse frequencies
// This is the FPGA frequency channel index, indexed by the local coarse frequency channel.
// TODO: this should really be a freq_id_t array
const std::string COARSE_FREQ("COARSE_FREQ"); // an array of int of size CHORD_META_MAX_FREQ
const std::string DATASET_ID("DATASET_ID"); // a 128bit hash of the system state, of type dset_id_t
const std::string RFI_NUM_BAD_INPUTS("RFI_NUM_BAD_INPUTS"); // a uint32_t of bad frames count
const std::string RFI_FLAGGED_SAMPLES(
    "RFI_FLAGGED_SAMPLES"); // a int32_t of FPGA frames flagged as containing RFI
const std::string LOST_TIMESAMPLES("LOST_TIMESAMPLES"); // a int32_t of samples lost
const std::string
    STREAM_ID("STREAM_ID"); // a uint64_t stream identifier set originally by the FPGA board
const std::string STREAM_IDS("STREAM_IDS");       // an array of uint32_t stream identifiers
const std::string FRAME_COUNTER("FRAME_COUNTER"); // an int

const std::string FIRST_PACKET_RECV_TIME(
    "FIRST_PACKET_RECV_TIME"); // The system time when the first packet in the frame was captured
const std::string TV_SEC("TV_SEC");   // the tv_sec member of a timeval
const std::string TV_USEC("TV_USEC"); // the tv_usec member of a timeval

const std::string
    FREQ_UPCHAN_FACTOR("FREQ_UPCHAN_FACTOR"); // an array of int of size CHORD_META_MAX_FREQ
const std::string
    FREQ_UPCHAN_INDEX("FREQ_UPCHAN_INDEX"); // an array of int of size CHORD_META_MAX_FREQ

const std::string RFI_FRAME_EXCISION_ENABLED(
    "RFI_FRAME_EXCISION_ENABLED"); // a bool noting whether RFI second stage excision (gpu frames)
                                   // is enabled
const std::string RFI_FRAME_EXCISION_THRESHOLDS(
    "RFI_FRAME_EXCISION_THRESHOLDS"); // an array of array<float, 2> of size MAX_NUM_RFI_THRESHOLDS


/// The coordinates of the tracking beams.
///
/// An unused entry is marked by a NaN right ascension and declination, and a
/// zero scaling. Since a NaN can never be a real coordinate, this identifies the
/// beams in use on its own, and no separate beam count is needed. @c from_json
/// marks the entries it does not fill in that way.
struct beamCoord {
    float right_ascension[MAX_NUM_BEAMS];
    float declination[MAX_NUM_BEAMS];
    uint32_t scaling[MAX_NUM_BEAMS];
};

// Defined in jsonMetadata.cpp. Found via ADL because beamCoord lives in this
// namespace.
void from_json(const nlohmann::json& j, beamCoord& c);
void to_json(nlohmann::json& j, const beamCoord& c);

} // namespace jsonMetadata

// Defined in jsonMetadata.cpp. Found via ADL because timeval lives in the
// global namespace.
void to_json(nlohmann::json& j, const timeval& tv);
void from_json(const nlohmann::json& j, timeval& tv);

#endif
