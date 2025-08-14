#ifndef JSON_METADATA
#define JSON_METADATA


#include "json.hpp"

#include <cstdint>
#include <string>
#include <ctime>
#include <vector>

#include <sys/time.h>

#pragma pack()

// json based metadata structures to hold "physics" metadata, ie. data that
// various at most once per frame

namespace jsonMetadata {
// these must match chimeMetadata and choordMetadata for now
// in namespace for C++ const
#define MAX_NUM_BEAMS 20
const int CHORD_META_MAX_FREQ = 1024;

typedef nlohmann::json metadata;

// known types of metadata, use name variables instead of bare strings to have
// the compiler catch (some) typos

const std::string BEAM_COORD("BEAM_COORD");            // a struct beamCoord
const std::string RIGHT_ASCENSION("RIGHT_ASCENSION");  // an array of float of size MAX_NUM_BEAMS
const std::string DECLINATION("DECLINATION");          // an array of float of size MAX_NUM_BEAMS
const std::string SCALING("SCALING");                  // an array of uint32 of size MAX_NUM_BEAMS
const std::string FPGA_SEQ_NUM("FPGA_SEQ_NUM");        // an uint64
const std::string NFREQ("NFREQ");                      // an int
const std::string COARSE_FREQ("COARSE_FREQ");          // an array of int of size CHORD_META_MAX_FREQ
const std::string DATASET_ID("DATASET_ID");            // a 128bit hash of the system state, of type dset_id_t
const std::string RFI_NUM_BAD_INPUTS("RFI_NUM_BAD_INPUTS"); // a uint32_t of bad frames count
const std::string RFI_FLAGGED_SAMPLES("RFI_FLAGGED_SAMPLES"); // a int32_t of FPGA frames flagged as containing RFI
const std::string LOST_TIMESAMPLES("LOST_TIMESAMPLES"); // a int32_t of samples lost
const std::string STREAM_ID("STREAM_ID");               // a uint64_t stream identifier set originally by the FPGA board

const std::string FIRST_PACKET_RECV_TIME("FIRST_PACKET_RECV_TIME"); // The system time when the first packet in the frame was captured
const std::string TV_SEC("TV_SEC");                     // the tv_sec memmber of a timeval
const std::string TV_USEC("TV_USEC");                   // the tv_usec memmber of a timeval


struct beamCoord {
    float right_ascension[MAX_NUM_BEAMS];
    float declination[MAX_NUM_BEAMS];
    uint32_t scaling[MAX_NUM_BEAMS];

    beamCoord() {
#ifdef DEBUG
        for(auto& ra : right_ascension)
            ra = std::nanf("");
        for(auto& dec : declination)
            dec = std::nanf("");
        for(auto& scale : scaling)
            scale = 0;
#endif
    }

    // TODO: turn into a from_json function?
    beamCoord(const metadata& md) {
        {
        const auto& ra(md[RIGHT_ASCENSION]);
        if(ra.size() > MAX_NUM_BEAMS)
                throw std::runtime_error("Number of beams request exceeds MAX_NUM_BEAMS");
        size_t i = 0;
        for(auto it = ra.cbegin() ; it != ra.cend() ; ++it) {
            it->get_to(right_ascension[i++]);
        }
#ifdef DEBUG
        for(size_t i = ra.size() ; i < MAX_NUM_BEAMS ; ++i) {
            right_ascension[i] = std::nanf("");
        }
#endif
        }

        {
        const auto& dec(md[DECLINATION]);
        if(dec.size() > MAX_NUM_BEAMS)
                throw std::runtime_error("Number of beams request exceeds MAX_NUM_BEAMS");
        size_t i = 0;
        for(auto it = dec.cbegin() ; it != dec.cend() ; ++it) {
                it->get_to(declination[i]);
        }
#ifdef DEBUG
        for(size_t i = dec.size() ; i < MAX_NUM_BEAMS ; ++i) {
            declination[i] = std::nanf("");
        }
#endif
        }

        {
        const auto& scale(md[SCALING]);
        if(scale.size() > MAX_NUM_BEAMS)
            throw std::runtime_error("Number of beams request exceeds MAX_NUM_BEAMS");
        size_t i = 0;
        for(auto it = scale.cbegin() ; it != scale.cend() ; ++it) {
            it->get_to(scaling[i]);
        }
#ifdef DEBUG
        for(size_t i = scale.size() ; i < MAX_NUM_BEAMS ; ++i) {
            scaling[i] = 0;
        }
#endif
        }
    }
};

struct coarseFreq {
        int nfreq;
        int coarse_freq[CHORD_META_MAX_FREQ];

        // TODO: turn into a from_json function?
        coarseFreq(const metadata& md) {
        md[NFREQ].get_to(nfreq);
        const auto& freq(md[COARSE_FREQ]);
        if(freq.size() > MAX_NUM_BEAMS)
            throw std::runtime_error("Number of frequencies requested exceeds CHORD_META_MAX_FREQ");
        size_t i = 0;
        for(auto it = freq.cbegin() ; it != freq.cend() ; ++it) {
            it->get_to(coarse_freq[i]);
        }
#ifdef DEBUG
        for(size_t i = freq.size() ; i < CHORD_META_MAX_FREQ ; ++i) {
            coarse_freq[i] = 0;
        }
#endif
        }
};

}

static inline
void to_json(nlohmann::json&j, const timeval& tv) {
  using namespace jsonMetadata;
  j[TV_SEC] = static_cast<std::int64_t>(tv.tv_sec);
  j[TV_USEC] = static_cast<std::int64_t>(tv.tv_usec);
}

static inline
void from_json(const nlohmann::json&j, timeval& tv) {
  using namespace jsonMetadata;
  tv.tv_sec = j[TV_SEC].template get<std::int64_t>();
  tv.tv_usec = j[TV_USEC].template get<std::int64_t>();
}

#endif
