
/*****************************************
@file
@brief Fixed length ring buffering file based on visFileRaw.
- visFileRing
*****************************************/
#ifndef VIS_FILE_RING_HPP
#define VIS_FILE_RING_HPP

#include "visFileRaw.hpp"

/** @brief A CHIME correlator ring-style buffer file in raw format.
 *
 * The class manages a CHIME correlator file of fixed length in time,
 * where additional times get written at the start of the file again,
 * effectively maintaining a ring buffer of the most recent data stream.
 *
 * The file format is provided by visFileRaw.
 *
 * @warning Unlike visFileRaw, this class will overwrite an existing
 *          file with the same path.
 *
 * @author Tristan Pinsonneault-Marotte
 **/
class visFileRing : public visFileRaw {

public:

    /**
     * Create a ring output file.
     *
     * This variant uses the datasetManager to look up properties of the
     * dataset that we are dealing with.
     *
     * @param name      Name of the file to write
     * @param log_level kotekan log level for any logging generated by the visFile instance
     * @param metadata  Textual metadata to write into the file.
     * @param dataset   ID of dataset we are writing.
     * @param max_time  Maximum number of times to write into the file.
     **/
    visFileRing(const std::string& name, const kotekan::logLevel log_level,
                const std::map<std::string, std::string>& metadata, dset_id_t dataset,
                size_t max_time);

    /**
     * @brief Extend the file to a new time sample.
     *        If the file has reached it's full length,
     *        start adding times to the start again.
     *
     * @param new_time The new time to add.
     * @return The index of the added time in the file.
     **/
    uint32_t extend_time(time_ctype new_time) override;

    void write_metadata();

private:
    size_t file_len;
    size_t cur_pos = 0;
};

#endif
