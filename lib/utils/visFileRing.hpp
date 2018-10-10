
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

    // Implement the create_file method
    void create_file(const std::string& name,
                     const std::map<std::string, std::string>& metadata,
                     dset_id_t dataset, size_t num_ev, size_t max_time) override;

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