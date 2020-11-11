/*****************************************
@file
@brief Read HFBFileRaw data.
- HFBRawReader : public kotekan::Stage
*****************************************/
#ifndef _HFB_RAW_READER_HPP
#define _HFB_RAW_READER_HPP

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "buffer.h"
#include "bufferContainer.hpp"
#include "datasetManager.hpp" // for dset_id_t
#include "visUtil.hpp"        // for freq_ctype (ptr only), input_ctype, prod_ctype, rstack_ctype

#include "json.hpp" // for json

#include <map>      // for map
#include <stddef.h> // for size_t
#include <stdint.h> // for uint32_t, uint8_t
#include <string>   // for string
#include <utility>  // for pair
#include <vector>   // for vector

/**
 * @class HFBRawReader
 * @brief Read and stream a raw 21cm absorber file.
 *
 * This will divide the file up into time-frequency chunks of set size and
 * stream out the frames with time as the *fastest* index. The dataset ID
 * will be restored from the dataset broker if `use_comet` is set. Otherwise
 * a new dataset will be created and the original ID stored in the frames
 * will be lost.
 *
 * @par Buffers
 * @buffer out_buf The data read from the raw file.
 *         @buffer_format VisBuffer structured
 *         @buffer_metadata VisMetadata
 *
 * @conf    readahead_blocks       Int. Number of blocks to advise OS to read ahead
 *                                 of current read.
 * @conf    chunk_size             Array of [int, int, int]. Read chunk size (freq,
 *                                 prod, time). If not specified will read file
 *                                 contiguously.
 * @conf    infile                 String. Path to the (data-meta-pair of) files to
 *                                 read (e.g. "/path/to/0000_000", without .data or
 *                                 .meta).
 * @conf    max_read_rate          Float. Maximum read rate for the process in MB/s.
 *                                 If the value is zero (default), then no rate
 *                                 limiting is applied.
 * @conf    sleep_time             Float. After the data is read pause this long in
 *                                 seconds before sending shutdown. If < 0, never
 *                                 send a shutdown signal. Default is -1.
 * @conf    update_dataset_id      Bool. Update the dataset ID with information about the
                                   file, for example which time samples does it contain.
 * @conf    use_dataset_broker     Bool. Restore dataset ID from dataset broker (i.e. comet).
 *                                 Should be disabled only for testing. Default is true.
 * @conf    use_local_dataset_man  Bool. Instead of using comet, register metadata on top
 *                                 of dataset ID in the file. Should only be used for
 *                                 testing or if original dataset IDs can be lost.
 *                                 Default is False.
 *
 * @author Richard Shaw, Tristan Pinsonneault-Marotte, Rick Nitsche
 */
class HFBRawReader : public kotekan::Stage {

public:
    /// default constructor
    HFBRawReader(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);

    ~HFBRawReader();

    /// Main loop over buffer frames
    void main_thread() override;

    /**
     * @brief Get the times in the file.
     **/
    const std::vector<time_ctype>& times() {
        return _times;
    }

    /**
     * @brief Get the frequencies in the file.
     **/
    const std::vector<std::pair<uint32_t, freq_ctype>>& freqs() {
        return _freqs;
    }

    /**
     * @brief Get the metadata saved into the file.
     **/
    const nlohmann::json& metadata() {
        return _metadata;
    }

private:
    /**
     * @brief Get the new dataset ID.
     *
     * If not using change the ID this just returns its input ID. If using the
     * broker, this will simply append a timeState to the current state.
     * Otherwise, we just use a static dataset_id constructed from the file
     * metadata.
     *
     * @param  ds_id  The ID of the read frame.
     * @returns       The replacement ID
     */
    dset_id_t get_dataset_state(dset_id_t ds_id);

    /**
     * @brief Read the next frame.
     *
     * The exact frame read is dependent on whether the reads are time ordered
     * or not.
     *
     * @param ind The frame index to read.
     **/
    void read_ahead(int ind);

    /**
     * @brief Map the index into a frame position in the file.
     *
     * The exact frame read dependent on whether the reads are time ordered
     * or not.
     *
     * @param ind The frame index.
     * @returns The frame index into the file.
     **/
    int position_map(int ind);

    Buffer* out_buf;

    // The metadata
    nlohmann::json _metadata;
    std::vector<time_ctype> _times;
    std::vector<std::pair<uint32_t, freq_ctype>> _freqs;
    std::vector<uint32_t> _beams;
    std::vector<uint32_t> _subfreqs;

    // whether to read in chunks
    bool chunked;

    // whether to update the dataset ID with info about the file
    bool update_dataset_id;

    // whether to use comet to track dataset IDs
    bool use_comet;

    // whether to use the local dataset manager
    bool local_dm;

    // Read chunk size (freq, prod, time)
    std::vector<int> chunk_size;
    // time chunk size
    size_t chunk_t;
    // freq chunk size
    size_t chunk_f;

    // Number of elements in a chunked row
    size_t row_size;

    // the input file
    std::string filename;
    int fd;
    uint8_t* mapped_file;

    size_t file_frame_size, metadata_size, data_size, nfreq, ntime, nbeam, nsubfreq;

    // Number of blocks to read ahead while reading from disk
    size_t readahead_blocks;

    // Dataset states constructed from metadata
    std::vector<state_id_t> states;

    // Dataset ID to assign to output frames if not using comet
    dset_id_t static_out_dset_id;

    // the dataset state for the time axis
    state_id_t tstate_id;

    // Map input to output dataset IDs for quick access
    std::map<dset_id_t, dset_id_t> ds_in_file;

    // The read rate
    double max_read_rate;

    // Sleep time after reading
    double sleep_time;
};

#endif
