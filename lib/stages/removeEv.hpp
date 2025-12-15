/*****************************************
@file
@brief Remove the eigenvalues/vectors from a buffer
- removeEv : public kotekan::Stage
*****************************************/
#ifndef REMOVE_EV_HPP
#define REMOVE_EV_HPP

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "datasetManager.hpp" // for dset_id_t, state_id_t

#include <map>    // for map
#include <string> // for string

/**
 * @class removeEv
 * @brief Remove any eigenvalues/vectors from a buffer.
 *
 * @par Buffers
 * @buffer in_buf Input buffer with eigensector.
 *         @buffer_format VisBuffer structured
 *         @buffer_metadata VisMetadata
 * @buffer out_buf Output buffer without eigensector.
 *         @buffer_format VisBuffer structured
 *         @buffer_metadata VisMetadata
 *
 * @par Dataset states
 * Removes `eigenvalue`/`eigenvector` dataset states; produces a dataset ID without those states.
 *
 * @par Example
 * @code
 * remove_ev:
 *   kotekan_stage: removeEv
 *   in_buf: vis_with_ev
 *   out_buf: vis_no_ev
 * @endcode
 *
 * @author Richard Shaw
 */
class removeEv : public kotekan::Stage {

public:
    removeEv(kotekan::Config& config, const std::string& unique_name,
             kotekan::bufferContainer& buffer_container);

    void main_thread() override;

private:
    void change_dataset_state(dset_id_t input_dset_id);

    Buffer* in_buf;
    Buffer* out_buf;

    state_id_t ev_state_id;

    std::map<dset_id_t, dset_id_t> dset_id_map;
};

#endif
