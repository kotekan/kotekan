#include "N2FrameView.hpp"

size_t N2FrameView::data_size() const {
    return calculate_frame_size(_metadata->num_elements, _metadata->num_ev);
}

void N2FrameView::zero_frame() {
    // // Fill data with zeros
    // std::memset(_frame, 0, data_size());
    // erms = 0;

    // // Set non-structural metadata
    // freq_id = 0;
    // dataset_id = dset_id_t::null;
    // time = std::make_tuple(0, timespec{0, 0});
    // n_valid_fpga_samples = 0;

    // // mark frame as empty by ensuring this is 0
    // fpga_seq_length = 0;
    // fpga_seq_total = 0;
}


