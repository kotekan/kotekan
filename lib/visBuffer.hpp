#ifndef VISBUFFER_HPP
#define VISBUFFER_HPP

#include <time.h>

#include "buffer.h"


struct visMetadata {

    // The time of the integration frame (as FPGA count and ctime)
    int64_t fpga_seq_num;
    timespec ctime;

    // ID of the frequency bin
    uint16_t freq_id;

    // ID of the dataset (vis, gatedvisX ...), main vis dataset = 0
    uint16_t dataset_id;

    // Sizes of data
    uint32_t num_elements;
    uint32_t num_prod;

    // Number of eigenvectors and values calculated
    uint16_t num_eigenvectors;

};


/// A view of the visibillity frame
class visFrameView {

public:

    // Create view without modifying layout (useful if buffer already created)
    visFrameView(Buffer * buf, int frame_id);

    // Create view and write layout (for buffer initialisation)
    visFrameView(Buffer * buf, int_frame_id, uint32_t num_elements,
                 uint16_t num_eigenvectors);
    visFrameView(Buffer * buf, int_frame_id, uint32_t num_elements,
                 uint32_t num_prod, uint16_t num_eigenvectors);

    // Sample metadata
    std::tuple<int64_t &, timespec &> time();
    uint16_t & freq_id();
    uint16_t & dataset_id();

    // Sample data
    complex_int * vis();
    double * eigenvalues();
    std::complex<double> * eigenvectors();
    double & rms();

    // These define the layout of the buffer, and cannot be modified after
    // creation time
    uint32_t num_elements();
    uint32_t num_prod();
    uint8_t num_eigenvectors();

private:

    visMetadata * metadata;

    complex_int * vis_ptr;
    double * eval_ptr;
    double * evec_ptr;
    double * rms_ptr;


    void check_and_set();
};



#endif
