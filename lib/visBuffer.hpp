#ifndef VISBUFFER_HPP
#define VISBUFFER_HPP

#include <time.h>
#include <sys/time.h>
#include <tuple>
#include <complex>

#include "visUtil.hpp"

#include "buffer.h"


struct visMetadata {

    // The time of the integration frame (as FPGA count and ctime)
    uint64_t fpga_seq_num;
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
    visFrameView(Buffer * buf, int frame_id, uint32_t num_elements,
                 uint16_t num_eigenvectors);
    visFrameView(Buffer * buf, int frame_id, uint32_t num_elements,
                 uint32_t num_prod, uint16_t num_eigenvectors);

    // Copy frame to new buffer and create view
    visFrameView(Buffer * buf, int frame_id,
                                visFrameView frame_to_copy);

    // Sample metadata
    std::tuple<uint64_t &, timespec &> time();
    uint16_t & freq_id();
    uint16_t & dataset_id();

    // Sample data
    std::complex<float> * vis();
    float * eigenvalues();
    std::complex<float> * eigenvectors();
    float & rms();

    // These define the layout of the buffer, and cannot be modified after
    // creation time
    uint32_t num_elements();
    uint32_t num_prod();
    uint32_t num_eigenvectors();

    // Return a summary of the visibility buffer contents
    std::string summary();

private:

    std::complex<float> * vis_ptr;
    float * eval_ptr;
    std::complex<float> * evec_ptr;
    float * rms_ptr;

    Buffer * const buffer;
    const int id;
    visMetadata * const  metadata;

    void check_and_set();
};



#endif
