#include "eigenVis.hpp"
#include "visBuffer.hpp"
#include "errors.h"
#include "fpga_header_functions.h"
#include "chimeMetadata.h"

eigenVis::eigenVis(Config& config,
                       const string& unique_name,
                       bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container, std::bind(&eigenVis::main_thread, this)) {

    apply_config(0);

    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());
    num_eigenvectors =  config.get_int(unique_name, "num_eigenvectors");
}

eigenVis::~eigenVis() {
}

void eigenVis::apply_config(uint64_t fpga_seq) {
}

void eigenVis::main_thread() {

    unsigned int frame_id = 0;
    const auto num_elements;
    bool initialized = false;
    std::complex<float> * vis_square;

    while (!stop_thread) {

        // Get input visibilities. We assume the shape of these doesn't change.
        if (wait_for_full_frame(input_buf, unique_name.c_str(),
                               frame_id) == nullptr) {
            break;
        }
        auto input_frame = visFrameView(input_buffer, frame_id);
        if (!initialized) {
            num_element = input_frame.num_elements();
            if (input_frame.num_eigenvectors() < num_eigenvectors) {
                throw std::runtime_error("Insufficient storage space for"
                                         " requested number of eigenvectors.");
            }
            vis_square = (std::complex<float>) malloc(num_elements * num_elements
                                                      * sizeof(*vis_square));
            if (vis_square != nullptr) ;
            initialized = true;
        }
        if (input_frame.num_prod() != num_elements * (num_elements - 1) / 2) {
            throw std::runtime_error("Eigenvectors requier full correlation"
                                     " triangle");
        }

        // Get output buffer for visibilities. Essentially identical to input buffers.
        if (wait_for_empty_frame(output_buf, unique_name.c_str(),
                                frame_id) == nullptr) {
            break;
        }
        auto output_frame = visFrameView(output_buf, frame_id,
                                         input_frame.num_elements(),
                                         input_frame.num_eigenvectors());

    }

    if (initialized) {
        free(vis_square);
    }
}
