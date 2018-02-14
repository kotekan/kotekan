#include "eigenVis.hpp"
#include "visBuffer.hpp"
#include "errors.h"
#include "fpga_header_functions.h"
#include "chimeMetadata.h"

#include <cblas.h>
#include <lapacke.h>


eigenVis::eigenVis(Config& config,
                       const string& unique_name,
                       bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container, std::bind(&eigenVis::main_thread, this)) {

    apply_config(0);

    input_buffer = get_buffer("in_buf");
    register_consumer(input_buffer, unique_name.c_str());
    output_buffer = get_buffer("out_buf");
    register_producer(output_buffer, unique_name.c_str());
    num_eigenvectors =  config.get_int(unique_name, "num_eigenvectors");
}

eigenVis::~eigenVis() {
}

void eigenVis::apply_config(uint64_t fpga_seq) {
}

void eigenVis::main_thread() {

    unsigned int input_frame_id = 0;
    unsigned int output_frame_id = 0;
    unsigned int num_elements;
    bool initialized = false;
    std::complex<float> * vis_square;
    std::complex<float> * evecs;
    float * evals;

    int info, ev_found, nside, nev;

    openblas_set_num_threads(1);

    while (!stop_thread) {

        // Get input visibilities. We assume the shape of these doesn't change.
        if (wait_for_full_frame(input_buffer, unique_name.c_str(),
                                input_frame_id) == nullptr) {
            break;
        }
        auto input_frame = visFrameView(input_buffer, input_frame_id);
        if (!initialized) {
            num_elements = input_frame.num_elements();
            if (input_frame.num_eigenvectors() < num_eigenvectors) {
                throw std::runtime_error("Insufficient storage space for"
                                         " requested number of eigenvectors.");
            }
            vis_square = (std::complex<float> *) malloc(num_elements * num_elements
                                                        * sizeof(*vis_square));
            evecs = (std::complex<float> *) malloc(num_elements * num_eigenvectors
                                                   * sizeof(*evecs));
            evals = (float *) malloc(num_elements * sizeof(*evals));

            if (vis_square == nullptr){
                // XXX What is the right thing to do here?
                break;
            }
            initialized = true;
        }
        INFO("%d, %d", input_frame.num_prod(), num_elements);
        if (input_frame.num_prod() != num_elements * (num_elements + 1) / 2) {
            throw std::runtime_error("Eigenvectors require full correlation"
                                     " triangle");
        }

        // Fill the upper half of the square version of the visibilities.
        int prod_ind = 0;
        for(int i = 0; i < num_elements; i++) {
            for(int j = i; j < num_elements; j++) {
                vis_square[i * num_elements + j] = input_frame.vis()[prod_ind];
                prod_ind++;
            }
        }

        nside = (int) num_elements;
        nev = (int) num_eigenvectors;
        info = LAPACKE_cheevr(LAPACK_COL_MAJOR, 'V', 'I', 'U', nside,
                              (lapack_complex_float *) vis_square, nside,
                              0.0, 0.0, nside - nev + 1, nside, 0.0,
                              &ev_found, evals, (lapack_complex_float *) evecs,
                              nside, NULL);

        // Get output buffer for visibilities. Essentially identical to input buffers.
        if (wait_for_empty_frame(output_buffer, unique_name.c_str(),
                                 output_frame_id) == nullptr) {
            break;
        }
        allocate_new_metadata_object(output_buffer, output_frame_id);
        auto output_frame = input_frame.copy_to_buffer(output_buffer,
                                                       output_frame_id);

        // Finish up interation.
        mark_frame_empty(input_buffer, unique_name.c_str(), input_frame_id);
        mark_frame_full(output_buffer, unique_name.c_str(),
                        output_frame_id);
        input_frame_id = (input_frame_id + 1) % input_buffer->num_frames;
        output_frame_id = (output_frame_id + 1) % output_buffer->num_frames;
    }

    if (initialized) {
        free(vis_square);
        free(evecs);
        free(evals);
    }
}
