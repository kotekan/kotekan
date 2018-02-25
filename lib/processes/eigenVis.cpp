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
    num_diagonals_filled =  config.get_int(unique_name, "num_diagonals_filled");
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

    // Memory for LAPACK interface.
    std::vector<cfloat> vis_square;
    std::vector<cfloat> evecs;
    std::vector<float> evals(num_eigenvectors, 0);

    int info, ev_found, nside, nev;

    // Storage space for the last eigenvector calculated (scaled by sqrt
    // the eigenvalue) for each freq_id.
    std::map<uint32_t, std::vector<cfloat>> last_evs;
    uint32_t freq_id;

    openblas_set_num_threads(1);

    while (!stop_thread) {

        // Get input visibilities. We assume the shape of these doesn't change.
        if (wait_for_full_frame(input_buffer, unique_name.c_str(),
                                input_frame_id) == nullptr) {
            break;
        }
        auto input_frame = visFrameView(input_buffer, input_frame_id);

        if (!initialized) {
            num_elements = input_frame.num_elements;
            if (input_frame.num_eigenvectors < num_eigenvectors) {
                throw std::runtime_error("Insufficient storage space for"
                                         " requested number of eigenvectors.");
            }

            vis_square.resize(num_elements * num_elements, 0);
            evecs.resize(num_elements * num_eigenvectors, 0);

            initialized = true;
        }
        if (input_frame.num_prod != num_elements * (num_elements + 1) / 2) {
            throw std::runtime_error("Eigenvectors require full correlation"
                                     " triangle");
        }

        freq_id = input_frame.freq_id;
        // Conditionally initialize eigenvector storage space.
        if (last_evs.find(freq_id) == last_evs.end()) {
            last_evs.emplace(std::piecewise_construct,
                             std::forward_as_tuple(freq_id),
                             std::forward_as_tuple(num_elements * num_eigenvectors, 0)
                             );
        }

        // Fill the upper half (lower in fortran order!) of the square version
        // of the visibilities.
        int prod_ind = 0;
        for(int i = 0; i < num_elements; i++) {
            for(int j = i; j < i + num_diagonals_filled && j < num_elements; j++) {
                cfloat value = 0;
                for (int ev_ind = 0; ev_ind < num_eigenvectors; ev_ind++) {
                    value += (std::conj(last_evs[freq_id][ev_ind * num_elements + i])
                              * last_evs[freq_id][ev_ind * num_elements + j]);
                }
                vis_square[i * num_elements + j] = value;
                prod_ind++;
            }
            for(int j = i + num_diagonals_filled; j < num_elements; j++) {
                vis_square[i * num_elements + j] = input_frame.vis[prod_ind];
                prod_ind++;
            }
        }

        nside = (int) num_elements;
        nev = (int) num_eigenvectors;
        info = LAPACKE_cheevr(LAPACK_COL_MAJOR, 'V', 'I', 'L', nside,
                              (lapack_complex_float *) vis_square.data(), nside,
                              0.0, 0.0, nside - nev + 1, nside, 0.0,
                              &ev_found, evals.data(),
                              (lapack_complex_float *) evecs.data(),
                              nside, NULL);

        // Update the stored eigenvectors for the next iteration.
        for(int ev_ind = 0; ev_ind < num_eigenvectors; ev_ind++) {
            for(int i = 0; i < num_elements; i++) {
                last_evs[freq_id][ev_ind * num_elements + i] =
                    std::sqrt(evals[ev_ind]) * evecs[ev_ind * num_elements + i];
            }
        }

        // Get output buffer for visibilities. Essentially identical to input buffers.
        if (wait_for_empty_frame(output_buffer, unique_name.c_str(),
                                 output_frame_id) == nullptr) {
            break;
        }
        allocate_new_metadata_object(output_buffer, output_frame_id);
        auto output_frame = visFrameView(output_buffer, output_frame_id, input_frame);

        // Report all eigenvalues to stdout.
        std::string str_evals = "";
        for (auto const& value: evals) str_evals += " " + std::to_string(value);
        INFO("Found eigenvalues:%s", str_evals.c_str());

        // Copy in eigenvectors and eigenvalues.
        gsl::span<cfloat> evecs_s{evecs};
        gsl::span<float> evals_s{evals};
        copy(evecs_s, output_frame.eigenvectors);
        copy(evals_s.subspan(0, num_eigenvectors), output_frame.eigenvalues);

        // Finish up interation.
        mark_frame_empty(input_buffer, unique_name.c_str(), input_frame_id);
        mark_frame_full(output_buffer, unique_name.c_str(),
                        output_frame_id);
        input_frame_id = (input_frame_id + 1) % input_buffer->num_frames;
        output_frame_id = (output_frame_id + 1) % output_buffer->num_frames;
    }
}
