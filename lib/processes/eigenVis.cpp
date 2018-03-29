#include "eigenVis.hpp"
#include "visBuffer.hpp"
#include "errors.h"
#include "fpga_header_functions.h"
#include "chimeMetadata.h"
#include "prometheusMetrics.hpp"
#include "fmt.hpp"
#include "visUtil.hpp"

#include <cblas.h>
#include <lapacke.h>
#include <time.h>

REGISTER_KOTEKAN_PROCESS(eigenVis);

eigenVis::eigenVis(Config& config,
                       const string& unique_name,
                       bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container, std::bind(&eigenVis::main_thread, this)) {

    apply_config(0);

    input_buffer = get_buffer("in_buf");
    register_consumer(input_buffer, unique_name.c_str());
    output_buffer = get_buffer("out_buf");
    register_producer(output_buffer, unique_name.c_str());
    num_eigenvectors =  config.get_int(unique_name, "num_ev");
    num_diagonals_filled =  config.get_int_default(unique_name,
                                                   "num_diagonals_filled", 0);
    // Read a list from the config, but permit it to be absent (implying empty).
    try {
        for (auto e : config.get_int_array(unique_name, "exclude_inputs")) {
            exclude_inputs.push_back((int32_t) e);
        }
    } catch (std::runtime_error const & ex) {
        // Missing, leave empty.
    }

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
    std::vector<float> evals;

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

        // Start the calculation clock.
        double start_time = current_time();

        if (!initialized) {
            num_elements = input_frame.num_elements;
            if (input_frame.num_ev < num_eigenvectors) {
                throw std::runtime_error("Insufficient storage space for"
                                         " requested number of eigenvectors.");
            }

            vis_square.resize(num_elements * num_elements, 0);
            evecs.resize(num_elements * num_eigenvectors, 0);
            evals.resize(num_elements, 0);

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
        for (int i = 0; i < num_elements; i++) {
            for (int j = i; j < i + num_diagonals_filled && j < num_elements; j++) {
                cfloat value = 0;
                for (int ev_ind = 0; ev_ind < num_eigenvectors; ev_ind++) {
                    value += (std::conj(last_evs[freq_id][ev_ind * num_elements + i])
                              * last_evs[freq_id][ev_ind * num_elements + j]);
                }
                vis_square[i * num_elements + j] = value;
                prod_ind++;
            }
            for (int j = i + num_diagonals_filled; j < num_elements; j++) {
                // Conjugate because Fortran interprets as lower triangle.
                vis_square[i * num_elements + j] = std::conj(input_frame.vis[prod_ind]);
                prod_ind++;
            }
        }

        // Go through and zero out data in excluded rows and columns.
        for (auto iexclude : exclude_inputs) {
            for (int j = 0; j < iexclude; j++) {
                vis_square[j * num_elements + iexclude] = {0, 0};
            }
            for (int j = iexclude; j < num_elements; j++) {
                vis_square[iexclude * num_elements + j] = {0, 0};
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

        DEBUG("LAPACK exit status: %d", info);
        if (info) {
            ERROR("LAPACK failed with exit code %d", info);
            break;
        }

        // Update the stored eigenvectors for the next iteration.
        for (int ev_ind = 0; ev_ind < num_eigenvectors; ev_ind++) {
            for (int i = 0; i < num_elements; i++) {
                last_evs[freq_id][ev_ind * num_elements + i] =
                    std::sqrt(evals[ev_ind]) * evecs[ev_ind * num_elements + i];
            }
        }

        // Calculate RMS residuals to eigenvalue approximation to the visibilities.
        double sum_sq = 0;
        int nprod_sum = 0;
        std::vector<int> ipts(num_elements); // Inputs to iterate over
        std::iota(ipts.begin(), ipts.end(), 0); // Fill sequentially

        for (auto iexclude : exclude_inputs) { // Remove excluded inputs
            ipts.erase(std::remove(ipts.begin(), ipts.end(), iexclude),ipts.end());
        }
        for(std::vector<int>::size_type idx = 0; idx != ipts.size(); idx++) {
            int i = ipts[idx];
            int j0 = i + num_diagonals_filled;
            unsigned int j0_idx;
            // Find start column:
            while(j0 < num_elements) {
                auto j0_iterator = std::find(ipts.begin(), ipts.end(), j0);
                if(j0_iterator == ipts.end()) {
                    j0++;
                } else {
                    j0_idx = std::distance(ipts.begin(), j0_iterator);
                    break;
                }
            }
            if(j0 >= num_elements) {
                break;
            }
            for(auto j_idx = j0_idx; j_idx != ipts.size(); j_idx++) {
                int j = ipts[j_idx];
                prod_ind = cmap(i,j,num_elements)
                cfloat residual = input_frame.vis[prod_ind];
                for (int ev_ind = 0; ev_ind < num_eigenvectors; ev_ind++) {
                    residual -= (last_evs[freq_id][ev_ind * num_elements + i]
                                 * std::conj(last_evs[freq_id][ev_ind * num_elements + j]));
                }
                sum_sq += std::norm(residual);
                nprod_sum++;
            }
        }
        double rms = pow(sum_sq / nprod_sum, 0.5);

        // Stop the calculation clock. This doesn't include time to copy stuff into
        // the buffers, but that has to wait for one to be available.
        double elapsed_time = current_time() - start_time;

        // Report all eigenvalues to stdout.
        std::string str_evals = "";
        for (int i = 0; i < num_eigenvectors; i++) {
            str_evals += " " + std::to_string(evals[i]);
        }
        INFO("Found eigenvalues:%s, with RMS residuals: %e, in %3.1f s.",
             str_evals.c_str(), rms,elapsed_time);

        // Update average write time in prometheus
        calc_time.add_sample(elapsed_time);
        prometheusMetrics::instance().add_process_metric(
            "kotekan_eigenvis_comp_time_seconds",
            unique_name, calc_time.average()
        );

        // Output eigenvalues to prometheus
        for(int i = 0; i < num_eigenvectors; i++) {
            std::string labels = fmt::format(
                "eigenvalue=\"{}\",freq_id=\"{}\",dataset_id=\"{}\"",
                i, freq_id, input_frame.dataset_id
            );
            prometheusMetrics::instance().add_process_metric(
                "kotekan_eigenvis_eigenvalue",
                unique_name, evals[num_eigenvectors - 1 - i], labels
            );
        }

        // Output RMS to prometheus
        std::string labels = fmt::format(
            "eigenvalue=\"rms\",freq_id=\"{}\",dataset_id=\"{}\"",
            freq_id, input_frame.dataset_id
        );
        prometheusMetrics::instance().add_process_metric(
            "kotekan_eigenvis_eigenvalue",
            unique_name, rms, labels
        );

        // Get output buffer for visibilities. Essentially identical to input buffers.
        if (wait_for_empty_frame(output_buffer, unique_name.c_str(),
                                 output_frame_id) == nullptr) {
            break;
        }
        allocate_new_metadata_object(output_buffer, output_frame_id);
        auto output_frame = visFrameView(output_buffer, output_frame_id, input_frame);

        // Copy in eigenvectors and eigenvalues.
        for(int i = 0; i < num_eigenvectors; i++) {
            int indr = num_eigenvectors - 1 - i;
            output_frame.eval[i] = evals[indr];

            for(int j = 0; j < num_elements; j++) {
                output_frame.evec[i * num_elements + j] = evecs[indr * num_elements + j];
            }
        }
        output_frame.erms = rms;

        // Finish up interation.
        mark_frame_empty(input_buffer, unique_name.c_str(), input_frame_id);
        mark_frame_full(output_buffer, unique_name.c_str(),
                        output_frame_id);
        input_frame_id = (input_frame_id + 1) % input_buffer->num_frames;
        output_frame_id = (output_frame_id + 1) % output_buffer->num_frames;
    }
}
