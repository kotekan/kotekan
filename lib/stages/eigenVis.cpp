#include "eigenVis.hpp"

#include "Config.hpp"            // for Config
#include "Hash.hpp"              // for operator!=
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"              // for mark_frame_empty, allocate_new_metadata_object, mark_fr...
#include "datasetState.hpp"      // for datasetState, eigenvalueState, state_uptr
#include "kotekanLogging.hpp"    // for DEBUG, ERROR, INFO
#include "prometheusMetrics.hpp" // for Metrics, Gauge, MetricFamily
#include "visBuffer.hpp"         // for visFrameView, visField, visField::erms, visField::eval
#include "visUtil.hpp"           // for cfloat, frameID, modulo, current_time, cmap, movingAverage

#include "fmt.hpp"      // for format, fmt
#include "gsl-lite.hpp" // for span

#include <algorithm>  // for fill, max, lower_bound, remove
#include <atomic>     // for atomic_bool
#include <cblas.h>    // for openblas_set_num_threads
#include <cmath>      // for pow, sqrt
#include <complex>    // for operator*, norm, complex
#include <cstdint>    // for uint32_t
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <lapacke.h>  // for LAPACKE_cheevr, LAPACK_COL_MAJOR
#include <map>        // for map, map<>::mapped_type, operator==, map<>::iterator
#include <memory>     // for make_unique
#include <numeric>    // for iota
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error
#include <time.h>     // for size_t
#include <tuple>      // for forward_as_tuple
#include <utility>    // for move, pair, piecewise_construct

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(eigenVis);

eigenVis::eigenVis(Config& config, const std::string& unique_name,
                   bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&eigenVis::main_thread, this)) {

    input_buffer = get_buffer("in_buf");
    register_consumer(input_buffer, unique_name.c_str());
    output_buffer = get_buffer("out_buf");
    register_producer(output_buffer, unique_name.c_str());
    num_eigenvectors = config.get<uint32_t>(unique_name, "num_ev");
    num_diagonals_filled = config.get_default<uint32_t>(unique_name, "num_diagonals_filled", 0);
    // Read a list from the config, but permit it to be absent (implying empty).
    try {
        for (uint32_t e : config.get<std::vector<uint32_t>>(unique_name, "exclude_inputs")) {
            exclude_inputs.push_back(e);
        }
    } catch (std::runtime_error const& ex) {
        // Missing, leave empty.
    }

    // Create the state describing the eigenvalues
    auto& dm = datasetManager::instance();
    state_uptr ev_state = std::make_unique<eigenvalueState>(num_eigenvectors);
    ev_state_id = dm.add_state(std::move(ev_state)).first;
}

dset_id_t eigenVis::change_dataset_state(dset_id_t input_dset_id) {
    auto& dm = datasetManager::instance();
    return dm.add_dataset(ev_state_id, input_dset_id);
}

void eigenVis::main_thread() {

    frameID input_frame_id(input_buffer);
    frameID output_frame_id(output_buffer);
    uint32_t num_elements = 0;
    bool initialized = false;
    size_t lapack_failure_total = 0;
    dset_id_t _output_dset_id = dset_id_t::null;

    // Memory for LAPACK interface.
    std::vector<cfloat> vis_square;
    std::vector<cfloat> evecs;
    std::vector<float> evals;

    int info, ev_found;
    int64_t nside, nev;

    // Storage space for the last eigenvector calculated (scaled by sqrt
    // the eigenvalue) for each freq_id.
    std::map<uint32_t, std::vector<cfloat>> last_evs;
    uint32_t freq_id;

    openblas_set_num_threads(1);

    auto& eigenvalue_metric = Metrics::instance().add_gauge(
        "kotekan_eigenvis_eigenvalue", unique_name, {"eigenvalue", "freq_id", "dataset_id"});

    auto& comp_time_seconds_metric =
        Metrics::instance().add_gauge("kotekan_eigenvis_comp_time_seconds", unique_name);

    // TODO: this should logically be a Counter
    auto& lapack_failure_counter = Metrics::instance().add_gauge(
        "kotekan_eigenvis_lapack_failure_total", unique_name, {"freq_id", "dataset_id"});

    while (!stop_thread) {

        // Get input visibilities. We assume the shape of these doesn't change.
        if (wait_for_full_frame(input_buffer, unique_name.c_str(), input_frame_id) == nullptr) {
            break;
        }
        auto input_frame = visFrameView(input_buffer, input_frame_id);


        // check if the input dataset has changed
        if (input_dset_id != input_frame.dataset_id) {
            input_dset_id = input_frame.dataset_id;
            _output_dset_id = change_dataset_state(input_dset_id);
        }

        // Start the calculation clock.
        double start_time = current_time();

        if (!initialized) {
            num_elements = input_frame.num_elements;
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
            last_evs.emplace(std::piecewise_construct, std::forward_as_tuple(freq_id),
                             std::forward_as_tuple(num_elements * num_eigenvectors, 0));
        }

        // Fill the upper half (lower in fortran order!) of the square version
        // of the visibilities.
        int prod_ind = 0;
        for (uint32_t i = 0; i < num_elements; i++) {
            for (uint32_t j = i; j < i + num_diagonals_filled && j < num_elements; j++) {
                cfloat value = 0;
                for (uint32_t ev_ind = 0; ev_ind < num_eigenvectors; ev_ind++) {
                    value += (std::conj(last_evs[freq_id][ev_ind * num_elements + i])
                              * last_evs[freq_id][ev_ind * num_elements + j]);
                }
                vis_square[i * num_elements + j] = value;
                prod_ind++;
            }
            for (uint32_t j = i + num_diagonals_filled; j < num_elements; j++) {
                // Conjugate because Fortran interprets as lower triangle.
                vis_square[i * num_elements + j] = std::conj(input_frame.vis[prod_ind]);
                prod_ind++;
            }
        }

        // Go through and zero out data in excluded rows and columns.
        for (auto iexclude : exclude_inputs) {
            for (uint32_t j = 0; j < iexclude; j++) {
                vis_square[j * num_elements + iexclude] = {0, 0};
            }
            for (uint32_t j = iexclude; j < num_elements; j++) {
                vis_square[iexclude * num_elements + j] = {0, 0};
            }
        }

        nside = (int32_t)num_elements;
        nev = (int32_t)num_eigenvectors;
        info = LAPACKE_cheevr(LAPACK_COL_MAJOR, 'V', 'I', 'L', nside,
                              (lapack_complex_float*)vis_square.data(), nside, 0.0, 0.0,
                              nside - nev + 1, nside, 0.0, &ev_found, evals.data(),
                              (lapack_complex_float*)evecs.data(), nside, nullptr);

        DEBUG("LAPACK exit status: {:d}", info);
        if (info) {
            ERROR("LAPACK failed with exit code {:d}", info);
            lapack_failure_total++;

            // Update prometheus metric about LAPACK failures
            lapack_failure_counter
                .labels({std::to_string(freq_id), input_frame.dataset_id.to_string()})
                .set(lapack_failure_total);

            // Clear frame and advance
            mark_frame_empty(input_buffer, unique_name.c_str(), input_frame_id++);
            continue;
        }

        // Update the stored eigenvectors for the next iteration.
        for (uint32_t ev_ind = 0; ev_ind < num_eigenvectors; ev_ind++) {
            for (uint32_t i = 0; i < num_elements; i++) {
                last_evs[freq_id][ev_ind * num_elements + i] =
                    std::sqrt(evals[ev_ind]) * evecs[ev_ind * num_elements + i];
            }
        }

        // Calculate RMS residuals to eigenvalue approximation to the visibilities.
        double sum_sq = 0;
        int nprod_sum = 0;
        std::vector<int> ipts(num_elements);    // Inputs to iterate over
        std::iota(ipts.begin(), ipts.end(), 0); // Fill sequentially

        for (auto iexclude : exclude_inputs) { // Remove excluded inputs
            ipts.erase(std::remove(ipts.begin(), ipts.end(), iexclude), ipts.end());
        }
        for (auto i = ipts.begin(); i != ipts.end(); i++) {
            auto jstart = std::lower_bound(i, ipts.end(), *i + num_diagonals_filled);
            for (auto j = jstart; j != ipts.end(); j++) {
                prod_ind = cmap(*i, *j, num_elements);
                cfloat residual = input_frame.vis[prod_ind];
                for (uint32_t ev_ind = 0; ev_ind < num_eigenvectors; ev_ind++) {
                    residual -= (last_evs[freq_id][ev_ind * num_elements + *i]
                                 * std::conj(last_evs[freq_id][ev_ind * num_elements + *j]));
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
        for (uint32_t i = 0; i < num_eigenvectors; i++) {
            str_evals = fmt::format(fmt("{:s} {}"), str_evals, evals[i]);
        }
        INFO("Found eigenvalues: {:s}, with RMS residuals: {:e}, in {:3.1f} s.", str_evals, rms,
             elapsed_time);

        // Update average write time in prometheus
        calc_time.add_sample(elapsed_time);
        comp_time_seconds_metric.set(calc_time.average());

        // Output eigenvalues to prometheus
        for (uint32_t i = 0; i < num_eigenvectors; i++) {
            eigenvalue_metric
                .labels({std::to_string(i), std::to_string(freq_id),
                         input_frame.dataset_id.to_string()})
                .set(evals[num_eigenvectors - 1 - i]);
        }

        // Output RMS to prometheus
        eigenvalue_metric
            .labels({"rms", std::to_string(freq_id), input_frame.dataset_id.to_string()})
            .set(rms);

        // Get output buffer for visibilities. Essentially identical to input buffers.
        if (wait_for_empty_frame(output_buffer, unique_name.c_str(), output_frame_id) == nullptr) {
            break;
        }
        allocate_new_metadata_object(output_buffer, output_frame_id);
        auto output_frame = visFrameView(output_buffer, output_frame_id, input_frame.num_elements,
                                         input_frame.num_prod, num_eigenvectors);

        // Copy over metadata and data, but skip all ev members which may not be
        // defined
        output_frame.copy_metadata(input_frame);
        output_frame.dataset_id = _output_dset_id;
        output_frame.copy_data(input_frame, {visField::eval, visField::evec, visField::erms});

        // Copy in eigenvectors and eigenvalues.
        for (uint32_t i = 0; i < num_eigenvectors; i++) {
            int indr = num_eigenvectors - 1 - i;
            output_frame.eval[i] = evals[indr];

            for (uint32_t j = 0; j < num_elements; j++) {
                output_frame.evec[i * num_elements + j] = evecs[indr * num_elements + j];
            }
        }
        output_frame.erms = rms;

        // Finish up interation.
        mark_frame_empty(input_buffer, unique_name.c_str(), input_frame_id++);
        mark_frame_full(output_buffer, unique_name.c_str(), output_frame_id++);
    }
}
