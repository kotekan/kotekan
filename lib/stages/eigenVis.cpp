#include "eigenVis.hpp"

#include "Config.hpp"            // for Config
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"            // for mark_frame_empty, allocate_new_metadata_object, mark_fr...
#include "kotekanLogging.hpp"    // for DEBUG, ERROR, INFO
#include "prometheusMetrics.hpp" // for Metrics, Gauge, MetricFamily
#include "N2FrameView.hpp"       // for N2FrameView
#include "N2Util.hpp"            // for cfloat, frameID, current_time, cmap

#include "fmt.hpp"      // for format, fmt
#include "gsl-lite.hpp" // for span

#include <algorithm>  // for lower_bound, remove
#include <cblas.h>    // for openblas_set_num_threads
#include <cmath>      // for pow, sqrt
#include <complex>    // for operator*, norm, complex
#include <cstdint>    // for uint32_t
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <lapacke.h>  // for LAPACKE_cheevr, LAPACK_COL_MAJOR
#include <map>        // for map, map<>::mapped_type, operator==, map<>::iterator
#include <numeric>    // for iota
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
    input_buffer->register_consumer(unique_name);

    output_buffer = get_buffer("out_buf");
    output_buffer->register_producer(unique_name);

    num_diagonals_filled = config.get_default<uint32_t>(unique_name, "num_diagonals_filled", 0);

    // Read a list from the config, but permit it to be absent (implying empty).
    try {
        for (uint32_t e : config.get<std::vector<uint32_t>>(unique_name, "exclude_inputs")) {
            exclude_inputs.push_back(e);
        }
    } catch (std::runtime_error const& ex) {
        // Missing, leave empty.
    }
}

void eigenVis::main_thread() {

    N2::frameID input_frame_id(input_buffer);
    N2::frameID output_frame_id(output_buffer);

    uint32_t num_elements = 0;
    bool initialized = false;
    size_t lapack_failure_total = 0;

    // Memory for LAPACK interface.
    std::vector<N2::cfloat> vis_square;
    std::vector<N2::cfloat> evecs;
    std::vector<float> evals;

    int info, ev_found;
    int64_t nside;
    uint32_t num_ev;

    // Storage space for the last eigenvector calculated (scaled by sqrt
    // the eigenvalue) for each freq_id.
    std::map<uint32_t, std::vector<N2::cfloat>> last_evs;

    openblas_set_num_threads(1);

    auto& eigenvalue_metric = Metrics::instance().add_gauge(
        "kotekan_eigenvis_eigenvalue", unique_name, {"eigenvalue", "freq_id"});

    auto& comp_time_seconds_metric =
        Metrics::instance().add_gauge("kotekan_eigenvis_comp_time_seconds", unique_name);

    auto& lapack_failure_counter = Metrics::instance().add_counter(
        "kotekan_eigenvis_lapack_failure_total", unique_name, {"freq_id"});
    
    while (!stop_thread) {

        // Get input visibilities. We assume the shape of these doesn't change.
        if (input_buffer->wait_for_full_frame(unique_name, input_frame_id) == nullptr) {
            break;
        }
        auto input_frame = N2FrameView(input_buffer, input_frame_id);
        num_ev = input_frame.num_ev;

        // Start the calculation clock.
        u_int64_t start_time = N2::current_time();

        if (!initialized) {
            num_elements = input_frame.num_elements;
            vis_square.resize(num_elements * num_elements, 0);
            evecs.resize(num_elements * num_ev, 0);
            evals.resize(num_elements, 0);

            // initialized = true;
        }
        if (input_frame.num_prod != num_elements * (num_elements + 1) / 2) {
            throw std::runtime_error("Eigenvector calculation requires a triangular correlation "
                                     "matrix.");
        }

        uint32_t freq_id = input_frame.freq_id;
        // Conditionally initialize eigenvector storage space.
        if (last_evs.find(freq_id) == last_evs.end()) {
            last_evs.emplace(std::piecewise_construct, std::forward_as_tuple(freq_id),
                             std::forward_as_tuple(num_elements * num_ev, 0));
        }

        // Fill the upper half (lower in fortran order!) of the square version
        // of the visibilities.
        int prod_ind = 0;
        for (uint32_t i = 0; i < num_elements; i++) {
            for (uint32_t j = i; j < i + num_diagonals_filled && j < num_elements; j++) {
                N2::cfloat value = 0;
                for (uint32_t ev_ind = 0; ev_ind < num_ev; ev_ind++) {
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

        nside = (int32_t) num_elements;
        info = LAPACKE_cheevr(LAPACK_COL_MAJOR, 'V', 'I', 'L', nside,
                              (lapack_complex_float*)vis_square.data(), nside, 0.0, 0.0,
                              nside - num_ev + 1, nside, 0.0, &ev_found, evals.data(),
                              (lapack_complex_float*)evecs.data(), nside, nullptr);

        DEBUG("LAPACK exit status: {:d}", info);
        if (info) {
            ERROR("LAPACK failed with exit code {:d}", info);
            lapack_failure_total++;

            // Update prometheus metric about LAPACK failures
            lapack_failure_counter
                .labels({std::to_string(freq_id)})
                .inc();

            // Clear frame and advance
            input_buffer->mark_frame_empty(unique_name, input_frame_id++);
            continue;
        }

        // Update the stored eigenvectors for the next iteration.
        for (uint32_t ev_ind = 0; ev_ind < num_ev; ev_ind++) {
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
                prod_ind = N2::cmap(*i, *j, num_elements);
                N2::cfloat residual = input_frame.vis[prod_ind];
                for (uint32_t ev_ind = 0; ev_ind < num_ev; ev_ind++) {
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
        u_int64_t elapsed_time = N2::current_time() - start_time;

        // Report all eigenvalues to stdout.
        std::string str_evals = "";
        for (uint32_t i = 0; i < num_ev; i++) {
            str_evals = fmt::format(fmt("{:s} {}"), str_evals, evals[i]);
        }
        INFO("Found eigenvalues: {:s}, with RMS residuals: {:e}, in {:3.1f} s.", str_evals, rms,
             elapsed_time);

        // Update average write time in prometheus
        calc_time.add_sample(elapsed_time);
        comp_time_seconds_metric.set(calc_time.average());

        // Output eigenvalues to prometheus
        for (uint32_t i = 0; i < num_ev; i++) {
            eigenvalue_metric
                .labels({std::to_string(i), std::to_string(freq_id)})
                .set(evals[num_ev - 1 - i]);
        }

        // Output RMS to prometheus
        eigenvalue_metric
            .labels({"rms", std::to_string(freq_id)})
            .set(rms);

        // Get output buffer for visibilities. Essentially identical to input buffers.
        if (output_buffer->wait_for_empty_frame(unique_name, output_frame_id) == nullptr) {
            break;
        }

        // Create view to output frame
        input_buffer->pass_metadata(input_frame_id, output_buffer, output_frame_id);
        N2FrameView output_frame(output_buffer, output_frame_id);

        // Copy over metadata and data, but skip all ev members which may not be defined
        output_frame.copy_data(input_frame, {N2Field::eval, N2Field::evec, N2Field::erms});

        // Copy in eigenvectors and eigenvalues.
        for (uint32_t i = 0; i < num_ev; i++) {
            int indr = num_ev - 1 - i;
            output_frame.eval[i] = evals[indr];

            for (uint32_t j = 0; j < num_elements; j++) {
                output_frame.evec[i * num_elements + j] = evecs[indr * num_elements + j];
            }
        }
        output_frame.emethod = N2EigenMethod::cheevr;
        output_frame.erms = rms;

        // Finish up interation.
        input_buffer->mark_frame_empty(unique_name, input_frame_id++);
        output_buffer->mark_frame_full(unique_name, output_frame_id++);
    }
}
