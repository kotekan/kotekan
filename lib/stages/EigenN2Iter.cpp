#include "EigenN2Iter.hpp"

#include "Config.hpp"            // for Config
#include "Hash.hpp"              // for operator!=, operator<
#include "LinearAlgebra.hpp"     // for EigConvergenceStats, eigen_masked_subspace, to_blaze_herm
#include "N2FrameView.hpp"       // for N2FrameView
#include "N2Util.hpp"            // for cfloat, frameID
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"            // for allocate_new_metadata_object, mark_frame_empty, mark_fr...
#include "kotekanLogging.hpp"    // for DEBUG
#include "prometheusMetrics.hpp" // for Gauge, Metrics, MetricFamily

#include "fmt.hpp"      // for format, fmt
#include "gsl-lite.hpp" // for span

#include <algorithm>     // for copy, copy_backward, equal, max, min
#include <atomic>        // for atomic_bool
#include <blaze/Blaze.h> // for DynamicMatrix, DMatDeclHermExpr, band, HermitianMatrix
#include <cblas.h>       // for openblas_set_num_threads
#include <complex>       // for complex
#include <cstdint>       // for uint32_t, int32_t
#include <deque>         // for deque
#include <exception>     // for exception
#include <functional>    // for _Bind_helper<>::type, bind, function
#include <iostream>      // for basic_ostream::operator<<, operator<<, basic_ostream<>:...
#include <memory>        // for make_unique
#include <regex>         // for match_results<>::_Base_type
#include <stdexcept>     // for runtime_error, out_of_range
#include <tuple>         // for tie, tuple

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(EigenN2Iter);

EigenN2Iter::EigenN2Iter(Config& config, const std::string& unique_name,
                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&EigenN2Iter::main_thread, this)),

    in_buf(get_buffer("in_buf")), out_buf(get_buffer("out_buf")),

    _num_eigenvectors(config.get<size_t>(unique_name, "num_ev")),

    // Convergence params
    _tol_eval(config.get_default<double>(unique_name, "tol_eval", 1e-6)),
    _tol_evec(config.get_default<double>(unique_name, "tol_evec", 1e-5)),
    _num_ev_conv(config.get_default<size_t>(unique_name, "num_ev_conv", 0)),
    _max_iterations(config.get_default<size_t>(unique_name, "max_iterations", 15)),
    _krylov(config.get_default<size_t>(unique_name, "krylov", 2)),
    _subspace(config.get_default<size_t>(unique_name, "subspace", 3)),

    // Masking params
    _exclude_inputs(config.get_default<std::vector<size_t>>(unique_name, "exclude_inputs", {})),
    _block_fill_size(config.get_default<size_t>(unique_name, "block_fill_size", 0)),
    _diagonal_bands_filled(config.get_default<std::vector<std::pair<size_t, size_t>>>(
        unique_name, "diagonal_bands_filled", {})),

    comp_time_seconds_metric(
        Metrics::instance().add_gauge("kotekan_eigenN2iter_comp_time_seconds", unique_name)),
    eigenvalue_metric(Metrics::instance().add_gauge("kotekan_eigenN2iter_eigenvalue", unique_name,
                                                    {"eigenvalue", "freq_id"})),
    iterations_metric(
        Metrics::instance().add_gauge("kotekan_eigenN2iter_iterations", unique_name, {"freq_id"})),
    eigenvalue_convergence_metric(Metrics::instance().add_gauge(
        "kotekan_eigenN2iter_eigenvalue_convergence", unique_name, {"freq_id"})),
    eigenvector_convergence_metric(Metrics::instance().add_gauge(
        "kotekan_eigenN2iter_eigenvector_convergence", unique_name, {"freq_id"})) {

    in_buf->register_consumer(unique_name);
    out_buf->register_producer(unique_name);

    if (in_buf->buffer_type != "N2" || out_buf->buffer_type != "N2")
        FATAL_ERROR("EigenN2Iter stage requires 'N2' buffers as input and output.");


    if (_num_ev_conv > _num_eigenvectors)
        FATAL_ERROR("The `num_ev_conv` config parameter ({:d}) must be less than or equal to "
                    "`num_ev` ({:d}).",
                    _num_ev_conv, _num_eigenvectors);
    if (_num_eigenvectors <= 0)
        FATAL_ERROR("The `num_ev` config parameter must be greater than zero.");
    if (_tol_eval <= 0)
        FATAL_ERROR("The `tol_eval` config parameter must be greater than zero.");
    if (_tol_evec <= 0)
        FATAL_ERROR("The `tol_evec` config parameter must be greater than zero.");
    if (_max_iterations == 0)
        FATAL_ERROR("The `max_iterations` config parameter must be greater than zero.");
}

void EigenN2Iter::main_thread() {

    frameID input_frame_id(in_buf);
    frameID output_frame_id(out_buf);

    DynamicHermitian<float> mask;
    uint32_t num_elements = 0;
    bool initialized = false;

    openblas_set_num_threads(1);

    while (!stop_thread) {

        // Containers for results
        eig_t<cfloat> eigpair;
        EigConvergenceStats stats;

        // Get input visibilities. We assume the shape of these doesn't change.
        INFO("EigenN2Iter waiting for input frame...");
        if (in_buf->wait_for_full_frame(unique_name, input_frame_id) == nullptr) {
            break;
        }
        INFO("EigenN2Iter got input frame.");
        N2FrameView input_frame(in_buf, input_frame_id);

        // Check that we have the full triangle
        if (input_frame.vis_layout != N2Layout::FullUpperTri) {
            FATAL_ERROR(
                "Eigenvector calculations require full correlation triangle. Got layout {:d}.",
                static_cast<int>(input_frame.vis_layout));
        }

        // Start the calculation clock.
        uint64_t start_time = current_time();

        // Initialise the mask
        if (!initialized) {
            num_elements = input_frame.num_elements;
            mask = calculate_mask(num_elements);
            initialized = true;
        }

        // Copy the visibilties into a blaze container
        DynamicHermitian<cfloat> vis = to_blaze_herm(input_frame.vis);

        // Perform the actual eigen-decomposition
        std::tie(eigpair, stats) =
            eigen_masked_subspace(vis, mask, _num_eigenvectors, _tol_eval, _tol_evec,
                                  _max_iterations, _num_ev_conv, _krylov, _subspace);
        auto& evals = eigpair.first;
        auto& evecs = eigpair.second;

        // Stop the calculation clock. This doesn't include time to copy stuff into
        // the buffers, but that has to wait for one to be available.
        uint64_t elapsed_time = current_time() - start_time;

        // Report all eigenvalues to stdout.
        std::string str_evals = "";
        for (uint32_t i = 0; i < _num_eigenvectors; i++) {
            str_evals = fmt::format("{} {}", str_evals, evals[i]);
        }
        DEBUG("Found eigenvalues: {:s}, with RMS residuals: {:e}, in {:d} s. Took {:d}/{:d} "
              "iterations.",
              str_evals, stats.rms, elapsed_time, stats.iterations, _max_iterations);

        // Update Prometheus metrics
        update_metrics(input_frame.freq_id, elapsed_time, eigpair, stats);

        /* Write out new frame */
        // Get output buffer for visibilities. Essentially identical to input buffers.
        if (out_buf->wait_for_empty_frame(unique_name, output_frame_id) == nullptr) {
            break;
        }

        // Create view to output frame
        in_buf->pass_metadata(input_frame_id, out_buf, output_frame_id);

        // Ensure metadata reflects the number of eigenvectors so the frame layout matches the
        // buffer's frame size (which is sized for the eigen data).
        auto out_meta = get_N2_metadata(out_buf, output_frame_id);
        if (out_meta)
            out_meta->num_ev = _num_eigenvectors;

        N2FrameView output_frame(out_buf, output_frame_id);
        // Copy over data, but skip all ev members which may not be defined
        output_frame.copy_data(input_frame, {N2Field::eval, N2Field::evec, N2Field::erms});

        // Copy in eigenvectors and eigenvalues.
        for (uint32_t i = 0; i < _num_eigenvectors; i++) {
            int indr = _num_eigenvectors - 1 - i;
            output_frame.eval[i] = evals[indr];

            for (uint32_t j = 0; j < num_elements; j++) {
                output_frame.evec[i * num_elements + j] = evecs(j, indr);
            }
        }
        // HACK: return the convergence state in the RMS field (negative == not
        // converged)
        output_frame.erms = stats.converged ? stats.rms : -stats.eps_eval;
        output_frame.emethod = N2EigenMethod::iterative;

        // Finish up interation.
        in_buf->mark_frame_empty(unique_name, input_frame_id++);
        out_buf->mark_frame_full(unique_name, output_frame_id++);
    }
}


void EigenN2Iter::update_metrics(int freq_id, u_int64_t elapsed_time, const eig_t<cfloat>& eigpair,
                                 const EigConvergenceStats& stats) {
    // Update average write time in prometheus
    auto& calc_time = calc_time_map[freq_id];
    calc_time.add_sample(elapsed_time);
    comp_time_seconds_metric.set(calc_time.average());

    // Output eigenvalues to prometheus
    for (uint32_t i = 0; i < _num_eigenvectors; i++) {
        eigenvalue_metric.labels({std::to_string(i), std::to_string(freq_id)})
            .set(eigpair.first[_num_eigenvectors - 1 - i]);
    }

    // Output RMS to prometheus
    eigenvalue_metric.labels({"rms", std::to_string(freq_id)}).set(stats.rms);

    // Output convergence stats
    std::vector<std::string> labels = {std::to_string(freq_id)};
    iterations_metric.labels(labels).set(stats.iterations);
    eigenvalue_convergence_metric.labels(labels).set(stats.eps_eval);
    eigenvector_convergence_metric.labels(labels).set(stats.eps_evec);
}


DynamicHermitian<float> EigenN2Iter::calculate_mask(size_t num_elements) const {
    blaze::DynamicMatrix<float, blaze::columnMajor> M;
    M.resize(num_elements, num_elements);

    // Construct the mask matrix ...
    // Go through and zero out data in excluded rows and columns.
    M = 1.0f;
    for (auto iexclude : _exclude_inputs) {
        for (size_t j = 0; j < num_elements; j++) {
            M(iexclude, j) = 0.0;
            M(j, iexclude) = 0.0;
        }
    }

    // Remove specified diagonal bands
    for (const auto& br : _diagonal_bands_filled) {
        std::cout << br.first << " " << br.second << std::endl;
        for (size_t i = br.first; i < br.second; i++) {
            blaze::band(M, i) = 0.0;
            blaze::band(M, -i) = 0.0;
        }
    }

    // Zero out blocks on the diagonal if requested
    if (_block_fill_size > 0) {
        unsigned int nb = num_elements / _block_fill_size;
        for (unsigned int ii = 0; ii < nb; ii++) {
            unsigned int start = ii * _block_fill_size;
            unsigned int width = std::min(num_elements - start, _block_fill_size);
            blaze::submatrix(M, start, start, width, width) = 0.0;
        }
    }

    return blaze::declherm(M);
}
