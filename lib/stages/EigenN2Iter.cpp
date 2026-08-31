#include "EigenN2Iter.hpp"

#include "Config.hpp"            // for Config
#include "Hash.hpp"              // for operator!=, operator<
#include "LinearAlgebra.hpp"     // for EigConvergenceStats, eigen_masked_subspace, to_blaze_herm
#include "N2FrameDesc.hpp"       // for N2FrameDesc
#include "N2FrameView.hpp"       // for N2FrameView
#include "N2Util.hpp"            // for cfloat, frameID
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"            // for allocate_new_metadata_object, mark_frame_empty, mark_fr...
#include "div.hpp"               // for div_ceil
#include "kotekanLogging.hpp"    // for DEBUG
#include "prometheusMetrics.hpp" // for Counter, Gauge, Metrics, MetricFamily

#include "fmt.hpp"      // for format, fmt
#include "gsl-lite.hpp" // for span

#include <algorithm>     // for find, min
#include <atomic>        // for atomic_bool
#include <blaze/Blaze.h> // for DynamicMatrix, DMatDeclHermExpr, band, HermitianMatrix
#include <cblas.h>       // for openblas_set_num_threads
#include <complex>       // for complex
#include <cstdint>       // for uint32_t, int32_t
#include <deque>         // for deque
#include <exception>     // for exception
#include <functional>    // for _Bind_helper<>::type, bind, function
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
    failed_buf(config.exists(unique_name, "failed_buf") ? get_buffer("failed_buf") : nullptr),

    _num_eigenvectors(config.get<size_t>(unique_name, "num_ev")),

    // Convergence params
    _tol_eval(config.get_default<double>(unique_name, "tol_eval", 1e-6)),
    _tol_evec(config.get_default<double>(unique_name, "tol_evec", 1e-5)),
    _num_ev_conv(config.get_default<size_t>(unique_name, "num_ev_conv", 0)),
    _max_iterations(config.get_default<size_t>(unique_name, "max_iterations", 15)),
    _krylov(config.get_default<size_t>(unique_name, "krylov", 2)),
    _subspace(config.get_default<size_t>(unique_name, "subspace", 3)),

    // Blaze SMP thread count
    _num_blaze_workers(config.get_default<uint32_t>(unique_name, "num_blaze_workers", 0)),

    // Masking params
    _exclude_inputs(config.get_default<std::vector<size_t>>(unique_name, "exclude_inputs", {})),
    _block_fill_size(config.get_default<size_t>(unique_name, "block_fill_size", 0)),
    _diagonal_bands_filled(config.get_default<std::vector<std::pair<size_t, size_t>>>(
        unique_name, "diagonal_bands_filled", {})),
    _mask_flagged_inputs(config.get_default<bool>(unique_name, "mask_flagged_inputs", true)),

    comp_time_seconds_metric(
        Metrics::instance().add_gauge("kotekan_eigenN2iter_comp_time_seconds", unique_name)),
    eigenvalue_metric(Metrics::instance().add_gauge("kotekan_eigenN2iter_eigenvalue", unique_name,
                                                    {"eigenvalue", "freq_id"})),
    iterations_metric(
        Metrics::instance().add_gauge("kotekan_eigenN2iter_iterations", unique_name, {"freq_id"})),
    eigenvalue_convergence_metric(Metrics::instance().add_gauge(
        "kotekan_eigenN2iter_eigenvalue_convergence", unique_name, {"freq_id"})),
    eigenvector_convergence_metric(Metrics::instance().add_gauge(
        "kotekan_eigenN2iter_eigenvector_convergence", unique_name, {"freq_id"})),
    num_failed_eigencalc(
        Metrics::instance().add_counter("kotekan_eigenN2iter_num_failed_eigencalc", unique_name)),
    masked_elements_metric(
        Metrics::instance().add_gauge("kotekan_eigenN2iter_masked_elements", unique_name)),
    num_mask_updates(
        Metrics::instance().add_counter("kotekan_eigenN2iter_num_mask_updates", unique_name)),
    num_insufficient_elements(Metrics::instance().add_counter(
        "kotekan_eigenN2iter_num_insufficient_elements", unique_name)) {

    in_buf->register_consumer(unique_name);
    out_buf->register_producer(unique_name);

    if (in_buf->buffer_type != "N2" || out_buf->buffer_type != "N2")
        FATAL_ERROR("EigenN2Iter stage requires an 'N2' buffer type for in_buf and out_buf.");

    // Validate that input and output buffers have N2 frame descriptors set
    auto in_desc = in_buf->require_frame_desc<kotekan::N2FrameDesc>();
    auto out_desc = out_buf->require_frame_desc<kotekan::N2FrameDesc>();
    // Validate num_elements and layout match
    if (in_desc->get_num_elements() != out_desc->get_num_elements()) {
        FATAL_ERROR("EigenN2Iter: Input and output buffer num_elements must match");
    }
    if (in_desc->get_n2_layout() != out_desc->get_n2_layout()) {
        FATAL_ERROR("EigenN2Iter: Input and output buffer n2_layout must match");
    }
    // Validate output buffer has correct num_ev for this stage
    if (out_desc->get_num_ev() != _num_eigenvectors) {
        FATAL_ERROR("EigenN2Iter: Output buffer num_ev ({:d}) does not match stage config ({:d})",
                    out_desc->get_num_ev(), _num_eigenvectors);
    }

    if (failed_buf) {
        failed_buf->register_producer(unique_name);
        // this replicates some of the tests on in_buf to not rely on previous tests
        if (failed_buf->buffer_type != "N2")
            FATAL_ERROR("EigenN2Iter stage requires 'N2' buffer type for failed_buf.");

        // Validate that input and failed buffers have N2 frame descriptors set
        auto failed_desc = failed_buf->require_frame_desc<kotekan::N2FrameDesc>();
        // Validate num_elements and layout match
        if (in_desc->get_num_elements() != failed_desc->get_num_elements()) {
            FATAL_ERROR("EigenN2Iter: Input and failed buffer num_elements must match");
        }
        if (in_desc->get_n2_layout() != failed_desc->get_n2_layout()) {
            FATAL_ERROR("EigenN2Iter: Input and failed buffer n2_layout must match");
        }
        // Validate failed buffer has correct num_ev for this stage
        if (failed_desc->get_num_ev() != 0) {
            FATAL_ERROR("EigenN2Iter: Failed buffer num_ev ({:d}) not 0",
                        failed_desc->get_num_ev());
        }
    }

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

// Whether a frame's flags match the binarized flags the cached mask was built
// from. The mask only depends on which flags are zero, so comparing binarized
// values keeps a fractional or NaN flag from forcing a rebuild of an identical
// mask every frame.
static bool flags_match(const std::vector<float>& applied, const gsl_lite::span<float>& flags) {
    if (applied.size() != flags.size())
        return false;
    for (size_t i = 0; i < applied.size(); i++) {
        if (applied[i] != (flags[i] != 0.0f ? 1.0f : 0.0f))
            return false;
    }
    return true;
}

void EigenN2Iter::main_thread() {

    frameID input_frame_id(in_buf);
    frameID output_frame_id(out_buf);
    // need to pass a valid buffer to avoid a segfault, even if failed_frame_id
    // is not used afterwards
    frameID failed_frame_id(failed_buf != nullptr ? failed_buf : out_buf);

    DynamicHermitian<float> mask;
    uint32_t num_elements = 0;
    // Binarized per-element flags the cached mask was built from (any non-zero
    // incoming flag counts as good). Empty until the first frame forces a build.
    std::vector<float> applied_flags;
    size_t num_good_elements = 0;
    // When a degenerate-mask warning was last logged
    double last_degenerate_warn = 0.0;

    // Force serial BLAS so Blaze owns intra-op parallelism via its own OpenMP
    // path (per-stage team, inherits this stage's cpu_affinity). With a
    // multithreaded OpenBLAS we'd otherwise hit its process-global pool whose
    // affinity is fixed at first-call time and bleeds between concurrent
    // eigen stages.
    openblas_set_num_threads(1);
    if (_num_blaze_workers > 0)
        blaze::setNumThreads(_num_blaze_workers);

    while (!stop_thread) {

        // Containers for results
        eig_t<cfloat> eigpair;
        EigConvergenceStats stats;
        bool failed = false;

        // Get input visibilities. We assume the shape of these doesn't change.
        DEBUG("EigenN2Iter waiting for input frame...");
        if (in_buf->wait_for_full_frame(unique_name, input_frame_id) == nullptr) {
            break;
        }
        DEBUG("EigenN2Iter got input frame.");
        N2FrameView input_frame(in_buf, input_frame_id);

        // Check that we have the full correlation triangle: a dense upper triangle over
        // the frame's own element axis holds exactly n(n+1)/2 products. The count is
        // necessary but not sufficient (a subset of the right size could still be
        // non-triangular); layouts that store the dense triangle in the canonical order
        // (FullUpperTri, and compact subsets like DishInputs) pass, and the solver
        // stays blind to where the frame's elements came from.
        if (input_frame.num_prod
            != (size_t)input_frame.num_elements * (input_frame.num_elements + 1) / 2) {
            FATAL_ERROR("Eigenvector calculations require a full correlation triangle: "
                        "{:d} elements need {:d} products, got {:d} (layout {}).",
                        input_frame.num_elements,
                        (size_t)input_frame.num_elements * (input_frame.num_elements + 1) / 2,
                        input_frame.num_prod, N2Layout_to_string(input_frame.n2_layout));
        }

        // Start the calculation clock.
        double start_time = current_time();

        // (Re)build the mask when the element count changes or when the flags
        // differ from the cached mask. Flags change infrequently in normal operation,
        // so this should not happen often.
        const bool flags_changed =
            _mask_flagged_inputs && !flags_match(applied_flags, input_frame.flags);
        if (num_elements != input_frame.num_elements || flags_changed) {
            num_elements = input_frame.num_elements;
            applied_flags.assign(num_elements, 1.0f);
            if (_mask_flagged_inputs) {
                for (uint32_t i = 0; i < num_elements; i++)
                    applied_flags[i] = input_frame.flags[i] != 0.0f ? 1.0f : 0.0f;
            }

            mask = calculate_mask(num_elements, applied_flags);

            // Count the elements neither the flags nor the config masks out.
            num_good_elements = 0;
            for (uint32_t i = 0; i < num_elements; i++) {
                if (applied_flags[i] == 0.0f)
                    continue;
                if (std::find(_exclude_inputs.begin(), _exclude_inputs.end(), i)
                    != _exclude_inputs.end())
                    continue;
                num_good_elements++;
            }

            masked_elements_metric.set(num_elements - num_good_elements);
            num_mask_updates.inc(1);
            INFO("Mask updated: {:d} of {:d} elements masked out.",
                 num_elements - num_good_elements, num_elements);
            if (num_good_elements <= _num_eigenvectors) {
                WARN("Only {:d} of {:d} elements are unmasked; solving for {:d} eigenpairs "
                     "needs more than that. Frames will be reported as failed until this "
                     "changes.",
                     num_good_elements, num_elements, _num_eigenvectors);
                last_degenerate_warn = current_time();
            }
        }

        // Perform the actual eigen-decomposition
        if (num_good_elements <= _num_eigenvectors) {
            // A decomposition needs more unmasked elements than the eigenpairs
            // requested. The mask update above warns once; re-warn every ten
            // minutes so a stream failing every frame stays visible in the logs.
            num_insufficient_elements.inc(1);
            failed = true;
            if (current_time() - last_degenerate_warn > 600.0) {
                WARN("Only {:d} of {:d} elements are unmasked; frames are still being "
                     "reported as failed.",
                     num_good_elements, num_elements);
                last_degenerate_warn = current_time();
            }
        } else {
            // Copy the visibilities into a blaze container
            DynamicHermitian<cfloat> vis = to_blaze_herm(input_frame.vis);

            try {
                std::tie(eigpair, stats) =
                    eigen_masked_subspace(vis, mask, _num_eigenvectors, _tol_eval, _tol_evec,
                                          _max_iterations, _num_ev_conv, _krylov, _subspace);
            } catch (const std::runtime_error& e) {
                ERROR("Could not find eigenvalues after {:d} for frame fpga_seq {:d}: {:s}",
                      _max_iterations, input_frame.fpga_start_tick, e.what());
                num_failed_eigencalc.inc(1);
                failed = true;
            }
        }

        // Failed frames are routed to failed_buf when configured; otherwise they fall
        // through to out_buf with zero-filled eigenpairs.
        const bool route_to_failed = failed && failed_buf != nullptr;
        Buffer* const dest_buf = route_to_failed ? failed_buf : out_buf;
        frameID& dest_frame_id = route_to_failed ? failed_frame_id : output_frame_id;

        if (dest_buf->wait_for_empty_frame(unique_name, dest_frame_id) == nullptr) {
            break;
        }

        in_buf->pass_metadata(input_frame_id, dest_buf, dest_frame_id);

        N2FrameView dest_frame(dest_buf, dest_frame_id);
        // Copy over data, but skip all ev members which we'll set explicitly below
        dest_frame.copy_data(input_frame, {N2Field::eval, N2Field::evec, N2Field::erms});

        if (failed) {
            // HACK: negative RMS indicates a failed/non-converged calculation.
            dest_frame.erms = -std::numeric_limits<float>::max();
            dest_frame.emethod =
                N2EigenMethod::failed_iterative; // Specific type indicates code failure
            // failed_buf has num_ev == 0 (validated in the constructor) and so has no
            // eval/evec storage; only zero them when falling through to out_buf.
            if (!route_to_failed) {
                for (uint32_t i = 0; i < _num_eigenvectors; i++) {
                    dest_frame.eval[i] = 0.0f;
                    for (uint32_t j = 0; j < num_elements; j++) {
                        dest_frame.evec[i * num_elements + j] = {0.0f, 0.0f};
                    }
                }
            }
        } else {
            // Stop the calculation clock. This doesn't include time to copy stuff into
            // output buffers.
            double elapsed_time = current_time() - start_time;

            auto& evals = eigpair.first;
            auto& evecs = eigpair.second;

            // Report eigenvalues to stdout.
            std::string str_evals = "";
            for (uint32_t i = 0; i < _num_eigenvectors; i++) {
                str_evals = fmt::format("{} {}", str_evals, evals[i]);
            }
            DEBUG("Found eigenvalues: {:s}, with RMS residuals: {:e}, in {:.3f} s. Took {:d}/{:d} "
                  "iterations.",
                  str_evals, stats.rms, elapsed_time, stats.iterations, _max_iterations);

            // Update Prometheus metrics
            update_metrics(input_frame.freq_id, elapsed_time, eigpair, stats);

            // Copy in eigenvectors and eigenvalues.
            for (uint32_t i = 0; i < _num_eigenvectors; i++) {
                int indr = _num_eigenvectors - 1 - i;
                dest_frame.eval[i] = evals[indr];

                for (uint32_t j = 0; j < num_elements; j++) {
                    dest_frame.evec[i * num_elements + j] = evecs(j, indr);
                }
            }
            // HACK: return the convergence state in the RMS field (negative == not
            // converged)
            dest_frame.erms = stats.converged ? stats.rms : -stats.eps_eval;
            dest_frame.emethod = N2EigenMethod::iterative;
        }

        // Finish up iteration.
        dest_buf->mark_frame_full(unique_name, dest_frame_id++);
        in_buf->mark_frame_empty(unique_name, input_frame_id++);
    }
}


void EigenN2Iter::update_metrics(int freq_id, double elapsed_time, const eig_t<cfloat>& eigpair,
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


DynamicHermitian<float> EigenN2Iter::calculate_mask(size_t num_elements,
                                                    const std::vector<float>& flags) const {
    // Blaze does not bounds check element access in a release build, so a
    // config containing an out of bounds element would corrupt memory.
    for (auto iexclude : _exclude_inputs) {
        if (iexclude >= num_elements)
            FATAL_ERROR("The `exclude_inputs` entry {:d} is out of range for frames with {:d} "
                        "elements. These are indices into the incoming frame's elements, which "
                        "for a subsetted buffer are the subset's own indices.",
                        iexclude, num_elements);
    }
    for (const auto& br : _diagonal_bands_filled) {
        if (br.first > br.second || br.second > num_elements)
            FATAL_ERROR("The `diagonal_bands_filled` range [{:d}, {:d}) is not a valid band range "
                        "for frames with {:d} elements.",
                        br.first, br.second, num_elements);
    }

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

    // Zero out the rows and columns of elements flagged bad. The flags here
    // are already binarized, as the solver's mask must be strictly binary.
    for (size_t i = 0; i < num_elements; i++) {
        if (flags[i] != 0.0f)
            continue;
        for (size_t j = 0; j < num_elements; j++) {
            M(i, j) = 0.0;
            M(j, i) = 0.0;
        }
    }

    // Remove specified diagonal bands
    for (const auto& br : _diagonal_bands_filled) {
        DEBUG("Masking diagonal bands [{:d}, {:d}).", br.first, br.second);
        // Signed so that -i below is a genuine negation, not size_t wraparound.
        for (int64_t i = br.first; i < (int64_t)br.second; i++) {
            blaze::band(M, i) = 0.0;
            blaze::band(M, -i) = 0.0;
        }
    }

    // Zero out blocks on the diagonal if requested
    if (_block_fill_size > 0) {
        unsigned int nb = kotekan::div_ceil(num_elements, _block_fill_size);
        for (unsigned int ii = 0; ii < nb; ii++) {
            unsigned int start = ii * _block_fill_size;
            unsigned int width = std::min(num_elements - start, _block_fill_size);
            blaze::submatrix(M, start, start, width, width) = 0.0;
        }
    }

    return blaze::declherm(M);
}
