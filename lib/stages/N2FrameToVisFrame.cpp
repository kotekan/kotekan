#include "N2FrameToVisFrame.hpp"

#include <time.h>               // for timespec, time_t, size_t
#include <gsl-lite.hpp>         // for span, span_iterator
#include <cassert>              // for assert
#include <complex>              // for complex, conj
#include <algorithm>            // for transform
#include <functional>           // for bind, function
#include <iterator>             // for back_insert_iterator, begin, end, back_inserter
#include <memory>               // for shared_ptr, __shared_ptr_access, dynamic_pointer_cast
#include <numeric>              // for iota
#include <tuple>                // for get, tuple

#include "Config.hpp"           // for Config
#include "N2FrameDesc.hpp"      // for N2FrameDesc
#include "N2FrameView.hpp"      // for N2FrameView
#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"           // for Buffer
#include "bufferContainer.hpp"  // for bufferContainer
#include "datasetState.hpp"     // for eigenvalueState, freqState, gatingState, inputState, meta...
#include "gateSpec.hpp"         // for gateSpec
#include "visBuffer.hpp"        // for VisFrameView
#include "visUtil.hpp"          // for prod_ctype, input_ctype, frameID, freq_ctype, modulo, par...
#include "Hash.hpp"             // for operator!=, Hash
#include "Telescope.hpp"        // for Telescope
#include "fmt.hpp"              // for compile_string_to_view
#include "kotekanLogging.hpp"   // for FATAL_ERROR, DEBUG, logLevel
#include "version.h"            // for get_git_commit_hash

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(n2FrameToVisFrame);

n2FrameToVisFrame::n2FrameToVisFrame(Config& config, const std::string& unique_name,
                                     bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&n2FrameToVisFrame::main_thread, this)),
    dm(datasetManager::instance()) {

    n2_buf = get_buffer("n2_buf");
    n2_buf->register_consumer(unique_name);

    vis_buf = get_buffer("vis_buf");
    vis_buf->register_producer(unique_name);

    // N2FrameView and VisFrameView have different data fields, so extract the basic
    // size parameters from the N2 buffer and compute the size of VisFrameView they correspond
    // to.
    // N2FrameDesc's are constructed on buffer creation (before stages) so this object will
    // exist with the correct data.
    const std::shared_ptr<const kotekan::N2FrameDesc> n2_frame_desc =
        n2_buf->require_frame_desc<kotekan::N2FrameDesc>();
    uint32_t n2_ne = n2_frame_desc->get_num_elements();
    uint32_t n2_np = n2_frame_desc->get_num_products();
    uint32_t n2_nev = n2_frame_desc->get_num_ev();
    size_t n2_conv_frame_size = VisFrameView::calculate_frame_size(n2_ne, n2_np, n2_nev);

    // Check the sizes are equivalent
    if (n2_conv_frame_size != vis_buf->frame_size) {
        FATAL_ERROR("Frame sizes between converted N2FrameView and VisFrameView must be identical, "
                    "but got {} (converted) and {}",
                    n2_conv_frame_size, vis_buf->frame_size);
    }

    fake_git_tag = config.get_default<std::string>(unique_name, "fake_git_tag", std::string());

    // Get everything we need for registering dataset states

    const auto& tel = Telescope::instance();

    // --> get metadata
    std::string instrument_name =
        config.get_default<std::string>(unique_name, "instrument_name", "chime");

    // Get the frequency IDs that are on this stream, check the config or just
    // assume all CHIME channels
    std::vector<uint32_t> freq_ids;
    if (config.exists(unique_name, "freq_ids")) {
        freq_ids = config.get<std::vector<uint32_t>>(unique_name, "freq_ids");
    } else {
        freq_ids.resize(tel.num_freq());
        std::iota(std::begin(freq_ids), std::end(freq_ids), 0);
    }

    // Create the frequency specification
    std::vector<std::pair<uint32_t, freq_ctype>> freqs;
    std::transform(std::begin(freq_ids), std::end(freq_ids), std::back_inserter(freqs),
                   [&tel](uint32_t id) -> std::pair<uint32_t, freq_ctype> {
                       return {id, {tel.to_freq_MHz(id), tel.freq_width_MHz(id)}};
                   });

    // The input specification from the config
    const auto& input_reorder = parse_reorder_default(config, unique_name);
    std::vector<input_ctype> inputs = std::get<1>(input_reorder);

    size_t num_elements = inputs.size();

    // Create the product specification
    std::vector<prod_ctype> prods;
    prods.reserve(num_elements * (num_elements + 1) / 2);
    for (uint16_t i = 0; i < num_elements; i++) {
        for (uint16_t j = i; j < num_elements; j++) {
            prods.push_back({i, j});
        }
    }

    // register base dataset states to prepare for getting dataset IDs for out frames
    register_base_dataset_states(instrument_name, freqs, inputs, prods);
}

n2FrameToVisFrame::~n2FrameToVisFrame() {}

void n2FrameToVisFrame::main_thread() {

    // base dataset_id of visAcummulate (before gating)
    dset_id_t base_dataset_id;
    // added state of eigenVis
    state_id_t evState_id;
    // incoming dataset_id
    dset_id_t ds_id_n2;
    // outgoing dataset_id (could be visibility or eigenvalues)
    dset_id_t ds_id_vis;

    // sanity checks
    uint32_t num_ev_first_time;

    frameID n2_buf_frame_id(n2_buf);
    frameID vis_buf_frame_id(vis_buf);
    bool first_time = true;
    while (!stop_thread) {

        if (n2_buf->wait_for_full_frame(unique_name, n2_buf_frame_id) == nullptr)
            return;
        const N2FrameView n2_frame(n2_buf, n2_buf_frame_id);

        if (vis_buf->wait_for_empty_frame(unique_name, vis_buf_frame_id) == nullptr)
            return;
        auto vis_frame = VisFrameView::create_frame_view(
            vis_buf, vis_buf_frame_id, n2_frame.num_elements, n2_frame.num_prod, n2_frame.num_ev);

        // items set in VisFrameView::fill_metadata
        vis_frame.freq_id = n2_frame.freq_id;
        const uint64_t fpga_seq = n2_frame.fpga_start_tick;
        const timespec ts = {.tv_sec = time_t(n2_frame.frame_start_time_ns / 1'000'000'000),
                             .tv_nsec = time_t(n2_frame.frame_start_time_ns % 1'000'000'000)};
        vis_frame.time = std::tuple(fpga_seq, ts);

        // items set in visAccumulate::initialise_output

        // dataset_id
        dset_id_t ds_id_n2_new = n2_frame.dataset_id;
        // in CHIME's visAccumulate the dataset id also changes due to state
        // changes via the REST endpoint for gated datasets. Since there is no
        // gating yet in N2Accumulate we do not worry about this here.
        if (first_time || (ds_id_n2_new != ds_id_n2)) {
            ds_id_n2 = ds_id_n2_new;

            // Register base dataset. If no dataset ID was was set in the incoming frame,
            // ds_id_in will be dset_id_t::null and thus cause a root dataset to
            // be registered.
            base_dataset_id = dm.add_dataset(base_dataset_states, ds_id_n2);
            DEBUG("Registered base dataset: {}", base_dataset_id);

            // no gating supported by N2Accumulate, only main visibility
            // from visAccumulate's maint_thread and register_gate_dataset
            const auto visSpec =
                gateSpec::create("uniform", "vis", kotekan::logLevel(_member_log_level));
            const state_id_t visState_id = dm.create_state<gatingState>(*visSpec).first;
            ds_id_vis = dm.add_dataset(visState_id, base_dataset_id);

            // eigenVis creates a new state by adding the number of eigenVectors
            if (vis_frame.num_ev > 0) {
                if (first_time) {
                    evState_id = dm.create_state<eigenvalueState>(vis_frame.num_ev).first;
                    num_ev_first_time = vis_frame.num_ev;
                } else {
                    (void)num_ev_first_time, /* pacify unused warning */
                        assert(vis_frame.num_ev == num_ev_first_time
                               && "Number of eigenvectors changed unexpectedly.");
                }
                ds_id_vis = dm.add_dataset(evState_id, ds_id_vis);
            }

            first_time = false;
        }
        vis_frame.dataset_id = ds_id_vis;

        vis_frame.fpga_seq_length = n2_frame.frame_length_fpga_ticks;

        // items set in visAccumulate::main_thread
        vis_frame.fpga_seq_total = n2_frame.n_valid_fpga_ticks;
        vis_frame.rfi_total = n2_frame.n_rfi_fpga_ticks;

        // CHIME and n2k use different conventions for the visiblity matrix,
        // with CHIME using the unusual choice V_ij = < x_i*(nu) x_j(nu) >
        // (CHIME_signs_conventions_note.pdf, Eq. (2)) with n2 using the
        // conventional V_ij = < x_i(nu) x*_j(nu) >.
        assert(n2_frame.vis.size() == vis_frame.vis.size());
        for (auto it_n2 = n2_frame.vis.begin(), it_vis = vis_frame.vis.begin();
             it_n2 != n2_frame.vis.end(); ++it_n2, ++it_vis)
            *it_vis = std::conj(*it_n2);

        assert(n2_frame.weight.size() == vis_frame.weight.size());
        for (auto it_n2 = n2_frame.weight.begin(), it_vis = vis_frame.weight.begin();
             it_n2 != n2_frame.weight.end(); ++it_n2, ++it_vis)
            *it_vis = *it_n2;

        assert(n2_frame.flags.size() == vis_frame.flags.size());
        for (auto it_n2 = n2_frame.flags.begin(), it_vis = vis_frame.flags.begin();
             it_n2 != n2_frame.flags.end(); ++it_n2, ++it_vis)
            *it_vis = *it_n2;

        assert(n2_frame.eval.size() == vis_frame.eval.size());
        for (auto it_n2 = n2_frame.eval.begin(), it_vis = vis_frame.eval.begin();
             it_n2 != n2_frame.eval.end(); ++it_n2, ++it_vis)
            *it_vis = *it_n2;

        assert(n2_frame.evec.size() == vis_frame.evec.size());
        for (auto it_n2 = n2_frame.evec.begin(), it_vis = vis_frame.evec.begin();
             it_n2 != n2_frame.evec.end(); ++it_n2, ++it_vis)
            *it_vis = std::conj(*it_n2);

        vis_frame.erms = n2_frame.erms;

        assert(n2_frame.gain.size() == vis_frame.gain.size());
        for (auto it_n2 = n2_frame.gain.begin(), it_vis = vis_frame.gain.begin();
             it_n2 != n2_frame.gain.end(); ++it_n2, ++it_vis)
            *it_vis = std::conj(*it_n2);

        vis_buf->mark_frame_full(unique_name, vis_buf_frame_id++);
        n2_buf->mark_frame_empty(unique_name, n2_buf_frame_id++);
    }
}

// from visAccumulate
void n2FrameToVisFrame::register_base_dataset_states(
    std::string& instrument_name, std::vector<std::pair<uint32_t, freq_ctype>>& freqs,
    std::vector<input_ctype>& inputs, std::vector<prod_ctype>& prods) {
    // weight calculation is hardcoded, so is the weight type name
    // TODO: check if this is still true
    const std::string weight_type = "inverse_var";
    const std::string git_tag = fake_git_tag.empty() ? get_git_commit_hash() : fake_git_tag;

    // create all the states
    base_dataset_states.push_back(dm.create_state<freqState>(freqs).first);
    base_dataset_states.push_back(dm.create_state<inputState>(inputs).first);
    base_dataset_states.push_back(dm.create_state<prodState>(prods).first);
    base_dataset_states.push_back(dm.create_state<eigenvalueState>(0).first);
    base_dataset_states.push_back(
        dm.create_state<metadataState>(weight_type, instrument_name, git_tag).first);
}
