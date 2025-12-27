#include "N2FrameView.hpp"

#include "FrameView.hpp" // for bind_span, bind_scalar, FrameView
#include "buffer.hpp"    // for Buffer

#include <assert.h> // for assert
#include <complex>  // for complex
#include <cstring>  // for memset, size_t

namespace {
std::shared_ptr<const kotekan::N2FrameDesc>
validate_desc(std::shared_ptr<const kotekan::FrameDesc> desc) {
    auto n2_desc = std::dynamic_pointer_cast<const kotekan::N2FrameDesc>(desc);
    if (!n2_desc) {
        throw std::runtime_error("N2FrameView: Buffer does not have a valid N2FrameDesc");
    }
    return n2_desc;
}
} // namespace

N2FrameView::N2FrameView(Buffer* buf, int frame_id) :

    FrameView(buf, frame_id),
    _metadata(std::static_pointer_cast<N2Metadata>(buf->metadata[frame_id])),
    _desc(validate_desc(buf->get_frame_description())),

    // Set the const refs to the structural metadata
    n2_layout(_desc->get_n2_layout()), num_elements(_desc->get_num_elements()),
    num_prod(_desc->get_num_products()), num_ev(_desc->get_num_ev()), nfreq(_metadata->nfreq),
    frame_layout(kotekan::N2FrameDesc::get_frame_layout(num_elements, num_ev, num_prod)),

    // Non-structural data
    freq_id(_metadata->freq_id), freq_MHz(_metadata->freq_MHz),
    abs_time_idx(_metadata->abs_time_idx),

    time_center_eop(_metadata->time_center_eop), bin_eop(_metadata->bin_eop),
    bin_start_ERA_deg(_metadata->bin_start_ERA_deg), bin_end_ERA_deg(_metadata->bin_end_ERA_deg),
    bin_start_LAST(_metadata->bin_start_LAST), bin_end_LAST(_metadata->bin_end_LAST),

    fpga_start_tick(_metadata->fpga_start_tick),
    frame_start_time_ns(_metadata->frame_start_time_ns),
    frame_length_fpga_ticks(_metadata->frame_length_fpga_ticks),
    n_valid_fpga_ticks(_metadata->n_valid_fpga_ticks),
    n_rfi_fpga_ticks(_metadata->n_rfi_fpga_ticks),

    vis(bind_span<N2::cfloat>(_frame, frame_layout[N2Field::vis])),
    weight(bind_span<float>(_frame, frame_layout[N2Field::weight])),
    flags(bind_span<float>(_frame, frame_layout[N2Field::flags])),
    eval(bind_span<float>(_frame, frame_layout[N2Field::eval])),
    evec(bind_span<N2::cfloat>(_frame, frame_layout[N2Field::evec])),
    emethod(bind_scalar<N2EigenMethod>(_frame, frame_layout[N2Field::emethod])),
    erms(bind_scalar<float>(_frame, frame_layout[N2Field::erms])),
    gain(bind_span<N2::cfloat>(_frame, frame_layout[N2Field::gain])) {

    assert(data_size() == buf->frame_size);
}

size_t N2FrameView::data_size() const {
    return kotekan::N2FrameDesc::calculate_frame_size(num_elements, num_ev, num_prod);
}

void N2FrameView::zero_frame() {
    // Fill data with zeros
    std::memset(_frame, 0, data_size());
}

N2FrameView N2FrameView::copy_frame(Buffer* buf_src, int frame_id_src, Buffer* buf_dest,
                                    int frame_id_dest) {
    FrameView::copy_frame(buf_src, frame_id_src, buf_dest, frame_id_dest);

    return N2FrameView(buf_dest, frame_id_dest);
}

void N2FrameView::copy_data(N2FrameView frame_to_copy_from, const std::set<N2Field>& skip_members) {
    auto copy_member = [&](N2Field member) { return (skip_members.count(member) == 0); };

    assert(nfreq == frame_to_copy_from.nfreq);

    if (copy_member(N2Field::vis) || copy_member(N2Field::weight) || copy_member(N2Field::flags)
        || copy_member(N2Field::evec) || copy_member(N2Field::gain)) {
        assert(num_elements == frame_to_copy_from.num_elements);
    }

    if (copy_member(N2Field::eval) || copy_member(N2Field::evec)) {
        assert(num_ev == frame_to_copy_from.num_ev);
    }

    if (copy_member(N2Field::vis))
        std::copy(frame_to_copy_from.vis.begin(), frame_to_copy_from.vis.end(), vis.begin());

    if (copy_member(N2Field::weight))
        std::copy(frame_to_copy_from.weight.begin(), frame_to_copy_from.weight.end(),
                  weight.begin());

    if (copy_member(N2Field::flags))
        std::copy(frame_to_copy_from.flags.begin(), frame_to_copy_from.flags.end(), flags.begin());

    if (copy_member(N2Field::eval))
        std::copy(frame_to_copy_from.eval.begin(), frame_to_copy_from.eval.end(), eval.begin());

    if (copy_member(N2Field::evec))
        std::copy(frame_to_copy_from.evec.begin(), frame_to_copy_from.evec.end(), evec.begin());

    if (copy_member(N2Field::erms))
        erms = frame_to_copy_from.erms;

    if (copy_member(N2Field::gain))
        std::copy(frame_to_copy_from.gain.begin(), frame_to_copy_from.gain.end(), gain.begin());
}

void N2FrameView::fill_prod_maps(std::vector<N2::prod_ctype>& prods) const {
    _desc->fill_prod_maps(prods);
}
