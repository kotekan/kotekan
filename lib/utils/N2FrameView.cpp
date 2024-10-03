#include "N2FrameView.hpp"
#include "buffer.hpp"        // for Buffer
#include "FrameView.hpp"     // for FrameView, bind_span

N2FrameView::N2FrameView(Buffer* buf, int frame_id) :

    FrameView(buf, frame_id), _metadata(std::static_pointer_cast<N2Metadata>(buf->metadata[id])),

    // Set the const refs to the structural metadata
    num_elements(_metadata->num_elements), num_prod(_metadata->num_prod), num_ev(_metadata->num_ev),
    frame_layout(get_frame_layout(_metadata->num_elements, _metadata->num_ev)),

    // Non-structural data
    freq_id(_metadata->freq_id),

    vis(bind_span<N2::cfloat>(_frame, frame_layout[N2Field::vis])),
    weight(bind_span<float>(_frame, frame_layout[N2Field::weight])),
    flags(bind_span<float>(_frame, frame_layout[N2Field::flags])),
    eval(bind_span<float>(_frame, frame_layout[N2Field::eval])),
    evec(bind_span<N2::cfloat>(_frame, frame_layout[N2Field::evec])),
    emethod(bind_scalar<N2EigenMethod>(_frame, frame_layout[N2Field::emethod])),
    erms(bind_scalar<float>(_frame, frame_layout[N2Field::erms])),
    gain(bind_span<N2::cfloat>(_frame, frame_layout[N2Field::gain])) {

    // assert frame size is correct
    assert(data_size() == buf->frame_size);
}

size_t N2FrameView::data_size() const {
    return calculate_frame_size(_metadata->num_elements, _metadata->num_ev);
}

void N2FrameView::zero_frame() {
    // Fill data with zeros
    std::memset(_frame, 0, data_size());
}

void N2FrameView::copy_data(N2FrameView frame_to_copy_from, const std::set<N2Field>& skip_members)
{
    auto copy_member = [&](N2Field member) { return (skip_members.count(member) == 0); };

    if (copy_member(N2Field::vis) || copy_member(N2Field::weight) || copy_member(N2Field::flags)
            || copy_member(N2Field::evec) || copy_member(N2Field::gain) ) {
        assert(num_elements == frame_to_copy_from.num_elements);
    }

    if (copy_member(N2Field::eval) || copy_member(N2Field::evec)) {
        assert(num_ev == frame_to_copy_from.num_ev);
    }

    if (copy_member(N2Field::vis))
        std::copy(frame_to_copy_from.vis.begin(), frame_to_copy_from.vis.end(), vis.begin());

    if (copy_member(N2Field::weight))
        std::copy(frame_to_copy_from.weight.begin(), frame_to_copy_from.weight.end(), weight.begin());

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


